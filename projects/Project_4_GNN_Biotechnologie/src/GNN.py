import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader as GraphLoader

# --- encodeurs atomes/liaisons : 3 emplacements possibles selon version PyG ---
try:
    from torch_geometric.nn import AtomEncoder, BondEncoder      # PyG <= 2.3
except ImportError:
    try:
        from torch_geometric.graphgym.models.encoder import AtomEncoder, BondEncoder  # PyG recent
    except ImportError:
        AtomEncoder = BondEncoder = None                         # ogb absent -> fallback manuel

# --- GATv2Conv / pool : ceux-ci n'ont JAMAIS demenage ---
from torch_geometric.nn import GATv2Conv, global_mean_pool
from tqdm import tqdm
from types import SimpleNamespace


from dataset import TASK_NAMES  # vos 12 noms de taches

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GAT(torch.nn.Module):
    """GATv2 multi-taches (12 sorties) pour Tox21.

    - AtomEncoder : les 9 features categorielles d'atome -> vecteur dense,
    - 3 couches GATv2Conv avec tete d'attention, residus et normalisation,
    - BondEncoder passe le type de liaison a l'attention (edge_dim),
    - global_mean_pool : resume du graphe -> 1 vecteur par molecule,
    - tete MLP partagee -> 12 logits (une par tache).
    """

    def __init__(self, hidden: int = 128, heads: int = 4,
                 n_layers: int = 3, n_tasks: int = 12, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        # features brutes de PyG (entiers) -> embeddings
        self.atom_enc = BondEncoder(emb_dim=hidden)

        # 3 couches : hidden -> (heads * out) par couche, avec residus
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden
            out_dim = hidden // heads          # pour que concat des tete = hidden
            self.convs.append(GATv2Conv(in_dim, out_dim, heads=heads,
                                        edge_dim=hidden, dropout=dropout,
                                        add_self_loops=True))
            self.norms.append(torch.nn.LayerNorm(hidden))
            # pas besoin de proj d'entree : in_dim == hidden des la 2e couche

        # tete de prediction multi-taches, partagee entre les 12 taches
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, n_tasks),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. encodage des atomes et des liaisons
        h = self.atom_enc(x)                    # (n_atoms_total, hidden)
        e = self.bond_enc(SimpleNamespace(edge_attr=edge_attr)) if hasattr(self.atom_enc, "bond_encoder") else edge_attr

        # 2. message passing attentionnel
        for conv, norm in zip(self.convs, self.norms):
            h = h + conv(h, edge_index, edge_attr=e)   # residu : gradient stable
            h = norm(h)                                 # normalisation par couche
            h = F.relu(h)

        # 3. une molécule = un vecteur (moyenne de ses atomes)
        mol = global_mean_pool(h, batch)        # (batch_size, hidden)

        # 4. 12 logits (pas de sigmoid : BCEWithLogitsLoss s'en charge)
        return self.head(mol)                  # (batch_size, 12)

class GATTrainer:
    """Meme protocole que Gaia : scaffold split, AUC masquee par NaN,
    multi-seeds. Difference : les entrees sont les GRAPHES PyG, pas X."""

    def __init__(self, data, params=None):
        """
        data : instance de Data (votre classe) — fournit df (liste de
        graphes PyG nettoyes), labels, murcko_scaffold_smiles, scaffold_split.
        """
        self.DATA = data
        self.params = {
            "hidden": 128, "heads": 4, "n_layers": 3,
            "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-5,
            "batch_size": 256, "epochs": 100, "patience": 15,
        }
        if params:
            self.params.update(params)

        self.Y = self.DATA.stack_labels(self.DATA.df)   # (n, 12) float64 avec NaN

        # pos_weight par tache : contrebalance le desequilibre des classes
        # (meme role que class_weight="balanced" dans vos SVM)
        masks = ~np.isnan(self.Y)
        pos = np.nansum(self.Y, axis=0)
        neg = masks.sum(axis=0) - pos
        self.pos_weight = torch.tensor(
            np.clip(neg / np.maximum(pos, 1), 1, 50), dtype=torch.float32
        ).to(DEVICE)

    def make_loaders(self, idx_train, idx_val):
        """idx (np.int64) -> GraphLoader. Le masque NaN est gere par la loss,
        pas par le loader : on garde TOUTES les molecules dans le batch."""
        ds = [self.DATA.df[i] for i in idx_train]
        dl = [self.DATA.df[i] for i in idx_val]
        train_loader = GraphLoader(ds, batch_size=self.params["batch_size"], shuffle=True)
        val_loader = GraphLoader(dl, batch_size=self.params["batch_size"], shuffle=False)
        return train_loader, val_loader
    def masked_bce(self, logits, y):
        """BCE par tache en ignorant les NaN (le point cle du multi-tache Tox21).

        logits, y : (B, 12). Retourne un scalaire."""
        mask = ~torch.isnan(y)                          # (B, 12) bool
        y_clean = torch.where(mask, y, torch.zeros_like(y))
        loss = F.binary_cross_entropy_with_logits(
            logits, y_clean,
            pos_weight=self.pos_weight.expand_as(logits),  # (12,) -> (B,12)
            reduction="none",
        )
        return loss[mask].mean()                        # moyenne sur les labels REELS
    def run_epoch(self, model, loader, optimizer=None):
        """Un passage. optimizer=None -> evaluation (pas de gradient)."""
        train_mode = optimizer is not None
        model.train(train_mode)
        total_loss = 0.0
        all_scores, all_ys = [], []

        for batch in loader:
            batch = batch.to(DEVICE)
            y = batch.y.squeeze(1).float()              # (B, 12) avec NaN
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            loss = self.masked_bce(logits, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # stabilite GAT
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            all_scores.append(logits.detach().cpu().numpy())
            all_ys.append(y.cpu().numpy())

        return (total_loss / len(loader.dataset),
                np.concatenate(all_scores), np.concatenate(all_ys))
    @staticmethod
    def aucs_per_task(y, scores):
        """AUC par tache : meme garde-fou que Gaia.evaluate
        (tache sans deux classes -> NaN). Retourne (auc_mean, auc_par_tache)."""
        from sklearn.metrics import roc_auc_score
        aucs = []
        for t in range(y.shape[1]):
            mask = ~np.isnan(y[:, t])
            y_t = y[mask, t].astype(int)
            if len(np.unique(y_t)) < 2:
                aucs.append(np.nan)
                continue
            aucs.append(roc_auc_score(y_t, scores[mask, t]))
        aucs = np.array(aucs)
        return np.nanmean(aucs), aucs
    def train_one_split(self, idx_train, idx_val, idx_test, seed):
        """Entrainement complet sur UN split. Retourne le dict de scores
        {val, test} du MEILLEUR checkpoint (early stopping sur val AUC)."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = GAT(hidden=self.params["hidden"], heads=self.params["heads"],
                    n_layers=self.params["n_layers"],
                    dropout=self.params["dropout"]).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=self.params["lr"],
                          weight_decay=self.params["weight_decay"])
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

        train_loader, val_loader = self.make_loaders(idx_train, idx_val)
        test_loader, _ = self.make_loaders(idx_train, idx_test)  # meme batch size

        best_val, best_state, patience = -1.0, None, 0
        for epoch in range(self.params["epochs"]):
            tr_loss, _, _ = self.run_epoch(model, train_loader, optimizer)
            with torch.no_grad():
                _, s_val, y_val = self.run_epoch(model, val_loader)
            val_auc, _ = self.aucs_per_task(y_val, s_val)
            scheduler.step(val_auc)

            if val_auc > best_val:
                best_val, patience = val_auc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.params["patience"]:
                    break

        # evaluation finale au meilleur checkpoint (jamais au dernier !)
        model.load_state_dict(best_state)
        with torch.no_grad():
            _, s_val, y_val = self.run_epoch(model, val_loader)
            _, s_te, y_te = self.run_epoch(model, test_loader)
        val_auc, val_per_task = self.aucs_per_task(y_val, s_val)
        test_auc, test_per_task = self.aucs_per_task(y_te, s_te)

        return {"val": {"auc_mean": val_auc, "auc_per_task": val_per_task},
                "test": {"auc_mean": test_auc, "auc_per_task": test_per_task},
                "epochs_run": epoch + 1}

    def pipeline(self, seed_list=(0, 1, 2), params=None, save_results=True):
        if params:
            self.params.update(params)

        val_means, test_means, test_per_task_all = [], [], []
        for seed in tqdm(seed_list, desc="GAT seeds"):
            tr, va, te = self.DATA.scaffold_split(self.DATA.df, seed=seed)
            res = self.train_one_split(tr, va, te, seed)
            val_means.append(res["val"]["auc_mean"])
            test_means.append(res["test"]["auc_mean"])
            test_per_task_all.append(res["test"]["auc_per_task"])
            print(f"  seed {seed} : val {res['val']['auc_mean']:.4f} "
                  f"test {res['test']['auc_mean']:.4f} ({res['epochs_run']} epochs)")

        results = {
            "model_name": "GATv2",
            "params": {k: v for k, v in self.params.items() if k != "epochs"},
            "n_seeds": len(seed_list),
            "val": {"auc_mean": np.nanmean(val_means),
                    "auc_mean_std": np.nanstd(val_means)},
            "test": {"auc_mean": np.nanmean(test_means),
                     "auc_mean_std": np.nanstd(test_means),
                     "auc_matrix": np.stack(test_per_task_all)},
        }
        if save_results:
            json.dumps(results["params"])  # verif serialisable, puis :
            with open(Path("results") / "GATv2_summary.json", "w") as f:
                json.dump({k: v for k, v in results.items() if k != "test"}, f, indent=2)
        return results