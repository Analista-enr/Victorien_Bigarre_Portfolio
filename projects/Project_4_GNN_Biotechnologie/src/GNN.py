"""GATv2 multi-taches (12 sorties) pour Tox21.

Meme protocole que Gaia : scaffold split, AUC masquee par NaN, multi-seeds.
Difference : les entrees sont les GRAPHES PyG (Data.df), pas les fingerprints.

Corrections par rapport a la version precedente :
  1. encodeurs atome/liaison maison (plus de dependance OGB, dims lues dans PyG),
  2. test_loader construit sur idx_test (et non idx_train),
  3. sauvegarde complete (val + test + AUC par tache) en JSON et CSV,
  4. pos_weight calcule sur le TRAIN uniquement (pas de fuite),
  5. split scaffold "balanced_random" : la variance entre seeds mesure enfin
     la variance de split (le split deterministe reste disponible pour comparer a Gaia),
  6. garde-fous, dropout sur les features, AdamW, pooling mean||add||max,
     best_epoch, loss equilibree par tache.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sauvegarde sans affichage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader as GraphLoader
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils.smiles import e_map, x_map
from tqdm import tqdm

from dataset import TASK_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TASKS = len(TASK_NAMES)
FIG_DIR = Path("results/figures")

# Cardinalites des features categorielles, lues directement dans PyG
# (MoleculeNet/from_smiles : 9 features d'atome, 3 de liaison).
ATOM_DIMS = [len(v) for v in x_map.values()]
BOND_DIMS = [len(v) for v in e_map.values()]


# ====================================================================== modele
class CatEncoder(torch.nn.Module):
    """Features categorielles (entiers) -> vecteur dense : somme d'embeddings,
    un embedding par colonne. Sert pour les atomes ET les liaisons."""

    def __init__(self, dims, emb_dim: int):
        super().__init__()
        self.embs = torch.nn.ModuleList(torch.nn.Embedding(d, emb_dim) for d in dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(emb(x[:, i]) for i, emb in enumerate(self.embs))


class GAT(torch.nn.Module):
    """GATv2 multi-taches.

    - CatEncoder atomes / liaisons (les liaisons alimentent l'attention via edge_dim),
    - n_layers GATv2Conv : residu + LayerNorm + ReLU + dropout,
    - pooling concatene mean || add || max (garde l'information de taille),
    - tete MLP partagee -> 12 logits (pas de sigmoid : la loss s'en charge).
    """

    def __init__(self, hidden: int = 128, heads: int = 4, n_layers: int = 3,
                 n_tasks: int = N_TASKS, dropout: float = 0.2):
        super().__init__()
        assert hidden % heads == 0, "hidden doit etre multiple de heads"
        self.dropout = dropout
        self.atom_enc = CatEncoder(ATOM_DIMS, hidden)
        self.bond_enc = CatEncoder(BOND_DIMS, hidden)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=hidden,
                          dropout=dropout, add_self_loops=True)
            )  # concat des tetes -> hidden : le residu est donc possible
            self.norms.append(torch.nn.LayerNorm(hidden))

        self.head = torch.nn.Sequential(
            torch.nn.Linear(3 * hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, n_tasks),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_enc(x)                 # (n_atomes_total, hidden)
        e = self.bond_enc(edge_attr)         # (n_liaisons_total, hidden)

        for conv, norm in zip(self.convs, self.norms):
            h = h + conv(h, edge_index, edge_attr=e)   # residu
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        mol = torch.cat(
            [global_mean_pool(h, batch), global_add_pool(h, batch), global_max_pool(h, batch)],
            dim=1,
        )                                    # (batch_size, 3 * hidden)
        return self.head(mol)                # (batch_size, 12)


# ==================================================================== entraineur
class GATTrainer:
    """data : instance de Data (fournit df = graphes PyG nettoyes, labels,
    murcko_scaffold_smiles, scaffold_split)."""

    def __init__(self, data, params=None):
        self.DATA = data
        self.params = {
            "hidden": 128, "heads": 4, "n_layers": 3,
            "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4,
            "batch_size": 256, "epochs": 100, "patience": 15,
            "task_balanced_loss": True,   # moyenne par tache puis sur les taches
            "max_pos_weight": 50.0,
            "tensorboard": False,         # logs en direct : tensorboard --logdir results/tensorboard
        }
        if params:
            self.params.update(params)

        self.Y = np.asarray(self.DATA.labels, dtype=np.float64)   # (n, 12), NaN conserves
        self.groups = self._scaffold_groups()                     # calcule UNE fois

    # ------------------------------------------------------------------ splits
    def _scaffold_groups(self):
        """Familles de scaffolds : liste de tableaux d'indices."""
        scaffold_to_idx = {}
        for i, mol in enumerate(self.DATA.df):
            scaffold_to_idx.setdefault(self.DATA.murcko_scaffold_smiles(mol), []).append(i)
        return [np.asarray(g, dtype=np.int64) for g in scaffold_to_idx.values()]

    def balanced_scaffold_split(self, seed, frac_train=0.8, frac_val=0.1, frac_test=0.1):
        """Scaffold split randomise (facon Chemprop) : les gros scaffolds (> moitie
        d'un jeu val/test) vont d'abord en train, le reste est melange par le seed.
        Contrairement au split deterministe, le seed change reellement les
        jeux val/test, donc l'ecart-type entre seeds reflete la variance de split."""
        assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
        rng = np.random.default_rng(seed)
        n = len(self.DATA.df)
        train_size, val_size = int(frac_train * n), int(frac_val * n)
        small_cut = min(frac_val, frac_test) * n / 2

        big = [k for k, g in enumerate(self.groups) if len(g) > small_cut]
        small = [k for k, g in enumerate(self.groups) if len(g) <= small_cut]
        order = list(rng.permutation(big)) + list(rng.permutation(small))

        train, val, test = [], [], []
        for k in order:
            g = self.groups[k]
            if len(train) + len(g) <= train_size:
                train.extend(g)
            elif len(val) + len(g) <= val_size:
                val.extend(g)
            else:
                test.extend(g)

        train, val, test = (np.asarray(s, dtype=np.int64) for s in (train, val, test))
        assert len(val) > 0 and len(test) > 0, "val/test vide"
        assert len(train) + len(val) + len(test) == n
        assert not (set(train) & set(val)) and not (set(train) & set(test)) \
            and not (set(val) & set(test))
        return train, val, test

    def get_split(self, seed, split="balanced_random"):
        if split == "balanced_random":
            return self.balanced_scaffold_split(seed)
        if split == "deterministic":   # meme split que Gaia : comparaison directe
            return self.DATA.scaffold_split(self.DATA.df, seed=seed)
        raise ValueError(f"split inconnu : {split!r}")

    # ----------------------------------------------------------- donnees / loss
    def make_loader(self, idx, shuffle=False):
        """idx (np.int64) -> GraphLoader. Le masque NaN est gere par la loss :
        on garde TOUTES les molecules dans le batch."""
        graphs = [self.DATA.df[i] for i in idx]
        return GraphLoader(graphs, batch_size=self.params["batch_size"], shuffle=shuffle)

    def compute_pos_weight(self, idx_train):
        """pos_weight par tache, calcule sur le TRAIN uniquement."""
        Y = self.Y[idx_train]
        pos = np.nansum(Y, axis=0)
        neg = (~np.isnan(Y)).sum(axis=0) - pos
        w = np.clip(neg / np.maximum(pos, 1), 1.0, self.params["max_pos_weight"])
        return torch.tensor(w, dtype=torch.float32, device=DEVICE)

    def masked_bce(self, logits, y, pos_weight):
        """BCE multi-taches en ignorant les NaN. logits, y : (B, 12)."""
        mask = ~torch.isnan(y)
        y_clean = torch.where(mask, y, torch.zeros_like(y))
        loss = F.binary_cross_entropy_with_logits(
            logits, y_clean, pos_weight=pos_weight.expand_as(logits), reduction="none"
        )
        loss = loss * mask
        if self.params["task_balanced_loss"]:
            per_task = loss.sum(0) / mask.sum(0).clamp(min=1)   # moyenne par tache
            return per_task[mask.any(0)].mean()                 # puis sur les taches presentes
        return loss.sum() / mask.sum().clamp(min=1)             # moyenne sur les labels reels

    # ------------------------------------------------------------ boucles / AUC
    def run_epoch(self, model, loader, pos_weight, optimizer=None):
        """Un passage. optimizer=None -> evaluation (pas de gradient)."""
        train_mode = optimizer is not None
        model.train(train_mode)
        total_loss, all_scores, all_ys = 0.0, [], []

        with torch.set_grad_enabled(train_mode):
            for batch in loader:
                batch = batch.to(DEVICE)
                y = batch.y.view(-1, N_TASKS).float()      # (B, 12) avec NaN
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = self.masked_bce(logits, y, pos_weight)

                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                total_loss += loss.item() * batch.num_graphs
                all_scores.append(logits.detach().cpu().numpy())
                all_ys.append(y.cpu().numpy())

        return (total_loss / len(loader.dataset),
                np.concatenate(all_scores), np.concatenate(all_ys))

    @staticmethod
    def aucs_per_task(y, scores):
        """AUC par tache (NaN si moins de deux classes). Retourne (moyenne, par tache)."""
        aucs = []
        for t in range(y.shape[1]):
            mask = ~np.isnan(y[:, t])
            y_t = y[mask, t].astype(int)
            if len(np.unique(y_t)) < 2:
                aucs.append(np.nan)
                continue
            aucs.append(roc_auc_score(y_t, scores[mask, t]))
        aucs = np.array(aucs)
        mean = np.nanmean(aucs) if not np.all(np.isnan(aucs)) else np.nan
        return mean, aucs

    def train_one_split(self, idx_train, idx_val, idx_test, seed):
        """Entrainement complet sur UN split. Early stopping sur l'AUC val ;
        val et test sont evalues au MEILLEUR checkpoint (jamais au dernier)."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        p = self.params
        model = GAT(hidden=p["hidden"], heads=p["heads"], n_layers=p["n_layers"],
                    dropout=p["dropout"]).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

        pos_weight = self.compute_pos_weight(idx_train)     # train seulement
        train_loader = self.make_loader(idx_train, shuffle=True)
        val_loader = self.make_loader(idx_val)
        test_loader = self.make_loader(idx_test)

        writer = self._make_writer(seed)
        history = []   # une entree par epoch : sert aux courbes et au CSV
        best_val, best_state, best_epoch, wait = -np.inf, None, 0, 0
        pbar = tqdm(range(p["epochs"]), desc=f"seed {seed}", leave=False)
        for epoch in pbar:
            tr_loss, _, _ = self.run_epoch(model, train_loader, pos_weight, optimizer)
            val_loss, s_val, y_val = self.run_epoch(model, val_loader, pos_weight)
            val_auc, _ = self.aucs_per_task(y_val, s_val)
            lr = optimizer.param_groups[0]["lr"]     # lr utilise pendant cette epoch
            if not np.isnan(val_auc):
                scheduler.step(val_auc)

            history.append({"epoch": epoch + 1, "train_loss": tr_loss,
                            "val_loss": val_loss, "val_auc": val_auc, "lr": lr})
            if writer is not None:
                writer.add_scalar("loss/train", tr_loss, epoch + 1)
                writer.add_scalar("loss/val", val_loss, epoch + 1)
                writer.add_scalar("auc/val", val_auc, epoch + 1)
                writer.add_scalar("lr", lr, epoch + 1)

            if val_auc > best_val:                          # NaN > x vaut False
                best_val, best_epoch, wait = val_auc, epoch + 1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1

            pbar.set_postfix(train=f"{tr_loss:.3f}", val=f"{val_loss:.3f}",
                             auc=f"{val_auc:.3f}", best=f"{best_val:.3f}@{best_epoch}",
                             wait=f"{wait}/{p['patience']}")
            if wait >= p["patience"]:
                break
        pbar.close()
        if writer is not None:
            writer.close()

        if best_state is None:                              # AUC val toujours NaN
            raise RuntimeError("Aucun checkpoint valide : AUC val indefinie a chaque epoch.")

        model.load_state_dict(best_state)
        _, s_val, y_val = self.run_epoch(model, val_loader, pos_weight)
        _, s_te, y_te = self.run_epoch(model, test_loader, pos_weight)
        val_auc, val_per_task = self.aucs_per_task(y_val, s_val)
        test_auc, test_per_task = self.aucs_per_task(y_te, s_te)

        return {
            "val": {"auc_mean": val_auc, "auc_per_task": val_per_task},
            "test": {"auc_mean": test_auc, "auc_per_task": test_per_task},
            "best_epoch": best_epoch,
            "epochs_run": epoch + 1,
            "history": history,
        }

    # ------------------------------------------------------------ visualisation
    def _make_writer(self, seed):
        """Writer TensorBoard (optionnel) : suivi en direct pendant l'entrainement."""
        if not self.params.get("tensorboard"):
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("tensorboard absent (pip install tensorboard) : suivi desactive.")
            return None
        return SummaryWriter(log_dir=f"results/tensorboard/GATv2_seed{seed}")

    @staticmethod
    def plot_history(history, best_epoch, seed, out):
        """Courbes d'UN entrainement : loss train/val, AUC val, learning rate.
        La ligne verte marque le checkpoint retenu (meilleure AUC val)."""
        ep = np.array([h["epoch"] for h in history])
        tr = np.array([h["train_loss"] for h in history])
        vl = np.array([h["val_loss"] for h in history])
        auc = np.array([h["val_auc"] for h in history])
        lr = np.array([h["lr"] for h in history])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        ax.plot(ep, tr, label="train", color="#4C72B0")
        ax.plot(ep, vl, label="val", color="#C44E52")
        ax.axvline(best_epoch, color="green", linestyle="--", label=f"checkpoint (epoch {best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE masquee (ponderee)")
        ax.set_title("Loss")
        ax.legend()

        ax = axes[1]
        ax.plot(ep, auc, color="#55A868")
        ax.axvline(best_epoch, color="green", linestyle="--")
        if 1 <= best_epoch <= len(auc):
            ax.scatter([best_epoch], [auc[best_epoch - 1]], color="green", zorder=3,
                       label=f"best = {auc[best_epoch - 1]:.4f}")
            ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC ROC moyenne (val)")
        ax.set_title("AUC de validation")

        ax = axes[2]
        ax.semilogy(ep, lr, color="#8172B3")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning rate (ReduceLROnPlateau)")

        fig.suptitle(f"GATv2 - entrainement, seed {seed}")
        fig.tight_layout()
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Figure sauvegardee : {out}")

    @staticmethod
    def plot_all_seeds(histories, seed_list, out):
        """Superpose les seeds : permet de juger la stabilite de l'entrainement
        (surapprentissage = train descend pendant que val remonte)."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        cmap = plt.get_cmap("tab10")
        titles = ["Loss train", "Loss val", "AUC val (point = checkpoint)"]
        keys = ["train_loss", "val_loss", "val_auc"]

        for k, (seed, hist) in enumerate(zip(seed_list, histories)):
            ep = [h["epoch"] for h in hist]
            for ax, key in zip(axes, keys):
                ax.plot(ep, [h[key] for h in hist], color=cmap(k % 10),
                        alpha=0.85, label=f"seed {seed}")
            aucs = np.array([h["val_auc"] for h in hist])
            if not np.all(np.isnan(aucs)):
                b = int(np.nanargmax(aucs))
                axes[2].scatter([ep[b]], [aucs[b]], color=cmap(k % 10), zorder=3)

        for ax, title in zip(axes, titles):
            ax.set_xlabel("Epoch")
            ax.set_title(title)
        axes[0].set_ylabel("BCE masquee (ponderee)")
        axes[2].set_ylabel("AUC ROC moyenne (val)")
        axes[2].legend(fontsize=8)

        fig.suptitle("GATv2 - courbes d'entrainement, toutes seeds")
        fig.tight_layout()
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Figure sauvegardee : {out}")

    @staticmethod
    def save_history(histories, seed_list, path):
        """Historique complet (seed, epoch, losses, AUC val, lr) en CSV."""
        rows = [{"seed": s, **h} for s, hist in zip(seed_list, histories) for h in hist]
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Historique : {path}")

    # ----------------------------------------------------------------- pipeline
    def pipeline(self, seed_list=(0, 1, 2), params=None, split="balanced_random",
                 save_results=True, out_dir=Path("results"), plot=True):
        if params:
            self.params.update(params)
        seed_list = list(seed_list)

        val_means, test_means, val_pt, test_pt = [], [], [], []
        histories, best_epochs = [], []
        for seed in tqdm(seed_list, desc="GAT seeds"):
            tr, va, te = self.get_split(seed, split)
            res = self.train_one_split(tr, va, te, seed)
            histories.append(res["history"])
            best_epochs.append(res["best_epoch"])
            if plot:
                self.plot_history(res["history"], res["best_epoch"], seed,
                                  FIG_DIR / f"GATv2_{split}_seed{seed}_training.png")
            val_means.append(res["val"]["auc_mean"])
            test_means.append(res["test"]["auc_mean"])
            val_pt.append(res["val"]["auc_per_task"])
            test_pt.append(res["test"]["auc_per_task"])
            print(f"  seed {seed} : val {res['val']['auc_mean']:.4f} "
                  f"test {res['test']['auc_mean']:.4f} "
                  f"(meilleure epoch {res['best_epoch']}/{res['epochs_run']})")

        results = {
            "model_name": "GATv2",
            "split": split,
            "params": dict(self.params),
            "n_seeds": len(seed_list),
            "val": {"auc_mean": np.nanmean(val_means),
                    "auc_mean_std": np.nanstd(val_means),
                    "auc_matrix": np.stack(val_pt)},
            "test": {"auc_mean": np.nanmean(test_means),
                     "auc_mean_std": np.nanstd(test_means),
                     "auc_matrix": np.stack(test_pt)},
        }
        print(f"\nGATv2 ({split}) | val {results['val']['auc_mean']:.4f} "
              f"+/- {results['val']['auc_mean_std']:.4f} | "
              f"test {results['test']['auc_mean']:.4f} "
              f"+/- {results['test']['auc_mean_std']:.4f}")

        results["histories"] = histories
        results["best_epochs"] = best_epochs

        if plot:
            self.plot_all_seeds(histories, seed_list,
                                FIG_DIR / f"GATv2_{split}_training_all_seeds.png")
        if save_results:
            self.save(results, out_dir)
            self.save_history(histories, seed_list,
                              Path(out_dir) / f"GATv2_{split}_history.csv")
        return results

    # ---------------------------------------------------------------- sauvegarde
    @staticmethod
    def save(results, out_dir=Path("results")):
        """2 fichiers : GATv2_{split}_summary.json (val + test + params) et
        GATv2_{split}_per_task.csv (AUC par tache, moyenne +/- std sur seeds)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"GATv2_{results['split']}"

        df = pd.DataFrame({"task": TASK_NAMES})
        for scope in ("val", "test"):
            mat = results[scope]["auc_matrix"]
            df[f"{scope}_auc"] = np.nanmean(mat, axis=0)
            df[f"{scope}_auc_std"] = np.nanstd(mat, axis=0)
        df.to_csv(out_dir / f"{tag}_per_task.csv", index=False)

        summary = {
            "model_name": results["model_name"],
            "split": results["split"],
            "n_seeds": results["n_seeds"],
            "params": results["params"],
            "per_task_csv": f"{tag}_per_task.csv",
        }
        for scope in ("val", "test"):
            summary[scope] = {
                "auc_mean": float(results[scope]["auc_mean"]),
                "auc_mean_std": float(results[scope]["auc_mean_std"]),
                "auc_matrix": results[scope]["auc_matrix"].tolist(),
            }
        path = out_dir / f"{tag}_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Resume : {path}")
        return path



