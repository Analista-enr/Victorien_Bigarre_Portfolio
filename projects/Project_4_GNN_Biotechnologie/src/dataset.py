from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.datasets import MoleculeNet

matplotlib.use("Agg")  # backend non interactif : sauvegarde sans affichage

TASK_NAMES = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "NR-Aromatase",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Generateur Morgan instancie UNE fois (nouvelle API RDKit, plus de deprecation)
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class Data:
    def __init__(self):
        self.df, self.df_report = self.clean_dataset(
            self.load_tox21()
        )  # dataset nettoyé
        self.features = self.build_feature_matrix(self.df)  # dataset fingerprint
        self.gram_tanimoto = self.tanimoto_kernel(self.features, self.features)
        self.labels = self.stack_labels(self.df)
        # self.EDA_resume = self.EDA_statistique(self.df)  # resume of dataset description
        # self.scaffold_distribution(self.df)

    # ------------------------------------------------------------- chargement dataset
    def load_tox21(self, root: str = "data/") -> MoleculeNet:
        """Tox21 via PyG : telecharge + cache au premier appel (dossier data/, ignore par git).
        Chaque molecule : x (atomes), edge_index (liaisons), y (12 taches, NaN = non teste),
        smiles."""
        dataset = MoleculeNet(root=root, name="Tox21")
        print(f"Tox21 : {len(dataset)} molecules")
        return dataset

    def clean_dataset(self, dataset) -> tuple:
        """Nettoyage AVANT tout le reste (split, features).

        1. SMILES invalides (MolFromSmiles -> None) : supprimes.
        2. Doublons (par SMILES canonique) :
           - labels identiques            -> une copie conservee
           - conflit 0/1 sur une tache    -> composé supprimé
           - annotations complementaires -> fusionnees (le non-NaN gagne)

        Retourne (dataset_clean, rapport: dict).
        """
        kept, seen = [], {}  # seen: canonique -> index dans kept
        n_invalid = n_dup_identical = n_conflict = n_merged = 0

        for d in dataset:
            mol = Chem.MolFromSmiles(d.smiles)
            if mol is None:
                n_invalid += 1
                continue
            canon = Chem.MolToSmiles(mol)  # la CLE : forme canonique

            if canon not in seen:
                seen[canon] = len(kept)
                kept.append(d)
                continue

            # --- doublon : comparer les labels ---
            first = kept[seen[canon]]
            y_new = np.asarray(d.labels)  # (12,) avec NaN
            y_old = np.asarray(first.labels)

            both = ~np.isnan(y_new) & ~np.isnan(y_old)  # taches annotées des 2 cotes
            conflict = both & (y_new != y_old)  # 0 vs 1 : contradiction

            if conflict.any():
                n_conflict += 1
                kept[seen[canon]] = None  # marqué : on supprimera
                # (le doublon courant n'est pas garde non plus)
            elif (
                (y_new[~np.isnan(y_new)] == y_old[~np.isnan(y_new)]).all()
                and np.isnan(y_old[both]).any() is False
                and (y_new[both] == y_old[both]).all()
            ):
                n_merged += 1  # fusion : combler les NaN de la copie garde
                fill = ~np.isnan(y_new) & np.isnan(y_old)
                if fill.any():
                    first.labels[fill] = y_new[fill]
            else:
                n_merged += 1
                fill = ~np.isnan(y_new) & np.isnan(y_old)
                if fill.any():
                    first.labels[fill] = y_new[fill]

        kept = [d for d in kept if d is not None]
        # renumerote seen inutile : kept est deja le dataset final

        report = {
            "n_initial": len(dataset),
            "n_invalid_smiles": n_invalid,
            "n_conflicting_duplicates_removed": n_conflict,
            "n_duplicates_merged": n_merged,
            "n_final": len(kept),
        }
        # for k, v in report.items():
        #     print(f"{k:>35} : {v}")
        return kept, report

    # ------------------------------------------------------------- vectorisation
    def murcko_scaffold_smiles(self, data) -> str:
        """SMILES canonique du scaffold Bemis-Murcko"""
        mol = Chem.MolFromSmiles(data.smiles)
        if mol is None:
            return ""
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))

    def smiles_to_fingerprint(self, smiles: str) -> np.ndarray:
        """ECFP4 : SMILES -> vecteur binaire (n_bits,). Sparse, sans fit -> pas de fuite."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"SMILES invalide : {smiles!r}")
        fp = MORGAN_GEN.GetFingerprint(mol)
        return np.asarray(fp, dtype=np.float32)

    def build_feature_matrix(self, dataset) -> np.ndarray:
        """Tout le dataset -> X (n_molecules, n_bits). Une seule fois, reutilise partout."""
        X = np.stack([self.smiles_to_fingerprint(d.smiles) for d in dataset])
        print(f"Features : X {X.shape}, densite {X.mean():.4f}")
        return X

    def tanimoto_kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Similarite Tanimoto vectorisee entre deux ensembles de fingerprints.
        K(i,j) = |Ai & Bj| / |Ai | Bj| - |Ai & Bj|   (bits communs / union des bits)"""
        inter = A @ B.T
        norm_a = A.sum(axis=1, keepdims=True)
        norm_b = B.sum(axis=1, keepdims=True).T
        union = norm_a + norm_b - inter
        return inter / union

    def stack_labels(self, dataset) -> np.ndarray:
        """y de PyG (n, 1, 12) -> Y (n, 12) avec NaN conserves."""
        return torch.stack([d.y for d in dataset]).squeeze(1).numpy()

    def EDA_statistique(self, dataset) -> pd.DataFrame:
        """Analyse par tache : taille du dataset annote, desequilibre et
        concentration des positifs par famille de scaffold.

        Retourne un DataFrame (12 lignes) :
          task, n_annotated, n_pos, pos_rate, n_scaffolds_pos,
          top3_share (part des positifs dans les 3 plus grosses familles).
        """
        # -- extraction directe depuis le dataset (plus d'argument Y externe) --
        Y = self.stack_labels(dataset)  # (n_mol, 12), NaN conserves

        # -- familles de scaffolds globales (identiques au split) --
        scaffold_to_idx = {}
        for i in range(len(dataset)):
            scaffold_to_idx.setdefault(
                self.murcko_scaffold_smiles(dataset[i]), []
            ).append(i)

        rows = []
        for t, name in enumerate(TASK_NAMES):
            y_t = Y[:, t]
            annotated = ~np.isnan(y_t)
            n_annotated = int(annotated.sum())
            pos_idx = np.where(y_t == 1)[0]  # indices des positifs

            if len(pos_idx) == 0:  # tache sans aucun positif
                rows.append(
                    {
                        "task": name,
                        "n_annotated": n_annotated,
                        "n_pos": 0,
                        "pos_rate": 0.0,
                        "n_scaffolds_pos": 0,
                        "top3_share": np.nan,
                    }
                )
                continue

            # familles des molécules positives (une famille peut apparaître plusieurs
            # fois si plusieurs positifs la partagent — c'est justement le signal)
            pos_family_sizes = [
                len(scaffold_to_idx[self.murcko_scaffold_smiles(dataset[i])])
                for i in pos_idx
            ]
            top3_share = float(np.sort(pos_family_sizes)[-3:].sum() / len(pos_idx))

            rows.append(
                {
                    "task": name,
                    "n_annotated": n_annotated,  # molécules testées sur cette tâche
                    "n_annotated_share": 100 * round(n_annotated / 7823, 2),
                    "n_pos": len(pos_idx),
                    "n_scaffolds_pos": len(
                        set(self.murcko_scaffold_smiles(dataset[i]) for i in pos_idx)
                    ),
                    "pos_rate (%) ": round(100 * len(pos_idx) / n_annotated, 2),
                    "top3_share (%)": round(top3_share, 2),
                }
            )
        df = pd.DataFrame(rows)
        df["AUC_SVM"] = [
            np.float64(0.7637564381759371),
            np.float64(0.8346816508577228),
            np.float64(0.8383128855542126),
            np.float64(0.7379983135662949),
            np.float64(0.7036555204015106),
            np.float64(0.7790513681066347),
            np.float64(0.7516067023086201),
            np.float64(0.7392137017117733),
            np.float64(0.7877337811030495),
            np.float64(0.719773581848089),
            np.float64(0.8325107776183542),
            np.float64(0.751375456204357),
        ]
        df["AUC_SVM_Tanimoto"] = [
            np.float64(0.76636379624712),
            np.float64(0.8508728081174223),
            np.float64(0.8511887392169402),
            np.float64(0.7576396341109417),
            np.float64(0.7127062707483469),
            np.float64(0.78081826858475),
            np.float64(0.773847006025392),
            np.float64(0.7706591668322907),
            np.float64(0.8018759451236958),
            np.float64(0.7385897030080996),
            np.float64(0.8483839917277287),
            np.float64(0.7801640207335038),
        ]
        df["AUC_LGBM"] = [
            np.float64(0.7148657893720418),
            np.float64(0.811033341218333),
            np.float64(0.8271887973309026),
            np.float64(0.7522284341744109),
            np.float64(0.6753752610872512),
            np.float64(0.7300185968333347),
            np.float64(0.7382615477061526),
            np.float64(0.7477153334130355),
            np.float64(0.789480199880695),
            np.float64(0.7326278490200222),
            np.float64(0.823201484145039),
            np.float64(0.7536326493010692),
        ]

        df = df.set_index("task")

        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        return df

    def extract(
        self,
        df: pd.DataFrame,
        title: str = "Matrice de corrélation",
        out: Path = FIG_DIR / "correlation_heatmap.png",
    ):
        """Heatmap de correlation (RdBu), divergente autour de 0.
        Rouge = correlation positive, bleu = negative (centree sur zero)."""
        corr = df.corr(numeric_only=True)

        fig, ax = plt.subplots(
            figsize=(2 + 0.6 * len(corr.columns), 1.5 + 0.5 * len(corr.columns))
        )
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

        # etiquettes : noms de variables sur les axes
        labels = corr.columns
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)

        # annotations : valeur dans chaque cellule
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    f"{corr.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
                )

        fig.colorbar(im, ax=ax, label="Corrélation de Pearson", shrink=0.8)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Figure sauvegardée : {out}")
        return fig

    def scaffold_distribution(
        self, dataset, out: Path = FIG_DIR / "scaffold_distribution.png"
    ):
        """Distribution des molecules par scaffold.

        Retourne un DataFrame descriptif (une ligne par famille) et produit
        une figure a double volet :
          - gauche : histogramme des tailles de familles (echelle log!)
          - droite  : courbe de Pareto (part cumulee des molecules selon
                      les k plus grosses familles) — LA lecture cle.
        """
        # -- 1. familles : scaffold -> liste des indices de ses molecules --
        scaffold_to_idx = {}
        for i in range(len(dataset)):
            scaffold_to_idx.setdefault(
                self.murcko_scaffold_smiles(dataset[i]), []
            ).append(i)

        sizes = np.array([len(v) for v in scaffold_to_idx.values()])

        # -- 2. statistiques descriptives --
        stats = {
            "n_molecules": len(dataset),
            "n_scaffolds": len(scaffold_to_idx),
            "mean_family_size": sizes.mean(),
            "median_family_size": np.median(sizes),
            "max_family_size": sizes.max(),
            "n_singletons": int((sizes == 1).sum()),  # familles d'UNE molecule
            "top10_share": np.sort(sizes)[-10:].sum()
            / len(dataset),  # part des 10 plus grosses
        }

        # -- 3. figure --
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        # volet 1 : histogramme des tailles (log-x : sinon la longue trainee ecrase tout)
        ax1.hist(np.log10(sizes), bins=40, color="#4C72B0", edgecolor="white")
        ax1.set_xlabel("log10(taille de la famille)")
        ax1.set_ylabel("Nombre de familles")
        ax1.set_title(f"{stats['n_scaffolds']} familles de scaffolds")
        ax1.axvline(
            np.log10(np.median(sizes)),
            color="red",
            linestyle="--",
            label=f"mediane = {stats['median_family_size']:.0f}",
        )
        ax1.legend()

        # volet 2 : Pareto — part cumulee des molecules couverte par les k plus grosses familles
        sizes_sorted = np.sort(sizes)[::-1]
        cumul = np.cumsum(sizes_sorted) / len(dataset)
        ax2.plot(np.arange(1, len(sizes_sorted) + 1), cumul, color="#C44E52")
        ax2.axhline(0.8, color="gray", linestyle=":", label="80% des molecules")
        ax2.set_xlabel("k plus grosses familles")
        ax2.set_ylabel("Part cumulee des molecules")
        ax2.set_title("Courbe de Pareto : concentration du dataset")
        ax2.legend()

        fig.suptitle("Distribution des molecules par scaffold (Tox21)")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Figure sauvegardee : {out}")

        for k, v in stats.items():
            print(f"{k:>22} : {v}")
        return stats

    def scaffold_split(
        self, dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed: int = 0
    ):
        """Scaffold split facon MoleculeNet/DeepChem :
        groupes tries par taille decroissante, chaque groupe affecte au split
        avec le plus gros quota restant. Garantit val/test non vides meme avec
        de tres gros scaffolds. Le seed brise les egalites de taille."""
        assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
        rng = np.random.default_rng(seed)

        scaffold_to_idx = {}
        for i in range(len(dataset)):
            scaffold_to_idx.setdefault(
                self.murcko_scaffold_smiles(dataset[i]), []  # self, pas self.DATA
            ).append(i)
        groups = [np.asarray(g, dtype=np.int64) for g in scaffold_to_idx.values()]

        # ETAPE 1 : tri par taille decroissante ; le seed melange a taille egale
        # (les familles de 1-2 molecules sont nombreuses : c'est la que le seed agit)
        sizes = np.array([len(g) for g in groups])
        noise = rng.random(len(groups))  # bruit uniforme, ordre fixe par seed
        order = np.lexsort((noise, -sizes))  # taille desc, puis bruit asc
        groups = [groups[k] for k in order]

        # ETAPE 2 : remplissage par deficit maximal
        n = len(dataset)
        quotas = {"train": frac_train * n, "val": frac_val * n, "test": frac_test * n}
        splits = {"train": [], "val": [], "test": []}

        for group in groups:
            # split avec le plus grand quota restant (jamais negatif => jamais vide)
            name = max(quotas, key=lambda k: quotas[k])
            splits[name].append(group)
            quotas[name] -= len(group)

        train = np.concatenate(splits["train"])
        val = np.concatenate(splits["val"])
        test = np.concatenate(splits["test"])

        # garde-fous
        assert (
            train.dtype == np.int64 and val.dtype == np.int64 and test.dtype == np.int64
        )
        assert len(val) > 0 and len(test) > 0, (
            f"val/test vide : {len(val)}/{len(test)} — augmentez frac_val/frac_test "
            f"ou verifiez qu'aucun scaffold ne domine le dataset"
        )
        assert len(train) + len(val) + len(test) == n
        assert len(np.intersect1d(train, test)) == 0

        return train, val, test