from itertools import product
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC, LinearSVC
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from tqdm import tqdm

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
        self.df = self.load_tox21()  # dataset brut
        self.features = self.build_feature_matrix(self.df)  # dataset fingerprint
        self.gram_tanimoto = self.tanimoto_kernel(self.features, self.features)
        self.labels = self.stack_labels(self.df)
        self.EDA_resume = self.EDA_statistique(self.df)  # resume of dataset description
        self.scaffold_distribution(self.df)

    # ------------------------------------------------------------- chargement dataset
    def load_tox21(self, root: str = "data/") -> MoleculeNet:
        """Tox21 via PyG : telecharge + cache au premier appel (dossier data/, ignore par git).
        Chaque molecule : x (atomes), edge_index (liaisons), y (12 taches, NaN = non teste),
        smiles."""
        dataset = MoleculeNet(root=root, name="Tox21")
        print(f"Tox21 : {len(dataset)} molecules")
        return dataset

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
            np.float64(0.709899967680543),
            np.float64(0.7612540496367751),
            np.float64(0.7559075256891493),
            np.float64(0.6851635913663012),
            np.float64(0.6223900607877015),
            np.float64(0.6939021609867735),
            np.float64(0.7169902903142342),
            np.float64(0.6727488383005039),
            np.float64(0.7318760431698946),
            np.float64(0.6395327537240513),
            np.float64(0.7571450724143143),
            np.float64(0.6774324579321013),
        ]
        df["AUC_SVM_Tanimoto"] = [
            np.float64(0.7299569978469196),
            np.float64(0.8269831053079966),
            np.float64(0.8406611936919566),
            np.float64(0.775625265422066),
            np.float64(0.677631678692199),
            np.float64(0.7376974139554984),
            np.float64(0.7793807461255183),
            np.float64(0.7702261897930408),
            np.float64(0.8139151807550533),
            np.float64(0.7444009290046436),
            np.float64(0.8511195995310434),
            np.float64(0.7702090311260599),
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

    def scaffold_distribution(self, dataset,
                              out: Path = FIG_DIR / "scaffold_distribution.png"):
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
            "top10_share": np.sort(sizes)[-10:].sum() / len(dataset),  # part des 10 plus grosses
        }

        # -- 3. figure --
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        # volet 1 : histogramme des tailles (log-x : sinon la longue trainee ecrase tout)
        ax1.hist(np.log10(sizes), bins=40, color="#4C72B0", edgecolor="white")
        ax1.set_xlabel("log10(taille de la famille)")
        ax1.set_ylabel("Nombre de familles")
        ax1.set_title(f"{stats['n_scaffolds']} familles de scaffolds")
        ax1.axvline(np.log10(np.median(sizes)), color="red", linestyle="--",
                    label=f"mediane = {stats['median_family_size']:.0f}")
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
