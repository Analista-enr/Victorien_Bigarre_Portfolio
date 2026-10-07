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
        df["AUC_SVM"] = np.array(
            [0.77, 0.84, 0.85, 0.74, 0.71, 0.81, 0.76, 0.73, 0.8, 0.74, 0.84, 0.77]
        ) - ([0.014 for i in range(12)])
        df["AUC_SVM_Tanimoto"] = [
            0.76,
            0.84,
            0.84,
            0.75,
            0.71,
            0.78,
            0.76,
            0.765,
            0.8,
            0.74,
            0.85,
            0.78,
        ]
        df["AUC_LGBM"] = [
            0.67,
            0.79,
            0.84,
            0.71,
            0.64,
            0.71,
            0.695,
            0.72,
            0.73,
            0.71,
            0.83,
            0.74,
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
