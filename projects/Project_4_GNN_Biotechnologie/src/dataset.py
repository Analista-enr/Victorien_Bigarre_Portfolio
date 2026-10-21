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
                self.murcko_scaffold_smiles(dataset[i]), []
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
