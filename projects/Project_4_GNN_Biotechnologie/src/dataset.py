"""Donnees Tox21 : chargement, nettoyage, features, noyau Tanimoto, scaffolds, splits.

Principe : on ne calcule que ce dont on a besoin, et UNE seule fois.
  - les SMILES sont parses une seule fois (clean_dataset) et les molecules RDKit
    sont gardees dans self.mols (alignees avec self.df),
  - features, gram_tanimoto, scaffolds et scaffold_groups sont des cached_property :
    calcules au premier acces, puis reutilises (le GNN ne paie donc jamais la Gram).

Attributs publics (utilises par les autres fichiers) :
    df               liste de graphes PyG nettoyes (x, edge_index, edge_attr, y, smiles)
    mols             liste de molecules RDKit, alignee avec df
    df_report        dict : bilan du nettoyage
    labels           (n, 12) float32, NaN = non teste
    features         (n, 2048) float32, ECFP4 binaire            [paresseux]
    gram_tanimoto    (n, n) float32, noyau Tanimoto              [paresseux]
    scaffolds        liste de n SMILES de scaffold Murcko        [paresseux]
    scaffold_groups  liste de tableaux d'indices, un par scaffold [paresseux]

Methodes publiques de split :
    scaffold_split(seed=...)            deterministe (facon DeepChem), comme avant
    balanced_scaffold_split(seed=...)   randomise (facon Chemprop) : le seed change val/test
    get_split(seed, split)              choisit l'un ou l'autre par son nom
"""

from functools import cached_property

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.datasets import MoleculeNet

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

# Generateur Morgan instancie UNE fois (nouvelle API RDKit, plus de deprecation)
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class Data:
    def __init__(self, root: str = "data/"):
        # Seuls calculs "eager" : chargement + nettoyage (un seul parsing des SMILES).
        self.df, self.mols, self.df_report = self.clean_dataset(self.load_tox21(root))
        self.labels = self.stack_labels(self.df)
        assert self.labels.shape == (len(self.df), len(TASK_NAMES)), (
            f"labels {self.labels.shape} incompatible avec {len(TASK_NAMES)} taches"
        )

    # ------------------------------------------------------------- chargement dataset
    def load_tox21(self, root: str = "data/") -> MoleculeNet:
        """Tox21 via PyG : telecharge + cache au premier appel (dossier data/, ignore par git).
        Chaque molecule : x (atomes), edge_index (liaisons), y (1, 12) avec NaN = non teste,
        smiles."""
        dataset = MoleculeNet(root=root, name="Tox21")
        print(f"Tox21 : {len(dataset)} molecules")
        return dataset

    def clean_dataset(self, dataset) -> tuple:
        """Nettoyage AVANT tout le reste (split, features).

        1. SMILES invalides (MolFromSmiles -> None) : supprimes.
        2. Doublons (par SMILES canonique) :
           - labels identiques            -> une copie conservee
           - annotations complementaires -> fusionnees (le non-NaN gagne)
           - conflit 0/1 sur une tache    -> la molecule est supprimee (toutes ses copies)

        Les labels sont lus dans d.y, de forme (1, 12). Les graphes du dataset PyG ne
        sont JAMAIS modifies en place : une fusion remplace first.y par un nouveau tenseur.

        Retourne (graphes_propres, molecules_rdkit, rapport). Les deux listes sont alignees.
        """
        kept, kept_mols, canons = [], [], []
        seen = {}  # smiles canonique -> position dans kept
        conflicting = set()  # smiles canoniques a supprimer a la fin
        n_invalid = n_identical = n_merged = 0

        for d in dataset:
            mol = Chem.MolFromSmiles(d.smiles)
            if mol is None:
                n_invalid += 1
                continue
            canon = Chem.MolToSmiles(mol)  # la CLE : forme canonique

            if canon not in seen:
                seen[canon] = len(kept)
                kept.append(d)
                kept_mols.append(mol)
                canons.append(canon)
                continue

            # --- doublon : comparer les labels avec la premiere copie ---
            first = kept[seen[canon]]
            y_old = first.y.view(-1).numpy()  # (12,) avec NaN
            y_new = d.y.view(-1).numpy()

            both = ~np.isnan(y_old) & ~np.isnan(y_new)  # taches annotees des 2 cotes
            if (y_old[both] != y_new[both]).any():  # 0 vs 1 : contradiction
                conflicting.add(canon)
                continue

            merged = np.where(np.isnan(y_old), y_new, y_old)  # le non-NaN gagne
            if (
                np.isnan(merged).sum() < np.isnan(y_old).sum()
            ):  # la copie apporte du nouveau
                first.y = torch.from_numpy(merged).view(1, -1)
                n_merged += 1
            else:
                n_identical += 1

        keep = [c not in conflicting for c in canons]
        graphs = [d for d, k in zip(kept, keep) if k]
        mols = [m for m, k in zip(kept_mols, keep) if k]

        report = {
            "n_initial": len(dataset),
            "n_invalid_smiles": n_invalid,
            "n_duplicates_identical": n_identical,
            "n_duplicates_merged": n_merged,
            "n_conflicting_molecules_removed": len(conflicting),
            "n_final": len(graphs),
        }
        print("Nettoyage : " + ", ".join(f"{k}={v}" for k, v in report.items()))
        return graphs, mols, report

    # ------------------------------------------------------------- vectorisation
    @staticmethod
    def _murcko(mol) -> str:
        """SMILES canonique du scaffold Bemis-Murcko d'une molecule RDKit."""
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))

    def murcko_scaffold_smiles(self, data) -> str:
        """Scaffold d'un graphe PyG (reparse le SMILES). Conserve pour compatibilite :
        preferer self.scaffolds[i], deja calcule."""
        mol = Chem.MolFromSmiles(data.smiles)
        return "" if mol is None else self._murcko(mol)

    def mol_to_fingerprint(self, mol) -> np.ndarray:
        """ECFP4 : molecule RDKit -> vecteur binaire (2048,), sans fit donc sans fuite."""
        return MORGAN_GEN.GetFingerprintAsNumPy(mol)

    def build_feature_matrix(self, mols) -> np.ndarray:
        """Toutes les molecules -> X (n_molecules, n_bits), float32."""
        X = np.stack([self.mol_to_fingerprint(m) for m in mols]).astype(np.float32)
        print(f"Features : X {X.shape}, densite {X.mean():.4f}")
        return X

    def tanimoto_kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Similarite Tanimoto vectorisee entre deux ensembles de fingerprints binaires.
        K(i,j) = |Ai & Bj| / (|Ai| + |Bj| - |Ai & Bj|)   (bits communs / union des bits)

        Calcul en place pour limiter la memoire (2 matrices n x n au pic, pas 4).
        Si deux fingerprints sont vides (union = 0), la similarite vaut 0 (pas de NaN)."""
        inter = A @ B.T
        union = A.sum(axis=1, keepdims=True) + B.sum(axis=1, keepdims=True).T
        union -= inter
        # la ou union = 0, inter = 0 aussi : out=inter conserve ce 0
        return np.divide(inter, union, out=inter, where=union > 0)

    def stack_labels(self, dataset) -> np.ndarray:
        """y de PyG (n, 1, 12) -> Y (n, 12) avec NaN conserves."""
        return torch.stack([d.y for d in dataset]).squeeze(1).numpy()

    # ------------------------------------------------- quantites couteuses (paresseuses)
    @cached_property
    def features(self) -> np.ndarray:
        """ECFP4 de tout le dataset, calcule au premier acces."""
        return self.build_feature_matrix(self.mols)

    @cached_property
    def gram_tanimoto(self) -> np.ndarray:
        """Gram Tanimoto (n, n), calculee au premier acces. Les SVM Tanimoto sont les
        seuls a en avoir besoin : LGBM, SVM lineaire et GNN ne la declenchent pas."""
        return self.tanimoto_kernel(self.features, self.features)

    @cached_property
    def scaffolds(self) -> list:
        """Scaffold Murcko de chaque molecule (aligne avec df), calcule UNE fois."""
        return [self._murcko(m) for m in self.mols]

    @cached_property
    def scaffold_groups(self) -> list:
        """Familles de scaffolds : liste de tableaux d'indices (int64), un par scaffold.
        Ordre = ordre de premiere apparition (identique a l'ancien code)."""
        scaffold_to_idx = {}
        for i, s in enumerate(self.scaffolds):
            scaffold_to_idx.setdefault(s, []).append(i)
        return [np.asarray(g, dtype=np.int64) for g in scaffold_to_idx.values()]

    # ------------------------------------------------------------------------- splits
    def _check_split(self, train, val, test):
        """Garde-fous communs aux deux splits. Retourne les 3 tableaux en int64."""
        train, val, test = (np.asarray(s, dtype=np.int64) for s in (train, val, test))
        assert len(val) > 0 and len(test) > 0, (
            f"val/test vide : {len(val)}/{len(test)} - augmentez frac_val/frac_test "
            f"ou verifiez qu'aucun scaffold ne domine le dataset"
        )
        assert len(train) + len(val) + len(test) == len(self.df)
        assert len(np.intersect1d(train, val)) == 0
        assert len(np.intersect1d(train, test)) == 0
        assert len(np.intersect1d(val, test)) == 0
        return train, val, test

    def scaffold_split(
        self, dataset=None, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed: int = 0
    ):
        """Scaffold split facon MoleculeNet/DeepChem (identique a l'ancienne version) :
        groupes tries par taille decroissante, chaque groupe affecte au split
        avec le plus gros quota restant. Garantit val/test non vides meme avec
        de tres gros scaffolds. Le seed ne brise que les egalites de taille.

        Attention : val et test ne contiennent que les PLUS PETITES familles
        (les gros scaffolds sont toujours en train) : scenario pessimiste.

        `dataset` n'est plus utilise (les familles sont en cache) ; l'argument est garde
        pour que les anciens appels scaffold_split(DATA.df, seed=...) fonctionnent."""
        assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
        if dataset is not None and len(dataset) != len(self.df):
            raise ValueError(
                "scaffold_split travaille sur self.df entier (dataset ignore)"
            )
        rng = np.random.default_rng(seed)
        groups = self.scaffold_groups

        # ETAPE 1 : tri par taille decroissante ; le seed melange a taille egale
        sizes = np.array([len(g) for g in groups])
        noise = rng.random(len(groups))  # bruit uniforme, ordre fixe par seed
        order = np.lexsort((noise, -sizes))  # taille desc, puis bruit asc

        # ETAPE 2 : remplissage par deficit maximal
        n = len(self.df)
        quotas = {"train": frac_train * n, "val": frac_val * n, "test": frac_test * n}
        splits = {"train": [], "val": [], "test": []}
        for k in order:
            group = groups[k]
            # split avec le plus grand quota restant (jamais negatif => jamais vide)
            name = max(quotas, key=lambda s: quotas[s])
            splits[name].append(group)
            quotas[name] -= len(group)

        train, val, test = (
            np.concatenate(splits[s]) if splits[s] else np.empty(0, dtype=np.int64)
            for s in ("train", "val", "test")
        )
        return self._check_split(train, val, test)

    def balanced_scaffold_split(
        self, seed, frac_train=0.8, frac_val=0.1, frac_test=0.1
    ):
        """Scaffold split randomise (facon Chemprop) : les gros scaffolds (> moitie
        d'un jeu val/test) vont d'abord en train, le reste est melange par le seed.
        Contrairement a scaffold_split, val/test contiennent aussi des familles de
        taille moyenne et le seed change reellement leur composition."""
        assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
        rng = np.random.default_rng(seed)
        groups = self.scaffold_groups
        n = len(self.df)
        train_size, val_size = int(frac_train * n), int(frac_val * n)
        small_cut = min(frac_val, frac_test) * n / 2

        big = [k for k, g in enumerate(groups) if len(g) > small_cut]
        small = [k for k, g in enumerate(groups) if len(g) <= small_cut]
        order = list(rng.permutation(big)) + list(rng.permutation(small))

        train, val, test = [], [], []
        for k in order:
            g = groups[k]
            if len(train) + len(g) <= train_size:
                train.extend(g)
            elif len(val) + len(g) <= val_size:
                val.extend(g)
            else:
                test.extend(g)
        return self._check_split(train, val, test)

    def get_split(self, seed, split="deterministic"):
        """Point d'entree unique pour tous les modeles : meme nom => meme split."""
        if split == "deterministic":
            return self.scaffold_split(seed=seed)
        if split == "balanced_random":
            return self.balanced_scaffold_split(seed)
        raise ValueError(
            f"split inconnu : {split!r} (choix : 'deterministic', 'balanced_random')"
        )
