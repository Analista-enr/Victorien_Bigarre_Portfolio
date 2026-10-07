from itertools import product
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dataset import Data
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC, LinearSVC
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from tqdm import tqdm

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


class Gaia:
    def __init__(self, model_name, seed_list, params):
        self.DATA = Data()
        self.TRAINERS = {
            "SVM_Linear": self._fit_svm_linear,
            "SVM_Tanimoto": self._fit_svm_tanimoto,
            "LGBM": self._fit_lgbm,
        }
        self.Hparams = {
            "SVM_Linear": 1,
            "SVM_Tanimoto": 1,
            "LGBM": {
                "L1": 0,
                "L2": 1,
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
            },
        }
        self.score = self.pipeline(model_name, seed_list, params)

    # ------------------------------------------------------------------ scaffold split
    def scaffold_split(
        self, dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed: int = 0
    ):
        """Scaffold split : toutes les molecules d'un meme scaffold dans le MEME split
        -> pas de fuite structurelle (contrairement au split aleatoire).
        Groupes tries par taille decroissante, remplissage train -> val -> test,
        ordre melange par seed -> split deterministe."""
        assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
        rng = torch.Generator().manual_seed(seed)

        scaffold_to_idx = {}
        for i in range(len(dataset)):
            scaffold_to_idx.setdefault(
                self.DATA.murcko_scaffold_smiles(dataset[i]), []
            ).append(i)

        # Le seed doit mélanger l'ordre DES GROUPES (pas leur interieur) :
        # permuter l'interieur d'un groupe ne change pas le split de ses membres,
        # d'ou des results rigoureusement identiques pour tous les seeds (bug fixe).
        groups = sorted(scaffold_to_idx.values(), key=len, reverse=True)
        order = torch.randperm(len(groups), generator=rng).tolist()
        groups = [groups[k] for k in order]

        n = len(dataset)
        train, val, test = [], [], []
        for group in groups:
            for idx in group:
                if len(train) < frac_train * n:
                    train.append(idx)
                elif len(val) < frac_val * n:
                    val.append(idx)
                else:
                    test.append(idx)
        return train, val, test

    # --- fitters : signature UNIFORME (Xtr, ytr, G_tt, params) ---

    def _fit_svm_linear(self, Xtr, ytr, G_tt, p):
        """SVM lineaire sur features. G_tt ignore (pas un modele a noyau precompute)."""
        return LinearSVC(C=p["C"], class_weight="balanced", max_iter=10_000).fit(
            Xtr, ytr
        )

    def _fit_svm_tanimoto(self, ytr, G_tt, p):
        """SVM a noyau Tanimoto precompute. Xtr ignore (le modele ne voit que la Gram)."""
        return SVC(kernel="precomputed", C=p["C"], class_weight="balanced").fit(
            G_tt, ytr
        )

    def _fit_lgbm(self, Xtr, ytr, G_tt, p):
        """LGBM sur features. G_tt ignore."""
        spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
        return lgb.LGBMClassifier(
            reg_lambda=p["L2"],
            reg_alpha=p["L1"],
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            num_leaves=p["num_leaves"],
            scale_pos_weight=spw,
            verbosity=-1,
            random_state=42,
        ).fit(Xtr, ytr)

    def training(self, model_name, idx_train, params):
        """Pipeline d'entrainement unique. Masque NaN calcule UNE fois par tache.
        Chaque fitter recoit (Xtr, ytr, G_tt, params) et utilise ce qu'il veut :
        l'orchestrateur n'a pas a savoir qui consomme quoi."""
        if model_name not in self.TRAINERS:
            raise ValueError(
                f"Modele inconnu : {model_name!r}. "
                f"Choix valides : {list(self.TRAINERS)}"
            )

        X, Y = self.DATA.features, self.DATA.labels
        G = self.DATA.gram_tanimoto
        fit = self.TRAINERS[model_name]
        models = []
        for t in range(Y.shape[1]):
            rows = idx_train[~np.isnan(Y[idx_train, t])]  # LE masque, une seule formule
            ytr = Y[rows, t].astype(int)
            models.append(fit(X[rows], ytr, G[np.ix_(rows, rows)], params))
        return models

    def _predict_svm_linear(
        self,
        clf,
        Xte,
        G_te,
    ):
        """Marge du SVM lineaire (ranking pour l'AUC, pas besoin de proba)."""
        return clf.decision_function(Xte)

    def _predict_svm_tanimoto(self, clf, Xte, G_te):
        """Marge du SVM Tanimoto : G_te = Gram (test masque x train masque)."""
        return clf.decision_function(G_te)

    def _predict_lgbm(self, clf, Xte, G_te):
        """Probabilite de la classe positive LGBM (probas natives)."""
        return clf.predict_proba(Xte)[:, 1]

    PREDICTORS = {
        "SVM_Linear": _predict_svm_linear,
        "SVM_Tanimoto": _predict_svm_tanimoto,
        "LGBM": _predict_lgbm,
    }

    def evaluate(self, model_name, models, idx_train, idx_test) -> dict:
        """Pipeline d'evaluation unique, miroir de training().
        Retourne {"auc_per_task": (12,), "auc_mean": float}.

        Invariant cle : le masque train de chaque tache est reconstruit ici
        par la MEME formule que dans training() — c'est ce qui garantit que
        la Gram croisee a la largeur attendue par le modele Tanimoto.
        """
        if model_name not in self.PREDICTORS:
            raise ValueError(
                f"Modele inconnu : {model_name!r}. "
                f"Choix valides : {list(self.PREDICTORS)}"
            )

        X, Y = self.DATA.features, self.DATA.labels
        G = self.DATA.gram_tanimoto
        predict = self.PREDICTORS[model_name]

        aucs = []
        for t, clf in enumerate(models):
            # test masque (lignes de la metrique)
            rows_te = idx_test[~np.isnan(Y[idx_test, t])]
            yte = Y[rows_te, t].astype(int)

            # garde-fou : AUC indefinie sans deux classes
            if len(np.unique(yte)) < 2:
                aucs.append(np.nan)
                continue

            # train masque : LA meme formule que training() (invariant de largeur)
            rows_tr = idx_train[~np.isnan(Y[idx_train, t])]
            G_te = G[np.ix_(rows_te, rows_tr)]  # Gram croisee test x train

            scores = predict(clf, X[rows_te], G_te)
            aucs.append(roc_auc_score(yte, scores))

        aucs = np.array(aucs)
        return {"auc_per_task": aucs, "auc_mean": np.nanmean(aucs)}

    # ------------------------------------------------------------------------- pipeline

    def plot_auc_per_task(
        self,
        model_name: str,
        auc_matrix: np.ndarray,
        out: Path = FIG_DIR / "auc_per_task.png",
    ):
        """Barres horizontales : AUC par tache (moyenne +/- std sur les seeds).
        Sauvegarde dans results/figures (versionne) + retourne la figure."""
        aucs = np.nanmean(auc_matrix, axis=0)
        stds = np.nanstd(auc_matrix, axis=0)
        order = np.argsort(aucs)  # triees de la pire a la meilleure

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            np.array(TASK_NAMES)[order],
            aucs[order],
            xerr=stds[order],
            color="#4C72B0",
            alpha=1,
        )
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="Hasard (0.5)")
        ax.axvline(
            np.mean(aucs),
            color="orange",
            linestyle="--",
            linewidth=1,
            label=" Mean of models",
        )
        ax.set_xlim(0.4, 1)
        ax.set_xlabel("AUC ROC (test, scaffold split)")
        ax.set_title(f"Baseline {model_name} + ECFP4 - AUC par tache")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"Mean  : {np.mean(aucs)}, std : {np.std(aucs)}")
        print(f"Figure sauvegardee : {out}")
        return fig

    # ------------------------------------------------------------------------- pipeline
    def pipeline(self, model_name="SVM", seed_list=(0, 1, 2), params=None):
        if params is None:  # jamais de dict par defaut mutable !
            params = {"C": 1, "L1": 1, "L2": 10, "n_estimators": 300,
                      "learning_rate": 0.05, "num_leaves": 31}

        val_means, test_means, test_aucs = [], [], []
        for seed in tqdm(seed_list):
            train, val, test = self.scaffold_split(self.DATA.df, seed=seed)
            models = self.training(model_name, train, params)

            res_val = self.evaluate(model_name, models, train, val)  # pour OPTI
            res_test = self.evaluate(model_name, models, train, test)  # pour le RAPPORT
            val_means.append(res_val["auc_mean"])
            test_means.append(res_test["auc_mean"])
            test_aucs.append(res_test["auc_per_task"])
        val_means = np.array(val_means)
        test_means = np.array(test_means)
        test_aucs = np.array(test_aucs)

        res_val = {"auc_mean": np.nanmean(val_means),
                        "auc_mean_std": np.nanstd(val_means)}
        res_test = {"auc_mean": np.nanmean(test_means),
                         "auc_mean_std": np.nanstd(test_means),
                         "auc_matrix": np.stack(test_aucs)}
        self.plot_auc_per_task(model_name, res_test["auc_matrix"])
        return res_val,res_test
