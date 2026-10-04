from itertools import product
from pathlib import Path
from typing import List, Tuple
import lightgbm as lgb

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
from dataset import Data



class Models:
    def __init__(self):
        self.DATA = Data()
        self.TRAINERS = {"SVM_Linear":self. _fit_svm_linear,
                         "SVM_Tanimoto": self._fit_svm_tanimoto,
                         "LGBM": self._fit_lgbm}
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
        return LinearSVC(C=p["C"], class_weight="balanced", max_iter=10_000)\
            .fit(Xtr, ytr)

    def _fit_svm_tanimoto(self, ytr, G_tt, p):
        """SVM a noyau Tanimoto precompute. Xtr ignore (le modele ne voit que la Gram)."""
        return SVC(kernel="precomputed", C=p["C"], class_weight="balanced")\
            .fit(G_tt, ytr)

    def _fit_lgbm(self, Xtr, ytr, G_tt, p):
        """LGBM sur features. G_tt ignore."""
        spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
        return lgb.LGBMClassifier(
            reg_lambda=p["L2"], reg_alpha=p["L1"],
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            num_leaves=p["num_leaves"],
            scale_pos_weight=spw,
            verbosity=-1, random_state=42,
        ).fit(Xtr, ytr)

    def training(self, model_name, idx_train, params):
        """Pipeline d'entrainement unique. Masque NaN calcule UNE fois par tache.
        Chaque fitter recoit (Xtr, ytr, G_tt, params) et utilise ce qu'il veut :
        l'orchestrateur n'a pas a savoir qui consomme quoi."""
        if model_name not in self.TRAINERS:
            raise ValueError(f"Modele inconnu : {model_name!r}. "
                             f"Choix valides : {list(self.TRAINERS)}")

        X, Y = self.DATA.features, self.DATA.labels
        G = self.DATA.gram_tanimoto
        fit = self.TRAINERS[model_name]
        models = []
        for t in range(Y.shape[1]):
            rows = idx_train[~np.isnan(Y[idx_train, t])]   # LE masque, une seule formule
            ytr = Y[rows, t].astype(int)
            models.append(fit(X[rows], ytr, G[np.ix_(rows, rows)], params))
        return models

    # ---------------------------------------------------------------------- evaluation

    def evaluate_svm(self, models, X: np.ndarray, Y: np.ndarray, idx) -> dict:
        """AUC par tache sur les lignes annotees du split, + moyenne.
        AUC = metrique de ranking : les marges signees suffisent."""
        aucs = []
        for t, clf in enumerate(models):
            mask = ~np.isnan(Y[idx, t])
            if mask.sum() < 2 or len(np.unique(Y[idx, t][mask])) < 2:
                aucs.append(np.nan)
                continue
            scores = clf.decision_function(X[idx][mask])
            aucs.append(roc_auc_score(Y[idx, t][mask].astype(int), scores))
        aucs = np.array(aucs)
        return {"auc_per_task": aucs, "auc_mean": np.nanmean(aucs)}

    def auc_for_task(self, clf, X, Y, idx_val, t) -> float:
        mask = ~np.isnan(Y[idx_val, t])
        if mask.sum() < 2 or len(np.unique(Y[idx_val, t][mask])) < 2:
            return np.nan
        return roc_auc_score(
            Y[idx_val, t][mask].astype(int), clf.decision_function(X[idx_val][mask])
        )





    def evaluate_svm_tanimoto(self, models, G_full, Y, idx_train, idx_test) -> dict:
        aucs = []
        for t, clf in enumerate(models):
            rows_tr = np.array([i for i in idx_train if not np.isnan(Y[i, t])])
            rows_te = np.array([i for i in idx_test if not np.isnan(Y[i, t])])
            G_te = G_full[
                np.ix_(rows_te, rows_tr)
            ]  # (n_test, n_train) — largeur du fit
            scores = clf.decision_function(G_te)
            aucs.append(roc_auc_score(Y[rows_te, t].astype(int), scores))
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
    # %%

    def evaluate_lgbm_per_task(models, X, Y, idx_test) -> dict:
        """AUC par tache sur lignes annotees, via predict_proba (probabilites
        natives de LGBM — pas besoin du hack decision_function du SVM)."""
        aucs = []
        for t, clf in enumerate(models):
            mask = ~np.isnan(Y[idx_test, t])
            if mask.sum() < 2 or len(np.unique(Y[idx_test, t][mask])) < 2:
                aucs.append(np.nan)
                continue
            proba = clf.predict_proba(X[idx_test][mask])[:, 1]
            aucs.append(roc_auc_score(Y[idx_test, t][mask].astype(int), proba))
        aucs = np.array(aucs)
        return {"auc_per_task": aucs, "auc_mean": np.nanmean(aucs)}

    def pipeline(
        self,
        model_name: str = "SVM",
        seed_list=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 45, 67),
        params: dict[float] = {
            "L1": 1,
            "L2": 10,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
        },
    ):

        X = self.DATA.df_features
        G = self.tanimoto_kernel(X, X)
        Y = self.DATA.df_labels

        all_means, all_aucs = [], []
        for seed in tqdm(seed_list):
            train, val, test = scaffold_split(dataset, seed=seed)
            if model_name == "SVM":
                models = train_svm_tanimoto(params, G, Y, train)
                res = evaluate_svm_tanimoto(models, G, Y, train, test)
            elif model_name == "LGBM":
                m_lgbm = train_lgbm_per_task(params, X, Y, train)
                res = evaluate_lgbm_per_task(m_lgbm, X, Y, test)
            # print(f"seed={seed} | AUC moyenne test = {res['auc_mean']:.4f}")
            # print("  par tache :", np.round(res["auc_per_task"], 3))
            all_means.append(res["auc_mean"])
            all_aucs.append(res["auc_per_task"])

        m = np.array(all_means)
        # print(f"\nSVM lineaire : AUC moyenne = {m.mean():.4f} +/- {m.std():.4f}")

        # Tableau console lisible
        auc_matrix = np.stack(all_aucs)  # (n_seeds, 12)

        plot_auc_per_task(model_name, auc_matrix)
        print("Ecart-type", m.std())
        return m.mean()
