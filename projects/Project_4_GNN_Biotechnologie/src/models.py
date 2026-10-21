from pathlib import Path

import lightgbm as lgb
import numpy as np

from dataset import Data
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC, LinearSVC
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

        self.score = self.pipeline(model_name, seed_list, params)

    # --- fitters : signature UNIFORME (Xtr, ytr, G_tt, params) ---

    def _fit_svm_linear(self, Xtr, ytr, G_tt, p):
        """SVM lineaire sur features. G_tt ignore (pas un modele a noyau precompute)."""
        return LinearSVC(C=p["C"], class_weight="balanced", max_iter=10_000).fit(
            Xtr, ytr
        )

    def _fit_svm_tanimoto(self, Xtr, ytr, G_tt, p):
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


    # ------------------------------------------------------------------------- pipeline
    def pipeline(self, model_name="SVM_Tanimoto", seed_list=(0, 1, 2), params=None,
                 evaluate_test=True, save_results=True):

        if params is None:
            params = {
                "C": 0.605,
                "num_leaves": 56,
                "n_estimators": 746,
                "learning_rate": 0.0997,
                "L2": 15.21,
                "L1": 0,
            }

        val_means, test_means, test_aucs = [], [], []
        for seed in tqdm(seed_list):
            self._current_seed = seed  # pour random_state LGBM
            train, val, test = self.DATA.scaffold_split(self.DATA.df, seed=seed)
            models = self.training(model_name, train, params)

            res_val = self.evaluate(model_name, models, train, val)
            val_means.append(res_val["auc_mean"])
            val_aucs = res_val["auc_per_task"]  # dernier seed, pour le CSV
            if evaluate_test:
                res_test = self.evaluate(model_name, models, train, test)
                test_means.append(res_test["auc_mean"])
                test_aucs.append(res_test["auc_per_task"])

        results = {
            "model_name": model_name,
            "params": params,
            "n_seeds": len(seed_list),
            "val": {
                "auc_mean": np.nanmean(val_means),
                "auc_mean_std": np.nanstd(val_means),
            },
        }
        if evaluate_test:
            results["test"] = {
                "auc_mean": np.nanmean(test_means),
                "auc_mean_std": np.nanstd(test_means),
                "auc_matrix": np.stack(test_aucs),
            }
        #if save_results:
            # self.summarize(results)

        return results


