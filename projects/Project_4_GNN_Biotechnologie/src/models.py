"""Catalogue des modeles classiques (SVM lineaire, SVM Tanimoto, LGBM) et entrainement
sur plusieurs seeds. Ce fichier ENTRAINE ; l'evaluation (AUC) est dans evaluate.py.

Chaque modele est une classe avec :
    name            nom utilise partout ("SVM_Linear", "SVM_Tanimoto", "LGBM")
    default_params  hyperparametres de depart
    search_space    espace de recherche pour l'optimisation (optimizer.py le lit ici) :
                    {param: (type, borne_min, borne_max, echelle_log)}
    fit(data, idx_train)    entraine UN classifieur par tache (12 au total)
    predict(data, idx)      scores (len(idx), 12) ; colonne NaN si la tache n'a pas pu
                            etre apprise (une seule classe dans le train)

Contrat avec evaluate.py : evaluate n'a PAS a connaitre les masques NaN du train ni la
largeur de la Gram. Il appelle model.predict(data, run.val) (ou run.test), puis compare
aux labels avec son propre masque NaN, en ignorant les taches dont le score est NaN.

Gaia.train(...) retourne une liste de Run (un par seed) : split + modele entraine.
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm

from dataset import TASK_NAMES, Data

N_TASKS = len(TASK_NAMES)


# ====================================================================== base
class PerTaskModel:
    """Un classifieur binaire par tache. Les sous-classes ne definissent que
    _fit_task (entrainer sur une tache) et _predict_task (scorer sur une tache)."""

    name = None
    default_params = {}
    search_space = {}

    def __init__(self, params=None, seed=0):
        params = dict(params or {})
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(
                f"{self.name} : hyperparametres inconnus {sorted(unknown)} "
                f"(valides : {sorted(self.default_params)})"
            )
        self.params = {**self.default_params, **params}
        self.seed = seed
        self.clfs = [None] * N_TASKS  # classifieur de chaque tache
        self.rows = [None] * N_TASKS  # lignes du train utilisees (masque NaN) par tache

    def fit(self, data, idx_train):
        """Entraine les 12 classifieurs. Le masque NaN est calcule ici, UNE fois par tache,
        et memorise dans self.rows (le SVM Tanimoto en a besoin pour predire)."""
        Y = data.labels
        for t in range(N_TASKS):
            rows = idx_train[~np.isnan(Y[idx_train, t])]
            y = Y[rows, t].astype(int)
            if len(np.unique(y)) < 2:  # impossible d'apprendre avec une seule classe
                continue
            self.rows[t] = rows
            self.clfs[t] = self._fit_task(data, rows, y)
        return self

    def predict(self, data, idx):
        """Scores (len(idx), 12). Toutes les molecules de idx sont scorees pour toutes les
        taches : c'est evaluate.py qui garde celles qui ont un label. NaN si tache non apprise."""
        scores = np.full((len(idx), N_TASKS), np.nan)
        for t, clf in enumerate(self.clfs):
            if clf is not None:
                scores[:, t] = self._predict_task(data, clf, self.rows[t], idx)
        return scores

    def _fit_task(self, data, rows, y):
        raise NotImplementedError

    def _predict_task(self, data, clf, rows, idx):
        raise NotImplementedError


# ==================================================================== modeles
class SVMLinear(PerTaskModel):
    name = "SVM_Linear"
    default_params = {"C": 0.605}
    search_space = {"C": ("float", 0.001, 10, True)}

    def _fit_task(self, data, rows, y):
        return LinearSVC(C=self.params["C"], class_weight="balanced", max_iter=10_000).fit(
            data.features[rows], y
        )

    def _predict_task(self, data, clf, rows, idx):
        return clf.decision_function(data.features[idx])  # marge : suffisant pour l'AUC


class SVMTanimoto(PerTaskModel):
    name = "SVM_Tanimoto"
    default_params = {"C": 0.605}
    search_space = {"C": ("float", 0.001, 10, True)}

    def _fit_task(self, data, rows, y):
        G = data.gram_tanimoto[np.ix_(rows, rows)]  # Gram train x train (masque de la tache)
        return SVC(kernel="precomputed", C=self.params["C"], class_weight="balanced").fit(G, y)

    def _predict_task(self, data, clf, rows, idx):
        # Gram croisee test x train, avec LES MEMES lignes train que pour le fit :
        # c'est pour cela que self.rows est memorise (largeur attendue par le modele)
        return clf.decision_function(data.gram_tanimoto[np.ix_(idx, rows)])


class LGBM(PerTaskModel):
    name = "LGBM"
    default_params = {
        "num_leaves": 56,
        "n_estimators": 746,
        "learning_rate": 0.0997,
        "L2": 15.21,  # -> reg_lambda
        "L1": 0.0,  # -> reg_alpha
        "min_child_samples": 20,  # defaut LightGBM : comportement inchange
        "colsample_bytree": 1.0,  # defaut LightGBM : comportement inchange
    }
    search_space = {
        "num_leaves": ("int", 8, 63, False),
        "n_estimators": ("int", 100, 800, False),
        "learning_rate": ("float", 0.01, 0.3, True),
        "L2": ("float", 0.001, 100, True),
        "L1": ("float", 1e-7, 100, True),
        "min_child_samples": ("int", 5, 100, False),
        "colsample_bytree": ("float", 0.2, 1.0, False),
    }

    def _fit_task(self, data, rows, y):
        p = self.params
        n_pos = y.sum()  # >= 1 : fit() ecarte les taches a une seule classe
        return lgb.LGBMClassifier(
            reg_lambda=p["L2"],
            reg_alpha=p["L1"],
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            num_leaves=p["num_leaves"],
            min_child_samples=p["min_child_samples"],
            colsample_bytree=p["colsample_bytree"],
            scale_pos_weight=(len(y) - n_pos) / n_pos,
            random_state=self.seed,
            force_col_wise=True,  # evite le test de strategie a chaque fit (12 fits / seed)
            verbosity=-1,
        ).fit(data.features[rows], y)

    def _predict_task(self, data, clf, rows, idx):
        return clf.predict_proba(data.features[idx])[:, 1]


MODELS = {cls.name: cls for cls in (SVMLinear, SVMTanimoto, LGBM)}


def get_model_class(model_name):
    if model_name not in MODELS:
        raise ValueError(f"Modele inconnu : {model_name!r}. Choix valides : {list(MODELS)}")
    return MODELS[model_name]


# ================================================================ entrainement
@dataclass
class Run:
    """Resultat d'entrainement pour UNE seed : le split utilise + le modele entraine.
    `split` est le NOM de la strategie ('deterministic', 'balanced_random') : evaluate.py
    l'ecrit dans ses resultats pour qu'on sache toujours sur quel protocole ils reposent."""

    seed: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    model: PerTaskModel
    split: str = "deterministic"


class Gaia:
    """Entraine un modele du catalogue sur plusieurs seeds. Ne calcule aucune metrique.
    Le constructeur ne lance rien : Gaia(data) est instantane."""

    def __init__(self, data=None):
        self.DATA = data if data is not None else Data()

    def train(self, model_name, seed_list=(0, 1, 2), params=None, split="deterministic",
              progress=True):
        """Retourne une liste de Run, un par seed.
        - split : 'deterministic' ou 'balanced_random' (voir Data.get_split),
        - params : hyperparametres du modele (les manquants prennent default_params)."""
        cls = get_model_class(model_name)
        runs = []
        for seed in tqdm(list(seed_list), desc=model_name, disable=not progress):
            train, val, test = self.DATA.get_split(seed, split)
            model = cls(params, seed=seed).fit(self.DATA, train)
            runs.append(
                Run(seed=seed, train=train, val=val, test=test, model=model, split=split)
            )
        return runs