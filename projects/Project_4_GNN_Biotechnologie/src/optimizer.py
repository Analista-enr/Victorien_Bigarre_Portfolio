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
import optuna
from models import Gaia


class Opti:
    def __init__(self, model_name: str, n_trials: int = 30):
        self.model_name = model_name
        self.seed_list = [0, 1, 2]
        self.n_trials = n_trials
        # un espace PAR modele, defini une fois
        self.spaces = {
            "SVM_Linear": {"C": lambda t: t.suggest_float("C", 1e-3, 10, log=True)},
            "SVM_Tanimoto": {"C": lambda t: t.suggest_float("C", 1e-3, 10, log=True)},
            "LGBM": {
                "num_leaves": lambda t: t.suggest_int("num_leaves", 8, 63),
                "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 800),
                "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "L2": lambda t: t.suggest_float("L2", 1e-3, 100, log=True),
                "L1": lambda t: t.suggest_float("L1", 1e-8, 100, log=True),
            },
        }

    def objective(self, trial):
        space = self.spaces[self.model_name]
        params = {**self.params_0, **{k: f(trial) for k, f in space.items()}}
        g = Gaia(model_name=self.model_name, seed_list=self.seed_list, params=params)
        return g.res_val["auc_mean"]  # VAL, jamais test

    def optimizer(self):
        study = optuna.create_study(direction="maximize",
                                    study_name=f"{self.model_name}_opti")
        study.optimize(self.objective, n_trials=self.n_trials)
        print(f"Meilleur val AUC : {study.best_value:.4f}")
        print(f"Meilleurs params : {study.best_params}")
        # le test est evalue UNE fois, a la fin, sur les meilleurs params
        return study.best_params, study.best_value
