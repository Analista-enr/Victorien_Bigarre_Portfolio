import optuna
from models import Gaia


class Apollon:
    def __init__(self, model_name: str, n_trials: int = 30):
        self.model_name = model_name
        self.n_trials = n_trials
        self.seed_list = [i for i in range(10)]
        self.params_0 = {
            "C": 0.001,
            "L1": 0,
            "L2": 1,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
        }
        self.g = Gaia(
            model_name=self.model_name,
            seed_list=self.seed_list,
            params=self.params_0,
        )

        if model_name in ("SVM_Linear", "SVM_Tanimoto"):
            self.search_space = {"C": (0.001, 10, True)}
        elif model_name == "LGBM":
            self.search_space = {
                "num_leaves": (8, 63, False),
                "n_estimators": (100, 800, False),
                "learning_rate": (0.01, 0.3, True),
                "L2": (0.001, 100, True),
                "L1": (0.0000001, 100, True),
            }
        else:
            raise ValueError(f"Modele inconnu : {model_name!r}")

        # self.mejor_HP,self.mejor_score=self.optimizer()

    def propose_params(self, trial):
        """Construit le dict params du trial, en partant des defauts
        puis en ecrasant avec les valeurs proposees par Optuna."""
        # etape 1 : copier les valeurs par defaut, une par une
        params = self.params_0.copy()
        # etape 2 : pour chaque parametre recherche, demander une valeur a Optuna
        for nom, (borne_min, borne_max, log) in self.search_space.items():
            if nom in ("num_leaves", "n_estimators"):
                valeur = trial.suggest_int(nom, borne_min, borne_max)
            elif log:
                valeur = trial.suggest_float(nom, borne_min, borne_max, log=True)
            else:
                valeur = trial.suggest_float(nom, borne_min, borne_max)
            params[nom] = valeur

        return params

    def objective(self, trial):
        params = self.propose_params(trial)
        g = self.g.pipeline(self.model_name, self.seed_list, params)
        return g.score["val"]["auc_mean"]

    def optimizer(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)

        print(f"Meilleur AUC val : {study.best_value:.4f}")
        print(f"Meilleurs params : {study.best_params}")

        return study.best_params, study.best_value
