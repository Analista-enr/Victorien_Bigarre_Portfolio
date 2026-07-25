import hexaly.optimizer
from typing import List, Tuple, Optional
from data_process import TrainScheduler
import pandas as pd

class HexalySolver(TrainScheduler):
    """
    Solveur utilisant Hexaly Optimizer pour l'affectation des trains aux quais.
    Hérite de TrainScheduler pour accéder aux données et à op_score.
    """

    def __init__(self, df: pd.DataFrame, nb_quai: int, penalty: int = 10):
        super().__init__(df, nb_quai, penalty)

    def solve(self, time_limit: int = 60, output_file: Optional[str] = None) -> Tuple[List[int], float]:
        """
        Résout le problème avec Hexaly Optimizer.

        Args:
            time_limit: Limite de temps en secondes (défaut: 60s).
            output_file: Fichier de sortie pour la solution (optionnel).

        Returns:
            Tuple[solution, score]: Solution optimale et score associé.
        """
        with hexaly.optimizer.HexalyOptimizer() as optimizer:
            model = optimizer.model

            # Variables : x[i] = quai du train i (1 à nb_quais)
            x = [model.int(1, self.nb_quais) for _ in range(self.nb_train)]

            # Contraintes : affectations fixes
            for train_idx, quai in self.fixed_assignments.items():
                model.constraint(x[train_idx] == quai)

            # Matrice des chevauchements (accès via model.array)
            overlap_array = model.array(self.overlap)

            # Objectif : minimiser la somme des chevauchements pondérés
            total_overlap = model.sum()
            for i in range(self.nb_train):
                for k in range(i + 1, self.nb_train):
                    total_overlap.add_operand(overlap_array[i][k] * (x[i] == x[k]))

            model.minimize(self.penalty * total_overlap)
            model.close()

            # Initialisation (optionnelle)
            for i in range(self.nb_train):
                if i in self.fixed_assignments:
                    x[i].set_value(self.fixed_assignments[i])
                else:
                    x[i].set_value(1)  # Quai 1 par défaut

            # Résolution
            optimizer.param.time_limit = time_limit
            optimizer.solve()

            # Récupération des résultats
            solution = [x[i].value for i in range(self.nb_train)]
            score = float(total_overlap.value * self.penalty)

            if output_file:
                with open(output_file, 'w') as f:
                    f.write(f"{score}\n")
                    f.write(" ".join(map(str, solution)) + "\n")

            return solution, score