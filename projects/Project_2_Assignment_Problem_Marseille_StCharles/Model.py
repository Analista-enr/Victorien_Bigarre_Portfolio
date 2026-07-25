import random
import math
import numpy as np
from typing import List, Tuple
from data_process import TrainScheduler

class HybridSolver(TrainScheduler):
    """
    Solveur hybride PSO+ASA pour l'affectation des trains aux quais.
    Hérite de TrainScheduler pour accéder aux données et à op_score.
    """

    def __init__(self, df: pd.DataFrame, nb_quai: int, penalty: int = 10):
        super().__init__(df, nb_quai, penalty)

    # --- Fonctions utilitaires pour les algorithmes ---
    def _neighbor(self, sol: List[int]) -> List[int]:
        """Génère un voisin aléatoire en respectant les contraintes fixes."""
        n = sol.copy()
        move = random.choice(['swap', 'move', 'reverse'])

        if move == 'swap':
            i, j = random.sample(range(len(n)), 2)
            if i in self.fixed_assignments or j in self.fixed_assignments:
                return self._neighbor(sol)
            n[i], n[j] = n[j], n[i]

        elif move == 'move':
            i = random.randint(0, len(n) - 1)
            if i in self.fixed_assignments:
                return self._neighbor(sol)
            n[i] = random.randint(1, self.nb_quais)

        else:  # reverse
            i, j = sorted(random.sample(range(len(n)), 2))
            if any(idx in self.fixed_assignments for idx in range(i, j + 1)):
                return self._neighbor(sol)
            n[i:j + 1] = n[i:j + 1][::-1]

        return n

    def _init_solution(self) -> List[int]:
        """Génère une solution initiale respectant les affectations fixes."""
        return [
            self.fixed_assignments[i] if i in self.fixed_assignments
            else random.randint(1, self.nb_quais)
            for i in range(self.nb_train)
        ]

    # --- PSO ---
    def _run_pso(self, n_particles: int = 50, max_iter: int = 100) -> Tuple[List[int], float]:
        """Exécute l'algorithme PSO."""
        w_start, w_end = 0.9, 0.4
        c1 = c2 = 1.5
        v_max = self.nb_quais / 2

        positions = np.array([self._init_solution() for _ in range(n_particles)]).astype(float)
        velocities = np.random.uniform(-v_max, v_max, (n_particles, self.nb_train))
        scores = np.array([self.op_score(list(pos)) for pos in positions])

        pbest_positions = positions.copy()
        pbest_scores = scores.copy()
        gbest_idx = np.argmin(pbest_scores)
        gbest_position = pbest_positions[gbest_idx]
        gbest_score = pbest_scores[gbest_idx]

        if gbest_score == 0:
            return list(gbest_position), 0.0

        for t in range(max_iter):
            w = w_start - (w_start - w_end) * (t / max_iter)
            r1, r2 = np.random.random((2, n_particles, self.nb_train))
            velocities = w * velocities + c1 * r1 * (pbest_positions - positions) + c2 * r2 * (gbest_position - positions)
            velocities = np.clip(velocities, -v_max, v_max)

            positions = np.clip(np.round(positions + velocities).astype(int), 1, self.nb_quais)

            # Réappliquer les contraintes fixes
            for i in range(n_particles):
                for idx in self.fixed_assignments:
                    positions[i, idx] = self.fixed_assignments[idx]

            scores = np.array([self.op_score(list(pos)) for pos in positions])

            improved = scores < pbest_scores
            pbest_positions[improved] = positions[improved]
            pbest_scores[improved] = scores[improved]

            current_best_idx = np.argmin(pbest_scores)
            if pbest_scores[current_best_idx] < gbest_score:
                gbest_position = pbest_positions[current_best_idx]
                gbest_score = pbest_scores[current_best_idx]
                if gbest_score == 0:
                    return list(gbest_position), 0.0

        return list(gbest_position), float(gbest_score)

    # --- ASA ---
    def _run_asa(self, initial_solution: List[int], niter: int = 50,
                 temp: float = 1.0, cooling: float = 0.99, target: float = 0.44) -> Tuple[List[int], float]:
        """Exécute l'algorithme ASA à partir d'une solution initiale."""
        cur = initial_solution.copy()
        cur_score = best_score = self.op_score(cur)
        best = cur.copy()
        acc, it = 0, 0

        for _ in range(niter):
            new = self._neighbor(cur)
            new_score = self.op_score(new)
            if new_score == 0:
                return new, 0.0
            delta = new_score - cur_score
            if delta < 0 or random.random() < math.exp(-delta / temp):
                cur, cur_score = new, new_score
                acc += 1
                if cur_score < best_score:
                    best, best_score = cur, cur_score
            it += 1
            if it % 10 == 0:
                rate = acc / it if it > 0 else 0.0
                temp *= math.exp((rate - target) / target)
                acc, it = 0, 0
            temp *= cooling
        return best, best_score

    # --- Hybridation PSO+ASA ---
    def solve(self, n_particles: int = 50, max_iter: int = 100,
              asa_niter: int = 50, asa_temp: float = 1.0,
              asa_cooling: float = 0.99) -> Tuple[List[int], float]:
        """
        Exécute l'hybridation PSO+ASA.

        Args:
            n_particles: Nombre de particules dans PSO.
            max_iter: Nombre d'itérations pour PSO.
            asa_niter: Nombre d'itérations pour ASA.
            asa_temp: Température initiale pour ASA.
            asa_cooling: Taux de refroidissement pour ASA.

        Returns:
            Tuple[solution, score]: Solution optimale et score associé.
        """
        sol, score = self._run_pso(n_particles, max_iter)
        if score == 0:
            return sol, 0.0
        return self._run_asa(sol, asa_niter, asa_temp, asa_cooling)