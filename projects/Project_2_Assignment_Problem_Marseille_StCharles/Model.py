from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from data_process import TrainScheduler


class HybridSolver(TrainScheduler):
    """
    Solveur hybride PSO + recuit simulé adaptatif (ASA) pour l'affectation
    des trains aux quais. Hérite de TrainScheduler (données, overlap, op_score).

    Changements principaux par rapport à l'ancienne version :
    - générateur aléatoire unique et graine (`seed`) pour la reproductibilité ;
    - la solution gloutonne est injectée dans l'essaim PSO ;
    - l'ASA évalue chaque voisin par un delta incrémental O(n) au lieu d'un
      score complet O(n²), ce qui permet des dizaines de milliers d'itérations ;
    - la température initiale est normalisée sur l'échelle des scores
      (penalty x chevauchement moyen) et décroît géométriquement ;
    - plus de récursion dans la génération de voisins ;
    - les contraintes fixes sont réappliquées par indexation vectorisée.

    Après `solve()` : `self.solution` (liste, index du train -> quai 1..nb_quais)
    et `self.score` (op_score).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        nb_quai: int,
        penalty: int = 10,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(df, nb_quai, penalty, **kwargs)
        self.rng = np.random.default_rng(seed)

        self.free_idx = np.array(
            [i for i in range(self.nb_train) if i not in self.fixed_assignments],
            dtype=int,
        )
        self._fixed_idx = np.array(list(self.fixed_assignments.keys()), dtype=int)
        self._fixed_val = np.array(list(self.fixed_assignments.values()), dtype=int)

        self.solution: Optional[List[int]] = None
        self.score: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Utilitaires
    # ------------------------------------------------------------------ #
    def _random_solution(self) -> np.ndarray:
        sol = self.rng.integers(1, self.nb_quais + 1, size=self.nb_train)
        sol[self._fixed_idx] = self._fixed_val
        return sol

    def _score_batch(self, positions: np.ndarray) -> np.ndarray:
        return np.array([self.op_score(p) for p in positions])

    # ------------------------------------------------------------------ #
    # Voisinage avec delta incrémental
    # ------------------------------------------------------------------ #
    def _propose(self, cur: np.ndarray) -> Optional[Tuple[str, int, int]]:
        """Propose un mouvement ('move', i, q_new) ou ('swap', i, j) non neutre."""
        if len(self.free_idx) == 0 or self.nb_quais == 1:
            return None

        if len(self.free_idx) >= 2 and self.rng.random() < 0.5:
            i, j = self.rng.choice(self.free_idx, size=2, replace=False)
            if cur[i] != cur[j]:
                return ("swap", int(i), int(j))

        i = int(self.rng.choice(self.free_idx))
        q = int(self.rng.integers(1, self.nb_quais))  # 1..nb_quais-1
        if q >= cur[i]:
            q += 1  # exclut le quai actuel
        return ("move", i, q)

    def _delta_move(self, cur: np.ndarray, i: int, q_new: int) -> int:
        ov = self.overlap[i]
        return self.penalty * int(ov[cur == q_new].sum() - ov[cur == cur[i]].sum())

    def _delta(self, cur: np.ndarray, move: Tuple[str, int, int]) -> int:
        kind, i, x = move
        if kind == "move":
            return self._delta_move(cur, i, x)
        # swap : deux déplacements successifs
        j = x
        a, b = cur[i], cur[j]
        d1 = self._delta_move(cur, i, b)
        cur[i] = b
        d2 = self._delta_move(cur, j, a)
        cur[i] = a  # restauration
        return d1 + d2

    @staticmethod
    def _apply(cur: np.ndarray, move: Tuple[str, int, int]) -> None:
        kind, i, x = move
        if kind == "move":
            cur[i] = x
        else:
            cur[i], cur[x] = cur[x], cur[i]

    # ------------------------------------------------------------------ #
    # PSO
    # ------------------------------------------------------------------ #
    def _run_pso(
        self,
        n_particles: int = 50,
        max_iter: int = 100,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
    ) -> Tuple[List[int], int]:
        """PSO sur positions entières (arrondies), avec une particule gloutonne."""
        n = self.nb_train
        v_max = max(self.nb_quais / 2, 1.0)

        positions = np.array([self._random_solution() for _ in range(n_particles)])
        positions[0] = self.greedy_init()
        velocities = self.rng.uniform(-v_max, v_max, (n_particles, n))
        scores = self._score_batch(positions)

        pbest_pos = positions.copy()
        pbest_scores = scores.copy()
        g = int(np.argmin(pbest_scores))
        gbest_pos = pbest_pos[g].copy()
        gbest_score = int(pbest_scores[g])

        if gbest_score == 0:
            return list(map(int, gbest_pos)), 0

        for t in range(max_iter):
            w = w_start - (w_start - w_end) * (t / max_iter)
            r1 = self.rng.random((n_particles, n))
            r2 = self.rng.random((n_particles, n))
            velocities = (
                w * velocities
                + c1 * r1 * (pbest_pos - positions)
                + c2 * r2 * (gbest_pos - positions)
            )
            velocities = np.clip(velocities, -v_max, v_max)

            positions = np.clip(
                np.rint(positions + velocities).astype(int), 1, self.nb_quais
            )
            positions[:, self._fixed_idx] = self._fixed_val

            scores = self._score_batch(positions)

            improved = scores < pbest_scores
            pbest_pos[improved] = positions[improved]
            pbest_scores[improved] = scores[improved]

            g = int(np.argmin(pbest_scores))
            if pbest_scores[g] < gbest_score:
                gbest_pos = pbest_pos[g].copy()
                gbest_score = int(pbest_scores[g])
                if gbest_score == 0:
                    break

        return list(map(int, gbest_pos)), gbest_score

    # ------------------------------------------------------------------ #
    # ASA
    # ------------------------------------------------------------------ #
    def _run_asa(
        self,
        initial_solution: List[int],
        niter: int = 20000,
        temp: Optional[float] = None,
        temp_end: Optional[float] = None,
        target: float = 0.44,
        adapt_every: int = 200,
    ) -> Tuple[List[int], int]:
        """
        Recuit simulé à partir d'une solution initiale.

        Args:
            temp: température initiale. Par défaut : penalty x chevauchement
                moyen (échelle d'un delta typique).
            temp_end: température finale (défaut : temp x 1e-3). Le
                refroidissement géométrique en est déduit.
            target: taux d'acceptation visé pour l'ajustement adaptatif
                (le facteur d'ajustement est borné à [0.8, 1.25]).
        """
        cur = np.asarray(initial_solution, dtype=int).copy()
        cur_score = self.op_score(cur)
        best, best_score = cur.copy(), cur_score

        if cur_score == 0 or len(self.free_idx) == 0:
            return list(map(int, best)), best_score

        positive = self.overlap[self.overlap > 0]
        if temp is None:
            temp = self.penalty * (float(positive.mean()) if positive.size else 1.0)
        if temp_end is None:
            temp_end = temp * 1e-3
        cooling = (temp_end / temp) ** (1.0 / max(niter, 1))

        acc = 0
        for it in range(1, niter + 1):
            move = self._propose(cur)
            if move is None:
                break
            delta = self._delta(cur, move)

            if delta <= 0 or self.rng.random() < math.exp(-delta / temp):
                self._apply(cur, move)
                cur_score += delta
                acc += 1
                if cur_score < best_score:
                    best, best_score = cur.copy(), cur_score
                    if best_score == 0:
                        break

            if it % adapt_every == 0:
                rate = acc / adapt_every
                factor = float(np.clip(math.exp((rate - target) / target), 0.8, 1.25))
                temp *= factor
                acc = 0
            temp = max(temp * cooling, 1e-9)

        return list(map(int, best)), int(best_score)

    # ------------------------------------------------------------------ #
    # Hybridation PSO + ASA
    # ------------------------------------------------------------------ #
    def solve(
        self,
        n_particles: int = 50,
        max_iter: int = 100,
        asa_niter: int = 20000,
        asa_temp: Optional[float] = None,
        asa_temp_end: Optional[float] = None,
    ) -> Tuple[List[int], int]:
        """
        Exécute PSO puis ASA sur la meilleure solution trouvée.

        Returns:
            (solution, score) ; également stockés dans self.solution / self.score.
        """
        sol, score = self._run_pso(n_particles, max_iter)
        if score > 0:
            sol, score = self._run_asa(
                sol, niter=asa_niter, temp=asa_temp, temp_end=asa_temp_end
            )

        self.check_solution(sol)
        score = self.op_score(sol)  # recalcul : garde-fou contre toute dérive du delta
        self.solution, self.score = sol, score
        return sol, score

    def plot(self, **kwargs):
        """Gantt de la dernière solution (voir TrainScheduler.plot_gantt)."""
        return self.plot_gantt(self.solution, **kwargs)
