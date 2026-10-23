from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import pulp as lp

from data_process import TrainScheduler


class Medee:
    """
    Solveur MILP (PuLP / CBC) : problème de faisabilité, sans conflit autorisé.

    Variables : matrice X de taille (nb_train x nb_quai), X[j][i] = 1 si le
    train j est affecté au quai i+1 (colonne i = quai i+1).

    Contraintes :
        sum_i X[j][i] = 1                 chaque train sur exactement un quai
        X[j][i] + X[k][i] <= 1            si j et k se chevauchent, pour tout quai i
        X[j][q-1] = 1                     affectations fixes

    Si le modèle est infeasible (concurrence > nb_quai, ou deux trains fixés
    en conflit sur le même quai), la solution gloutonne est renvoyée avec un
    avertissement : son score est alors > 0.

    Après résolution : solution (index du train -> quai 1..nb_quai), score
    (op_score) et status (statut PuLP).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        nb_quai: int,
        penalty: int = 10,
        time_limit: Optional[float] = None,
        msg: bool = False,
        solve_now: bool = True,
    ):
        self.DATA = TrainScheduler(df, nb_quai, penalty)
        self.nb_quai = self.DATA.nb_quais
        self.nb_train = self.DATA.nb_train
        self.conflictions = self.DATA.overlap
        self.time_limit = time_limit
        self.msg = msg

        self.status: Optional[str] = None
        self.solution: Optional[List[int]] = None
        self.score: Optional[int] = None

        if solve_now:
            self.solve()

    def solve(self) -> List[int]:
        n, nq = self.nb_train, self.nb_quai
        pairs = np.argwhere(np.triu(self.conflictions > 0, 1))

        prob = lp.LpProblem("train_platform_assignment", lp.LpMinimize)

        # Matrice de variables binaires : X[j][i] = 1 <=> train j sur le quai i+1
        X = [[lp.LpVariable(f"X_{j}_{i}", cat="Binary") for i in range(nq)]
             for j in range(n)]

        prob += 0  # problème de faisabilité : pas d'objectif

        # Un train = un quai
        for j in range(n):
            prob += lp.lpSum(X[j]) == 1

        # Deux trains qui se chevauchent ne partagent pas de quai
        for j, k in pairs:
            for i in range(nq):
                prob += X[j][i] + X[k][i] <= 1

        # Affectations fixes
        for j, q in self.DATA.fixed_assignments.items():
            prob += X[j][q - 1] == 1

        prob.solve(lp.PULP_CBC_CMD(msg=self.msg, timeLimit=self.time_limit))
        self.status = lp.LpStatus[prob.status]

        values = np.array(
            [[lp.value(X[j][i]) for i in range(nq)] for j in range(n)], dtype=object
        )
        if prob.status == lp.LpStatusInfeasible or (values == None).any():  # noqa: E711
            warnings.warn(
                f"Pas de solution sans conflit (statut {self.status}) ; "
                f"repli sur la solution gloutonne. Concurrence max : "
                f"{self.DATA.max_concurrency()} pour {nq} quais."
            )
            solution = list(map(int, self.DATA.greedy_init()))
        else:
            solution = [int(np.argmax(values[j].astype(float))) + 1 for j in range(n)]

        self.DATA.check_solution(solution)
        self.solution = solution
        self.score = self.DATA.op_score(solution)
        return solution

    def plot(self, **kwargs):
        """Gantt de la solution (voir TrainScheduler.plot_gantt)."""
        return self.DATA.plot_gantt(self.solution, **kwargs)