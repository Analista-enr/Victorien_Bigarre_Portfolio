from typing import List, Tuple, Union

import numpy as np
import pulp as lp
from deap import algorithms, base, creator, tools
from pyswarm import pso
from scipy.optimize import differential_evolution, dual_annealing, shgo


class Apollon:
    def __init__(self, df, Tau: int):
        self.week = df.columns[:-2]
        self.n_weeks = len(self.week) // 7
        self.T = 7 * 24 * 60 * self.n_weeks  # Minutes totales
        self.H = 7 * 24 * self.n_weeks  # Heures totales
        assert self.T % 60 == 0, "T doit être un multiple de 60"
        assert self.H == self.T // 60, "H doit être égal à T // 60"

        self.arrivals = self.llegada_fast(df)
        self.E = np.array(
            [[1] * 8 + [0] * 16, [0] * 8 + [1] * 8 + [0] * 8, [0] * 16 + [1] * 8]
        )  # (3, 24)
        self.W = np.tile(
            np.array(
                [
                    [1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1],
                    [1, 0, 0, 1, 1, 1, 1],
                    [1, 1, 0, 0, 1, 1, 1],
                    [1, 1, 1, 0, 0, 1, 1],
                    [1, 1, 1, 1, 0, 0, 1],
                ]
            ),
            (1, self.n_weeks),
        )  # (7, 7 * n_weeks)

        self.N_opt, self.Q_opt = self.solve(Tau)
        self.N_naive, self.Q_naive = self.naive(Tau)
        self.N_long = self.to_long(self.N_opt)

    def llegada_fast(self, df):
        Q = np.zeros(self.T)
        for day_idx, day in enumerate(self.week):
            mask = (
                (df["Hour"] >= 0)
                & (df["Hour"] < 24)
                & (df["Minute"] >= 0)
                & (df["Minute"] < 60)
            )
            hours = df.loc[mask, "Hour"].values
            minutes = df.loc[mask, "Minute"].values
            day_values = df.loc[mask, day].values
            idx = hours * 60 + minutes
            np.add.at(Q, idx + day_idx * 24 * 60, day_values)
        return Q

    def naive(self, Tau):
        Qh = np.zeros(self.H)
        for t in range(self.T):
            Qh[t // 60] += self.arrivals[t]
        Nh = np.round(Qh / Tau).astype(int)

        N = np.zeros((7, 3), dtype=int)
        for i in range(7):  # workers
            for j in range(3):  # catégories
                total = 0
                count = 0
                for h in range(self.H):
                    day = h // 24
                    hour = h % 24
                    if self.W[i, day] == 1 and self.E[j, hour] == 1:
                        total += Nh[h]
                        count += 1
                N[i, j] = total // count if count > 0 else 0
        return N.tolist(), Qh.tolist()

    def solve(self, Tau):
        prob = lp.LpProblem("fast_model", lp.LpMinimize)
        L, K = self.W.shape[0], self.E.shape[0]
        N = [
            [lp.LpVariable(f"N_{j}_{k}", lowBound=0, cat="Integer") for k in range(K)]
            for j in range(L)
        ]

        Q = [lp.LpVariable(f"Q_{t}", lowBound=0) for t in range(self.T + 1)]

        # Condition initiale
        prob += Q[0] == 0

        prod_hour = [lp.LpAffineExpression() for _ in range(self.H)]
        for h in range(self.H):
            day = h // 24
            hour = h % 24
            for j in range(L):
                for k in range(K):
                    prod_hour[h] += N[j][k] * self.E[k][hour] * self.W[j][day]

        for t in range(self.T):
            h = t // 60
            prob += Q[t + 1] >= Q[t] + self.arrivals[t] - prod_hour[h]

        # Contrainte moyenne sur Q
        prob += lp.lpSum(Q) <= Tau * self.T

        # Objectif : minimiser la somme de N
        prob += lp.lpSum(N[j][k] for j in range(L) for k in range(K))

        prob.solve(lp.PULP_CBC_CMD(msg=0))

        total_douaniers = sum(lp.value(n) for row in N for n in row)
        print(f"### === nombre de douaniers : {total_douaniers} === ###")

        return ([[lp.value(n) for n in row] for row in N], [lp.value(q) for q in Q])

    def flux(self, N: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Calcule la taille de la queue à chaque minute.
        Args:
            N: (n_workers, 3) - nombre de douaniers par worker et catégorie.
        Returns:
            Q: (T,) - taille de la queue à chaque minute.
        """
        N = np.asarray(N, dtype=float)
        assert N.ndim == 2 and N.shape[1] == 3, (
            f"N doit être de forme (n, 3), got {N.shape}"
        )

        Q = np.zeros(self.T)
        prod_hour = np.zeros(self.H)

        # Vectorisation : Remplace les 3 boucles imbriquées par une opération matricielle
        for h in range(self.H):
            day = h // 24
            hour = h % 24
            e_vals = self.E[:, hour]  # (3,) - coefficients pour les 3 catégories
            w_vals = self.W[:, day]  # (n_workers,) - workers actifs ce jour
            # prod_hour[h] = sum_{j,k} N[j,k] * e_vals[k] * w_vals[j]
            # = sum_j w_vals[j] * (N[j] @ e_vals)
            prod_hour[h] = np.sum(w_vals * (N @ e_vals))

        # Calcul de Q (vectorisable aussi, mais moins critique)
        for t in range(self.T - 1):
            hour_idx = t // 60
            Q[t + 1] = max(0, Q[t] - prod_hour[hour_idx] + self.arrivals[t])

        return Q

    def to_long(self, N: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Convertit N (n_workers, 3) en un vecteur long de taille T.
        Args:
            N: (n_workers, 3)
        Returns:
            dN: (T,) - nombre total de douaniers actifs à chaque minute.
        """
        N = np.asarray(N, dtype=float)
        assert N.ndim == 2 and N.shape[1] == 3, (
            f"N doit être de forme (n, 3), got {N.shape}"
        )

        dN = np.zeros(self.T)
        for t in range(self.T):
            day = t // (24 * 60)
            hour = (t % (24 * 60)) // 60
            w_mask = self.W[:, day] == 1  # (n_workers,)
            e_mask = self.E[:, hour] == 1  # (3,)
            # dN[t] = sum_{j,k} N[j,k] * w_mask[j] * e_mask[k]
            dN[t] = np.sum(w_mask * (N @ e_mask))

        return dN

    def op_particule(
        self,
        N: Union[np.ndarray, List[List[float]]],
        Tau: float,
        lambda_N: float = 1.0,
        lambda_mean: float = 1.0,
        lambda_p95: float = 1.0,
    ) -> float:
        """
        Fonction de coût pour l'optimisation.
        Args:
            N: (n_workers, 3) ou (n_workers * 3,) - vecteur de décision.
            Tau: Temps d'attente cible.
        Returns:
            Coût total (float).
        """
        N = np.asarray(N, dtype=float)
        if N.ndim == 1:
            N = N.reshape(-1, 3)  # Convertit (n*3,) en (n, 3)

        Q = self.flux(N)
        mean_wait = np.mean(Q)
        p95_wait = np.percentile(Q, 95)

        cost_N = lambda_N * (np.sum(N) / 100)
        cost_mean = lambda_mean * (mean_wait / Tau - 1) ** 2
        cost_p95 = lambda_p95 * (p95_wait / (1.5 * Tau) - 1) ** 2
        return float(cost_N + cost_mean + cost_p95)
