import numpy as np
import scipy as sp
from Optimizer import Apollon
from Process import P0


class R0:
    def __init__(
        self,
        arrival_df,
        average_df,
        tau: float,
        cN: float,
        cT: float,
    ):
        self.tau = tau
        self.cN = cN
        self.cT = cT

        self.data = P0(arrival_df, average_df).data
        self.obj = Apollon(self.data, self.tau)
        self.arrivals = self.obj.arrivals
        self.N_opt, self.Q_opt = self.obj.N_opt, self.obj.Q_opt
        self.N_naive, self.Q_naive = self.obj.N_naive, self.obj.Q_naive
        self.best_T = self.best()
        self.N_long = self.obj.N_long
        self.best_N = Apollon(self.data, self.best_T).N_opt

    def op_optimal(self):
        opt = Apollon(self.data, self.tau)
        return opt.N_opt, opt.Q_opt

    def best(self):
        result = sp.optimize.minimize_scalar(
            self.op_pareto, bounds=(1, self.tau * 10), method="bounded"
        )
        return result.x

    def op_pareto(self, T):
        opt = Apollon(self.data, T)
        N = np.array(opt.N_opt).sum()
        loss = (self.cN * N + (self.cT * T) ** 2) ** 2
        return loss
