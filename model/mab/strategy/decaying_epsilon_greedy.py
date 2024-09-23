import numpy as np

from model.mab.mab import BernoulliBandit
from model.mab.strategy.solver import Solver


class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit: BernoulliBandit, init_prob: float = 1.0) -> None:
        super().__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1. / self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += (1. / (self.counts[k] + 1)) * (r - self.estimates[k])
        return k
