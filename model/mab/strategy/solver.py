from abc import abstractmethod, ABC

import numpy as np

from model.mab.mab import BernoulliBandit


class Solver(ABC):
    def __init__(self, bandit: BernoulliBandit) -> None:
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.actions = []
        self.regrets = []

    def update_regret(self, k: int) -> None:
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    @abstractmethod
    def run_one_step(self):
        pass

    def run(self, episode: int) -> None:
        for _ in range(episode):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)
