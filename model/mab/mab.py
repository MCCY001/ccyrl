import numpy as np


class BernoulliBandit:
    def __init__(self, k: int) -> None:
        self.probs = np.random.uniform(size=k)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = k

    def step(self, k: int) -> int:
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
