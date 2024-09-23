import numpy as np

from model.mab.mab import BernoulliBandit
from model.mab.strategy.solver import Solver


class EpsilonGreedy(Solver):
    def __init__(self, bandit: BernoulliBandit, epsilon: float = 0.01, init_prob: float = 1.0) -> None:
        super().__init__(bandit)
        self.epsilon = epsilon
        # 为每个臂分配一个初始概率估计值，所有臂的初始值都相同（默认为1.0），采用“乐观初始值”的策略
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        # 增量式更新对应臂的概率估计
        self.estimates[k] += (1. / (self.counts[k] + 1)) * (r - self.estimates[k])
        return k
