import numpy as np

from model.mab.mab import BernoulliBandit
from model.mab.strategy.solver import Solver


# 上置信界（upper confidence bound，UCB）算法是一种经典的基于不确定性的策略算法，
# 它的思想用到了一个非常著名的数学原理：霍夫丁不等式（Hoeffding's inequality）。
class UCB(Solver):
    """ UCB算法,继承Solver类 """

    def __init__(self, bandit: BernoulliBandit, coef: float, init_prob=1.0) -> None:
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k
