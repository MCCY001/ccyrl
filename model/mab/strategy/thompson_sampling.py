import numpy as np

from model.mab.mab import BernoulliBandit
from model.mab.strategy.solver import Solver

# MAB 中还有一种经典算法——汤普森采样（Thompson sampling），
# 先假设拉动每根拉杆的奖励服从一个特定的概率分布，然后根据拉动每根拉杆的期望奖励来进行选择。
# 汤普森采样是一种计算所有拉杆的最高奖励概率的蒙特卡洛采样方法。
class ThompsonSampling(Solver):
    def __init__(self, bandit: BernoulliBandit) -> None:
        super().__init__(bandit)
        # 列表,表示每根拉杆奖励为1的次数
        self._a = np.ones(self.bandit.K)
        # 列表,表示每根拉杆奖励为0的次数
        self._b = np.ones(self.bandit.K)

    def run_one_step(self):
        # 为每个臂构造一个 beta 分布并采样
        samples = np.random.beta(self._a, self._b)
        # 选取采样最大值对应的臂
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self._a[k] += r
        self._b[k] += (1 - r)
        return k
