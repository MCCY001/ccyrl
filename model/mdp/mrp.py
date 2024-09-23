import numpy as np


class MRP:
    def __init__(self, states: list[str],
                 transition_probs: dict[tuple[str, str], float],
                 rewards: dict[str, float], gamma: float = 0.9) -> None:
        """
        初始化MRP
        :param states: 状态空间 (list)
        :param transition_probs: 状态转移概率 (dict)
        :param rewards: 奖励函数 (dict)
        :param gamma: 折扣因子 (float)
        """
        self.states = states
        self.transition_probs = transition_probs
        self.rewards = rewards
        self.gamma = gamma

    def get_transition_prob(self, state: str, next_state: str) -> float:
        """
        获取状态转移概率
        :param state: 当前状态 (str)
        :param next_state: 下一个状态 (str)
        :return: 状态转移的概率值
        """
        return self.transition_probs.get((state, next_state), 0.0)

    def get_reward(self, state: str) -> float:
        """
        获取即时奖励
        :param state: 当前状态 (str)
        :return: 奖励值
        """
        return self.rewards.get(state, 0.0)

    def is_terminal(self, state: str) -> bool:
        """
        判断状态是否为终止状态
        :param state: 当前状态 (str)
        :return: 是否终止状态 (bool)
        """
        return not any(transition[0] == state for transition in self.transition_probs.keys())

    def transition_matrix(self) -> np.ndarray:
        """
        生成状态转移矩阵
        :return: 状态转移矩阵 (numpy.ndarray)
        """
        matrix = np.zeros((len(self.states), len(self.states)))
        for i, state in enumerate(self.states):
            for j, next_state in enumerate(self.states):
                matrix[i, j] = self.get_transition_prob(state, next_state)

        return matrix

    def compute_value(self) -> np.ndarray:
        """
        MRP 价值函数计算，使用贝尔曼方程解析解直接计算
        :return: 每个状态对应的价值 (numpy.ndarray)
        """
        p = self.transition_matrix()
        r = np.array([self.get_reward(state) for state in self.states])
        i = np.eye(len(self.states))
        v = np.linalg.inv(i - self.gamma * p).dot(r)

        return v
