import numpy as np

from model.mdp.mrp import MRP


class MDP:
    def __init__(self, states: list[str], actions: list[str],
                 transition_probs: dict[tuple[str, str, str], float],
                 rewards: dict[tuple[str, str], float], gamma: float = 0.9) -> None:
        """
        初始化MDP
        :param states: 状态空间 (list)
        :param actions: 动作空间 (list)
        :param transition_probs: 状态转移概率 (dict)
        :param rewards: 奖励函数 (dict)
        :param gamma: 折扣因子 (float)
        """
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs
        self.rewards = rewards
        self.gamma = gamma

    def get_transition_prob(self, state: str, action: str, next_state: str) -> float:
        """
        获取状态转移概率
        :param state: 当前状态 (str)
        :param action: 采取的动作 (str)
        :param next_state: 下一个状态 (str)
        :return: 状态转移的概率值 (float)
        """
        return self.transition_probs.get((state, action, next_state), 0.0)

    def get_reward(self, state: str, action: str) -> float:
        """
        获取即时奖励
        :param state: 当前状态 (str)
        :param action: 采取的动作 (str)
        :return: 奖励值 (float)
        """
        return self.rewards.get((state, action), 0.0)

    def is_terminal(self, state: str) -> bool:
        """
        判断状态是否为终止状态
        :param state: 当前状态 (str)
        :return: 是否终止状态 (bool)
        """
        return not any(transition[0] == state for transition in self.transition_probs.keys())

    def to_mrp(self, pi: dict[str, dict[str, float]]) -> MRP:
        """
        将MDP转换为MRP，根据给定的策略计算期望的奖励函数和状态转移概率。

        :param pi: 策略 (dict)，格式为 {状态: {动作: 概率}}。在每个状态下定义动作及其执行概率。

        :return: 返回一个新的 MRP 实例。该 MRP 包含边缘化后的状态转移概率和期望奖励。

        MRP 的转化过程基于以下两个公式：
            1. 奖励函数 r'(s) = sum_a π(a|s) * r(s, a)
            2. 状态转移函数 P'(s'|s) = sum_a π(a|s) * P(s'|s, a)
        """
        mrp_transition_probs = {}
        mrp_rewards = {}

        for state in self.states:
            # 计算期望奖励
            reward = 0
            for action, action_prob in pi[state].items():
                reward += action_prob * self.get_reward(state, action)
            mrp_rewards[state] = reward

            # 计算期望状态转移概率
            for next_state in self.states:
                transition_prob = 0
                for action, action_prob in pi[state].items():
                    transition_prob += action_prob * self.get_transition_prob(state, action, next_state)
                if transition_prob > 0:
                    mrp_transition_probs[(state, next_state)] = transition_prob
        return MRP(self.states, mrp_transition_probs, mrp_rewards, self.gamma)

    def compute_state_value(self, pi: dict[str, dict[str, float]]) -> np.ndarray:
        """
        计算 MDP 的状态价值函数，将其转换成对应的 MRP 计算其价值函数，两者值相等。
        因为对于 MDP ，其价值函数与其采用的策略有关，因为不同的策略会采取不同的动作，
        从而之后会遇到不同的状态，以及获得不同的奖励，所以它们的累积奖励的期望也就不同，即状态价值不同。
        :param pi: 策略 (dict)，格式为 {状态: {动作: 概率}}。在每个状态下定义动作及其执行概率。
        :return: 每个状态对应的价值向量 (numpy.ndarray)
        """
        return self.to_mrp(pi).compute_value()

    def compute_action_value(self, pi: dict[str, dict[str, float]]) -> np.ndarray:
        """
        计算 MDP 的所有的对应动作价值函数
        :param pi: 策略 (dict)，格式为 {状态: {动作: 概率}}。在每个状态下定义动作及其执行概率。
        :return: 状态价值函数矩阵，行代表对应状态，列代表对应动作 (numpy.ndarray)
        """
        v = self.compute_state_value(pi)
        q = np.zeros((len(self.states), len(self.actions)))
        for i, state in enumerate(self.states):
            for j, action in enumerate(self.actions):
                q_sa = self.get_reward(state, action)
                for k, next_state in enumerate(self.states):
                    q_sa += self.gamma * self.get_transition_prob(state, action, next_state) * v[k]
                q[i, j] = q_sa
        return q
