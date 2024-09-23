import numpy as np

from model.mdp.mdp import MDP


class MonteCarlo:
    def __init__(self, episode_num: int, timestep_max: int) -> None:
        self.episodes = []
        self.episode_num = episode_num
        self.timestep_max = timestep_max

    def sample_trajectory(self, mdp: MDP, pi: dict[str, dict[str, float]]) -> None:
        """
        给定 MDP，从随机状态开始，根据策略 pi 采样一条轨迹，直到终止状态，重复 self.episode_num 次。
        :param mdp: 马尔可夫决策过程
        :param pi: 策略
        :return: 轨迹上的累积回报
        """
        for _ in range(self.episode_num):
            episode = []
            timestep = 0
            random_start = np.random.randint(len(mdp.states))
            while mdp.is_terminal(mdp.states[random_start]):
                random_start = np.random.randint(len(mdp.states))
            s = mdp.states[random_start]
            a = 0
            r = 0
            s_next = 0
            while timestep < self.timestep_max and not mdp.is_terminal(s):
                timestep += 1
                rand, temp = np.random.rand(), 0
                # 在状态 s 下根据策略选择动作
                for a_opt in mdp.actions:
                    if a_opt in pi[s]:
                        temp += pi[s][a_opt]
                    if temp > rand:
                        a = a_opt
                        r = mdp.get_reward(s, a)
                        break
                rand, temp = np.random.rand(), 0
                # 根据状态转移概率得到下一个状态 s_next
                for s_opt in mdp.states:
                    temp += mdp.get_transition_prob(s, a, s_opt)
                    if temp > rand:
                        s_next = s_opt
                        break
                episode.append((s, a, r, s_next))
                s = s_next
            self.episodes.append(episode)

    def compute_state_value(self, mdp: MDP, pi: dict[str, dict[str, float]]) -> np.ndarray:
        """
        采样获得轨迹，并根据采样得到的轨迹估计状态价值
        :param mdp: 马尔可夫决策过程
        :param pi: 策略
        :return: MDP 每个状态的价值估计
        """
        self.sample_trajectory(mdp, pi)

        counts = np.zeros(len(mdp.states))  # 初始化状态计数器
        values = np.zeros(len(mdp.states))  # 初始化状态价值估计
        for episode in self.episodes:
            g = 0
            for i in range(len(episode) - 1, -1, -1):
                s, a, r, s_next = episode[i]
                g = r + mdp.gamma * g
                index = mdp.states.index(s)
                counts[index] += 1
                values[index] += (g - values[index]) / counts[index]
        return values

    def compute_occupancy(self, mdp: MDP, pi: dict[str, dict[str, float]], s: str, a: str, gamma: float) -> float:
        """
        采样获得轨迹，并根据采样得到的轨迹估计对应 (s, a) 的占用度量
        :param mdp: 马尔可夫决策过程
        :param pi: 策略
        :param s: 状态
        :param a: 动作
        :param gamma: 归一化因子
        :return: 给定 MDP 和 pi 下对应 (s, a) 的占用度量
        """
        self.sample_trajectory(mdp, pi)

        rho = 0
        total_times = np.zeros(self.timestep_max)
        occur_times = np.zeros(self.timestep_max)
        for episode in self.episodes:
            for i in range(len(episode)):
                s_opt, a_opt, r, s_next = episode[i]
                total_times[i] += 1
                if s == s_opt and a == a_opt:
                    occur_times[i] += 1

        for i in reversed(range(self.timestep_max)):
            if total_times[i]:
                rho += gamma ** i * (occur_times[i] / total_times[i])
        return (1 - gamma) * rho
