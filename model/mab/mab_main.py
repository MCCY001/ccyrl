import numpy as np
from matplotlib import pyplot as plt

from model.mab.mab import BernoulliBandit
from model.mab.strategy.decaying_epsilon_greedy import DecayingEpsilonGreedy
from model.mab.strategy.epsilon_greedy import EpsilonGreedy
from model.mab.strategy.thompson_sampling import ThompsonSampling
from model.mab.strategy.ucb import UCB


def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(1)
    K = 10
    bandit_10_arm = BernoulliBandit(10)
    print("随机生成了一个%d臂伯努利老虎机" % K)
    print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
          (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

    np.random.seed(1)
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

    np.random.seed(1)
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
    decaying_epsilon_greedy_solver.run(5000)
    print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

    np.random.seed(1)
    coef = 1  # 控制不确定性比重的系数
    UCB_solver = UCB(bandit_10_arm, coef)
    UCB_solver.run(5000)
    print('上置信界算法的累积懊悔为：', UCB_solver.regret)
    plot_results([UCB_solver], ["UCB"])

    np.random.seed(1)
    thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(5000)
    print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])
