import gym
import numpy as np

env = gym.make("CliffWalking-v0")
n_observation, n_action = env.observation_space.n, env.action_space.n
Q = np.zeros((n_observation, n_action))
learning_rate = 0.9  # 学习率
gamma = 0.9
max_episode_count = 200  # 训练使用的 episode 数


def epsilon_greedy(obs, epsilon=0.1):
    """
    根据 obs 选择 action
    有 1-epsilon+(epsilon/|actions|) 的概率选择价值最大的action，其他的action则均分剩下的概率
    """
    if np.random.uniform(0, 1) < (1.0 - epsilon):
        now_Q = Q[obs, :]
        max_Q = np.max(now_Q)
        candidate_action = np.where(now_Q == max_Q)[0]
        next_action = np.random.choice(candidate_action)
    else:
        next_action = np.random.choice(n_action)
    return next_action


def update_Q(obs, act, now_reward, next_obs, is_done):
    """
    通过这一次转移更新Q表
    :param obs: 上一状态
    :param act: 先前状态转移到当前状态采取的action
    :param now_reward: 该状态对应的action下对应的reward
    :param is_done: 该episode是否结束
    :param next_obs: 转移到的下一状态
    :return: void
    """
    predict = Q[obs, act]
    if is_done:
        target = now_reward
    else:
        target = now_reward + gamma * np.max(Q[next_obs, :])
    Q[obs, act] += learning_rate * (target - predict)


def train_running():
    """
    进行训练，训练得出一张 Q-Table
    :return:void
    """
    print("\n======================================================")
    print("TRAINNING")
    print("======================================================")
    observation = env.reset()  # 创建新的 episode
    train_total_reward = 0
    train_total_step = 0
    while True:
        action = epsilon_greedy(observation, epsilon=0.1)
        next_observation, reward, done, _ = env.step(action)
        update_Q(observation, action, reward, next_observation, done)
        observation = next_observation
        train_total_reward += reward
        train_total_step += 1
        if done:
            break
    return train_total_reward, train_total_step


def test_running():
    """
    利用训练出的 Q-Table 对环境进行测试
    :return:
    """
    print("\n======================================================")
    print("TESTING")
    print("======================================================")
    observation = env.reset()
    total_reward = 0
    total_step = 0
    while True:
        action = epsilon_greedy(observation, epsilon=0)
        next_observation, reward, done, info = env.step(action)
        observation = next_observation
        total_reward += reward
        total_step += 1
        print(action, end=' ')
        if done:
            break

    print("\nTotal Reward: " + str(total_reward))
    print("Total Step: " + str(total_step))


for episode_count in range(max_episode_count):
    train_total_reward, train_total_step = train_running()
    print("Episode #" + str(episode_count + 1) + "  \tReward: " + str(train_total_reward) + " \tSteps: " + str(
        train_total_step))
test_running()
