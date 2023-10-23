import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('CartPole-v0')

print(env.action_space) # 返回动作空间类型，如:Discrete(2)
print(env.observation_space) # 返回环境空间
print(env.action_space.sample()) # 从动作空间中随机取一个向量
print(env.observation_space.sample()) # 从环境空间中随机取一个向量

episodes = 5  # 游戏轮次数为5
for episode in range(1, episodes + 1):
    state = env.reset()  # 游戏初始化，返回初始游戏状态
    done = False  # done判断游戏是否结束
    score = 0  # score存储每一轮游戏的总分数

    while not done:
        env.render()  # 渲染游戏画面
        action = env.action_space.sample()  # 按照游戏的动作空间随机抽样(动作空间的类型和大小是事先定好的，只要是动作空间中的一个向量，都可以作为游戏的输入)
        n_state, reward, done, info = env.step(action)  # 游戏向前迈进一步，返回当前状态，当前这步的奖励，游戏是否结束，以及其他信息
        score += reward

print(env.action_space) # 返回动作空间类型，如:Discrete(2)
print(env.observation_space) # 返回环境空间
print(env.action_space.sample()) # 从动作空间中随机取一个向量
print(env.observation_space.sample()) # 从环境空间中随机取一个向量

env.close()  # 关闭游戏环境

