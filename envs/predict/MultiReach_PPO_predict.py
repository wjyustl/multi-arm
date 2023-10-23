import numpy as np
import math
from multi_arm.pybullet import PyBullet
from multi_arm.envs.my_core import MyRobotTaskEnv
from multi_arm.envs.robots.panda import Panda
from multi_arm.envs.tasks.MultiReach import MultiReach
from stable_baselines3 import PPO


class MultiReachPPOEnv(MyRobotTaskEnv):
    def __init__(
            self,
            reward_type: str = "dense",
            control_type: str = "ee",) -> None:
        self.sim = PyBullet(render=True)
        robot1 = Panda(self.sim, body_name="panda", block_gripper=True, base_position=np.array([-1, 1.5, 0.0]),
                       control_type=control_type)
        robot2 = Panda(self.sim, body_name="panda2", block_gripper=True, base_position=np.array([-0.9, -1.5, 0.0]),
                       control_type=control_type)
        robot3 = Panda(self.sim, body_name="panda3", block_gripper=True, base_position=np.array([1, -1.5, 0.0]),
                       control_type=control_type)
        robot4 = Panda(self.sim, body_name="panda4", block_gripper=True, base_position=np.array([0.9, 1.5, 0.0]),
                       control_type=control_type)
        self.sim.set_base_pose(body="panda", position=np.array([-1, 1.5, 0.0]),
                               orientation=np.array([0, 0, -math.pi / 2]))
        self.sim.set_base_pose(body="panda2", position=np.array([-0.9, -1.5, 0.0]),
                               orientation=np.array([0, 0, math.pi / 2]))
        self.sim.set_base_pose(body="panda3", position=np.array([1, -1.5, 0.0]),
                               orientation=np.array([0, 0, math.pi / 2]))
        self.sim.set_base_pose(body="panda4", position=np.array([0.9, 1.5, 0.0]),
                               orientation=np.array([0, 0, -math.pi / 2]))
        robots = [robot1, robot2, robot3, robot4]
        task = MultiReach(sim=self.sim, robots=robots, reward_type=reward_type)
        super().__init__(
            robots,
            task
        )


env = MultiReachPPOEnv()
print(env.action_space)                 # 返回动作空间类型，如:Discrete(2)
print(env.observation_space)            # 返回环境空间
print(env.action_space.sample())        # 从动作空间中随机取一个向量
print(env.observation_space.sample())   # 从环境空间中随机取一个向量

model = PPO.load(
    r'D:\pyProject\my-multi-arm\multi_arm\envs\train\model\MultiReach_PPO_2023-10-17_111123.zip',
    print_system_info=True,
    device="auto",
)

obs = env.reset()   # 环境初始化
score = 0
steps = 0

for i in range(1000):
    action, _state = model.predict(obs, deterministic=False)     # 使用model来预测动作,返回预测的动作和下一个状态
    obs, reward, done, info = env.step(action)  # 向前迈进、返回当前状态/奖励
    print(obs)
    score += reward
    env.render(mode="human")    # 画面渲染

    if done:
        print("Finished after {} steps".format(steps))
        obs = env.reset()
        steps = 0
    else:
        steps += 1

env.close()
print("score=", score)
