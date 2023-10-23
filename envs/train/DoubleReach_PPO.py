import numpy as np
from typing import Optional
import pybullet as p
import math
from multi_arm.pybullet import PyBullet
from multi_arm.envs.core import RobotTaskEnv
from multi_arm.envs.my_core import MyRobotTaskEnv
from multi_arm.envs.robots.panda import Panda
from multi_arm.envs.tasks.DoubleReach import DoubleReach

import torch as th 
from stable_baselines3 import PPO,DDPG
from stable_baselines3.common.env_util import make_vec_env


class MultiReachPPOEnv(MyRobotTaskEnv):
    def __init__(
        self,
        reward_type: str = "dense",
        control_type: str = "ee",
    ) -> None:
        self.sim = PyBullet(render=False)
        robot1 = Panda(self.sim,body_name="panda",block_gripper=True, base_position=np.array([-1, 1.5, 0.0]), control_type=control_type)
        robot2 = Panda(self.sim,body_name="panda2",block_gripper=True,base_position=np.array([-0.9, -1.5, 0.0]),control_type=control_type)
        self.sim.set_base_pose(body="panda",position= np.array([-1, 1.5, 0.0]),orientation= np.array([0,0,-math.pi/2]))
        self.sim.set_base_pose(body="panda2",position= np.array([-0.9,-1.5, 0.0]),orientation= np.array([0,0,math.pi/2]))
        robots = [robot1,robot2]
        task = DoubleReach(sim=self.sim,robots=robots,reward_type=reward_type)
        super().__init__(
            robots,
            task
        )

env = MultiReachPPOEnv()
# total training timesteps





total_timesteps = 1.6e6
# Learning rate for the PPO optimizer
learning_rate = 0.001
# Batch size for training the PPO model
batch_size = 2048
tb_log_name = f"DoubleReach_PPO_{learning_rate}" 
model_save_name = f"DoubleReach_PPO_{learning_rate}.zip"
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="runs",
    batch_size=batch_size,
    learning_rate=learning_rate,
    #train_freq=(2048*3,"step")
    
)
model.learn(
    total_timesteps=total_timesteps,
    tb_log_name=tb_log_name,
    progress_bar=True
)


model.save(model_save_name)



