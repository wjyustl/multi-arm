from typing import Optional
import numpy as np

from multi_arm.envs.tasks.reach import Reach
from multi_arm.envs.core import RobotTaskEnv
from multi_arm.pybullet import PyBullet
from multi_arm.envs.robots.panda import Panda

import torch as th
from stable_baselines3 import PPO


class PandaReachEnv(RobotTaskEnv):
    def __init__(
            self,
            reward_type: str = "dense",
            control_type: str = "ee",
    ) -> None:
        sim = PyBullet(render=True)
        robot = Panda(sim, body_name='panda', block_gripper=True, base_position=np.array([-0.5, 0.0, 0.0]),
                      control_type=control_type)
        task = Reach(sim=sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(
            robot,
            task
        )


env = PandaReachEnv()
# total training timesteps
total_timesteps = 20000 * 2
# Learning rate for the PPO optimizer
learning_rate = 0.001
# Batch size for training the PPO model
batch_size = 64
# Number of hidden units for the policy networkzl
pi_hidden_units = [64, 64]
# Number of hidden units for the value function network
vf_hidden_units = [64, 64]
# Custom actor (pi) and value function (vf) networks
# of two layers of size 64 each with Relu activation function
policy_kwargs = dict(
    activation_fn=th.nn.ReLU, net_arch=[dict(pi=pi_hidden_units, vf=vf_hidden_units)]
)

tb_log_name = f"Reach_PPO_{learning_rate}"
model_save_name = f"Reach_PPO_{learning_rate}.zip"
log_dir = './tensorboard'

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    batch_size=batch_size,
    normalize_advantage=True,
    learning_rate=learning_rate,
    policy_kwargs=policy_kwargs,
)
model.learn(
    total_timesteps=total_timesteps,
    tb_log_name=tb_log_name
)

model.save(model_save_name)
