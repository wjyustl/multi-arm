import numpy as np
import math
import time
from multi_arm.pybullet import PyBullet
from multi_arm.envs.my_core import MyRobotTaskEnv
from multi_arm.envs.robots.panda import Panda
from multi_arm.envs.tasks.MultiReach import MultiReach

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class MultiReachPPOEnv(MyRobotTaskEnv):
    def __init__(
            self,
            reward_type: str = "dense",
            control_type: str = "ee",) -> None:
        self.sim = PyBullet(render=False)
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

total_timesteps = int(1.6e6)
learning_rate = 0.001
batch_size = 2048

tb_log_name = f"MultiReach_PPO_{time.strftime('%Y-%m-%d_%H%M%S')}"
model_save_path = fr"model\MultiReach_PPO_{time.strftime('%Y-%m-%d_%H%M%S')}.zip"
log_dir = r"tensorboard"

model = PPO(
    policy="MultiInputPolicy",              # sb3策略之一,决定了Agent应该如何在环境中运行
    env=env,
    verbose=1,                              # 1冗余信息输出、0不输出、2debug
    tensorboard_log=log_dir,
    batch_size=batch_size,
    learning_rate=learning_rate,
    # buffer_size=100000,
    # replay_buffer_class=HerReplayBuffer,
    device="auto"
)

model.learn(
    total_timesteps=total_timesteps,
    tb_log_name=tb_log_name,                # TensorBoard日志运行名称
    progress_bar=True
)

model.save(model_save_path)
