from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import gym
import gym.spaces
import gym.utils.seeding
import numpy as np

from panda_gym.pybullet import PyBullet
from multi_arm.envs.core import PyBulletRobot
from multi_arm.envs.core import Task


class MyRobotTaskEnv(gym.Env):
    """Robotic task goal env, as the junction of a task and a robot.

    Args:
        robot (PyBulletRobot): The robot.
        task (Task): The task.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, robots: list, task: Task) -> None:
        assert robots[0].sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robots[0].sim
        self.robots = robots
        if self.robots[0].block_gripper == True:
            self.action_n = 3
        else:
            self.action_n = 4
        self.task = task
        self.seed()  # required for init; can be changed later
        obs = self.reset()
        observation_shape = obs["observation"].shape
        achieved_goal_shape = obs["achieved_goal"].shape
        desired_goal_shape = obs["achieved_goal"].shape
        self.observation_space = gym.spaces.Dict(
            dict(
                observation=gym.spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                desired_goal=gym.spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
                achieved_goal=gym.spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
            )
        )
        self.action_space = gym.spaces.Box(-10.0, 10.0, shape=(len(robots) * self.action_n,), dtype=np.float32)
        self.compute_reward = self.task.compute_reward

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robots[0].get_obs()
        for i in range(len(self.robots) - 1):
            robot_obs = np.concatenate([robot_obs, self.robots[i + 1].get_obs()])
        task_obs = self.task.get_obs()  # object position, velococity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.task.get_achieved_goal()
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal(),
        }

    def reset(self) -> Dict[str, np.ndarray]:
        with self.sim.no_rendering():
            for i in range(len(self.robots)):
                self.robots[i].reset()
            self.task.reset()
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        for i in range(len(self.robots)):
            self.robots[i].set_action(action[i * self.action_n:(i + 1) * self.action_n])
        self.sim.step()
        obs = self._get_obs()
        done = False

        info = {"is_success": self.task.is_success(obs["achieved_goal"], self.task.get_goal())}
        reward = self.task.compute_reward(obs["achieved_goal"], self.task.get_goal(), info)
        assert isinstance(reward, float), "the reward must be float,but got {0}".format(
            type(reward))  # needed for pytype cheking
        done = info["is_success"]
        return obs, reward, done, info

    def seed(self, seed: Optional[int] = None) -> int:
        """Setup the seed."""
        return self.task.seed(seed)

    def close(self) -> None:
        self.sim.close()

    def render(
            self,
            mode: str,
            width: int = 720,
            height: int = 480,
            target_position: np.ndarray = np.zeros(3),
            distance: float = 1.4,
            yaw: float = 45,
            pitch: float = -30,
            roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.

        If mode is "human", make the rendering real-time. All other arguments are
        unused. If mode is "rgb_array", return an RGB array of the scene.

        Args:
            mode (str): "human" of "rgb_array". If "human", this method waits for the time necessary to have
                a realistic temporal rendering and all other args are ignored. Else, return an RGB array.
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Rool of the camera. Defaults to 0.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        return self.sim.render(
            mode,
            width=width,
            height=height,
            target_position=target_position,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )
