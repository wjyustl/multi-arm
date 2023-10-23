from typing import Any, Dict
import numpy as np

from multi_arm.envs.core import Task
from multi_arm.envs.robots.panda import Panda
from multi_arm.utils import distance
from multi_arm.pybullet import PyBullet


class DoubleReach(Task):
    def __init__(
            self,
            sim: PyBullet,
            robots: list,
            reward_type="sparse",
            distance_threshold=0.1,
            goal_range=0.1
    ) -> None:
        super().__init__(sim)
        assert type(robots[0]) == Panda, "the robot's type must be Panda"
        self.robots = robots
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.goal_range_low = np.array([-goal_range / 4, -goal_range / 4, 0])
        self.goal_range_high = np.array([goal_range / 4, goal_range / 4, goal_range / 4])
        self.success_state = [0] * len(self.robots)
        self.success_num = 0
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.3, yaw=30, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_box(
            body_name="bed",
            half_extents=np.array([0.5, 0.25, 0.06]),
            mass=500000.,
            position=np.array([0, 0, 0.06]),
            rgba_color=np.array([0.75, 0.82, 0.94, 1]),

        )
        self.sim.loadURDF(
            body_name="plane",
            fileName="plane.urdf"
        )
        self.sim.create_sphere(
            body_name="target1",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.array([-0.5, 1.125, 0.17]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3])
        )
        self.sim.create_sphere(
            body_name="target2",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.array([-0.5, -1.125, 0.17]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3])
        )

    def get_obs(self) -> np.ndarray:
        ##reach environment's observation 
        return np.array([])

    def get_achieved_goal(self) -> np.ndarray:
        ## return array whose shape is (3*len(robots),)
        ee_positions = self.robots[0].get_ee_position()
        for i in range(len(self.robots) - 1):
            ee_positions = np.concatenate([ee_positions, self.robots[i + 1].get_ee_position()])

        return ee_positions

    def reset(self) -> None:
        self.success_num += 1
        self.goal = self._sample_goal()
        print("self.success_num:", self.success_num)

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        nose = np.random.uniform(self.goal_range_low, self.goal_range_high)
        target1_position = np.array([-0.5, 0.75, 0.17])
        target2_position = np.array([-0.5, -0.75, 0.17])
        target1_position += nose
        target2_position += nose
        self.sim.set_base_pose("target1", target1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target2", target2_position, np.array([0.0, 0.0, 0.0, 1.0]))
        goal = np.concatenate([target1_position, target2_position])
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:

        d1 = distance(achieved_goal[0:3], desired_goal[0:3])
        d2 = distance(achieved_goal[3:6], desired_goal[3:6])

        return np.array(d1 < self.distance_threshold and d2 < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray,
                       info: Dict[str, Any] = ...) -> np.ndarray:
        punishment = False
        d1 = distance(achieved_goal[0:3], desired_goal[0:3])
        d2 = distance(achieved_goal[3:6], desired_goal[3:6])
        for i in range(len(self.robots) - 1):
            for j in range(i + 1, len(self.robots)):
                if (self.sim.getContactPoints(self.robots[i].bodyID, self.robots[j].bodyID)):
                    punishment = True

        d = d1 + d2
        if punishment:
            d += 0.5
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d
