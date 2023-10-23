from panda_gym.pybullet import PyBullet
from multi_arm.envs.robots.panda import Panda
from multi_arm.envs.tasks.MultiReach import MultiReach

sim = PyBullet(render_mode="human")
robot1 = Panda(sim, body_name="panda", block_gripper=True)
robot=[robot1]
task = MultiReach(sim, robot)

task.reset()
print(task.get_obs())
print(task.get_achieved_goal())
# print(task.is_success(task.get_achieved_goal(), task.get_goal()))
# print(task.compute_reward(task.get_achieved_goal(), task.get_goal()))
