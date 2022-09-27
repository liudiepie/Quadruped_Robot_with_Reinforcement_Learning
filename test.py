import pybullet as p
import pybullet_envs
import pybullet_data
import torch as th
import gym
import time
from stable_baselines3 import ppo 
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

print(th.version.cuda)


# env = gym.make()

# env.render(mode='human')

# MAX_AVERAGE_SCORE = 271

# policy_kwargs = dict(activation_fn=th.nn.LeakyReLU,net_arch=[512,512])
# model = PPO()

# setup simulation
p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0,0,-9.8)
p.setRealTimeSimulation(0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())


plane = p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
# testudog = p.loadURDF("franka_panda/panda.urdf",useFixedBase = True)
testudog = p.loadURDF("/ENPM690/final_project/urdf/testudog.urdf",[0,0,0.25],[0,0,0,1])
jointid = 0
num_joints = p.getNumJoints(testudog)
lower = p.getJointInfo(testudog, jointid)[8]
upper = p.getJointInfo(testudog, jointid)[9]

p.setJointMotorControlArray(testudog,range(4),p.POSITION_CONTROL,targetPositions=[1.5]*4)
for step in range(500):
    p.stepSimulation()
    print(p.getJointStates(testudog,[jointid]))
    time.sleep(1/240)