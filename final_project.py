from click import pass_context
import pybullet as p
import pybullet_envs
import pybullet_data
import torch 
import gym
from gym import spaces
import time
from stable_baselines3 import PPO 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import numpy as np        
import math
import os

# see tensorboard : tensorboard --logdir=log
    
class TestudogEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(TestudogEnv, self).__init__()
        self.state = self.init_state()
        self.action_space = spaces.Box(low=-1, high=1, shape=(24,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(37,), dtype=np.float32)
        
    def init_state(self):
        p.connect(p.GUI)
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
        self.testudogid = p.loadURDF("/ENPM690/final_project/urdf/testudog.urdf",[0,0,0.25],[0,0,0,1])
        focus,_ = p.getBasePositionAndOrientation(self.testudogid)
        p.resetDebugVisualizerCamera(cameraDistance=1,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=focus)
        
        # observation --> body: pos, rot, lin_vel ang_vel / joints: pos, vel / foot position? / foot contact?
        body_pos = p.getLinkState(self.testudogid,0)[0]
        body_rot = p.getLinkState(self.testudogid,0)[1]
        body_lin_vel = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[6]
        body_ang_vel = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[7]
        joint_pos = []
        joint_vel = []
        for i in range(12):
            joint_pos.append(p.getJointState(self.testudogid,i)[0])
            joint_vel.append(p.getJointState(self.testudogid,i)[1])        
            
        obs = list(body_pos) + list(body_rot) + list(body_lin_vel) + list(body_ang_vel) + joint_pos + joint_vel
        obs = np.array(obs).astype(np.float32)
        return obs
    
    def reset(self):
        p.disconnect()
        obs = self.init_state()
        self.state = obs
        return obs
        
    def step(self,action):
        p.setJointMotorControlArray(self.testudogid,[0,1,2,3,4,5,6,7,8,9,10,11],p.VELOCITY_CONTROL,\
            targetPositions=math.pi*action[0:12],targetVelocities=2*math.pi*action[12:24])
        # p.setJointMotorControlArray(self.testudogid,[0,1,2,3,4,5,6,7,8,9,10,11],p.POSITION_CONTROL,targetPositions=math.pi*action)
        p.stepSimulation()
        
        # observation --> body: pos, rot, lin_vel ang_vel / joints: pos, vel / foot position? / foot contact?
        body_pos = p.getLinkState(self.testudogid,0)[0]
        body_rot = p.getLinkState(self.testudogid,0)[1]
        body_lin_vel = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[6]
        body_ang_vel = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[7]
        joint_pos = []
        joint_vel = []
        joint_pow = []
        for i in range(12):
            joint_pos.append(p.getJointState(self.testudogid,i)[0])
            joint_vel.append(p.getJointState(self.testudogid,i)[1]) 
            joint_pow.append(p.getJointState(self.testudogid,i)[1]*p.getJointState(self.testudogid,i)[3]) # pow = torque*vel
                   
        obs = list(body_pos) + list(body_rot) + list(body_lin_vel) + list(body_ang_vel) + joint_pos + joint_vel
        obs = np.array(obs).astype(np.float32)
        info = {}
        
        # terminal fail condition eg robot fall  
        body_rot_rpy = p.getEulerFromQuaternion(body_rot) 
        if (body_rot_rpy[1]<0):
            # self.reset()
            reward = -10
            done = True
            return obs, reward, done, info
        
        done = False
        vmax = 0.06
        w1 = 2
        w2 = 0.008
        w3 = 2
        dt = 1/240
        
        # reward --> forward velocity + consumed energy
        reward = w1*(obs[1]-self.state[1]) - w2*sum(joint_pow)*dt + w3*(body_pos[2]-0.2) + 0.01
        print(reward)
        
        self.state = obs
        return obs, reward, done, info 

if (__name__ == '__main__'):
    # set save directory
    model_dir ="ENPM690/final_project/models/PPO"
    log_dir = "ENPM690/final_project/log"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # train loop    
    env = TestudogEnv()
    # check_env(env)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    TIMESTEPS = 2000
    count = 1
    while(True):
        print(count)
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{model_dir}/{TIMESTEPS*count}")
        count += 1
        if count == -1:
            break
    
    # run trained model
    episodes = 1
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            time.sleep(1/240)
                
