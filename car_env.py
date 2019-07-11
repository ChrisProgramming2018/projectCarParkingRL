from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import airsim
import cv2
import os
import setup_path
import time
import matplotlib.pyplot as plt

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


def step(action, car_controls, client):
    if action == 1:
        print("action 1")
        # go forward
        car_controls.throttle = 0.5
        car_controls.steering = 0
        client.setCarControls(car_controls)
        print("Go Forward")

    if action == 2:
        print("action 2")
        # Go forward + steer right
        car_controls.throttle = 0.5
        car_controls.steering = 1
        client.setCarControls(car_controls)
        print("Go Forward, steer right")
    
    if action == 3:
        print("action 3")
        # Go forward + steer left
        car_controls.throttle = 0.5
        car_controls.steering = -1
        client.setCarControls(car_controls)
    
    if action == 4:
        print("action 4")
        # Go forward + steer left
        car_controls.throttle = 0
        car_controls.steering = 0
        client.setCarControls(car_controls)
    
    if action == 5:
        print("action 5")
        # Go forward + steer left
        car_controls.throttle = 0
        car_controls.steering = 0
        car_controls.brake = 1
        client.setCarControls(car_controls)
    pose = client.simGetVehiclePose()
    print("x={}, y={}, z={}".format(pose.position.x_val,
    pose.position.y_val, pose.position.z_val))
    angles = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
    print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))

    state = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
        airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
        airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
        airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGB array
    reward = 0
    done = False
    info = None
    return state, reward, done, info 




class CarEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
                shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        print("reset")
        self._state = 0
        self._episode_ended = False
        self.client = airsim.CarClient()
        return ts.restart(np.array([self._state], dtype=np.int32))
    
    def _step(self, action):
        if action == 1:
            print("action 1")
            # go forward
            car_controls.throttle = 0.5
            car_controls.steering = 0
            self.client.setCarControls(car_controls)
            print("Go Forward")

        if action == 2:
            print("action 2")
            # Go forward + steer right
            car_controls.throttle = 0.5
            car_controls.steering = 1
            self.client.setCarControls(car_controls)
            print("Go Forward, steer right")
    
        if action == 3:
            print("action 3")
            # Go forward + steer left
            car_controls.throttle = 0.5
            car_controls.steering = -1
            self.client.setCarControls(car_controls)
    
        if action == 4:
            print("action 4")
            # Go forward + steer left
            car_controls.throttle = 0
            car_controls.steering = 0
            self.client.setCarControls(car_controls)
    
        if action == 5:
            print("action 5")
            # Go forward + steer left
            car_controls.throttle = 0
            car_controls.steering = 0
            car_controls.brake = 1
            self.client.setCarControls(car_controls)
        pose = self.client.simGetVehiclePose()
        print("x={}, y={}, z={}".format(pose.position.x_val,
        pose.position.y_val, pose.position.z_val))
        angles = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))

        state = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
            airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
            airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
            airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGB array
        reward = 0
        done = False
        info = None
        return state, reward, done, info 



get_new_card_action = 0
end_round_action = 1

environment = CarEnv()
train_env = tf_py_environment.TFPyEnvironment(environment)
eval_env = tf_py_environment.TFPyEnvironment(environment)

print(train_env.observation_spec)

print("load env")
print(train_env.reset())
#print(train_env.step(1))



print("took step")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")

amount_of_steps = 5
num_episode = 1
state = step(6, car_controls,client)
for episode in range(num_episode):
    client.reset()
    print("new episode")
    for steps in range(amount_of_steps):
        print(steps)
        state, reward,  = step(np.random.choice([1,2,3]), car_controls, client)
        print("state length")




print("end env")











# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
amount_of_steps = 5
num_episode = 1
state = step(6, car_controls,client)
for episode in range(num_episode):
    client.reset()
    print("new episode")
    for steps in range(amount_of_steps):
        print(steps)
        state = step(np.random.choice([1,2,3]), car_controls, client)
        print("state length")
        print(len(state))
        #render(state)
state = step(6, car_controls,client)
        #restore to original state
                        
client.reset()
client.enableApiControl(False)









