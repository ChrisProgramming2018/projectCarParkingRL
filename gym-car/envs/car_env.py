import gym
import numpy as np
import copy
import random
import math
import queue
import time
from gym import spaces, error
import airsim
import cv2
import os
import setup_path
import time
import matplotlib.pyplot as plt
from PIL import Image




class CarEnv(gym.Env):
    def __init__(self):
        self.client = airsim.CarClient("10.8.105.156")
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        self.action_space = spaces.Discrete(4)
        self.time_step = 0 
        self.x_pos_goal = 0.6   #0.545647 0.6
        self.y_pos_goal = -2.5  #-2.5
        self.z_pos_goal = 0.2   #0.2
        self.counter_no_state = 0 
        self.w_rot_goal = 1.0    # 0.999967
        self.x_rot_goal = 0.0    # 
        self.y_rot_goal = 0.0    # -0.000095
        self.z_rot_goal = 0.02    # 0.019440
        self.max_step = 10  # steps to check if blocked 
        self.last_states = queue.deque(maxlen=self.max_step)        
        self.height = 80
        self.width = 80  # old 320 
        # self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(low = 0, high = 255, shape=(1, self.height, self.width))
        self.debug_mode = False
        self.goal_counter = 0
        self.count = 0
        self.state = None
    
    def reset(self):
        """
        This function resets the environment and returns the game state.
        """
        self.client.reset()
        
        self.state = self._get_state("3")  # forward camera
        # state2 = self._get_state("4") # backward camera
        #state_n = np.zeros((self.height, self.width))
        #print("reset", state_n.shape) 
        #state = np.stack((state_n, state_n))
        #print("reset 1", state.shape)
        #state_n = np.expand_dims(state_n, axis=0)
        #state = np.concatenate((state, state_n))
        #state = np.concatenate((state, state_n))
        #print("reset add", state.shape) 
        #self.state = state
        pose = self.client.simGetVehiclePose()
        reward = self._get_reward(pose)
        self.time_step = 0 
        #print("reset end", state.shape) 
        return np.array(self.state)

        

    def render(self, responses, mode='human', close=False):
        """
        This function renders the current game state in the given mode.
        """
        wait = 0.2
        for response in responses:
            img1d = np.fromstring(response.image_data_uint8, dtype= np.uint8) # get numpy array
            if img1d.shape[0] == 268800:
                img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image array H X W>
                #cv2.imshow("video", img_rgb)
                im = plt.imshow(img_rgb)
                im.set_data(img_rgb)
                plt.pause(wait)



    def step(self, action):
        """

        Parameters
        ----------
        action : int
            The action is an angle between 0 and 180 degrees, that
            decides the direction of the bubble.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing the
                state of the environment.
            reward (float) :
                amount of reward achieved by the previous action.
            episode_over (bool) :
                whether it's time to reset the environment again.
            info (dict) :
                diagnostic information useful for debugging.
        """
        self.time_step += 1 
        self.car_controls.brake = 0
        #print("action", action)
        if action == 0:
            # go forward
            self.car_controls.throttle = 0.5
            self.car_controls.steering = 0
            self.client.setCarControls(self.car_controls)
        
        if action == 1:
            # Go forward + steer right
            self.car_controls.throttle = 0.5
            self.car_controls.steering = 1
            self.client.setCarControls(self.car_controls)
    
        if action == 2:
            # Go forward + steer left
            self.car_controls.throttle = 0.5
            self.car_controls.steering = -1
            self.client.setCarControls(self.car_controls)
    
        if action == 3:
            #  stop
            self.car_controls.throttle = 0
            self.car_controls.steering = 0
            self.car_controls.brake = 1
            self.client.setCarControls(self.car_controls)
    
        if action == 4:

            # Go backward
            self.car_controls.throttle = -0.5
            self.car_controls.steering = 0
            self.client.setCarControls(self.car_controls)
        
        if action == 5:
            # Go backward + steer right
            self.car_controls.throttle = -0.5
            self.car_controls.steering = 1
            self.client.setCarControls(self.car_controls)
        
        if action == 6:
            # Go backward + steer left
            self.car_controls.throttle = -0.5
            self.car_controls.steering = -1
            self.client.setCarControls(self.car_controls)
        # act  for certain time and stop
        time.sleep(0.2)

        #print("throttle", self.car_controls.throttle)
        #print("steering", self.car_controls.steering)
        self.car_controls.throttle = 0
        self.car_controls.steering = 0
        self.car_controls.brake = 1
        self.client.setCarControls(self.car_controls)
        
        pose = self.client.simGetVehiclePose()
        reward = self._get_reward(pose)
        done  = False
        self.state = self._get_state("3")  # forward camera

        # state2 = self._get_state("4") # backward camera
        #state = self.state[1:,:,:]
        #print("STEP", state.shape)
        #self.state = np.concatenate((state, np.expand_dims(state1,axis=0)))
        #print("exit", self.state.shape)
        #print(" add", self.state.shape)  
        #if reward > -2:
        #    reward = reward + 2 
        
        if self._is_goal(pose):
            reward = 1
        return self.state, reward, done, pose 
    
    def set_debug(self, debug=False):
        """
        """
        print("my env")
        self.debug_mode = debug

    def process_image(self, state):
        """ 
        """
        if self.debug_mode:
            img = Image.fromarray(state, 'RGB')
            img.show()
            img.close()

        state =  cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state,(self.width, self.height))
        # normailze (values between 0 and 1) done in the main program
        
        return state

    
    def _get_reward(self, pose):
        """
        This function calculates the reward.
        """
        x_pos = pose.position.x_val
        y_pos = pose.position.y_val
        z_pos = pose.position.z_val
        
        x_rot = pose.orientation.x_val
        y_rot = pose.orientation.y_val
        z_rot = pose.orientation.z_val

        # calculate difference between current and goal
        dif = 0
        x_dif = math.sqrt((x_pos - self.x_pos_goal)**2)
        y_dif = math.sqrt((y_pos - self.y_pos_goal)**2)
        z_dif = math.sqrt((z_pos - self.z_pos_goal)**2)

        x_r_dif = math.sqrt((x_rot - self.x_rot_goal)**2)
        y_r_dif = math.sqrt((y_rot - self.y_rot_goal)**2)
        z_r_dif = math.sqrt((z_rot - self.z_rot_goal)**2)
        dif = x_dif + y_dif + z_dif + x_r_dif + y_r_dif + z_r_dif
        if y_dif >= 3:
            reward = -1
        else:
            reward  = max(-dif, -3)
        
        if dif < 2:
            reward = 0.5**abs(dif)
        if dif < 0.3:
            reward = 1

        
        return reward




    def _is_goal(self, pose):
        x_pos = pose.position.x_val
        y_pos = pose.position.y_val
        z_pos = pose.position.z_val
        
        x_rot = pose.orientation.x_val
        y_rot = pose.orientation.y_val
        z_rot = pose.orientation.z_val

        # calculate difference between current and goal
        
        x_dif = math.sqrt((x_pos - self.x_pos_goal)**2)
        y_dif = math.sqrt((y_pos - self.y_pos_goal)**2)
        z_dif = math.sqrt((z_pos - self.z_pos_goal)**2)

        x_r_dif = math.sqrt((x_rot - self.x_rot_goal)**2)
        y_r_dif = math.sqrt((y_rot - self.y_rot_goal)**2)
        z_r_dif = math.sqrt((z_rot - self.z_rot_goal)**2)
        eps = 0.3
        if self.debug_mode:
            debug_message = '                                position difference of x: {:.2f}, '
            debug_message += 'y: {:.2f}, z: {:.2f}, x_r: '
            debug_message += '{:.2f}, y_r: {:.2f}, z_r: {:.2f}' 
            debug_message = debug_message.format(x_dif, y_dif, z_dif, x_r_dif, y_r_dif , z_r_dif)
            print(debug_message, end='\r', flush=True) 
        if x_dif < eps and y_dif < eps and z_dif < eps and x_r_dif < eps  and y_r_dif < eps  and z_r_dif < eps:
            self.goal_counter +=1
            return True
        return False 

        


    def _get_state(self, camera):
        """ if camera is 3 forward and 4 backward
        """
        # get back camera 
        responses = self.client.simGetImages([airsim.ImageRequest(camera,
            airsim.ImageType.DepthVis),
            airsim.ImageRequest("3", airsim.ImageType.DepthPerspective, True), #depth in pe>
            airsim.ImageRequest(camera, airsim.ImageType.Scene), #scene
            airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)])
        state_exists = False
        for response in responses:
            img1d = np.fromstring(response.image_data_uint8, dtype= np.uint8) # get numpy array
            if img1d.shape[0] == 268800:
                state_exists = True
                state= img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image
                #print("state", state)
                #img = Image.fromarray(state, 'RGB')
                #text = 'my-{}.png'.format(self.count)
                #self.count +=1
                #img.save(text)

                state = self.process_image(state) 
        if state_exists == False:
            state = np.zeros((self.height, self.width))
        return state
