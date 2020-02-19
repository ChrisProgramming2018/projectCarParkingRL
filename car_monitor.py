import setup_path 
import airsim
import math
import cv2 #conda install opencv
import time
import gym
import gym_car
import numpy as np
import gym
import gym_car

# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
car_controls = airsim.CarControls()

start = time.time()

print("Time,Speed,Gear,PX,PY,PZ,OW,OX,OY,OZ")





x_pos_goal = 0.6   #0.545647
y_pos_goal = -2.5  #-1.419126
z_pos_goal = 0.2   #0.176768
w_rot_goal = 1.0    # 0.999967
x_rot_goal = 0.0    # 
y_rot_goal = 0.0    # -0.000095
z_rot_goal = 0.02    #

def get_reward(pose):
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
        x_dif = math.sqrt((x_pos - x_pos_goal)**2)
        y_dif = math.sqrt((y_pos - y_pos_goal)**2)
        z_dif = math.sqrt((z_pos - z_pos_goal)**2)

        x_r_dif = math.sqrt((x_rot - x_rot_goal)**2)
        y_r_dif = math.sqrt((y_rot - y_rot_goal)**2)
        z_r_dif = math.sqrt((z_rot - z_rot_goal)**2)
        dif = x_dif + y_dif + z_dif + x_r_dif + y_r_dif + z_r_dif
        print("\n", dif)
        if y_dif >= 2:
            reward = 0
        else:
            reward  = max(-dif, 0)
        if dif < 3:
            reward = 0.7**abs(dif)
        if dif < 0.3:
            reward = 1
        return reward

def goal(pose):
    x_pos = pose.position.x_val
    y_pos = pose.position.y_val
    z_pos = pose.position.z_val
     
    x_rot = pose.orientation.x_val
    y_rot = pose.orientation.y_val
    z_rot = pose.orientation.z_val
    
    x_dif = math.sqrt((x_pos - x_pos_goal)**2)
    
    y_dif = math.sqrt((y_pos - y_pos_goal)**2)
    
    z_dif = math.sqrt((z_pos - z_pos_goal)**2)
    
    x_r_dif = math.sqrt((x_rot - x_rot_goal)**2)
    y_r_dif = math.sqrt((y_rot - y_rot_goal)**2)
    
    z_r_dif = math.sqrt((z_rot - z_rot_goal)**2)
    
    eps = 0.15
    debug_message = ' position difference of x: {:.2f}, '
    debug_message += 'y: {:.2f}, z: {:.2f}, x_r: '
    debug_message += '{:.2f}, y_r: {:.2f}, z_r: {:.2f}'
    
    if x_dif < eps and y_dif < eps and z_dif < eps and x_r_dif < eps  and y_r_dif < eps  and z_r_dif < eps:
        text =  ' goal'
    else:
        text =  ' noth'
    
    debug_message += text
    debug_message = debug_message.format(x_dif, y_dif, z_dif, x_r_dif, y_r_dif , z_r_dif)
    print(debug_message, end='\r', flush=True)






# monitor car state while you drive it manually.
#env = gym.make('Car-v0')

while (cv2.waitKey(1) & 0xFF) == 0xFF:
#while (False):
    # get state of the car
    car_state = client.getCarState()
    #pos = car_state.kinematics_estimated.position
    pose = client.simGetVehiclePose()
    time.sleep(1)
    #pose = client.simGetVehiclePose()
    
    print("reward", get_reward(pose))
    #goal(pose)
    #orientation = car_state.kinematics_estimated.orientation
    milliseconds = (time.time() - start) * 1000
    print("%f,%f,%f,%f,%f,%f,%f" % \
       ( pose.position.x_val, pose.position.y_val, pose.position.z_val, 
        pose.orientation.w_val,pose.orientation.z_val, pose.orientation.y_val, pose.orientation.z_val))
    time.sleep(0.1)




