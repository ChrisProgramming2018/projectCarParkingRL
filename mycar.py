import airsim
import cv2
import numpy as np
import os
import setup_path 
import time
import matplotlib.pyplot as plt


def drive():
    for idx in range(3):
        # get state of the car
        car_state = client.getCarState()
        print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

        # go forward
        car_controls.throttle = 0.5
        car_controls.steering = 0
        client.setCarControls(car_controls)
        print("Go Forward")
        time.sleep(3)   # let car drive a bit

        # Go forward + steer right
        car_controls.throttle = 0.5
        car_controls.steering = 1
        client.setCarControls(car_controls)
        print("Go Forward, steer right")
        time.sleep(3)   # let car drive a bit

        # go reverse
        car_controls.throttle = -0.5
        car_controls.is_manual_gear = True;
        car_controls.manual_gear = -1
        car_controls.steering = 0
        client.setCarControls(car_controls)
        print("Go reverse, steer right")
        time.sleep(3)   # let car drive a bit
        car_controls.is_manual_gear = False; # change back gear to auto
        car_controls.manual_gear = 0  

        # apply brakes
        car_controls.brake = 1
        client.setCarControls(car_controls)
        print("Apply brakes")
        time.sleep(3)   # let car drive a bit
        car_controls.brake = 0 #remove brake
    
        # get camera images from the car
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
            airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
            airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
            airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGB array
        print('Retrieved images: %d', len(responses))

        for response in responses:
            filename = 'c:/temp/py' + str(idx)
            if not os.path.exists('c:/temp/'):
                os.makedirs('c:/temp/')
            if response.pixels_as_float:
                print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
            elif response.compress: #png format
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
            else: #uncompressed array
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image array H X W X 3
                cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png 


def main():
    # connect to the AirSim simulator 
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()
    amount_of_steps = 50
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
    return state 

def render(responses):
    wait = 0.2
    for response in responses:
        img1d = np.fromstring(response.image_data_uint8, dtype= np.uint8) # get numpy array
        if img1d.shape[0] == 268800:
            img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image array H X W X 3
            #cv2.imshow("video", img_rgb)
            im = plt.imshow(img_rgb)
            im.set_data(img_rgb)
            plt.pause(wait) 

if __name__ == "__main__":
    print("Hello car")
    main()
