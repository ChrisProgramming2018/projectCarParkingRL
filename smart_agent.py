from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import numpy as np
import time
import os
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image
import tensorflow as tf
import queue
import csv
from collections import  deque
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()
import gym
import gym_car
import time

print(dqn_agent.__file__)

print("load env ..")
env_name =("Car-v0")
#env = gym.make("Car-v0")

fc_layer_params = (512,)
  
batch_size = 32  # @param
learning_rate = 0.00025  # @param



train_py_env = suite_gym.load(env_name, discount=0.99, max_episode_steps=1000)
eval_py_env = suite_gym.load(env_name, discount=0.99, max_episode_steps=1000)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)




q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)



optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

tf_agent = dqn_agent.DdqnAgent(train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        epsilon_greedy=0.9,
        td_errors_loss_fn=dqn_agent.element_wise_squared_loss,
        train_step_counter=train_step_counter)

tf_agent.initialize()

print("ready to go")

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
        train_env.action_spec())
print("policy")

def compute_avg_return(environment, policy, num_episodes=2):
    total_return = 0.0
    test = 0
    for _ in range(num_episodes):
        print("Episode", _)
        test += 1
        time_step = environment.reset()
        episode_return = 0.0
        #while not time_step.is_last():
        for step in range(50):
            action_step = policy.action(time_step)
            #print("step:  ", step ,"action: ", action_step.action.numpy()[0])
            #time.sleep(1)
            time_step = environment.step(action_step.action.numpy()[0])
            episode_return += time_step.reward.numpy()[0]
            print("current reward {}".format(time_step.reward.numpy()[0]))
        print("reward sum ", episode_return)
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return



#FLAGS = flags.FLAGS
#root_dir = FLAGS.root_dir
root_dir = "/home/leiningc/project/data"
root_dir = os.path.expanduser(root_dir)
train_dir = os.path.join(root_dir, 'parking')
eval_dir = os.path.join(root_dir, 'eval')
print("root_dir", root_dir)
print("train_dir", train_dir)
train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
        ] 

global_step = tf.compat.v1.train.get_or_create_global_step()
train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))

train_checkpointer.initialize_or_restore()
eval_policy = tf_agent.policy

print("trained agent")
print(compute_avg_return(eval_env, eval_policy, 10))

print("random agent")
print(compute_avg_return(eval_env, random_policy, 10))
