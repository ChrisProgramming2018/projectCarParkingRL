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
import argparse
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
import time


def main(arg, pars):
    """
    
    
    """
    print("load env ..")
    env_name =("CartPole-v0")
    #env = gym.make("Car-v0")
    max_steps = 200
    start = time.time()
    env = suite_gym.load(env_name, discount=0.99, max_episode_steps=1000)
    print(arg.tau)
    print_parameter(arg, pars)



def print_parameter(args, parser):
    """
    """
    for para in vars(args):
        print(para, parser.get_default(para))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', default=500)
    parser.add_argument('--max_t', default=2000)
    parser.add_argument('--eps_start', default=1.0)
    parser.add_argument('--eps_end', default=0.01)
    parser.add_argument('--eps_decay', default=0.990)
    parser.add_argument('--buffer-size', default=1e5, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--noise', default=False)
    parser.add_argument('--tau', default=1e-3)
    parser.add_argument('--lr', default=5e-4)
    parser.add_argument('--update-every', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--hidden_size_1', default=128)
    parser.add_argument('--hidden_size_2', default=64, type=int)
    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--replay-frequency', type=int, default=10, metavar='k', help='Frequency of sampling from memory')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='sigma', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--learn-start', type=int, default=int(800), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='eps', help='Adam epsilon')
    parser.add_argument('--priority-exponent', type=float, default=0.0, metavar='omega', help='Prioritised experience replay exponent (originally denoted alpha)')
    parser.add_argument('--priority-weight', type=float, default=0.8, metavar='beata', help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--evaluation-size', type=int, default=50000, metavar='N', help='Number of transitions to use for validating Q')
    parser.add_argument('--history-length', type=int, default=1, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--discount', type=float, default=0.99, metavar='gamma', help='Discount factor')
    parser.add_argument('--multi-step', type=int, default=1, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--target-update', type=int, default=int(4), metavar='tau', help='Number of steps after which to update target network')
    parser.add_argument('--reward-clip', type=int, default=0, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--model_num', default=1)
    arg = parser.parse_args()
    main(arg, parser)
