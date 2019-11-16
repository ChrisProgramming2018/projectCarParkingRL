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
print(dqn_agent.__file__)
tf.compat.v1.enable_v2_behavior()
import gym
import gym_car
import time
from helper import collect_data, print_parameter, compute_avg_return, train

def main(arg, pars):
    """
    
    
    """
    print("load env ..")
    env_name =("Car-v0")
    #env = gym.make("Car-v0")
    env = suite_gym.load(env_name, discount=arg.gamma, max_episode_steps=arg.max_t)
    print_parameter(arg, pars)
    train_py_env = suite_gym.load(env_name, discount=arg.gamma, max_episode_steps=arg.max_t)
    eval_py_env = suite_gym.load(env_name, discount=arg.gamma, max_episode_steps=arg.max_t)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    print("env loaded")
    train_dir = os.path.join(arg.root_dir, 'network_weights')
    eval_dir = os.path.join(arg.root_dir, 'eval')
    
    train_env.reset()
    fc_layer_params = (arg.hidden_size_1,)
    q_net = q_network.QNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=fc_layer_params)
              
                
                 
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=arg.lr)
    train_step_counter = tf.compat.v2.Variable(0)
                     
    tf_agent = dqn_agent.DdqnAgent(train_env.time_step_spec(), train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            epsilon_greedy=arg.eps_start,
            td_errors_loss_fn=dqn_agent.element_wise_squared_loss,
            train_step_counter=train_step_counter)
    
    
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
    
    
    
    if arg.continue_training == False:
        tf_agent.initialize()
        if os.path.exists("network_weights/*"):
            os.remove("network_weights/*")
    else:
        print("Continue Training")
        train_checkpointer.initialize_or_restore()
    print("ready to go")
    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=arg.buffer_size)
    tf_agent.collect_data_spec
    tf_agent.collect_data_spec._fields
    collect_data(train_env, random_policy, replay_buffer, steps=arg.learn_start, max_t=40)
    print("create dataset")
    dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=arg.batch_size, num_steps=2).prefetch(3)
    iterator = iter(dataset)
    
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)
    # Reset the train step
    tf_agent.train_step_counter.assign(0)
    avg_return = compute_avg_return(eval_env, tf_agent.policy, arg.num_eval_episodes)
    returns = [avg_return]
    returns_average = [avg_return]
    train_loss_average = [1]
    score = 0
    scores_window = deque(maxlen=100)       # last 100 scores
    total_train_loss = deque(maxlen=100)       # last 100 scores
    

    train(arg, tf_agent, train_env, eval_env, replay_buffer, iterator, train_checkpointer) 




def print_parameter(args, parser):
    """
    print the default parameter to the terminal

    """
    print("Default parameter")

    for para in vars(args):
        text = str(para) +  " : "
        print(text + str(parser.get_default(para)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', default=25000,type=int)
    parser.add_argument('--num_eval_episodes', default=3)
    parser.add_argument('--save_weights_every', default=100, type=int)
    parser.add_argument('--eval_interval', default=100, type=int)
    parser.add_argument('--max_t', default=30, type=int)
    parser.add_argument('--eps_start', default=1.0, type=float)
    parser.add_argument('--eps_end', default=0.1, type=float)
    parser.add_argument('--eps_decay', default= 0.99993, type=float)
    parser.add_argument('--buffer-size', default=450000, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--continue_training', default=False, type=bool)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=1e-3, type=float)
    parser.add_argument('--lr', default=0.00025,type=float)
    parser.add_argument('--repeat_training', default=1, type=int)
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--hidden_size_1', default=512)
    parser.add_argument('--learn-start', type=int, default=int(300), metavar='STEPS', help='Number of Episodes before start learning')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--evaluation-size', type=int, default=50000, metavar='N', help='Number of transitions to use for validating Q')
    parser.add_argument('--history-length', type=int, default=1, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--discount', type=float, default=0.99, metavar='gamma', help='Discount factor')
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--root_dir', default="", type=str)
    parser.add_argument('--model_num', default=1)
    arg = parser.parse_args()
    main(arg, parser)
