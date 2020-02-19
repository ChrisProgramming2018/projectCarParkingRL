from __future__ import division
import os
import numpy as np
import torch
import gym
import gym_car

from torch import optim
from memory import ReplayMemory
from model import DQN
from agent import Agent
from test import test
import gym

import argparse
import cv2
from collections import deque
from datetime import datetime
import tqdm

parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS',
        help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(40), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T',
        help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ',
        help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=5, metavar='VALUE',
        help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625,
        metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε',
        help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE',
        help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(2000),
        metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=10000,
        metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=2, metavar='N',
        help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N',
        help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=10000, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
parser.add_argument('--device', type=str, default='cpu')
# Setup
args = parser.parse_args()

env = gym.make('Car-v0')

action_space = env.action_space.n
print("show action space", action_space)
# Agent
dqn = Agent(args, env)

# If a model is provided, and evaluate is fale, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
    if not args.memory:
        raise ValueError('Cannot resume training without memory save path. Aborting...')
    elif not os.path.exists(args.memory):
        raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))
    mem = load_memory(args.memory, args.disable_bzip_memory)
else:
    mem = ReplayMemory(args, args.memory_capacity)


priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
results_dir = os.path.join('results', args.id)

print("Test agent")


dqn.eval()  # Set DQN (online network) to evaluation mode
T_rewards = []
for _ in range(args.evaluation_episodes):
    done = True
    for step in range(50):
        if done:
            state, reward_sum, done = env.reset(), 0, False
            state = cv2.resize(state[:,:,0], (84, 84), interpolation=cv2.INTER_LINEAR)
            state = torch.tensor(state, dtype=torch.float32, device=args.device).div_(255)
            zeros = torch.zeros_like(state)
            state_buffer = deque([], maxlen=args.history_length)
            state_buffer.append(zeros)
            state_buffer.append(zeros)
            state_buffer.append(zeros)
            state_buffer.append(state)
            state = torch.stack(list(state_buffer), 0)
            print(state.shape)
        if step < 10:
            action = np.random.randint(0, action_space)
        else:
            action = dqn.act_e_greedy(state)  # Choose an action greedily
        print("action", action)
        state, reward, done, _ = env.step(action)  # Step
        state = cv2.resize(state[:,:,0], (84, 84), interpolation=cv2.INTER_LINEAR)
        state = torch.tensor(state, dtype=torch.float32, device=args.device).div_(255)
        state_buffer.append(state)
        state = torch.stack(list(state_buffer), 0)
        reward_sum += reward
    print(" episode reward", reward_sum)
    T_rewards.append(reward_sum)

print(T_rewards)
