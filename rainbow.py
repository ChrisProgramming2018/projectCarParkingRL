from __future__ import division
import os
import numpy as np
import torch
from PIL import Image
import gym
import gym_car

from torch import optim
from memory import ReplayMemory
from model import DQN
from agent import Agent
from test import test, _plot_line
import gym
import sys
import argparse
import bz2
import pickle
import cv2
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import tqdm



def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, 'rb') as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)




def save_and_plot(num_iterations, returns_average, model=1):
    """
    """
    os.system('mkdir -p results/model-{}'.format(model))
    fig, ax = plt.subplots()
    iterations = range(0, len(returns_average))
    ax.plot(iterations, returns_average,'b', label='average')
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('C0')
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    #plt.ylim(top=25)
    plt.savefig('results/model-1/scores-{}.png'.format(num_iterations))
    print("plot saved")





def eval_policy(args, env):
    action_space = env.action_space.n
    print("show action space", action_space)
    # Agent
    dqn = Agent(args, env)
  
    size = 84
    episode_reward = 0
    eps = 0.1
    for episode in range(2):
        print("Episode ", episode)
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=args.device).div_(255)
        zeros = torch.zeros_like(state)
        state_buffer = deque([], maxlen=args.history_length)
        state_buffer.append(zeros)
        state_buffer.append(zeros)
        state_buffer.append(zeros)
        state_buffer.append(state)                        
        state = torch.stack(list(state_buffer),0)
        for step in range(2):
            action = dqn.act_e_greedy(state, eps)  # Choose an action greedily (with noisy weights)
            next_state, reward, done, _ = env.step(action)
            print(reward)
            episode_reward += reward
            if step == 39:
                done = True
            next_state = cv2.resize(next_state, (size, size), interpolation=cv2.INTER_LINEAR)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=args.device).div_(255)
            state_buffer.append(next_state)
            state = torch.stack(list(state_buffer), 0)
        print("Epiosde reward ", episode_reward)


def train(args, env):
    action_space = env.action_space.n
    print("show action space", action_space)
    print("state space", env.observation_space)
    # Agent
    dqn_1 = Agent(args, env)
    dqn_2 = Agent(args, env)
    
    results_dir = os.path.join('results', args.id)
    print("result dir", results_dir)
    
    T, done = 0, True
    # If a model is provided, and evaluate is fale, presumably we want to resume, so try to load memory
    print(" ags training", args.continue_training)
    
    args.continue_training = False
    if args.continue_training:
        print("Continue Training Load buffer 1 ...")
        
        args.memory = results_dir + "/val_mem_1/memory.pkl"
        mem_1 = load_memory(args.memory, args.disable_bzip_memory)
        val_mem_1 = ReplayMemory(args, args.evaluation_size)
        print("loaded memory buffer 1")
        print("Continue Training Load buffer 2 ...")
        args.memory = results_dir + "/val_mem_2/memory.pkl"
        mem_2 = load_memory(args.memory, args.disable_bzip_memory)
        val_mem_2 = ReplayMemory(args, args.evaluation_size)
        print("loaded memory buffer 2")
    
    else:
        print("use empty Buffers")
        args.memory = results_dir + "/val_mem_1/memory.pkl"
        path = results_dir + "/val_mem_1"
        print("save memory", args.memory)
        os.makedirs(path, exist_ok=True)
        val_mem_1 = ReplayMemory(args, args.evaluation_size)
        mem_1 = ReplayMemory(args, args.memory_capacity)
        args.memory = results_dir + "/val_mem_2/memory.pkl"
        path = results_dir + "/val_mem_2"
        print("save memory", args.memory)
        os.makedirs(path, exist_ok=True)
        val_mem_2 = ReplayMemory(args, args.evaluation_size)
        mem_2 = ReplayMemory(args, args.memory_capacity)
    
    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
    metrics = {'steps': [], 'rewards': [], 'Qs': [], 'step_rewards': [], 'train_rewards': [] , 'best_avg_reward': -float('inf')}
    
    args.continue_training = True
    def write_into_file(text, file_name='document.csv'):
        """
        """
        with open(file_name,'a', newline='\n') as fd:
            fd.write(str(text)+"\n")
    
    def log(s):
        text = '[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s
        write_into_file(text)
        print(text)
    
    if torch.cuda.is_available():
        print("cuda")
    
    def save_memory(memory, memory_path, disable_bzip):
      if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
          pickle.dump(memory, pickle_file)
      else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
          pickle.dump(memory, zipped_pickle_file)
    
    
    ("Create eval memory of size {} ".format(args.evaluation_size))
    # Construct validation memory
    
    size = 84
    print("Fill eval memory")
    
    # fill both memories at same time 
    # use the reward function for each
    try: 
        while T < args.evaluation_size:
            T += 1
            print("steps ", T)
            if done:
                t = 0
                done = False
                state = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=args.device).div_(255)
                zeros = torch.zeros_like(state)
                state_buffer = deque([], maxlen=args.history_length)
                state_buffer.append(zeros)
                state_buffer.append(zeros)
                state_buffer.append(zeros)
                state_buffer.append(state)                        
                state = torch.stack(list(state_buffer),0)
            t += 1
            if t == args.max_episode_length:
            #if t == 5:
                t = 0
                done = True
            next_state, _, _ ,_= env.step(np.random.randint(0, action_space))
            
            val_mem_1.append(state, None, None, done)
            val_mem_2.append(state, None, None, done)
            
            next_state = torch.tensor(next_state, dtype=torch.float32, device=args.device).div_(255)
            state_buffer.append(next_state)
            state = torch.stack(list(state_buffer), 0)
        eps_1 = 1
        eps_end_1 = 0.05
        eps_decay_1 = 0.999978   # reaches 10% at 105000
        
        eps_2 = 1
        eps_end_2 = 0.05
        eps_decay_2 = 0.999978   # reaches 10% at 10500
        #args.evaluate = True
        if args.evaluate:
            print("Test")
            dqn.eval()  # Set DQN (online network) to evaluation mode
            #avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, results_dir, env, evaluate=True)  # Test
            avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir, env)  # Test
            print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
        else:
            if args.continue_training:
                print("Start Training")
                T = args.learn_start + 500
            # Training loop
            dqn_1.train()
            dqn_2.train()
            episode = 0
            episode_reward = 0
            mean_reward = deque(maxlen=100)
            plot_rewards = []
            print("Fill both memory buffers ")
            while T < args.learn_start:
                if T % args.max_episode_length == 0:
                    state, done = env.reset(), False 
                    state = torch.tensor(state, dtype=torch.float32, device=args.device).div_(255)
                    zeros = torch.zeros_like(state)
                    state_buffer = deque([], maxlen=args.history_length)
                    state_buffer.append(zeros)
                    state_buffer.append(zeros)
                    state_buffer.append(zeros)
                    state_buffer.append(state)                        
                    state = torch.stack(list(state_buffer), 0)
                # choose action at random
                action = np.random.randint(0, action_space) 
                next_state, reward, done, reward_2 = env.step(action)  # Step
                text = "Step {} of {} ".format(T, args.learn_start)
                print(text,  end='\r', file=sys.stdout, flush=True)
                # set done on the last transition
                if (T+1) % args.max_episode_length == 0:
                    done = True
                mem_1.append(state, action, reward, done)
                mem_2.append(state, action, reward_2, done)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=args.device).div_(255)
                state_buffer.append(next_state)
                state = torch.stack(list(state_buffer), 0)
                T +=1
                if T >= args.learn_start:
                    args.memory = results_dir + "/val_mem_1/memory.pkl"
                    print("save memory 1", args.memory)
                    save_memory(mem_1, args.memory, args.disable_bzip_memory)
                    args.memory = results_dir + "/val_mem_2/memory.pkl"
                    print("save memory 2", args.memory)
                    save_memory(mem_2, args.memory, args.disable_bzip_memory)
                    break
            print("Start Training")
            #for T in tqdm.trange(args.learn_start, args.T_max + 1):
            for T in tqdm.trange(0, args.T_max + 1):
                if T % args.max_episode_length == 0:
                    mean_reward.append(episode_reward)
                    print("Epiosde: {}  Reward: {} Mean Reward: {}  Goal1 {}".format(episode,
                        episode_reward, np.mean(mean_reward), env.goal_counter_1))
                    plot_rewards.append(np.mean(mean_reward))
                    save_and_plot(T, plot_rewards)
                    episode_reward = 0
                    episode += 1
                    state, done = env.reset(), False 
                    state = torch.tensor(state, dtype=torch.float32, device=args.device).div_(255)
                    zeros = torch.zeros_like(state)
                    state_buffer = deque([], maxlen=args.history_length)
                    state_buffer.append(zeros)
                    state_buffer.append(zeros)
                    state_buffer.append(zeros)
                    state_buffer.append(state)                        
                    state = torch.stack(list(state_buffer), 0)
                    g = 0
                    set_input = True
                    secondTask = False
                
                
                if T % args.replay_frequency == 0:
                    pass
                    #dqn.reset_noise()  # Draw a new set of noisy weights
                
                
                """
                if env.task_one_complete or secondTask:
                    action = dqn_2.act_e_greedy(state, eps_2)  # Choose an action greedily (with noisy weights)
                    secondTask = True
                else:
                    action = dqn_1.act_e_greedy(state, eps_1)  # Choose an action greedily (with noisy weights)
                """
                if set_input:
                    set_input = False
                    g = input("Enter action : ") 
                    action = int(g)
                    g = input("Enter steps : ") 
                    g = int(g)
                if g <= 0:
                    set_input = True
                g -=1
                
                #print("step : {} action: {} eps: {}".format(T, action, eps))
                next_state, reward, done, reward_2 = env.step(action)  # Step
                
                if args.reward_clip > 0:
                    reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
                    reward_2 = max(min(reward_2, args.reward_clip), -args.reward_clip)  # Clip rewards
                         
                if env.task_one_complete or secondTask:
                    episode_reward += reward_2
                    eps_2 = max(eps_end_2, eps_decay_2*eps_2)
                    mem_2.priority_weight = min(mem_2.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
                else:
                    episode_reward += reward
                    eps_1 = max(eps_end_1, eps_decay_1*eps_1)
                    mem_1.priority_weight = min(mem_1.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                #print(reward)
                #print(reward_2)
                # incase the last action set done to True
                if T + 1 % args.max_episode_length == 0:
                    done = True
                
                mem_1.append(state, action, reward, done)  # Append transition to memory
                mem_2.append(state, action, reward_2, done)  # Append transition to memory
                
                # Train and test
                 
                next_state = torch.tensor(next_state, dtype=torch.float32, device=args.device).div_(255)
                # print("Main shape of  next_state", next_state.shape) 
                state_buffer.append(next_state)
                state = torch.stack(list(state_buffer), 0)
                continue
                # print("Main shape of  state", state.shape) 
                if T % args.replay_frequency == 0:
                    dqn_1.learn(mem_1)  # Train with n-step distributional double-Q learning
                    dqn_2.learn(mem_2)  # Train with n-step distributional double-Q learning
                
                if T % args.evaluation_interval == 0:
                    dqn_1.eval()  # Set DQN (online network) to evaluation mode
                    print("Eval epsilon 1 {} epsilon 2 {} ".format(eps_1,eps_2))
                    avg_reward, avg_Q = test(args, T, dqn_1, val_mem_1, metrics, results_dir, env, 1)  # Test
                    log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                    dqn_1.train()  # Set DQN (online network) back to training mode
                    dqn_2.eval()  # Set DQN (online network) to evaluation mode
                    avg_reward, avg_Q = test(args, T, dqn_2, val_mem_2, metrics, results_dir, env, 2)  # Test
                    log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                    dqn_2.train()  # Set DQN (online network) back to training mode
                    
                    
                # Update target network
                if T % args.target_update == 0:
                    dqn_1.update_target_net()
                    dqn_2.update_target_net()
        
                # checkpoint the network
                if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                    #print("save memory", args.memory)
                    #save_memory(mem, args.memory, args.disable_bzip_memory)
                    print("epsilon 1: ", eps_1)
                    print("epsilon 2: ", eps_2)
                    print("Save model at ", results_dir)
                    dqn_1.save(results_dir, '{}-checkpoint.pth'.format(T))
                    dqn_2.save(results_dir, '{}-2-checkpoint.pth'.format(T))
    except KeyboardInterrupt:
        print("Keybaord error")
    finally:
        print("save state....")
        print("Save model at ", results_dir)
        dqn_1.save(results_dir, '{}-checkpoint.pth'.format(T))
        dqn_2.save(results_dir, '{}-2-checkpoint.pth'.format(T))
        args.memory = results_dir + "/val_mem_1/memory.pkl"
        print("save memory 1  ...", args.memory)
        save_memory(mem_1, args.memory, args.disable_bzip_memory)
        args.memory = results_dir + "/val_mem_2/memory.pkl"
        print("save memory 2 ...", args.memory)
        save_memory(mem_2, args.memory, args.disable_bzip_memory)
        print("Save model at ", results_dir)
        dqn_1.save(results_dir, '{}-checkpoint.pth'.format(T))
        dqn_2.save(results_dir, '{}-2-checkpoint.pth'.format(T))
        print("... done Saving State")
        sys.exit()





if __name__ == "__main__":
    print("Main")
    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS',
            help='Number of training steps (4x number of frames)')
    parser.add_argument('--max-episode-length', type=int, default=int(100), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
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
    parser.add_argument('--evaluation-size', type=int, default=50, metavar='N',
            help='Number of transitions to use for validating Q')
    parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
    parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
    parser.add_argument('--checkpoint-interval', default=20000, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
    parser.add_argument('--memory', help='Path to save/load the memory from')
    parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--continue-training', type=bool, default='False')
    # Setup
    args = parser.parse_args()
    env = gym.make('Car-v0')
    
    #eval_policy(args, env)
    train(args, env)
