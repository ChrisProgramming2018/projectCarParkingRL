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

print("load env ..")
env_name =("Car-v0")
#env = gym.make("Car-v0")

env = suite_gym.load(env_name, discount=0.99, max_episode_steps=1000)




start = time.time()
num_iterations = 200000  # @param

initial_collect_steps = 10000  # @param
collect_steps_per_iteration = 1  # @param
replay_buffer_max_length = 250000 
fc_layer_params = (512,)

batch_size = 32  # @param
learning_rate = 0.00005  # @param
log_interval = 200  # @param

num_eval_episodes = 10  # @param
eval_interval = 10  # @param
repeat_training = 4

train_py_env = suite_gym.load(env_name, discount=0.99, max_episode_steps=1000)
eval_py_env = suite_gym.load(env_name, discount=0.99, max_episode_steps=1000)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

print("env loaded")

train_env.reset()


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
        epsilon_greedy=1,
        td_errors_loss_fn=dqn_agent.element_wise_squared_loss,
        train_step_counter=train_step_counter)

tf_agent.initialize()

print("ready to go")

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
        train_env.action_spec())
print("policy")

from helper import compute_avg_return


    return avg_return

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

tf_agent.collect_data_spec
tf_agent.collect_data_spec._fields


def collect_step(environment, policy, buffer):
    """ """
    debug = True
    if policy.name == 'random_tf_policy':
            time_step = environment.reset()
            for step in range(50):
                action_step = policy.action(time_step)
                next_time_step = environment.step(action_step.action)
                traj = trajectory.from_transition(time_step, action_step, next_time_step)

                # Add trajectory to the replay buffer
                buffer.add_batch(traj)
                time_step = next_time_step
            return

    time_step = environment.reset()
    # seed = np.random.randint(0,7)
    seed = np.random.uniform()
    episode_reward = 0
    debug = False 
    num_greedy_actions = 0
    for step in range(50):
        action_step = policy.action(time_step, seed=seed)
        #if action_step.action.numpy()[0] == policy.greedy_action.numpy()[0]:
         #   num_greedy_actions += 1
        if debug:
            if action_step.action.numpy()[0] == policy.greedy_action.numpy()[0]:
                print("greedy action")
            else:
                print("random action")
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        episode_reward += time_step.reward.numpy()[0]
        # Add trajectory to the replay buffer
        buffer.add_batch(traj)
        # train 
        experience, unused_info = next(iterator)    
        train_loss = tf_agent.train(experience).loss
        time_step = next_time_step
        if time_step.is_last().numpy()[0] and step > 20:
            print("Finished after time step", step)
            break
    return episode_reward,(num_greedy_actions / step)
def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)

print("collect data  ...")
data_collection_steps = 1
collect_data(train_env, random_policy, replay_buffer, steps=data_collection_steps)


print("create dataset")
dataset = replay_buffer.as_dataset(num_parallel_calls=3,  sample_batch_size=batch_size, num_steps=2).prefetch(3)
iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
num_eval_episodes = 1
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]
returns_average = [avg_return]
train_loss_average = [1]
print("save")
#FLAGS = flags.FLAGS
#root_dir = FLAGS.root_dir

root_dir = "/home/leiningc/b_project/gym_env/gym-car"
root_dir = os.path.expanduser(root_dir)
train_dir = os.path.join(root_dir, 'network_weights')
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

                                       
print("End save para")

print("start Training")
start = time.time()
save_weights_every = 100
score = 0
scores_window = deque(maxlen=100)       # last 100 scores
total_train_loss = deque(maxlen=1000)       # last 100 scores
def data_loger(start, num_greedy_actions, episode_reward, scores_window, eps):
    time_passed = time.time() - start
    time_passed = time_passed
    minutes = time_passed / 60
    hours = time_passed / 3600
    text = 'Episode: {}: reward: {:.0f} average reward: {:.0f}  eps: {:.3f} '
    text += 'td_loss: {:.4f} time: {:.0f} h {:.0f} min {:.0f} sec '
    text= text.format(episode, episode_reward, np.mean(scores_window), eps, num_greedy_actions, hours, minutes % 60, time_passed % 60 )    
    return text


def write_into_file(text, file_name='document.csv'):
    with open(file_name,'a', newline='\n') as fd:
            fd.write(str(text)+"\n")


def save_and_plot(num_iterations, returns_average, returns, td_loss, model=1):
    os.system('mkdir -p results/model-{}'.format(model))
    fig, ax = plt.subplots()
    iterations = range(0, len(returns))
    ax.plot(iterations, returns,'r--' ,label="return")
    ax.plot(iterations, returns_average,'b', label='average')

    
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('C0')
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=25)
    plt.savefig('results/model-1/scores-{}.png'.format(num_iterations))
    fig, ax = plt.subplots()
    ax.plot(iterations, td_loss, label='loss')
    plt.ylabel('td_loss')
    plt.xlabel('Iterations')
    plt.ylim(top=1)
    plt.savefig('results/model-1/loss-{}.png'.format(num_iterations))
    print("save plot")


num_iterations = 20 * 1000
eval_interval = 10
save_weights_every = 10
eps = 1
eps_end = 0.05
eps_decay = 0.9995  # reach 50 % at around 7000 episodes
for episode in range(num_iterations):
    step = tf_agent.train_step_counter.numpy()
    
    
    if episode % save_weights_every  == 0:
        print("Save Networkweights")
        train_checkpointer.save(global_step=step)
    
    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        episode_reward, num_greedy_actions = collect_step(train_env, tf_agent.collect_policy, replay_buffer)
        
    # Sample a batch of data from the buffer and update the
    # agent's network.
    
    # train several times
    for _ in range(repeat_training):
        experience, unused_info = next(iterator)    
        train_loss = tf_agent.train(experience).loss
    scores_window.append(episode_reward)
    total_train_loss.append(train_loss)

    
    if episode + 1 % log_interval == 0:
        print('step = {0}: loss = {1}'.format(episode, train_loss))
    
    if episode % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        text = 'Episode = {}: Average Return = {:.0f}'
        text = text.format(episode, avg_return)
        print(text, end="\n")
        returns.append(avg_return)
        returns_average.append(np.mean(scores_window))
        train_loss_average.append(np.mean(total_train_loss))
        save_and_plot(episode, returns_average, returns, train_loss_average)
    
    text = data_loger(start, train_loss, episode_reward, np.mean(scores_window),
            tf_agent.collect_policy._get_epsilon())
    write_into_file(text)
    print(text, end="\r", flush=True)
    eps = max(eps_end, eps_decay*eps)
    tf_agent.collect_policy._epsilon = eps 
save_and_plot(num_iterations, returns)

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







