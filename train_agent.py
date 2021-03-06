from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path = ['',
        '/home/leiningc/anaconda3/envs/sim/lib/python35.zip',
        '/home/leiningc/anaconda3/envs/sim/lib/python3.5',
        '/home/leiningc/anaconda3/envs/sim/lib/python3.5/plat-linux',
        '/home/leiningc/anaconda3/envs/sim/lib/python3.5/lib-dynload',
        '/home/leiningc/anaconda3/envs/sim/lib/python3.5/site-packages',
        '/home/leiningc/anaconda3/envs/sim/lib/python3.5/site-packages/IPython/extensions',
        '/home/leiningc/.ipython',
        '/home/leiningc/project/gym_env/gym-bubbleshooter']

print(sys.path)
import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image
import tensorflow as tf
import gym_bubbleshooter
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
print("load env ..")
sys.exit()
env_name =("BubbleShooter-v0")
num_iterations = 8000  # @param
print("load env completed")

#initial_collect_steps = 1000  # @param
initial_collect_steps = 10  # @param
collect_steps_per_iteration = 1  # @param
replay_buffer_capacity = 100000  # @param

fc_layer_params = (100,)

batch_size = 64  # @param
learning_rate = 1e-3  # @param
log_interval = 200  # @param

num_eval_episodes = 10  # @param
eval_interval = 1000  # @param


env = suite_gym.load(env_name, discount=0.99, max_episode_steps=1000)

env.reset()

print('Observation Spec:')
print(env.time_step_spec().observation)
print('Action Spec:')
print(env.action_spec())

time_step = env.reset()
print('Time step:')
print(time_step)

action = 1

next_time_step = env.step(action)
print('Next time step:')
print(next_time_step)
train_py_env = suite_gym.load(env_name, discount=0.99, max_episode_steps=1000)
eval_py_env = suite_gym.load(env_name, discount=0.99, max_episode_steps=1000)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)



train_env.reset()
print('Observation Spec:')
print(train_env.time_step_spec().observation)
print('board:')
print("env")
print(eval_env.reset())
print('end')



q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)



optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

tf_agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=dqn_agent.element_wise_squared_loss,
        train_step_counter=train_step_counter)

tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
print("policy")

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

print("compute_avg_return ... ")
#compute_avg_return(eval_env, random_policy, num_eval_episodes)
compute_avg_return(eval_env, random_policy, 5)




replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec, 
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

print(" init buffer ... ")
def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

for _ in range(initial_collect_steps):
    collect_step(train_env, random_policy)
print("collected")
dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

print("dataset")
iterator = iter(dataset)

print("train")
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):
    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, tf_agent.collect_policy)
        
    # Sample a batch of data from the buffer and update the
    # agent's network.
    experience, unused_info = next(iterator)
    train_loss = tf_agent.train(experience)
    step = tf_agent.train_step_counter.numpy()
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=250)
plt.show()









