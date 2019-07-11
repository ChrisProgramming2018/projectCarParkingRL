from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image
import os
import tensorflow as tf
from absl import flags
import sys 
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
from tf_agents.metrics import py_metrics
tf.compat.v1.enable_v2_behavior()



env_name = 'CartPole-v0'  # @param
num_iterations = 15000  # @param

initial_collect_steps = 1000  # @param
collect_steps_per_iteration = 1  # @param
replay_buffer_capacity = 100000  # @param

fc_layer_params = (100,)

batch_size = 64  # @param
learning_rate = 1e-3  # @param
log_interval = 200  # @param

num_eval_episodes = 10  # @param
eval_interval = 1000  # @param


env = suite_gym.load(env_name)
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

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
# save nn
print("init save para")
flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), 'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000, 'Total number train/eval iterations to perform.')
flags.DEFINE_bool('use_ddqn', False, 'If True uses the DdqnAgent instead of the DqnAgent.')
flags.mark_flag_as_required('root_dir')

train_checkpoint_interval=10000
policy_checkpoint_interval=5000
rb_checkpoint_interval=20000
log_interval=1000
summary_interval=1000
summaries_flush_secs=10
agent_class=dqn_agent.DqnAgent
debug_summaries=False
summarize_grads_and_vars=False


#FLAGS = flags.FLAGS
#root_dir = FLAGS.root_dir
root_dir = "/home/leiningc/project/data"
root_dir = os.path.expanduser(root_dir)
train_dir = os.path.join(root_dir, 'train')
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

#train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir, flush_millis=summaries_flush_secs * 1000)
#train_summary_writer.set_as_default()
train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics,
            'train_metrics'))



print("End save para")


#eval_summary_writer = tf.compat.v2.summary.create_file_writer(
#        eval_dir, flush_millis=summaries_flush_secs * 1000)
#eval_metrics = [
#        py_metrics.AverageReturnMetric(buffer_size=num_eval_episodes), 
#        py_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),]




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


compute_avg_return(eval_env, random_policy, num_eval_episodes)


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

def collect_step(environment, policy):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  replay_buffer.add_batch(traj)

for _ in range(initial_collect_steps):
  collect_step(train_env, random_policy)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]



print("start training")
print(" ")
print(" ")
print(" ")
print(" ")


train_checkpointer.initialize_or_restore()
trained = True
if trained:
    print("show trained agent")
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    print('Average Return = {0}'.format(avg_return))




for num  in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, tf_agent.collect_policy)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = tf_agent.train(experience)

  step = tf_agent.train_step_counter.numpy()

  if step % eval_interval == 0:
      print("save the agent")
      train_checkpointer.save(global_step=step)
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
