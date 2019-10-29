"""   """
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tf_agents.trajectories import trajectory


def collect_step(environment, policy, buffer, max_t):
    """ """
    goal = 0
    time_step = environment.reset()
    # seed = np.random.randint(0,7)
    seed = np.random.uniform()
    episode_reward = 0
    for step in range(max_t):
        action_step = policy.action(time_step, seed=seed)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        episode_reward += time_step.reward.numpy()[0]
        if time_step.reward.numpy()[0] == 10:
            goal += 1
        # Add trajectory to the replay buffer
        buffer.add_batch(traj)
        time_step = next_time_step
        if time_step.is_last().numpy()[0]:
            break
    return episode_reward, goal

def collect_data(env, policy, buffer, steps, max_t):
    for _ in range(steps):
        collect_step(env, policy, buffer, max_t)
        text = "collect Episode {} of {} "
        print(text.format(_, steps), end="\r", flush=True)





def print_parameter(args, parser):
    """
    print the default parameter to the terminal
    """
    print("Default parameter")
    for para in vars(args):
        text = str(para) +  " : "
        print(text + str(parser.get_default(para)))





def data_loger(episode, start, episode_reward, scores_window, eps, td_loss, goal):
    """

    """
    time_passed = time.time() - start
    time_passed = time_passed
    minutes = time_passed / 60
    hours = time_passed / 3600
    text = 'Episode: {}: reward: {:.0f} average reward: {:.0f}  eps: {:.3f} '
    text += 'td_loss: {:.4f} goal: {} time: {:.0f} h {:.0f} min {:.0f} sec '
    text= text.format(episode, episode_reward, np.mean(scores_window), eps,
            td_loss, goal, hours, minutes % 60, time_passed % 60 )
    return text



def write_into_file(text, file_name='document.csv'):
    """
    """
    with open(file_name,'a', newline='\n') as fd:
        fd.write(str(text)+"\n")


def save_and_plot(num_iterations, returns_average, returns, td_loss, model=1):
    """
    """
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
    print("plot saved")


def compute_avg_return(environment, policy, num_episodes=2):
    """


    """
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
        print("reward sum ", episode_return)
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return





def train(arg, tf_agent, train_env, eval_env, replay_buffer, iterator, train_checkpointer):
    """
    
    """
    print("start taining")
    eps = arg.eps_start
    returns = []
    returns_average = []
    train_loss_average = []
    score = 0
    scores_window = deque(maxlen=100)       # last 100 scores
    total_train_loss = deque(maxlen=1000)       # last 100 scores
    start = time.time()
    sum_goals = 0
    for episode in range(arg.n_episodes):
        step = tf_agent.train_step_counter.numpy()
        
        if episode % arg.save_weights_every  == 0:
            print("Save Networkweights", end='\n')
            train_checkpointer.save(global_step=step)
            
        # Collect a few steps using collect_policy and save to the replay buffer.
    
        episode_reward, goal = collect_step(train_env, tf_agent.collect_policy, replay_buffer, arg.max_t)
        sum_goals += goal
        # Sample a batch of data from the buffer and update the
        # agent's network.
        
        # train several times
        for _ in range(arg.repeat_training):
            experience, unused_info = next(iterator)
            train_loss = tf_agent.train(experience).loss
        scores_window.append(episode_reward)
        total_train_loss.append(train_loss)
        
        if episode % arg.eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, arg.num_eval_episodes)
            text = 'Episode = {}: Average Return = {:.0f}'
            text = text.format(episode, avg_return)
            print(text, end="\n")
            returns.append(avg_return)
            returns_average.append(np.mean(scores_window))
            train_loss_average.append(np.mean(total_train_loss))
            save_and_plot(episode, returns_average, returns, train_loss_average)
        text = data_loger(episode, start, episode_reward, np.mean(scores_window),
                tf_agent.collect_policy._get_epsilon(), train_loss, sum_goals)
        write_into_file(text)
        print(text, end="\r", flush=True)
        eps = max(arg.eps_end, arg.eps_decay*eps)
        tf_agent.collect_policy._epsilon = eps
