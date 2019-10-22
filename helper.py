"""   """
import numpy as np
from tf_agents.trajectories import trajectory



def collect_step(environment, policy, buffer, max_t):
    """ """
    
    time_step = environment.reset()
    # seed = np.random.randint(0,7)
    seed = np.random.uniform()
    episode_reward = 0
    for step in range(max_t):
        action_step = policy.action(time_step, seed=seed)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        episode_reward += time_step.reward.numpy()[0]
        # Add trajectory to the replay buffer
        buffer.add_batch(traj)
        time_step = next_time_step
        if time_step.is_last().numpy()[0]:
            break
    return episode_reward

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
238     iterations = range(0, len(returns))
239     ax.plot(iterations, returns,'r--' ,label="return")
240     ax.plot(iterations, returns_average,'b', label='average')
241
242
243     legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
244     legend.get_frame().set_facecolor('C0')
245     plt.ylabel('Average Return')
246     plt.xlabel('Iterations')
247     plt.ylim(top=25)
248     plt.savefig('results/model-1/scores-{}.png'.format(num_iterations))
249     fig, ax = plt.subplots()
250     ax.plot(iterations, td_loss, label='loss')
251     plt.ylabel('td_loss')
252     plt.xlabel('Iterations')
253     plt.ylim(top=1)
254     plt.savefig('results/model-1/loss-{}.png'.format(num_iterations))
255     print("save plot")

