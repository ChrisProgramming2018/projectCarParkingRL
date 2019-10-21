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
