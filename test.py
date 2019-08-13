import gym
import gym_car


def random_agent(episodes=100):
    env = gym.make("Car-v0")
    env.reset()
    total_reward = 0
    for e in range(episodes):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        #env.render()
        total_reward +=reward
        if done:
            print(total_reward)
            break

if __name__ == "__main__":
    random_agent()
    print("hello")
