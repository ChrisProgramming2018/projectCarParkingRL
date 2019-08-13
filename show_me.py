import gym
import gym_car


env = gym.make("Car-v0")
print(env)
def random_agent(episodes=100):
    env = gym.make("Car-v0")
    env.reset()
    total_reward = 0
    for e in range(episodes):
        action = env.action_space.sample()
        #action = 1
        state, reward, done, _ = env.step(action)
        #env.render(state)
        #print("new state")
        print("reward", reward)
        total_reward +=reward
        if done:
            break
    print(total_reward)
    env.reset()
if __name__ == "__main__":
    random_agent()
    print("hello")
