import gym
import time

env = gym.make("CartPole-v1")

print("Action Space :", env.action_space)
print("Observation Space :", env.observation_space)
print("Observation Space Low :", env.observation_space.low)
print("Observation Space High :", env.observation_space.high)


for episode in range(20):
    obs = env.reset()
    for t in range(200):
        # time.sleep(0.05)
        env.render()
        # print(obs)
        action = env.action_space.sample()
        # print(action)
        obs, reward, done, info = env.step(action)
        print(reward)
        if (done):
            print(f"Episode finished after {t+1} steps")
            break
env.close()
