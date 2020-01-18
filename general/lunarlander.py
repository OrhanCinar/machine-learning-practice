import gym
from stable_baselines import DQN


env = gym.make("LunarLander-v2")

model = DQN("MlpPolicy", env, learning_rate=1e-3,
            prioritized_replay=True, verbose=1)

model.learn(total_timesteps=int(2e5))

model.save("sql_lunar")

del model

model = DQN.load("dqn_lunar")

obs = env.reset()

for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
