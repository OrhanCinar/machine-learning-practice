import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


env = gym.make("CartPole-v1")

env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=0)


def evaluate(model, num_steps=1000):
    episde_rewards = [0.0]

    obs = env.reset()

    for i in range(num_steps):
        action, _states = model.predict(obs)

        obs, rewards, dones, info = env.step(action)

        episode_rewards[-1] += rewards[0]

        if dones[0]:
            obs = env.reset()
            episode_rewards.append(0.0)
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print(
        f"Mean reward {mean_100ep_reward}, Num episodes: {len(episde_rewards)}")
    return mean_100ep_reward


mean_reward_before_train = evaluate(model, num_steps=10000)

model.learn(total_timesteps=10000)

mean_reward = evaluate(model, num_steps=10000)
