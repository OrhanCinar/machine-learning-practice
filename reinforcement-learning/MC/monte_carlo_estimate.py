from lib import plotting
from lib.envs.blackjack import BlackjackEnv
from collections import defaultdict
import sys
import numpy as np
import matplotlib
import gym
%matplotlib inline


if "../" not in sys.path:
    sys.path.append("../")

matplotlib.style.use('ggplot')
# https://github.com/dennybritz/reinforcement-learning/blob/master/MC/MC%20Prediction%20Solution.ipynb

env = BlackjackEnv()


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    V = defaultdict(float)

    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 = 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            first_occurence_idx = next(
                i for i, x in enumerate(episode) if x[0] == state)
            G = sum([x[2] * (discount_factor**i)
                     for i, x in enumerate(episode[first_occurence_idx:])])
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]
    return V


def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score > = 20 else 1


V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")
