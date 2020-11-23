from lib import plotting
from lib.envs.blackjack import BlackjackEnv
import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
    sys.path.append("../")

matplotlib.style.use('ggplot')


env = BlackjackEnv()

# https://github.com/dennybritz/reinforcement-learning/blob/master/MC/MC%20Control%20with%20Epsilon-Greedy%20Policies%20Solution.ipynb


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.opes(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn
