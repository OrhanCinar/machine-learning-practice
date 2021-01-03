from lib import plotting
from lib.envs.cliff_walking import CliffWalkingEnv
from collections import defaultdict
import sys
import pandas as pd
import numpy as np
import matplotlib
import itertools
import gym
%matplotlib inline


if "../" not in sys.path:
    sys.path.append("../")


matplotlib.style.use('ggplot')

# https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Cliff%20Environment%20Playground.ipynb


env = CliffWalkingEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn
