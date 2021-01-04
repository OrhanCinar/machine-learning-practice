from lib import plotting
from lib.envs.windy_gridworld import WindyGridworldEnv
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


env = WindyGridworldEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn
