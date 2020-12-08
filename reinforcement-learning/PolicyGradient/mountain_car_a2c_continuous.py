from sklearn.kernel_approximation import RBFSampler
from lib import plotting
from lib.envs.cliff_walking import CliffWalkingEnv
import sklearn.preprocessing
import sklearn.pipeline
import collections
import tensorflow as tf
import sys
import numpy as np
import matplotlib
import itertools
import gym
%matplotlib inline


if "../" not in sys.path:
    sys.path.append("../")


matplotlib.style.use('ggplot')


env = gym.make("MountainCarContinuous-v0")
env.observation_space.sample()

observation_examples = np.array(
    [env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandartScaler()
scaler.fit(observation_examples)

featurizer = sklearn.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])

featurizer .fit(scaler.transform(observation_examples))


def featurizer_state(state):
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]
