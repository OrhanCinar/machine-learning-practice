from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from lib import plotting
import sklearn.preprocessing
import sklearn.pipeline
import sys
import numpy as np
import matplotlib
import itertools
import gym
%matplotlib inline


if "../" not in sys.path:
    sys.path.append("../")


env = gym.envs.make("MountainCar-v0")

observation_examples = np.array(
    [env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1": RBFSampler(gamme=5.0, n_components=100)),
    ("rbf2": RBFSampler(gamme=2.0, n_components=100)),
    ("rbf3": RBFSampler(gamme=1.0, n_components=100)),
    ("rbf4": RBFSampler(gamme=0.5, n_components=100))
])

featurizer.fit(scaler.transform(observation_examples))
