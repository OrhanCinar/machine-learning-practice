import gym
import random
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D

global episode
episode = 0
EPISODES = 8000000
env_name = "BreakoutDeterministic-v4"
