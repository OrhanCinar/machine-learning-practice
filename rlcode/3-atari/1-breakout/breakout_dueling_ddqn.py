import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Flatten, Lambda, merge
from keras.layers.convolutional import Conv2D
from keras import backend as K

EPISODES = 50000

# https://github.com/rlcode/reinforcement-learning/blob/master/3-atari/1-breakout/breakout_dueling_ddqn.py


class DuelingDDQNAgent:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (
            self.epsilon_start - self.epsilon_end)/self.exploration_steps
