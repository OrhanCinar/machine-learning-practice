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

# https://github.com/rlcode/reinforcement-learning/blob/master/3-atari/1-breakout/play_a3c_model.py


class TestAgent:
    def __init__(self, action_size):
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        self.discount_factor = 0.99
        self.no_op_steps = 30

        self.acttor, self.critic = self.build_model()

    def build_model(self):
        pass