import tensorflow as tf
import numpy as np
from vizdoom import *

import random
import time
from skimage import transform

from collections import deque
import matplotlib.pyplot as plt

import warnings
warnings.filters("ignore")

# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb


def create_environment():
    game = DoomGame()
    game.load_config("deadly_corridor.cfg")
    game.set_doom_scenario_path("deadly_corridor.wad")
    game.init()

    possible_actions = np.identity(7, type=int).tolist()
    return game, possible_actions


game, possible_actions = create_environment()

"""
    preprocess_frame:
    Take a frame.
    Resize it.
        __________________
        |                 |
        |                 |
        |                 |
        |                 |
        |_________________|
        
        to
        _____________
        |            |
        |            |
        |            |
        |____________|
    Normalize it.
    
    return preprocessed_frame
    
    """


def preprocess_frame(frame):
    cropped_frame = frame[15:-5, 20:-20]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(cropped_frame, [100, 120])
    return preprocessed_frame  # 100x120x1 frame


stack_size = 4
stacked_frames = deque([np.zeros((100, 120), dtype=np.int)
                        for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros((100, 120), dtype=np.int)
                                for i in range(stack_size)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stack_frames


stack_size = [100, 120, 4]
action_size = game.get_available_buttons_size()
learning_rate = 0.00025
total_episodes = 5000
max_steps = 5000
batch_size = 64
max_tau = 10000
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00005
gamma = 0.95
pretrain_length = 100000
memory_size = 100000
training = False
episode_render = False


class DDDQNet:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate,
        self.name = name

        with tf.variable_scope(self.name):
            self.inputs_ = tf.placeholder(
                tf.float32, [None, *state_size], name="inputs")
            self.ISWeigths_ = tf.placeholder(
                tf.float32, [None, 1], name="IS_Weigths")
            self.actions_ = tf.placeholder(
                tf.float32, [None, state_size], name="actions_")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            ELU
            """

            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")
            self.conv3_out = tf.nn.elu(self.conv2, name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out)

            self.value_fc = tf.layers.dense(inputs=self.flatten,
                                            units=512,
                                            activation=None,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="value_fc")

            self.value = tf.layers.dense(inputs=self.value_fc,
                                         units=512,
                                         activation=tf.nn.elu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name="value")

            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                units=512,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                name="advantage_fc")

            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                             name="advantages")

            self.output = self.value + \
                tf.subtract(self.advantage, tf.reduce_mean(
                    self.advantage, axis=1, keepdims=True))

            self.Q = tf.reduce_sum(tf.multiply(
                self.output, self.actions_), axis=1)

            self.absolute_errors = tf.abs(self.target_Q - self.Q)

            self.loff = tf.reduce_mean(
                self.ISWeights_ * tf.squared_difference(self.target_Q), self.Q)

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)


tf.reset_default_graph()
DQNetwork = DDDQNet(state_size, action_size, learning_rate, name="DQNetwork")

TargetNetwork = DDDQNet(state_size, action_size,
                        learning_rate, name="TargetNetwork")
