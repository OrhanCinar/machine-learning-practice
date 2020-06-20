import tensorflow as tf
import numpy as np
from vizdoom import *
import random
import time
from skimage import transform
from collections import deque
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def create_environment():
    game = DoomGame()
    game.load_config("health_gathering.cfg")
    game.set_doom_scenario_path("health_gathering.wad")
    game.init()
    possible_actions = np.identity(3, dtype=int).tolist()
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
    cropped_frame = frame[80:, :]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_form, [84, 84])
    return preprocessed_frame


stack_size = 4
stacked_frames = deque([np.zeros((84, 84), dtype=np.int)
                        for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int)
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


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamme + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / std
    return discounted_episode_rewards


state_size = [84, 84, 4]
action_size = game.get_available_buttons_size()
stack_size = 4
learning_rate = 0.002
num_epochs = 500
batch_size = 1000
gamme = 0.95
training = True


class PQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name="PQNetwork"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            with tf.name_scope("inputs"):
                self.inputs_ = tf.placeholder(
                    tf.float32, [None, *state_size], name="inputs_")
                self.actions = tf.placeholder(
                    tf.float32, [None, *state_size], name="actions")
                self.discounted_episode_rewards_ = tf.placeholder(
                    tf.float32, [None, ], name="discounted_episode_rewards_")
                self.mean_rewards_ = tf.placeholder(
                    tf.float32, name="mean_rewards_")

            with tf.name_scope("conv1"):
                """
               First convnet:
               CNN
               BatchNormalization
               ELU
               """
                self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                              filters=32,
                                              kernel_size=[8, 8],
                                              strides=[4, 4],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name="conv1")
                self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                                     training=True,
                                                                     epsilon=1e-5,
                                                                     name="batch_norm1")
                self.conv1_out = tf.nn.elu(
                    self.conv1_batchnorm, name="conv1_out")  # --> [20, 20, 32]

            with tf.name_scope("conv2"):
                """
                  Second convnet:
                  CNN
                  BatchNormalization
                  ELU
                  """
                self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                              filters=64,
                                              kernel_size=[4, 4],
                                              strides=[2, 2],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name="conv2")
                self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                                     training=True,
                                                                     epsilon=1e-5,
                                                                     name="batch_norm2")
                self.conv2_out = tf.nn.elu(
                    self.conv1_batchnorm, name="conv2_out")  # --> [9, 9, 64]

            with tf.name_scope("conv3"):
                """
                  Third convnet:
                  CNN
                  BatchNormalization
                  ELU
                  """
                self.conv3 = tf.layers.conv2d(inputs=self.conv3_out,
                                              filters=128,
                                              kernel_size=[4, 4],
                                              strides=[2, 2],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name="conv3")
                self.conv3_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                                     training=True,
                                                                     epsilon=1e-5,
                                                                     name="batch_norm3")
                self.conv3_out = tf.nn.elu(
                    self.conv1_batchnorm, name="conv3_out")  # --> [3, 3, 128]

            with tf.name_scope("flatten"):
                self.flatten = tf.layers.flatten(self.conv3_out)  # --> [1152]

            with tf.name_scope("fc1"):
                self.fc = tf.layers.dense(inputs=self.flatten,
                                          units=512,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="fc1")

            with tf.name_scope("logits"):
                self.logits = tf.layers.dense(inputs=self.fc,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              units=3,
                                              activation=None)
            with tf.name_scope("softmax"):
                self.action_distribution = tf.nn.softmax(self.logits)

            with tf.name_scope("loss"):
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.actions)
                self.loss = tf.reduce_mean(
                    self.neg_log_prob * self.discounted_episode_rewards_)

            with tf.name_scope("traing"):
                self.train_opt = tf.trains.RMSPropOptimizer(
                    self.learning_rate).minimize(self.loss)


tf.reset_default_graph()
PGNetwork = PGNetwork(state_size, action_size, learning_rate)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

writer = tf.summary.FileWrtier("/tensorboard/pg/test")
tf.summary.scalar("Loss", PGNetwork.loss)
tf.summary.scalar("Reward_mean", PGNetwork.mean_rewards_)
write_op = tf.summary.merge_all()
