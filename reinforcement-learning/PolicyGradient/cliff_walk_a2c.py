from lib import plotting
from lib.envs.cliff_walking import CliffWalkingEnv
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

env = CliffWalkingEnv()


class PolicyEstimator():
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.in32, [], "state")
            self.state = tf.placeholder(tf.in32, [], "action")
            self.state = tf.placeholder(tf.in32, [], "target")

            state_one_hot = tf.one_hot(
                self.state, int(env.observation_space.n))
            self.putput_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=env.action_space.n,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gathet(self.action_probs, self.action)

            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_rate)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.target: target,
                     self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():
    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tfelf.state = tf.placeholder(tf.int32, [], "state")
        self.target = tf.placeholder(dtype=tf.float32, name="target")

        state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
        self.output_layer = tf.contrib.layers.full_connected(
            inputs=tf.expand_dims(state_one_hot, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.zeros_initializer)

        self.value_estimate = tf.squeeze(self.output_layer)
        self.loss = tf.squared_difference(self.value_estimate, self.target)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(
            self.loss, global_Step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
