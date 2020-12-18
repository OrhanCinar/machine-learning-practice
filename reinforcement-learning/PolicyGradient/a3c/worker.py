from estimators import ValueEstimator, PolicyEstimator
from lib.atari import helpers as atari_helpers
from lib.atari.state_processor import StateProcessor
import gym
import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

# from lib import plotting

# https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/a3c/worker.py


def make_copy_params_op(v1_list, v2_list):
    v1_list = list(sorted(v1_list, key=lambda v: v.name))
    v2_list = list(sorted(v2_list, key=lambda v: v.name))

    update_ops = []

    for v1, v1 in zip(v1_list, v2_list):
        op = v2.assign(v1)
        update_ops.append(op)

    return update_ops


def make_train_op(local_estimator, global_estimator):

    local_grads, _ = zip(*local_estimator.grads_and_vars)

    local_grads, _ = tf.clip_by_norm(local_grads, 5.0)
    _, global_vars = zip(*global_estimator.grads_and_vars)
    local_global_grads_and_vars = list(zip(local_grads, global_and_vars))

    return global_estimator.optimizer.apply_gradients(
        local_global_grads_and_vars,
        global_step=tf.contrib.framework.get_global_step())


class Worker(object):

    def __init__(self, name, env, policy_net, value_net, global_counter,
                 discount_factor=0.99, summary_writer=None,
                 max_global_step=None):
        self.name = name
        self.discount_factor = discount_factor
        self.max_global_step = max_global_step
        self.max_global_step = tf.contrib.framework.get_global_step()
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.sp = StateProcessor()
        self.summary_writer = summary_writer
        self.env = env

        with tf.variable_scope(name):
            self.policy_net = PolicyEstimator(policy_net.num_outputs)
            self.value_net = ValueEstimator(reuse=True)

        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(
                scope="global",
                collection=tf.GraphKeys.TRAINABLE_VARIABLES)
            tf.contrib.slim.get_variables(
                scope=self.name+'/',
                ollection=tf.GraphKeys.TRAINABLE_VARIABLES)
        )

        self.vnet_train_op = make_copy_params_op(
            self.value_net, self.global_value_net)
        self.pnet_train_op = make_copy_params_op(
            self.policy_net, self.global_policy_net)

        self.state = None
