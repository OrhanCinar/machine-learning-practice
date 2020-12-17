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
