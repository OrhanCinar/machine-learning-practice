import matplotlib.pyplot as plt
from worker import make_copy_params_op
from estimators import ValueEstimator, PolicyEstimator
from lib.atari import helpers as atari_helpers
from lib.atari.state_processor import StateProcessor
import gym
from gym.wrappers import Monitor
import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf
import time

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)


# https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/a3c/policy_monitor.py

class PolicyMonitor(object):

    def __init__(self, env, policy_net, summary_writer, saver=None):
        self.video_dir = os.path.join(summary_writer.get_logdir(), "../videos")
        self.video_dir = os.path.abspath(self.video_dir)

        self.env = Monitor(env, directory, self.video_dir,
                           video_callable=lambda x: True, resume=True)
        self.global_policy_net = policy_net
        self.summary_writer = summary_writer
        self.saver = saver
        self.sp = StateProcessor()

        self.checkpoint_path = os.path.abspath(os.path.join(
            summary_writer.get_logdir(), "../checkpoints/model"))

        try:
            os.makedirs(self.video_dir)
        except expression as identifier:
            pass

        with tf.variable_scope("policy_eval"):
            self.contrib.slim.get_variables(
                scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES)
            self.contrib.slim.get_variables(
                scope="policy_eval", collection=tf.GraphKeys.TRAINABLE_VARIABLES)
