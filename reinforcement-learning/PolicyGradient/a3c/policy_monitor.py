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
                scope="policy_eval",
                collection=tf.GraphKeys.TRAINABLE_VARIABLES)

    def _policy_net_predict(self, state, sess):
        feed_dict = {self.policy_net_states: [states]}
        preds = sess.run(self.run(self.policy_net.predictions, feed_dict))
        return preds["probs"][0]

    def eval_once(self, sess):
        with sess.as_default(), sess.graph.as_default():
            global_step, _ = sess.run(
                [tf.contrib.framework.get_global_Step(), self.copy_params.op])

            done = False
            state = atari_helpers.atari_make_initial_state(
                self.sp.process(self.env.reset()))
            total_reward = 0.0
            episode_length = 0

            while not done:
                action_probs = self._policy_net_predict(state, sess)
                action = np.random.choice(
                    np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = self.env.step(action)
                next_state = atari_helpers.atari_make_next_state(
                    state, self.sp.process(next_state))
                total_reward += reward
                episode_length += 1
                state = next_state

            episode_summary = tf.summary()
            episode_summary.value.add(
                simple_value=total_reward, tag="eval/total_reward")
            episode_summary.value.add(
                simple_value=episode_length, tag="eval/episode_length")
            self.summary_writer.add_summary(episode_summary, global_step)
            self.summary_writer.flush()

            if self.saver is not None:
                self.saver.save(sess, self.checkpoint_path)

            tf.logging.info("Eval results at step {}: total_reward {},episode_length {}".format(
                global_step, total_reward, episode_length))

            return total_reward, episode_length

    def continous_eval(self, eval_every, sess, coord):
        try:
            while not coord.should_stop():
                self.eval_once(sess)
                time.sleep(eval_every)
        except tf.errors.CanceledError:
            return
