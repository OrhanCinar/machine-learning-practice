
from worker import Worker
from policy_monitor import PolicyMonitor
from estimators import ValueEstimator, PolicyEstimator
from lib.atari import helpers as atari_helpers
import unittest
import gym
import sys
import os
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)


tf.flags.DEFINE_string("model_dir", "/tmp/a3c",
                       "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_string("env", "Breakout-v0",
                       "Name of gym Atari environment, e.g. Breakout-v0")
tf.flags.DEFINE_integer(
    "t_max", 5, "Number of steps before performing an update")
tf.flags.DEFINE_integer("max_global_steps", None,
                        "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer(
    "eval_every", 300, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean(
    "reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer(
    "parallelism", None, "Number of threads to run. If not set we run [num_cpu_cores] threads.")

# https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/a3c/train.py

FLAGS = tf.flags.FLAGS


def make_env(wrap=True):
    env = gym.env.make(FLAGS.env)

    env = env.env

    if wrap:
        env = atari_helpers.AtariEnvWrapper(env)
    return env


env_ = make_env()

if FLAGS.env == "Pong-v0" or FLAGS.env == "Breakout-v0":
    VALID_ACTIONS = list(range(4))
else:
    VALID_ACTIONS = list(range(env_.action_space.n))

env_.close()


NUM_WORKERS = multiprocessing.cpu_count()

if FLAGS.parallelism:
    NUM_WORKERS = FLAGS.parallelism

MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")


if FLAGS.reset:
    shutil.rmtree(MODEL_DIR, ignore_errors=True)

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))
