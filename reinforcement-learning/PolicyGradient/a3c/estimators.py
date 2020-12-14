import numpy as np
import tensorflow as tf

# https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/a3c/estimators.py


def build_shared_network(X, add_summaries=False):
    conv1 = tf.contrib.layers.conv2d(
        X, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
    conv2 = tf.contrib.layers.conv2d(
        X, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2")

    fc1 = tf.contrib.layers.fully_connected(
        inputs=tf.contrib.layers.flatten(conv2),
        num_ouputs=256,
        scope="fc1")

    if add_summaries:
        tf.contrib.layers.summarize_activation(conv1)
        tf.contrib.layers.summarize_activation(conv2)
        tf.contrib.layers.summarize_activation(fc1)

    return fc1
