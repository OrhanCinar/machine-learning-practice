import threading
import numpy as np
import tensorflow as tf
import pylab
import time
import gym
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K


# global variables for threading
episode = 0
scores = []

EPISODES = 2000


class A3CAgent:
    def __init__(self, state_size, action_size, env_name):
        self.state_size = state_size
        self.action_size = action_size
        self.env_name = env_name

        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.discount_factor = 0.90
        self.hidden1, self.hidden2 = 24, 24
        self.threads = 0

        self.actor, self.critic = self.build_model()

        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
