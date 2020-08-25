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

    def build_model(self):
        state = Input(batch_shape=(None, self.state_size))
        shared = Dense(self.hidden1, input_dim=self.state_size,
                       activation='relu',
                       kernel_initializer='glorot_uniform')(state)

        actor_hidden = Dense(self.hidden2, activation='relu',
                             kernel_initializer='glorot_uniform')(shared)
        action_prob = Dense(self.action_size, activation='softmax',
                            kernel_initializer='glorot_uniform')(actor_hidden)

        value_hidden = Dense(self.hidden2, activation='relu',
                             kernel_initializer='he_uniform')(shared)
        state_value = Dense(1, activation='linear',
                            kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()
        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.action_size))
        advantages = K.placeholder(shape=(None, ))

        policy = self.actor.output

        good_prob = K.sum(action*policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        loss = K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        actor_loss = loss + 0.01 * entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(
            self.actor.trainable_weigths, [], actor_loss)
        train = K.function(
            [self.actor.input, action, advantages], [], updates=updates)
        return train
