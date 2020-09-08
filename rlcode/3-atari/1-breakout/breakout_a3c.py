import gym
import time
import random
import threading
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K

# https://github.com/rlcode/reinforcement-learning/blob/master/3-atari/1-breakout/breakout_a3c.py

global episode
episode = 0
EPISODES = 8000000

env_name = "BreakoutDeterministic-v4"


class A3CAgent:
    def __init__(self, action_size):
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        self.discount_factor = 0.99
        self.no_op_steps = 30

        self.actor_lr = 2.5-4
        self.critic_lr = 2.5-4
        self.threads = 8

        self.actor, self.critic = self.build_model()

        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.summar_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary/breackout_a3c', self.sess.graph)

    def train(self):
        agents = [Agent(self.action_size, self.state_size, [],
                        self.sess, self.optimizer, self.discount_factor,
                        [self.summary_op, self.summar_placeholders,
                         self.update_ops, self.summary_writer]) for _ in range(self.threads)]

        for agent in agents:
            time.sleep(1)
            agent.start()

        while True:
            time.sleep(60*10)
            self.save_model("./save_model/breakout_a3c")

    def build_model(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(input=input, outputs=policy)
        critic = Model(input=input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])
        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * advantages
        actor_loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        loss = actor_loss + 0.01 * entropy
        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages], [loss],
                           updates=updates)

        return train

    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))
        value = self.critic.output
        loss = K.mean(K.square(discounted_reward - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reawrd], [loss],
                           updates=updates)
        return train
