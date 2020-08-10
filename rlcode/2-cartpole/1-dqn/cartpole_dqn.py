import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 300

# https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/1-dqn/cartpole_dqn.py


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        self.memory = deque(maxlen=(2000))
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_tarhet_model()

        if self.load_model:
            self.model.load_weigths("./save_model/cartpole_dqn.h5")

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size,
                        activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
