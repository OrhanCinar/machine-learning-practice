# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb
import numpy as np
import gym
import random
import time
import pickle


class FrozenLake:

    def __init__(self):
        self.env = gym.make('FrozenLake-v0')

        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.n
        self.q_table = np.zeros((self.state_size, self.action_size))

        # print(action_size)
        # print(state_size)
        # print(q_table)

        self.total_episodes = 15000
        self.learning_rate = 0.8
        self.max_steps = 99
        self.gamma = 0.95

        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.005
        self.start_q_table = "qtable-1583610430.pickle"
        self.rewards = []

    def train(self):
        for episode in range(self.total_episodes):
            state = self.env.reset()
            step = 0
            done = False
            total_rewards = 0

            if episode % 100 == 0:
                print(f"Episode {episode}")

            for step in range(self.max_steps):
                exp_exp_tradeoff = random.uniform(0, 1)

                # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
                if exp_exp_tradeoff > self.epsilon:
                    action = np.argmax(self.q_table[state, :])
                else:
                    action = self.env.action_space.sample()

                new_state, reward, done, info = self.env.step(action)

                self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (
                    reward+self.gamma * np.max(self.q_table[new_state, :]) - self.q_table[state, action])

                total_rewards += reward

                state = new_state

                if done == True:
                    break

            self.epsilon = self.min_epsilon + (self.max_epsilon-self.min_epsilon) * \
                np.exp(-self.decay_rate*episode)
            self.rewards.append(total_rewards)

        print("Score over time: " + str(sum(self.rewards)/self.total_episodes))
        with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
            pickle.dump(self.q_table, f)
        # print(self.q_table)

    def play(self):
        # play game
        with open(self.start_q_table, "rb") as f:
            self.q_table = pickle.load(f)
        print(self.q_table)

        for episode in range(1):
            state = self.env.reset()
            step = 0
            done = False
            print("****************************************************")
            print("EPISODE ", episode)

            for step in range(self.max_steps):

                # Take the action (index) that have the maximum expected future reward given that state
                action = np.argmax(self.q_table[state, :])

                new_state, reward, done, info = self.env.step(action)
                self.env.render()

                if done:
                    # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
                    self.env.render()

                    # We print the number of step it took.
                    print("Number of steps", step)
                    break
                state = new_state
        self.env.close()


c = FrozenLake()

# c.train()
c.play()
