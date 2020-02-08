import gym
import random
import numpy as np
from collections import defaultdict
from gym.spaces import Discrete

env = gym.make('CartPole-v1')

# print(dir(env.action_space))

steps = 10000
epsilon = 0.1
#action_space = len(env.get_action_meanings())

print('observation_space', env.observation_space)
print('observation_space_high', env.observation_space.high)
print('observation_space_low', env.observation_space.low)
print('observation_action_space', env.action_space.n)
# print(env.get_action_meanings())

# print(action_space)
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high.astype(np.float64) -
                        env.observation_space.low.astype(np.float64)) / DISCRETE_OS_SIZE

q_table = np.random.uniform(
    low=-5, high=5, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print("DISCRETE_OS_SIZE", DISCRETE_OS_SIZE)
print("discrete_os_win_size", discrete_os_win_size)
print("q_table", q_table.shape)


# rewards = []
# reward = 0.0


# class Runner:
#     def __init__(self, env, gamma=0.99):

#         self.Q = defaultdict(float)
#         self.actions = range(env.action_space.n)
#         self.env = env
#         assert isinstance(env.action_space, Discrete)
#         assert isinstance(env.observation_space, Discrete)

#         # self.action_space = len(env.get_action_meanings())
#         # print('action_space', type(self.action_space))
#         # print('actions', self.actions)

#     def act(self, state, eps=0.1):
#         # return self.env.action_space.sample()
#         if np.random.random() < eps:
#             return self.env.action_space.sample()
#         # print("self.actions", self.actions)
#         print("state", state.shape)
#         qvals = {a: self.Q[state, a] for a in self.actions}

#         max_q = max(qvals.values())

#         action_with_max_q = [a for a, q in qvals.items() if q == max_q]
#         return np.random.choice(action_with_max_q)


# runner = Runner(env)

# for _ in range(steps):
#     done = False
#     obv = env.reset()

#     # print('r', r)

#     print('render')
#     while not done:
#         action = runner.act(obv)
#         # randomAction = random.randrange(0, action_space)
#         obv_next, r, done, _ = env.step(action)

#         env.render()
#         # print('done')


# env.close()


# # obv = env.reset()
