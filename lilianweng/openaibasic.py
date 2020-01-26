import gym
import random
import numpy as np

env = gym.make('MsPacman-v0')
steps = 10000
epsilon = 0.1
action_space = len(env.get_action_meanings())

# print(gym.spaces.Box)
# print(env.get_action_meanings())
# print(env.action_space)
# print(action_space)


obv = env.reset()
rewards = []
reward = 0.0


def act(self, obv):
    if np.random.random() < epsilon:
        return random.randrange(0, action_space)
    qvals = {a: self.Q[state, a] for a in env.get_action_meanings()}
    max_q = max(qvals.values())

    action_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(action_with_max_q)


for _ in range(steps):
    #randomAction = random.randrange(0, action_space)
    action = act(obv)
    obv_next, r, done, _ = env.step(action)
    #print('r', r)
    env.render()
    if done:
        gameDone = done
        print('done')
        obv = env.reset()

env.close()
#obv = env.reset()
