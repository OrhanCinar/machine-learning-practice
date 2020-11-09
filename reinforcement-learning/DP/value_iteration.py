

from lib.envs.gridworld import GridworldEnv
import numpy as np
import pprint
import sys
if "../" not in sys.path:
    sys.path.append("../")
# https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


def value_iteration(env, thetra=0.0001, discount_factory=1.0):
    def one_step_lookahed(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factory * V[next_state])
        return A

    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            A = one_step_lookahed(s, V)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value-V[s]))
            V[s] = best_action_value
        if delta < theta:
            break

    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        A = one_step_lookahed(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    return policy, V


policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")
