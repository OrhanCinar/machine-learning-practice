from lib.envs.gridworld import GridworldEnv
import numpy as np
import pprint
import sys
if "../" not in sys.path:
    sys.path.append("../")

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    V = np.zeros(env.nS)
    while True:
        delta = 0

        for s in range(env.nS):
            v = 0

            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * \
                        (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v-V[s]))
            V[s] = v

        if delta < thera:
            break

    return np.array(V)


random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")
