import numpy as np
import sys
import matplotlib.pyplot as plt
if "../" not in sys.path:
    sys.path.append("../")
# https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Gamblers%20Problem%20Solution.ipynb


def value_iteration_for_gamblers(p_h, theta=0.0001, discount_factor=1.0):
    rewards = np.zeros(101)
    rewards[100] = 1

    V = np.zeros(101)

    def one_step_lookahead(s, V, rewards):
        A = np.zeros(101)
        stahes = range(1, min(s, 100-s)+1)
        for a in stakes:
            A[a] = p_h * (rewards[s+a] + V[s+a] * discount_factor) + \
                (1-p_h) * (rewards[s-a] + V[s-a]*discount_factor)
        return A

    while True:
        delta = 0
        for s in range(1, 100):
            A = one_step_lookahead(s, V, rewards)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            break

    policy = np.zeros(100)
    for s in range(1, 100):
        A = one_step_lookahead(s, V, rewards)
        best_action = np.argmax(A)
        policy[s] = best_action

    return policy, V


policy, v = value_iteration_for_gamblers(0.25)

print("Optimized Policy:")
print(policy)
print("")

print("Optimized Value Function:")
print(v)
print("")

# x axis values
x = range(100)
# corresponding y axis values
y = v[:100]

# plotting the points
plt.plot(x, y)

# naming the x axis
plt.xlabel('Capital')
# naming the y axis
plt.ylabel('Value Estimates')

# giving a title to the graph
plt.title('Final Policy (action stake) vs State (Capital)')

# function to show the plot
plt.show()

# x axis values
x = range(100)
# corresponding y axis values
y = policy

# plotting the bars
plt.bar(x, y, align='center', alpha=0.5)

# naming the x axis
plt.xlabel('Capital')
# naming the y axis
plt.ylabel('Final policy (stake)')

# giving a title to the graph
plt.title('Capital vs Final Policy')

# function to show the plot
plt.show()
