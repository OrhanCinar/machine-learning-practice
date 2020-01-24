import gym
from gym.spaces import Box, Discrete
import numpy as np
from collections import defaultdict

Q = defaultdict(float)
gamma = 0.99  # Dsciount factor
alpha = 0.5

env = gym.make('CartPole-v0')
# actions = range(env.action_space)


def update_Q(s, r, a, s_next, done):
    max_q_next = max(Q[s_next, a] for a in actions)
    # Do not include the next state's value if currently at the terminal state.
    Q[s, a] += alpha * (r+gamme*max_q_next*(1.0-done)-Q[s, a])


class DiscretizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_bins=0, low=None, high=None):
        super().__init__(env)

        assert isinstance(env.observation_space, Box)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        low = np.array(low)
        high = np.array(high)

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins+1)
                         for l, h in zip(low.flatten(), high.flatten())]
        print("New ob space:", Discrete((n_bins + 1) ** len(low)))
        self.observation_space = Discrete(n_bins ** len(low))

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins+1)**i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0] for x, bins in zip(
            observation.flatten(), self.val_bins)]
        return self._convert_to_one_number


env = DiscretizedObservationWrapper(
    env,
    n_bins=8,
    low=[-2.4, -2.0, -0.42, -3.5],
    high=[2.4, 2.0, 0.42, 3.5]
)
