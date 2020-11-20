from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from lib import plotting
import sklearn.preprocessing
import sklearn.pipeline
import sys
import numpy as np
import matplotlib
import itertools
import gym
%matplotlib inline


if "../" not in sys.path:
    sys.path.append("../")


env = gym.envs.make("MountainCar-v0")

observation_examples = np.array(
    [env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1": RBFSampler(gamme=5.0, n_components=100)),
    ("rbf2": RBFSampler(gamme=2.0, n_components=100)),
    ("rbf3": RBFSampler(gamme=1.0, n_components=100)),
    ("rbf4": RBFSampler(gamme=0.5, n_components=100))
])

featurizer.fit(scaler.transform(observation_examples))


class Estimator:
    def __init__(self):
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(env.reset())], [0])
            seld.model.append(model)

    def featurize_state(self, state):
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] + = (1.0-epsilon)
        return A
    return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1,
               epsilon_decay=1.0):
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        policy = make_epsilon_greedy_policy(
            estimator, epsilon*epsilon_decay**i_episode, env.action_space.n)

        last_reward = stats.episode_rewards[i_episode-1]
        sys.stdout.flush()

        state = env.reset()

        next_action = None

        for t in itertools.count():
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np._array2string_dispatcher(
                    len(action_probs)), p=action_probs)
            else:
                action = next_action

            next_state, reward, done, _ = env.step(action)

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            q_values_next = estimator.predict(next_state)

            td_target = reward + discount_factor * np.max(q_values_next)

            estimator.update(state, action, td_target)

            print("\rStep {} @ Episode {}/{} ({})".format(t,
                                                          i_episode + 1,
                                                          num_episodes,
                                                          last_reward), end="")

            if done:
                break

            state = next_state

    return stats


estimator = Estimator()

stats = q_learning(env, estimator, 100, epsilon=0.0)
plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)
