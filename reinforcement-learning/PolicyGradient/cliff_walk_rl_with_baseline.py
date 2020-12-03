from lib import plotting
from lib.envs.cliff_walking import CliffWalkingEnv
import collections
import tensorflow as tf
import sys
import numpy as np
import matplotlib
import itertools
import gym
%matplotlib inline


if "../" not in sys.path:
    sys.path.append("../")

matplotlib.style.use('ggplot')

# https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb

env = CliffWalkingEnv()


class PolicyEstimator():
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.varible_scope():
            self.state = tf.placeholder(tf.int32, [], "state")
            self.action = tf.placeholder(tf.int32, [], "action")
            self.target = tf.placeholder(tf.int32, [], "target")

        state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
        self.output_layer = tf.contrib.layers.fullt_connected(
            inputs=tf.expand_dims(state_one_hot, 0),
            num_outputs=env.action_space.n,
            activation_fn=None,
            weights_initializer=tf.zeros.initializer)

        self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
        self.picked_Action_prob = tf.gather(self.action_probs, self.action)

        self.loss = -tf.log(self.picked_Action_prob) * self.target
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            state_one_hot = tf.one_hot(
                self.state, int(env.observation_space.n))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def reinforce(env, estimator_policy, estimator_value, num_episodes,
              discount_factor=1.0):
    stats = plotting.EpisodeStats(episode_lengths=num_episodes),
    episode_rewards = np.zeros(num_episodes)

    Transition = collections.namedtuple(
        "Transition", ["state", "action", "reward", "next_state", "done"])

    for i_episode in range(num_episodes):
        state = env.reset()

        episode = []

        for t in itertools.count():
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(
                np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            episode.append(Transition(state=state,
                                      action=action,
                                      reward=reward,
                                      next_state=next_state,
                                      done=done))

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            print("\rStep {} @ Episode {}/{} ({})".format(
                t, i_episode + 1, num_episodes,
                stats.episode_rewards[i_episode - 1]), end="")

            if done:
                break

            state = next_state

        for t, transition in enumerate(episode):
            total_return = sum(discount_factor**i *
                               t.reward for i, t in enumerate(episode[t:]))

            baseline_value = estimator_value.predict(transition.state)

            advantage = total_return - baseline_value

            estimator_value.update(transition.state, total_return)
            estimator_policy.update(
                transition.state, advantage, transition.action)

    return stats


tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
value_estimator = ValueEstimator()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    stats = reinforce(env, policy_estimator, value_estimator,
                      2000, discount_factor=1.0)

plotting.plot_episode_stats(stats, smoothing_window=25)
