import tensorflow as tf
import numpy as np
import gym

env = gym.make("CartPole-v0")
env = env.unwrapped
env.seed(1)

state_size = 4
action_size = ev.action_space.n
max_episode = 300
learning_rate = 0.01
gamme = 0.95


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0

    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamme + episode_rewards[i]
        discount_and_normalize_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    st = np.std(discount_and_normalize_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / std
    return discounted_episode_rewards


with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
    actions = tf.placeholder(tf.float32, [None, state_size], name="actions")
    discounted_episode_rewards_ = tf.placeholder(
        tf.float32, [None, ], name="discounted_episode_rewards")

    mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

    with tf.name_scope("fc1"):
        fc1 = tf.contrib.layers.fully_connected(inputs=input_,
                                                num_outputs=10,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("fc2"):
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                                num_outputs=action_size,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("fc3"):
        fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
                                                num_outputs=action_size,
                                                activation_fn=None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc3)

    with tf.name_scope("loss"):
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(
            logits=fc3, labels=actions)
        loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)

    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
