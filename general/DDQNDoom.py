import tensorflow as tf
import numpy as np
from vizdoom import *

import random
import time
from skimage import transform

from collections import deque
import matplotlib.pyplot as plt

import warnings
warnings.filters("ignore")

# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb


def create_environment():
    game = DoomGame()
    game.load_config("deadly_corridor.cfg")
    game.set_doom_scenario_path("deadly_corridor.wad")
    game.init()

    possible_actions = np.identity(7, type=int).tolist()
    return game, possible_actions


game, possible_actions = create_environment()

"""
    preprocess_frame:
    Take a frame.
    Resize it.
        __________________
        |                 |
        |                 |
        |                 |
        |                 |
        |_________________|
        
        to
        _____________
        |            |
        |            |
        |            |
        |____________|
    Normalize it.
    
    return preprocessed_frame
    
    """


def preprocess_frame(frame):
    cropped_frame = frame[15:-5, 20:-20]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(cropped_frame, [100, 120])
    return preprocessed_frame  # 100x120x1 frame


stack_size = 4
stacked_frames = deque([np.zeros((100, 120), dtype=np.int)
                        for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros((100, 120), dtype=np.int)
                                for i in range(stack_size)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stack_frames


stack_size = [100, 120, 4]
action_size = game.get_available_buttons_size()
learning_rate = 0.00025
total_episodes = 5000
max_steps = 5000
batch_size = 64
max_tau = 10000
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00005
gamma = 0.95
pretrain_length = 100000
memory_size = 100000
training = False
episode_render = False


class DDDQNet:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate,
        self.name = name

        with tf.variable_scope(self.name):
            self.inputs_ = tf.placeholder(
                tf.float32, [None, *state_size], name="inputs")
            self.ISWeigths_ = tf.placeholder(
                tf.float32, [None, 1], name="IS_Weigths")
            self.actions_ = tf.placeholder(
                tf.float32, [None, state_size], name="actions_")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            ELU
            """

            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")
            self.conv3_out = tf.nn.elu(self.conv2, name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out)

            self.value_fc = tf.layers.dense(inputs=self.flatten,
                                            units=512,
                                            activation=None,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="value_fc")

            self.value = tf.layers.dense(inputs=self.value_fc,
                                         units=512,
                                         activation=tf.nn.elu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name="value")

            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                units=512,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                name="advantage_fc")

            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                             name="advantages")

            self.output = self.value + \
                tf.subtract(self.advantage, tf.reduce_mean(
                    self.advantage, axis=1, keepdims=True))

            self.Q = tf.reduce_sum(tf.multiply(
                self.output, self.actions_), axis=1)

            self.absolute_errors = tf.abs(self.target_Q - self.Q)

            self.loss = tf.reduce_mean(
                self.ISWeights_ * tf.squared_difference(self.target_Q), self.Q)

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)


tf.reset_default_graph()
DQNetwork = DDDQNet(state_size, action_size, learning_rate, name="DQNetwork")

TargetNetwork = DDDQNet(state_size, action_size,
                        learning_rate, name="TargetNetwork")


class SumTree(object):

    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)
        self.data = np.zeros(capacity, dtype == object)

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        while tree_index != 0:
            tree_index = (tree_index-1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity+1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self):
        return self.tree[0]


class Memory(object):
    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4
    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, experience):
        max_priority = np.max(self.tree[-self.tree.capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.tree.add(max_priority, experience)

    def sample(self, n):
        memory_b = []
        b_idex, b_ISWeights = np.empty(
            (n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority / n

        self.PER_b = np.min(
            [1., self.PER_b + self.PER_b_increment_per_sampling])

        p_min = np.min(self.tree[-self.tree.capacity:]
                       ) / self.tree.total_priority
        max_weight = (p_min*n)(-self.PER_b)

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i+1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)

            sampling_probabilities = priority / self.tree.total_priority
            b_ISWeights[i, 0] = np.power(
                n*sampling_probabilities, -self.PER_b)/max_weight

            b_idx[i] = index
            experience = [data]
            memory_b.append(experience)
        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        pd = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


memory = Memory(memory_size)

game.new_episode()

for i in range(pretrain_length):
    if i == 0:
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    action = random.choice(possible_actions)
    reward = game.make_action(action)
    done = game.is_episode_finished()

    if done:
        next_state = np.zeros(state.shape)
        experience = state, action, reward, next_state, done
        memory.store(experience)
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    else:
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(
            stacked_frames, next_state, False)
        experience = state, action, reward, next_state, done
        memory.store(experience)
        state = next_state

writer = tf.summary.FileWriter("/tensorboard/ddqn/1")
tf.summary.scalar("Loss", DQNetwork.loss)
write_op = tf.summary.merge_all()


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, acions):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + \
        (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if(explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)
    else:
        Qs = sess.run(DQNetwork.output, feed_dict={
                      DQNetwork.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
    return action, explore_probability


def update_target_graph():
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABES, "DQNetwork")
    to_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABES, "TargerNetwork")

    op_holder = []

    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        decay_step = 0
        tau = 0
        game.init()
        update_target = update_target_graph()
        sess.run(update_target)

    for episode in range(total_episodes):
        step = 0
        episode_rewards = []
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        while step < max_steps:
            step += 1
            tau += 1
            decay_step += 1

            action, explore_probability = predict_action(
                explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

            reward = game.make_action(action)
            done = game.is_episode_finished()
            episode_rewards.append(reward)

            if done:
                next_state = np.zeros((120, 140), dtype=np.int)
                next_state, stacked_frames = stack_frames(
                    stacked_frames, next_state, False)
                step = max_steps
                total_reward = np.sum(episode_rewards)

                print('Episode: {}'.format(episode),
                      'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore P: {:.4f}'.format(explore_probability))

                experience = state, action, reward, next_state, done
                memory.store(experience)
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(
                    stacked_frames, next_state, False)
                experience = state, action, reward, next_state, done
                state = next_state

            tree_idx, batch, ISWeights_mb = memory.sample(batch_size)

            states_mb = np.array([each[0] for each in batch], ndmin=3)
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch], ndmin=3)
            dones_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []

            q_next_state = sess.run(DQNetwork.output, feed_dict={
                                    DQNetwork.inputs_: next_states_mb})

            q_target_next_state = sess.run(TargetNetwork.output, feed_dict={
                                           TargetNetwork.inputs_: next_states_mb})

            for i in range(0, len(batch)):
                terminal = dones_mb[i]

                action = np.argmax(q_next_state[i])

                if terminal:
                    target_Qs_batch.append(rewards_mb[i])
                else:
                    target = rewards_mb[i] + gamme * \
                        q_target_next_state[i][action]
                    target_Qs_batch.append(target)

            targets_mb = np.array([each for each in target_Qs_batch])

            _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                                feed_dict={DQNetwork.inputs_: states_mb,
                                                           DQNetwork.target_Q: targets_mb,
                                                           DQNetwork.actions_: actions_mb,
                                                           DQNetwork.ISWeigths_: ISWeights_mb})

            memory.batch_update(tree_idx, abs_errors)

            summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                    DQNetwork.target_Q: targets_mb,
                                                    DQNetwork.actions_: actions_mb,
                                                    DQNetwork.ISWeigths_: ISWeights_mb})
            writer.add_summary(summary, episode)
            writer.flush()

            if tau > max_tau:
                update_target = update_target_graph()
                sess.run(update_target)
                tau = 0
                print("Model Updated")

        if episode % 5 == 0:
            save_path = saver.save(sess, "./model,model.chpt")
            print("Model Saved")

with tf.Session() as sess:

    game = DoomGame()

    # Load the correct configuration (TESTING)
    game.load_config("deadly_corridor_testing.cfg")

    # Load the correct scenario (in our case deadly_corridor scenario)
    game.set_doom_scenario_path("deadly_corridor.wad")

    game.init()

    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    game.init()

    for i in range(10):
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        while not game.is_episode_finished():
            exp_exp_tradeoff = np.random.rand()
            explore_probability = 0.01

            if (explore_probability > exp_exp_tradeoff):
                action = random.choice(possible_actions)

            else:
                Qs = sess.run(DQNetwork.output, feed_dict={
                              DQNetwork.inputs_: state.reshape((1, *state.shape))})
                choice = np.argmax(Qs)
                action = possible_actions[int(choice)]

            game.make_action(action)
            done = game.is_episode_finished()

            if done:
                break

            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(
                    stacked_frames, next_state, False)
                state = next_state

        score = game.get_total_reward()
        print("Score: ", score)

    game.close()
