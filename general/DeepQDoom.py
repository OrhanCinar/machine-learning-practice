import tensorflow as tf
import numpy as np
from vizdoom import *

import random
import time
from skimage import transform

from collections import deque
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Doom/Deep%20Q%20learning%20with%20Doom.ipynb


def create_environment():
    game = DoomGame()
    game.load_config('basic.cfg')
    game.set_doom_scnerio_path('basic.wad')
    game.init()

    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]


def test_environment():
    game = DoomGame()
    game.load_config('basic.cfg')
    game.set_doom_scnerio_path('basic.wad')
    game.init()

    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    actions = [left, right, shoot]
    episodes = 10

    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print('\treward:', reward)
            time.sleep(0.02)
        print("Result : ", game.get_total_reward())
        time.sleep(2)
    game.close()


game, possible_actions = create_environment()


def preprocess_frame(frame):
    cropped_frame = frame[30:-10, 30:-30]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame


stack_size = 4


def stack_frames(stacked_frame, state, is_new_episode):
    frmae = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype == np.int)
                                for i in range(stack_size)], maxlen=4)

        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stack_frames.append(frame)
        stacked_state = np.stack(stacked_frames, acis=2)

    return stacked_state, stack_frames


state_size = [84, 84, 4]
action_size = game.get_available_buttons_size()
learning_rate = 0.0002

total_episodes = 500
max_steps = 100
batch_size = 64

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

gamme = 0.95

pretrain_length = batch_size
memory_size = 1000000

training = True
episode_render = False


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(
                tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(
                tf.float32, [None, *3], name="actions_")

            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
                First convnet:
                CNN
                BatchNormalization
                ELU
            """

            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filter=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name="batch_norm1")

            self.conv1_out = tf.nn.elu(
                self.conv1_batchnorm, name="conv1_out")  # --> [20, 20, 32]

            """
                Second convnet:
                CNN
                BatchNormalization
                ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filter=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name="batch_norm2")

            self.conv2_out = tf.nn.elu(
                self.conv2_batchnorm, name="conv2_out")  # --> [9, 9, 64]

            """
                   Third convnet:
                   CNN
                   BatchNormalization
                   ELU
            """

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filter=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name="batch_norm3")

            self.conv3_out = tf.nn.elu(
                self.conv3_batchnorm, name="conv3_out")  # --> [3, 3, 128]

            self.flatten = tf.layers.flatten(self.conv3_out)  # --> [1152]

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=3,
                                          activation=None)

            self.Q = tf.reduce_sum(tf.multiply(
                self.output, self.actions_, axis=1))

            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimum(self.loss)


tf.reset_default_graph()
DQNetwork = DQNetwork(state_size, action_size, learning_rate)


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(explore_start)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size), size=batch_size, replace=False)
        return (self.buffer[i] for i in index)


memory = Memory(max_size=memory_size)

game.new_episode()

for i in range(pretrain_length):
    if i == 0:
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    action = random.choice(possible_actions)

    reward = game.make_action(action)
    done = game.is_episode_finished()

    if done:
        next_state = np.zeros(state, shape)
        memory.add((state, action, reward, next_state, done))

        game.new_episode()

        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    else:
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(
            stacked_frames, next_state, False)
        memory.add((state, action, reward, next_state, done))
        state = next_state


writer = tf.summary.FileWriter("/tensorboard/dqn/1")

tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + \
        (explore_start - explore_stop) * np.exp(-decay_rate*decay_step)

    if (explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)
    else:
        Qs = sess.run(DQNetwork.output, feed_dict={
                      DQNetwork.inputs_: state.redshape((1, *state_shape))})

        choise = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability


saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        decay_step = 0

        game.init()

        for episode in range(total_episodes):
            step = 0
            episode_rewards = []
            game.new_episode()
            state = game.get_state().screen_buffer

            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step > max_steps:
                step += 1
                decay_step += 1

                action, explore_probability = predict_action(
                    explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

                reward = game.make_action(action)

                done = game.is_episode_finished()
                episode_rewards.append(reward)

                if done:
                    next_state = np.zeros((84, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(
                        stacked_frames, next_state, False)
                    step = max_steps

                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability))

                    memory.add((state, action, reward, next_state, done))
                else:
                    next_state = game.get_state().screen_buffer
                    next_state, stacked_frames = stack_frames(
                        stacked_frames, next_state, False)

                    memory.add((state, action, reward, next_state, done))
                    state = next_state

            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch], ndmin=3)
            done_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []
            Qs_next_state = sess.run(DQNetwork.output, feed_dict={
                                     DQNetwork.inputs_: next_states_mb})

            for i in range(0, len(batch)):
                terminal = done_mb[i]

                if terminal:
                    target_Qs_batch.append(rewards_mb[i])
                else:
                    target = rewards_mb[i] + gamme * np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)

            argets_mb = np.array([each for each in target_Qs_batch])

            loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                               feed_dict={DQNetwork.inputs_: states_mb,
                                          DQNetwork.target_Q: targets_mb,
                                          DQNetwork.actions_: actions_mb})

            # Write TF Summaries
            summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                    DQNetwork.target_Q: targets_mb,
                                                    DQNetwork.actions_: actions_mb})
            writer.add_summary(summary, episode)
            writer.flush()

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")


with tf.Session() as sess:

    game, possible_actions = create_environment()

    totalScore = 0

    saver.restore(sess, "./models/model.ckpt")
    game.init()
    for i in range(1):

        done = False

        game.new_episode()

        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        while not game.is_episode_finished():
            Qs = sess.run(DQNetwork.output, feed_dict={
                          DQNetwork.inputs_: state.reshape((1, *state.shape))})

            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]

            game.make_action(action)
            done = game.is_episode_finished()
            score = game.get_total_reward()

            if done:
                break

            else:
                print("else")
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(
                    stacked_frames, next_state, False)
                state = next_state

        score = game.get_total_reward()
        print("Score: ", score)
    game.close()
