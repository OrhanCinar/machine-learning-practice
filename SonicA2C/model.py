import os
import time
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
import cv2
import matplotlib.pyplot as plt
from baselines.a2c.utils import cat_entropy, mse
from utilities import make_path, find_traniable_variables, discount_with_done


class Model(object):
    def __init__(self,
                 policy,
                 ob_space,
                 action_space,
                 nenvs,
                 nsteps,
                 ent_coef,
                 vf_coef,
                 max_grad_norm):

        sess = tf.get_default_session()

        actions_ = tf.placeholder(tf.int32, [None], name="actions_")
        advantages_ = tf.placeholder(tf.int32, [None], name="advantages_")
        rewards_ = tf.placeholder(tf.int32, [None], name="rewards_")
        lr_ = tf.placeholder(tf.int32, [None], name="learning_rate_")

        step_model = policy(sess, ob_space, action_space,
                            nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, action_space,
                             nenvs*nsteps, nsteps, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=train_model.pi, labels=actions_)

        pg_loss = tf.reduce_mean(advantages_ * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squezze(train_model.vf), rewards_))
        entropy = tf.reduce_mean(train_model.pd.entropy())
        loss = pgloss - entropy * ent_coef + vf_loss * vf_coef

        params = find_tranaible_variables("model")

        grads = tf.gradients(loss, params)

        if max_grad_norm is not None:
            grands, grand_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        trainer = tf.train.RMSPropOptimizer(
            learning_rate_, decay=0.99, epsilon=1e-5)

        _train = trainer.apply_gradient(grads)

        def train(states_in, actions, returns, values, lr):
            advantages = returns - values

            td_map = {train_model.inputs_: states_in,
                      actions_: actions,
                      advantages_: advantages,
                      rewards_: returns,
                      lr_: lr}

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train], td_map)
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            saver = tf.train.Saver()
            saver.save(sess, save_path)

        def load(load_path):
            saver = tf.train.Saver()
            print("Loading " + load_path)
            saver.restore(sess, load_path)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)
