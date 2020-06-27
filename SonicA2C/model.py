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


class Runner(AbtractEnvRunner):
    def __init__(selv, env, model, nsteps, total_timesteps, gamma, lam):
        super().__init(env=env, model=model, nsteps=nsteps)

        self.gamma = gamma
        self.kam = lam
        self.total_timesteps = total_timesteps

    def run(self):
        mb_obs, mb_actions, mb_rewards, mb_values, mb_dones = [], [], [], [], []

        for n in range(self.nsteps):
            actions, values = self.model.step(self.obs, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_dones.append(values)
            self.obs[:], rewards, self.dones, _ = self.env.step(actions)
            mb_rewards.append(rewards)

        mb_obs = np.asarray(mb_obs, dtype=np.uint8)
        mb_rewards = np.asarray(mb_rewards, dtype=np.uint8)
        mb_actions = np.asarray(mb_actions, dtype=np.uint8)
        mb_values = np.asarray(mb_values, dtype=np.uint8)
        mb_dones = np.asarray(mb_dones, dtype=np.uint8)
        last_values = self.model.value(self.obs)

        mb_returns = np.zeros_like(mb_rewards)
        mb_advantages = np.zeros_like(mb_rewards)

        lastgaelam = 0

        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * \
                nextvalues * nextnonterminal - mb_values[t]

            mb_advantages[t] = lastgaelam = delta + \
                self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advantages + mb_values

        return map(sf01, (mb_obs, mb_actions, mb_returns, mb_values))


def sf01(arr):
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0]*s[1], *s[2:])


def learn(policy,
          env,
          nsteps,
          total_timesteps,
          gamma,
          lam,
          vf_coef,
          ent_coef,
          lr,
          max_grad_norm,
          log_interval
          ):
    noptepochs = 4
    nminibatches = 8
    nenvs = env.num_envs

    ob_space = env.observation_space
    ac_space = env.action_space

    batch_size = nenvs * nsteps
    batch_train_size = batch_size // nminibatches

    assert batch_size % nminibatches == 0

    model = Model(policy=policy, ob_space=ob_space, action_space=ac_space, nenvs=nenvs,
                  nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)

    load_path = "./models/260/model.ckpt"
    model.load(load_path)

    runner = Runner(env, model, nsteps, total_timesteps, gamma, lam)

    tfirststart = time.time()

    for update in range(1, total_timesteps//batch_size+1):
        tstart = time.time()
        obs, actions, returns, values = runner.run()
        mb_losses = []
        total_batches_train = 0
        indices = np.arange(batch_size)

        for _ in range(noptepochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, batch_train_size):
                end = start + batch_train_size
                mbinds = indices[start:end]
                slices = (arr[mbinds]
                          for arr in (obs, actions, returns, values))
                mb_losses.append(model.train(*slices, lr))

        lossvalues = np.mean(mb_losses, axis=0)
        tnow = time.time()

        fps = int(batch_size / (tnow - tstart))

        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*batch_size)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_loss", float(lossvalues[0]))
            logger.record_tabular("policy_entropy", float(lossvalues[2]))
            logger.record_tabular("value_loss", float(lossvalues[1]))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("time elapsed", float(tnow - tfirststart))
            logger.dump_tabular()

            savepath = "./models/" + str(update) + "/model.ckpt"
            model.save(savepath)
            print('Saving to', savepath)

    env.close()


def play(policy, env):
    ob_space = env.observation_space
    ac_space = env.action_space

    model = Model(policy, ob_space, ac_space, nenvs=1, nsteps=1,
                  ent_coef=0, vf_coef=0, max_grad_norm=0)

    load_path = "./model/260/model.ckpt"
    model.load(load_path)

    obs = env.reset()

    score = 0
    boom = 0
    done = False

    while done == False:
        boom += 1
        actions, values = model.step(obs)
        obd, rewards, done, _ = env.step(actions)
        score += rewards
        env.render()

    print("Score : ", score)
    env.close()
