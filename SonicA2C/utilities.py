import tensorflow as tf


def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()


def make_path(f):
    return os.makedirs(f, exists_ok=True)


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]
