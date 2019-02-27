import numpy as np
import tensorflow as tf


def spectral_norm_wrapper(scope=''):
    def spectral_norm(w, iteration=1):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable('%s/u' % scope, [1, w_shape[-1]], initializer=tf.random_normal_initializer(),
                            trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm

    return spectral_norm


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, sn=False):
    scope, i = tf.get_variable_scope().name, 0
    for h in hidden_sizes[:-1]:
        kc = spectral_norm_wrapper('%s/%d' % (scope, i)) if sn else None
        x = tf.layers.dense(x, units=h, activation=activation, kernel_constraint=kc)
        i += 1

    last_sn = spectral_norm_wrapper('%s/%d' % (scope, i)) if sn else None
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, kernel_constraint=last_sn)


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


"""
Actor-Critics
"""


def mlp_actor_critic(x, a, hidden_sizes=(400, 300), activation=tf.nn.relu,
                     output_activation=tf.tanh, action_space=None):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('pi'):
        pi = act_limit * mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
    with tf.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([x, a], axis=-1), list(hidden_sizes) + [1], activation, None, True), axis=1)
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x, pi], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)
    return pi, q, q_pi
