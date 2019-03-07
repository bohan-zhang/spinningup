import numpy as np
import tensorflow as tf
import time
from spinup.algos.sac import core
from spinup.algos.sac.core import get_vars
from spinup.utils.logx import EpochLogger
from spinup.algos.ood.pairwise_distance import pairwise_distance

"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""


class SAC:
    def __init__(self, sess, replay_buffer, env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
                 steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, polyak=0.995, lr=1e-3, alpha=0.2,
                 batch_size=100, start_steps=10000, max_ep_len=1000, logger_kwargs=dict(), save_freq=1, name='sac'):
        """

        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: A function which takes in placeholder symbols
                for state, ``x_ph``, and action, ``a_ph``, and returns the main
                outputs from the agent's Tensorflow computation graph:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                               | given states.
                ``pi``       (batch, act_dim)  | Samples actions from policy given
                                               | states.
                ``logp_pi``  (batch,)          | Gives log probability, according to
                                               | the policy, of the action sampled by
                                               | ``pi``. Critical: must be differentiable
                                               | with respect to policy parameters all
                                               | the way through action sampling.
                ``q1``       (batch,)          | Gives one estimate of Q* for
                                               | states in ``x_ph`` and actions in
                                               | ``a_ph``.
                ``q2``       (batch,)          | Gives another estimate of Q* for
                                               | states in ``x_ph`` and actions in
                                               | ``a_ph``.
                ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and
                                               | ``pi`` for states in ``x_ph``:
                                               | q1(x, pi(x)).
                ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and
                                               | ``pi`` for states in ``x_ph``:
                                               | q2(x, pi(x)).
                ``v``        (batch,)          | Gives the value estimate for states
                                               | in ``x_ph``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the actor_critic
                function you provided to SAC.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target
                networks. Target networks are updated towards main networks
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to
                inverse of reward scale in the original SAC paper.)

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """

        params = locals()
        params.pop('sess')
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(params)

        tf.set_random_seed(seed)
        np.random.seed(seed)

        env, test_env = env_fn(), env_fn()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = env.action_space.high[0]

        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = env.action_space

        # Inputs to computation graph
        x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('%s/main' % name):
            mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

        # Target value network
        with tf.variable_scope('%s/target' % name):
            _, _, _, _, _, _, _, v_targ = actor_critic(x2_ph, a_ph, **ac_kwargs)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in
                           ['%s/%s' % (name, v) for v in ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main']])
        print(('\nNumber of parameters: \t pi: %d, \t' +
               'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n') % var_counts)

        # Min Double-Q:
        min_q_pi = tf.minimum(q1_pi, q2_pi)

        # Targets for Q and V regression
        q_backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * v_targ)
        v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

        # Soft actor-critic losses
        pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
        v_loss = 0.5 * tf.reduce_mean((v_backup - v) ** 2)
        value_loss = q1_loss + q2_loss + v_loss

        # Policy train op
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('%s/main/pi' % name))

        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        value_params = get_vars('%s/main/q' % name) + get_vars('%s/main/v' % name)
        with tf.control_dependencies([train_pi_op]):
            # train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)
            # Calculate gradients for Q function
            variables = get_vars('%s/main/q' % name) + [x_ph, a_ph]
            grads = tf.gradients(value_loss, variables)
            gvs = zip(grads[:-2], variables[:-2])
            train_value_op = value_optimizer.apply_gradients(gvs)

            s_a_grads = tf.concat(grads[-2:], axis=1)
            s_a_norm = tf.norm(s_a_grads, axis=1)

            pairwise_q1_dist = pairwise_distance(tf.expand_dims(q1, 1))
            pairwise_q2_dist = pairwise_distance(tf.expand_dims(q2, 1))
            pairwise_s_a_dist = pairwise_distance(tf.concat([x_ph, a_ph], axis=1))
            pairwise_q1_sa_ratio = tf.reshape(pairwise_q1_dist / (pairwise_s_a_dist + 1e-5), [-1])
            pairwise_q2_sa_ratio = tf.reshape(pairwise_q2_dist / (pairwise_s_a_dist + 1e-5), [-1])

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([train_value_op]):
            target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                      for v_main, v_targ in
                                      zip(get_vars('%s/main' % name), get_vars('%s/target' % name))])

        # All ops to call during one training step
        step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi,
                    train_pi_op, train_value_op, target_update]

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('%s/main' % name), get_vars('%s/target' % name))])

        sess.run(tf.global_variables_initializer())
        sess.run(target_init)

        # Setup model saving
        logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                              outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v})

        # parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        self.replay_buffer = replay_buffer
        self.save_freq = save_freq
        self.sess = sess
        self.start_steps = start_steps
        self.steps_per_epoch = steps_per_epoch

        # variables
        self.logger = logger
        self.env, self.test_env = env, test_env
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = x_ph, a_ph, x2_ph, r_ph, d_ph
        self.mu, self.pi, self.logp_pi = mu, pi, logp_pi
        self.q1, self.q2, self.q1_pi, self.q2_pi, v = q1, q2, q1_pi, q2_pi, v
        self.step_ops = step_ops
        self.s_a_norm = s_a_norm
        self.pairwise_q1_sa_ratio = pairwise_q1_sa_ratio
        self.pairwise_q2_sa_ratio = pairwise_q2_sa_ratio

    def get_action(self, o, deterministic=False):
        act_op = self.mu if deterministic else self.pi
        return self.sess.run(act_op, feed_dict={self.x_ph: o.reshape(1, -1)})[0]

    def test_agent(self, n=10):
        # global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = self.test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def update(self, batch, step):
        feed_dict = {self.x_ph: batch['obs1'],
                     self.x2_ph: batch['obs2'],
                     self.a_ph: batch['acts'],
                     self.r_ph: batch['rews'],
                     self.d_ph: batch['done'],
                     }

        outs = self.sess.run(self.step_ops + [self.s_a_norm, self.pairwise_q1_sa_ratio, self.pairwise_q2_sa_ratio],
                             feed_dict)
        self.logger.store(
            LossPi=outs[0],
            LossQ1=outs[1],
            LossQ2=outs[2],
            LossV=outs[3],
            Q1Vals=outs[4],
            Q2Vals=outs[5],
            VVals=outs[6],
            LogPi=outs[7],
            Norm=outs[11],
            Q1Sa=outs[12],
            Q2Sa=outs[13]
        )

    def wrap_up_epoch(self, epoch, t, start_time):
        # Save model
        if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
            self.logger.save_state({'env': self.env}, None)

        # Test the performance of the deterministic version of the agent.
        self.test_agent()

        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', t)
        self.logger.log_tabular('Q1Vals', with_min_and_max=True)
        self.logger.log_tabular('Q2Vals', with_min_and_max=True)
        self.logger.log_tabular('VVals', with_min_and_max=True)
        self.logger.log_tabular('LogPi', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ1', average_only=True)
        self.logger.log_tabular('LossQ2', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('Norm', with_min_and_max=True)
        self.logger.log_tabular('Q1Sa', with_min_and_max=True)
        self.logger.log_tabular('Q2Sa', with_min_and_max=True)
        self.logger.log_tabular('Time', time.time() - start_time)
        self.logger.dump_tabular()
