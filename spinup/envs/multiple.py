import numpy as np
import tensorflow as tf
from itertools import chain
import time
import sys
from spinup.algos.sac import core as sac_core
from spinup.algos.ddpg import core as ddpg_core
from spinup.algos.td3 import core as td3_core
from spinup.algos.ood.sac import SAC
from spinup.algos.ood.ddpg import DDPG
from spinup.algos.ood.td3 import TD3
from spinup.envs.double_hill import DoubleHillEnv
from spinup.envs.double_zig_zag import DoubleZigZag


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def run_multiple(algorithms, sample_from, replay_buffer, batch_size=100, epochs=100, max_ep_len=1000, start_steps=1000,
                 steps_per_epoch=200):
    start_time = time.time()
    interactions = 0
    steps_per_epoch //= len(sample_from)
    total_steps = steps_per_epoch * epochs
    steps = [[a.env.reset(), None, 0, False, 0, 0] for a in algorithms]  # o, o2, r, d, ep_ret, ep_len

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        for i in sample_from:
            algorithm = algorithms[i]
            o, o2, r, d, ep_ret, ep_len = steps[i]

            if t > start_steps:
                a = algorithm.get_action(o)
            else:
                a = algorithm.env.action_space.sample()

            # Step the env
            o2, r, d, _ = algorithm.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == max_ep_len else d

            # Store experiences to replay buffer
            replay_buffer.store(o, a, r, o2, d)
            interactions += 1

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            if d or ep_len == max_ep_len:
                algorithm.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = algorithm.env.reset(), 0, False, 0, 0

            # Update steps
            steps[i] = [o, o2, r, d, ep_ret, ep_len]

        batch = replay_buffer.sample_batch(batch_size)

        algorithm = algorithms[0]
        feed_dict = {algorithm.x_ph: batch['obs1'],
                     algorithm.x2_ph: batch['obs2'],
                     algorithm.a_ph: batch['acts'],
                     algorithm.r_ph: batch['rews'],
                     algorithm.d_ph: batch['done']
                     }
        update_ops, callbacks = zip(*[a.get_batch_update_ops(t) for a in algorithms])

        outs = algorithm.sess.run(list(chain.from_iterable(update_ops)), feed_dict)

        lens = np.cumsum([0] + [len(o) for o in update_ops])
        for i, (start, end) in enumerate(zip(lens[:-1], lens[1:])):
            callbacks[i](outs[start:end])

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            for algorithm in algorithms:
                algorithm.wrap_up_epoch(epoch, interactions, start_time)
                sys.stdout.flush()


def make_env(env_name):
    if env_name == 'DoubleHill':
        return DoubleHillEnv()
    elif env_name == 'DoubleZigZag':
        return DoubleZigZag()
    return DoubleHillEnv()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('algorithms', type=str)
    parser.add_argument('--env', type=str, default='DoubleHill')
    parser.add_argument('--hid', type=int, default=100)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--buffer_size', type=int, default=1e6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--exp_name', type=str, default='multiple')
    parser.add_argument('--sample_from', type=str, default='', help='0-based index of algorithm to sample action from')
    parser.add_argument('--spectral_norm', type=float, default=0.0)
    parser.add_argument('--regularizer', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--no_gpu', type=bool, default=False)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--max_ep_len', type=int, default=1)
    parser.set_defaults(spectral_norm=False)
    args = parser.parse_args()

    if args.no_gpu:
        session = tf.Session()
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
    env = make_env(args.env)
    rb = ReplayBuffer(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        size=int(args.buffer_size)
    )

    from spinup.utils.run_utils import setup_logger_kwargs

    all_algorithms = []
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    phs = sac_core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    if args.activation == 'relu':
        act = tf.nn.relu
    elif args.activation == 'tanh':
        act = tf.nn.tanh
    else:
        raise NotImplementedError

    for k, algo in enumerate(args.algorithms.split(',')):
        algorithm_name = '%s-%s-%s' % (args.exp_name, args.env, algo)
        logger_kwargs = setup_logger_kwargs(algorithm_name, args.seed)

        if algo == 'sac':
            all_algorithms.append(
                SAC(session, rb, lambda: make_env(args.env), actor_critic=sac_core.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l, activation=act), gamma=args.gamma, seed=args.seed,
                    epochs=args.epochs, logger_kwargs=logger_kwargs, name=algorithm_name, phs=phs,
                    max_ep_len=args.max_ep_len)
            )
        elif algo == 'sac_alpha':
            all_algorithms.append(
                SAC(session, rb, lambda: make_env(args.env), actor_critic=sac_core.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l, activation=act), gamma=args.gamma, seed=args.seed,
                    epochs=args.epochs, logger_kwargs=logger_kwargs, name=algorithm_name, alpha=args.alpha, phs=phs,
                    max_ep_len=args.max_ep_len)
            )
        elif algo == 'ddpg':
            all_algorithms.append(
                DDPG(session, rb, lambda: make_env(args.env), actor_critic=ddpg_core.mlp_actor_critic,
                     ac_kwargs=dict(hidden_sizes=[args.hid] * args.l, activation=act, sn=args.spectral_norm,
                                    reg=args.regularizer), gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                     logger_kwargs=logger_kwargs, name=algorithm_name, phs=phs, max_ep_len=args.max_ep_len)
            )
        elif algo == 'td3':
            all_algorithms.append(
                TD3(session, rb, lambda: make_env(args.env), actor_critic=td3_core.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma, seed=args.seed,
                    epochs=args.epochs, logger_kwargs=logger_kwargs, name=algorithm_name, phs=phs,
                    max_ep_len=args.max_ep_len)
            )

    sf = tuple(
        [int(ind) for ind in args.sample_from.split(',') if len(ind) > 0 and int(ind) < len(all_algorithms)]) if len(
        args.sample_from) > 0 else tuple(range(len(all_algorithms)))

    run_multiple(all_algorithms, sf, rb, epochs=args.epochs, batch_size=args.batch_size, max_ep_len=args.max_ep_len)
