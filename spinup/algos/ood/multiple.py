import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.sac import core as sac_core
from spinup.algos.ddpg import core as ddpg_core
from spinup.algos.td3 import core as td3_core
from spinup.algos.ood.sac import SAC
from spinup.algos.ood.ddpg import DDPG
from spinup.algos.ood.td3 import TD3


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


def run_multiple(algorithms, sample_from, replay_buffer, batch_size=100, epochs=100, max_ep_len=1000, start_steps=10000,
                 steps_per_epoch=5000):
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
                for j in range(ep_len):
                    batch = replay_buffer.sample_batch(batch_size)

                    for alg in algorithms:
                        alg.update(batch, j)

                for alg in algorithms:
                    alg.logger.store(EpRet=ep_ret, EpLen=ep_len)

                o, r, d, ep_ret, ep_len = algorithm.env.reset(), 0, False, 0, 0

            # Update steps
            steps[i] = [o, o2, r, d, ep_ret, ep_len]

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            for algorithm in algorithms:
                algorithm.wrap_up_epoch(epoch, interactions, start_time)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('algorithms', type=str)
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=1e6)
    parser.add_argument('--exp_name', type=str, default='multiple')
    parser.add_argument('--sample_from', type=str, default='', help='0-based index of algorithm to sample action from')
    args = parser.parse_args()

    session = tf.Session()
    env = gym.make(args.env)
    rb = ReplayBuffer(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        size=int(args.buffer_size)
    )

    from spinup.utils.run_utils import setup_logger_kwargs

    all_algorithms = []
    for k, algo in enumerate(args.algorithms.split(',')):
        algorithm_name = '%s-%s-%d-%s' % (args.exp_name, args.env, k, algo)
        logger_kwargs = setup_logger_kwargs(algorithm_name, args.seed)

        if algo == 'sac':
            all_algorithms.append(
                SAC(session, rb, lambda: gym.make(args.env), actor_critic=sac_core.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                    gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                    logger_kwargs=logger_kwargs, name=algorithm_name)
            )
        elif algo == 'sac_zero_alpha':
            all_algorithms.append(
                SAC(session, rb, lambda: gym.make(args.env), actor_critic=sac_core.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                    gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                    logger_kwargs=logger_kwargs, name=algorithm_name, alpha=0.0)
            )
        elif algo == 'ddpg':
            all_algorithms.append(
                DDPG(session, rb, lambda: gym.make(args.env), actor_critic=ddpg_core.mlp_actor_critic,
                     ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                     gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                     logger_kwargs=logger_kwargs, name=algorithm_name)
            )
        elif algo == 'td3':
            all_algorithms.append(
                TD3(session, rb, lambda: gym.make(args.env), actor_critic=td3_core.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                    gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                    logger_kwargs=logger_kwargs, name=algorithm_name)
            )

    sf = tuple(
        [int(ind) for ind in args.sample_from.split(',') if len(ind) > 0 and int(ind) < len(all_algorithms)]) if len(
        args.sample_from) > 0 else tuple(range(len(all_algorithms)))

    run_multiple(all_algorithms, sf, rb, epochs=args.epochs)
