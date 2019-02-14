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
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
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


def run_multiple(algorithms, replay_buffer, env_fn, batch_size=100, epochs=100, max_ep_len=1000, start_steps=10000,
                 steps_per_epoch=5000, sample_from=tuple([])):
    start_time = time.time()
    total_steps = steps_per_epoch * epochs

    envs = SubprocVecEnv([env_fn for _ in algorithms])
    obs = envs.reset()
    steps = [[obs[i], None, 0, False, 0, 0] for i in range(len(algorithms))]  # o, o2, r, d, ep_ret, ep_len

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        actions = []
        for i in range(len(algorithms)):
            o = steps[i][0]

            if t > start_steps:
                a = algorithms[i].get_action(o)
            else:
                a = algorithms[i].env.action_space.sample()

            actions.append(a)

        o2_s, r_s, d_s, _ = envs.step(actions)

        for i in range(len(algorithms)):
            o, _, _, _, ep_ret, ep_len = steps[i]
            o2 = o2_s[i]
            r = r_s[i]

            ep_ret += r_s[i]
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == max_ep_len else d_s[i]

            # Store experiences to replay buffer
            if len(sample_from) == 0 or i in sample_from:
                replay_buffer.store(o, actions[i], r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # Update steps
            steps[i] = [o, o2, r, d, ep_ret, ep_len]

        done = any(step[3] for step in steps)
        reached_max_ep_len = any(step[5] == max_ep_len for step in steps)

        if done or reached_max_ep_len:
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(max(step[5] for step in steps)):
                batch = replay_buffer.sample_batch(batch_size)

                for algorithm in algorithms:
                    algorithm.update(batch, j)

            for i in range(len(algorithms)):
                algorithm = algorithms[i]
                _, _, _, _, ep_ret, ep_len = steps[i]
                algorithm.logger.store(EpRet=ep_ret, EpLen=ep_len)

            obs = envs.reset()
            steps = [[obs[i], None, 0, False, 0, 0] for i in range(len(algorithms))]

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            for algorithm in algorithms:
                algorithm.wrap_up_epoch(epoch, t, start_time)


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
    parser.add_argument('--exp_name', type=str, default='multiple')
    parser.add_argument('--sample_from', type=str, default='', help='0-based index of algorithm to sample action from')
    args = parser.parse_args()

    session = tf.Session()
    env = gym.make(args.env)
    rb = ReplayBuffer(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        size=int(1e6)
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

    sample_from = tuple([int(ind) for ind in args.sample_from.split(',')
                         if len(ind) > 0 and int(ind) < len(all_algorithms)])

    run_multiple(all_algorithms, rb, lambda: gym.make(args.env), sample_from=sample_from, epochs=args.epochs)
