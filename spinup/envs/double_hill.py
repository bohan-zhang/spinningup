from gym import spaces
import numpy as np
import gym


class DoubleHillEnv(gym.Env):
    def __init__(self):
        self.min_position = -1.0
        self.max_position = 1.0

        self.action_space = spaces.Box(self.min_position, self.max_position, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(np.array([0]), np.array([0]), dtype=np.float32)

    def step(self, action):
        x = action[0]

        # calculate reward based on step-wise defined function
        if -1.0 <= x < 0.0:
            reward = x + 1
        elif 0.0 <= x < 1.0:
            reward = 5 - 5 * x
        else:
            reward = 0.0

        return np.array([0]), reward, True, {}

    def reset(self):
        return np.array([0])

    def render(self, mode='human'):
        pass
