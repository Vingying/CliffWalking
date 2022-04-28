import os
import time

import numpy as np
import gym

env = gym.make("CliffWalking-v0")
n_observation = env.observation_space.n
n_action = env.action_space.n


