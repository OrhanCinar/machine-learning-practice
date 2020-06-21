import numpy as np
import gym

from retro_contest.local import make
from retro import make as make_retro

from baselines.common.atari_wrappers import FrameStack

import cv2
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/A2C%20with%20Sonic%20the%20Hedgehog/sonic_env.py

cv2.ocl.setUseOpenCL(false)


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        frame = fame[:, :, None]
        return frame
