import copy
import pylab
import numpy as np
from environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

EPISODES = 2500
