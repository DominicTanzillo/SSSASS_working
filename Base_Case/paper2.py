import jax.random as random
import jax.numpy as np
import numpy
import h5py
import itertools
from jax.api import jit, grad
from jax.config import config
from jax.experimental import optimizers
from jax.experimental.optimizers import Optimizer

import helper_functions

layer_sizes = [] #input to output

params = init_network_params(layer_sizes, random.PRNGKey(0))

learning_rate = .01

num_steps = 100

key = random.PRNGKey(123)