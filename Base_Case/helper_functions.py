import jax.random as random
import jax.numpy as np
import numpy
import h5py
import itertools
from jax.api import jit, grad
from jax.config import config
from jax.experimental import optimizers
from jax.experimental.optimizers import Optimizer

# Generate Randomness

def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]



def loss():
    return