import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
import h5py
import matplotlib.pyplot as plt
from jaxlib.experimental import optimizers
from jax.experimental.optimizers import Optimizer
import os
from scipy.special import softmax

learning_rate = .0001

s = 10 #size of layers

layer_sizes = [s, s, s, s]

# Parameter Generating Functions

def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

#Loss Function
def loss_fn(x,y,params):
    return 0

#

opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(params)

def step(step, opt_state):
  value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state))
  opt_state = opt_update(step, grads, opt_state)
  return value, opt_state

for step in range(num_steps):
  value, opt_state = step(step, opt_state)