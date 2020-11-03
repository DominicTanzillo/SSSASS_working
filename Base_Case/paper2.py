from jax import value_and_grad
import jax.random as random
import jax.numpy as np
import numpy
import h5py
import itertools
from jax.api import jit, grad
from jax.config import config
from jax.experimental import optimizers
from jax.experimental.optimizers import Optimizer
from . import helper_functions

layer_sizes = [] #input to output

params = init_network_params(layer_sizes, random.PRNGKey(0))

learning_rate = .01

num_steps = 100

key = random.PRNGKey(123)

def update(step, opt_state, batch,s):
    params = get_params(opt_state)
    value, grad = value_and_grad(loss)(params, batch)
    opt_state = opt_update(step, grad, opt_state)
    return value, opt_state

def step(step, opt_state, batch,s):
  value, grads = jax.value_and_grad(loss)(get_params(opt_state),batch,s)
  opt_state = opt_update(step, grads, opt_state)
  return value, opt_state