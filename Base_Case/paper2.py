from jax import value_and_grad
import jax.random as random
import jax.numpy as np
import numpy
import h5py
import scipy
import itertools
from jax.api import jit, grad
from jax.config import config
from jax.experimental import optimizers
from jax.experimental.optimizers import Optimizer
from . import helper_functions

layer_sizes = [] #input to output



def update(step, opt_state, batch,s):
    params = get_params(opt_state)
    value, grad = value_and_grad(loss)(params, batch)
    opt_state = opt_update(step, grad, opt_state)
    return value, opt_state

def step(step, opt_state, batch,s):
  value, grads = jax.value_and_grad(loss)(get_params(opt_state),batch,s)
  opt_state = opt_update(step, grads, opt_state)
  return value, opt_state

### Convolusion

l = 1 #length of templates, each template has squared norm of 1

Data = [1] #W by

templates = []

params = init_network_params(layer_sizes, random.PRNGKey(0))

learning_rate = .01

num_steps = 100

key = random.PRNGKey(123)

### Step 0: Initialize c spike "templates":



### Step 1: Get Data

def update_Y(Y,current_idy,WinSize,n1):
    Y = np.concatenate((Y[-500:],n1[current_idy:current_idy+WinSize,3]))
    current_idy += WinSize
    return Y, current_idy

### Step 2: Find Spikes

def find_peaks(Y):
    return scipy.signal.find_peaks(Y, -10)

### Step 3: Minimize X

def loss(Y,X,B) =

### Step 4: Reconvolve and Save End



# Update Dictionary

if __name__ == '__main__':

    #Initialization Params

    WinSize = 3200 # .1 second

    rawdata = h5py.File('ViSAPy_somatraces.h5','r')
    n1 = rawdata.get('data')
    n1 = np.array(n1)
    n1 = np.transpose(n1)

    current_idy = 42000
    Y = n1[1500:current_idy + WinSize,3] # first Y window


    ## Will Need to Update in Future
    B = find_peaks(Y)

    ## Currently Based on the Fact that First Window Finds Two Peaks, This will not work othewise

    template1 = Y[B[0][0] - 6:B[0][0] + 20]
    template2 = Y[B[0][1] - 6:B[0][1] + 20]

    Dict = [template1, template2]

    for i in range(5):
        Y, current_idy = update_Y(Y,current_idy,WinSize,n1)

