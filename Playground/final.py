import DevW
import jax.numpy as np
import h5py
import matplotlib.pyplot as plt
from jaxlib.experimental import optimizers
from jax.experimental.optimizers import Optimizer
import os

rawdata = h5py.File('ViSAPy_somatraces.h5','r')

n1 = rawdata.get('data')
n1 = np.array(n1)

# Plot Check

plt.plot(n1[3,28000:29000])
plt.show()

# Specify Size of Window

n1 = np.transpose(n1)

s = 10

## Step 1: Initialize WQ, WK, WV

[WQ, WK, WV] = DevW.three_W(s)

## Step 2: Receive input Signal

index = 27900

X = np.zeros(shape=(s,s+1))

for i in range(s+1):
    X[:,i] = n1[index:index+s,3]
    index+=1

X_prev = X[:,0:s]
X_next = X[:,1:s+1]

## Step 3: Calculate Attention Weight Signal

Z = calculate_Z(s,X_prev,WQ,WK,WV)

#print(Z)

## Step 5: Get w voltage samples x_next



## Step 6: Take gradient and optimize with Adam



# Messing Around

learning_rate = 0.001

opt = optimizers.adam(learning_rate)
opt_state = opt.init(params)

def step(step, opt_state):
  value, grads = jax.value_and_grad(loss_func)(opt.get_params(opt_state))
  opt_state = opt.update(step, grads, opt_state)
  return value, opt_state

for step in range(num_steps):
  value, opt_state = step(step, opt_state)


#print(n1)
#print(n1.shape)
#print(rawdata)

#print(rawdata.keys())