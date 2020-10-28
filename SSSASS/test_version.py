import jax
import jax.random as random
import jax.numpy as np
import numpy
import h5py
import itertools
from scipy.special import softmax
from jax.api import jit, grad
from jax.config import config
from jax.experimental import optimizers
from jax.experimental.optimizers import Optimizer

#import matplotlib.pyplot as plt
#import os

s = 10 #Define Window Size

### Param Functions

def three_W(key,s):

    key1, key2, key3 = jax.random.randint(key,(3,),1,10)

    key1 = random.PRNGKey(key1)
    key2 = random.PRNGKey(key2)
    key3 = random.PRNGKey(key3)

    WQ = generate_W(key1,s)
    WK = generate_W(key2,s)
    WV = generate_W(key3,s)

    return WQ, WK, WV

def generate_W(key,s):
    W = jax.random.normal(key,shape=(s,s))
    return W

### Math Formulae

def softmax_s(K,Q,s):

    a = (1/np.sqrt(s))*(np.transpose(K)*Q)
    v = a[np.triu_indices(a.shape[0], k = 0)]

    # Find Upper Triangular

    X = np.zeros((s, s))
    X[np.triu_indices(X.shape[0], k=0)] = v

    A = softmax(X,axis = 0)

    return A

def calculate_Z(X_prev,params):

    size = X_prev.size(-1)

    WQ = params[0]
    WK = params[1]
    WV = params[2]

    Q = WQ * X_prev
    K = WK * X_prev
    V = WV * X_prev

    A = softmax_s(K, Q, size)

    Z = V * A

    return Z

def predict(params,inputs,s):
    """
    :param params: [WQ, WK, WV] in order
    :param inputs: X_prev
    :return: Z - guess of next elements

    This is an attempt to match the formulae for a predict function.

    """

    activations = inputs
    count = 0
    for W in params:
        if count == 0:
            outputs = np.matmul(W,activations)
            activations = outputs
        elif count == 1:
            K = np.matmul(W,inputs)
            outputs = np.transpose(K)*activations

            ### Attempting to Apply Attention Metric Here

            d_k = s
                #outputs.size(-1)
            a = (1 / np.sqrt(d_k)) * outputs
            v = a[np.triu_indices(a.shape[0], k=0)]
            m=(d_k, d_k)
            X = np.zeros(m)
            jax.ops.index_update(X,np.triu_indices(X.shape[0], k=0), v)

            activations = softmax(X, axis=0)
        else:
            V = np.matmul(W,inputs)
            outputs = np.matmul(V,activations)
            activations = outputs
        count +=1

    return outputs

def loss(params, batch,s):
  inputs, targets = batch
  Z = predict(params, inputs,s)
  return np.linalg.norm((targets - Z), ord=2)


def loss_function(Y,X_prev,params,s):
    Z = calculate_Z(X_prev,params,s)
    return np.linalg.norm((Y-Z), ord=2)


### Adam Gradients

learning_rate = .01

num_steps = 100

key = random.PRNGKey(123)

init_params = three_W(key,s)

opt_init, opt_update, get_params = optimizers.adam(learning_rate)

opt_state = opt_init(init_params)

#@jit # when using Jit get an error?
def update(step, opt_state, batch,s):
    params = get_params(opt_state)
    value, grad = jax.value_and_grad(loss)(params, batch, s)
    opt_state = opt_update(step, grad, opt_state)
    return value, opt_state

def step(step, opt_state, batch,s):
  value, grads = jax.value_and_grad(loss)(get_params(opt_state),batch,s)
  opt_state = opt_update(step, grads, opt_state)
  return value, opt_state


### Get Batches
num_batches = 20

rawdata = h5py.File('ViSAPy_somatraces.h5','r')

n1 = rawdata.get('data')
n1 = np.array(n1)
n1 = np.transpose(n1)


def get_batch(source,size,index=27000):

    X = numpy.zeros(shape=(size, size + 1))

    for j in range(size + 1):
        X[:, j] = source[index:index + size, 3]
        index += 1

    X_prev = X[:, 0:size]
    X_next = X[:, 1:size + 1]

    return X_prev, X_next

batches = []

indices = numpy.random.randint(100000, size=num_batches) #pulling random sections of the data (might be overlap)

for i in range(num_batches):
    batches.append(get_batch(n1,s,indices[i]))


### Training
num_epochs = 1

key = random.PRNGKey(123)
opt_state = opt_init(init_params)
print(opt_state)
# itercount = itertools.count()
for i in range(num_batches):
    for step in range(num_steps):
        value, opt_state = update(step, opt_state, batches[i],s)
params = get_params(opt_state)

print(params)

print("Done")




#opt_init, opt_update, get_params = optimizers.adam(learning_rate)
#opt_state = opt_init(params)

# def step(step, opt_state):
#  value, grads = jax.value_and_grad(loss_function)(get_params(opt_state))
#  opt_state = opt_update(step, grads, opt_state)
#  return value, opt_state

#for step in range(num_steps):
#  value, opt_state = step(step, opt_state)

