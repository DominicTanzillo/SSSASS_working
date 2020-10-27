import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
import h5py
import matplotlib.pyplot as plt
from jaxlib.experimental import optimizers
from jax.experimental.optimizers import Optimizer
import os
from scipy.special import softmax




s = 10

def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

#

layer_sizes = [s, s, s, s]
param_scale = 0.1
step_size = 0.01
num_epochs = 10
batch_size = 128
n_targets = 10
params = init_network_params(layer_sizes, random.PRNGKey(0))



opt = optimizers.adam(learning_rate)
opt_state = opt.init(params)

def predict(params, x):
    count = 0
    for w in params:
        if count == 0:
            outputs = np.matmul(w,x)
        if count == 1:
            d_K = size
            a = np.transpose(np.matmul(w,x))
            np.matmul(a,outputs)
        outputs = np.matmul(w, activations) + b


        final_w, final_b = params[-1]
        logits = jnp.dot(final_w, activations) + final_b
        return logits - logsumexp(logits)



def loss_func(params) =

def step(step, opt_state):
  value, grads = jax.value_and_grad(loss_func)(opt.get_params(opt_state))
  opt_state = opt.update(step, grads, opt_state)
  return value, opt_state

for step in range(num_steps):
  value, opt_state = step(step, opt_state)


# Doesn't work with a batch
random_flattened_images = random.normal(random.PRNGKey(1), (10, 28 * 28))

batched_predict = vmap(predict, in_axes=(None, 0))

batched_preds = batched_predict(params, random_flattened_images)


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)


@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]


###

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = np.matmul(np.transpose(key),query) \
             / np.sqrt(d_k)

    # Upper Triangular Mask

    v = scores[np.triu_indices(scores.shape[0], k=0)]

    # upper Triangular

    X = np.zeros((d_k, d_k))
    X[np.triu_indices(X.shape[0], k=0)] = v

    p_attn = softmax(X, axis=0)

    # if mask is not None:
        # scores = np.ma.filled(mask == 0, -1e9)
    #p_attn = scores.softmax(scores, dim=-1)
    #if dropout is not None:
        # p_attn = dropout(p_attn)

    return np.matmul(value,p_attn), p_attn

### Training