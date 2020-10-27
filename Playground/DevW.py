import numpy as np
from scipy.special import softmax
np.set_printoptions(precision=10)

def three_W(s):
    WQ = generate_W(s)
    WK = generate_W(s)
    WV = generate_W(s)

    return WQ, WK, WV

def generate_W(s):
    W = np.random.normal(0,1,size=(s,s))
    return W

def softmax_s(K,Q,s):

    a = (1/np.sqrt(s))*(np.transpose(K)*Q)
    v = a[np.triu_indices(a.shape[0], k = 0)]

    #upper Triangular

    X = np.zeros((s, s))
    X[np.triu_indices(X.shape[0], k=0)] = v

    A = softmax(X,axis = 0)

    return A

def calculate_Z(s,X_prev,WQ,WK,WV):

    Q = WQ * X_prev
    K = WK * X_prev
    V = WV * X_prev

    A = DevW.softmax_s(K, Q, s)

    Z = V * A

    return Z

def loss_func(Z,X_next,s):

    dif = Z - X_next

    loss = 0

    for i in range(s):
        loss += np.dot(dif[:, 0], dif[:, 0])

    return loss

if __name__ == '__main__':
    pass