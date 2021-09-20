# This file contains utility functions

import numpy as np
from math import exp
from scipy.special import expit
from scipy.special import softmax as softmaxxx


# --- ACTIVATION FUNCTIONS ---

# activation functions are defined in pairs
# f is the forward function
# f_ is the derivative, multiplied by dy (output error) to backprop error

# efficiency: vectorize < built-in < basic operators
# but do use vectorize to avoid calling exponential on entire array


# sigmoid
expit_ = lambda x, y, dy:  dy * y * (1-y)


# tanh
tanh = lambda x: np.tanh(x)
tanh_ = lambda x, y, dy: dy / np.cosh(x)


# softmax
# oversimplified version of backward function
# only works with cross entropy loss function
softmax  = lambda x: softmaxxx(x, axis=1)
#softmax_ = lambda x, y, dy: dy
softmax_ = lambda x, y, dy: dy * y * (1-y)


# ReLU
relu  = lambda x: x * (x>=0)
relu_ = lambda x, y, dy: dy * (x>=0)


# leaky ReLU
lrelu  = lambda x, alpha=0.001: np.maximum(x, alpha*x)
lrelu_ = lambda x, y, dy, alpha=0.001: dy * ((x<0)*alpha + 1-alpha)


# ELU
def single_elu(x, alpha=1.0):
    return x if x>=0 else alpha*(exp(x)-1.0)

def single_elu_(x, y, dy, alpha=1.0):
    return dy if y>0 else exp(x)*dy

elu  = np.vectorize(single_elu)
elu_ = np.vectorize(single_elu_)



# --- CONVOLUTIONS ---

def correlate1(X, K):
    """1d convolution"""
    xw = len(X)
    kw = len(K)
    ow = len(X) - len(K) + 1
    
    if ow >= 1:
        return np.fromiter((np.sum(X[i:i+kw] * K) for i in range(ow)), dtype=float)
    else:
        return correlate1(K, X)

def convolve1(X, K):
    return correlate1(X, np.flip(K))


# corr2 is seperated from corr1 for efficiency
def correlate2(X, K):
    """1d convolution with height (2d)"""
    
    xw = X.shape[1]
    kw = K.shape[1]
    ow = xw - kw + 1
    
    if ow >= 1:
        return [np.sum(X[:, i:i+kw] * K) for i in range(ow)]
    else:
        return correlate2(K, X)  # correlation is communitive

def convolve2(X, K):
    return correlate2(X, np.flip(K, axis=1))


def correlate3(XS, KS):
    """1d convolution with batches & multiple kernels (3d)"""
    xshape = XS.shape
    kshape = KS.shape
    
    assert (len(xshape)==3 and len(kshape)==3), "3d input required"
    assert xshape[1] == kshape[1], "kernels not spanning input height"
    
    return np.array([[correlate2(x, k) for k in KS] for x in XS])

def convolve3(XS, KS):
    return correlate3(XS, np.flip(KS, axis=1))