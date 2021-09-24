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

def correlate1(X, K, stride=1):
    """1d correlation"""
    
    ow = len(X) - len(K)
    assert ow % stride == 0, "Invalid stride"
    
    if ow >= 0:
        return np.correlate(X, K, mode='valid')[::stride]
    else:
        return np.correlate(K, X, mode='valid')[::stride]   

def convolve1(X, K, stride=1):
    return correlate1(X, np.flip(K), stride=stride)



def correlate2(xs, ks, stride=1):
    """1d correlation with height (2d)"""

    ow = xs.shape[1] - ks.shape[1]
    assert ow % stride == 0, "Invalid stride"
    
    if ow < 0:
        return correlate2(ks, xs, stride=stride)
    
    # divide into 1d conrrelation
    out = np.array([np.correlate(x, k, mode='valid')[::stride] for x, k in zip(xs, ks)])
    
    # sum into 1d output
    return np.sum(out, axis=0)
    
def convolve2(xs, ks, stride=1):
    return correlate2(xs, np.flip(ks, axis=1), stride=stride)



def correlate3(xss, kss, stride=1):
    """1d correlation with height & batch (3d)"""
    
    xshape = xss.shape
    kshape = kss.shape
    
    assert (len(xshape)==3 and len(kshape)==3), "3d input required"
    assert xshape[1] == kshape[1], "kernels not spanning input height"
    
    # divide into single input and single kernel and pass to correlate2
    return np.array([[correlate2(xs, ks, stride=stride) for ks in kss] for xs in xss])

def convolve3(xss, kss, stride=1):
    return correlate3(xss, np.flip(xss, axis=2), stride=stride)