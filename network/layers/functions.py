# This file contains utility functions

import numpy as np
from math import exp
from scipy.special import expit
from scipy.special import softmax as softmaxxx

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