# This file contains utility functions

import numpy as np
import scipy.special
from math import exp

tanh = lambda x: np.tanh(x)
sech = lambda x: 1 / np.cosh(x)
expit = lambda x: scipy.special.expit(x)
stable_softmax = lambda x: scipy.special.softmax(x, axis=1)
elu = np.vectorize(lambda x, alpha: x if x >= 0 else alpha*(exp(x)-1.0))
elu_prime = np.vectorize(lambda x, alpha: 1.0 if x >= 0 else exp(x))