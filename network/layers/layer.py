# This file contains Layer objects to be used in neural networks

import numpy as np
from pprint import pprint
from collections import namedtuple

class Layer:
    """This is a blank template for layers,
       provides basic functions and an optimizer."""

    delta = namedtuple('delta',
                      ('dw', 'm1', 'm2'))
    
    def __init__(self):
        self.type = "blank"
        
    def reset(self):
        pass
        
    def forward(self, X, param):
        return X
    
    def backward(self, dout, param):
        return dout
    
    @staticmethod
    def optimize(delta, param):
        """
        Optimizers for any parameter,
        processe gradient "dw" into actual weight change,

        (delta) delta: a delta (namedtuple defined above) containing information to update weights
                         moment1 is shared between 'Momentum' & 'Adam'
        (dict) param: hyperparameters, should include learning_rate, momentum, epsilon, beta, epoch, method,

        Returns adjusted delta_w (value to be updated)
        """

        optimizer = param.get('optimizer', 'momentum')
        lr = param.get('lr', 1e-3)
        batch = param.get('batch', 16)

        momentum = param.get('momentum', 0.9)
        beta1, beta2 = param.get('beta', (0.9, 0.999))

        eps = param.get('eps', 1e-16)
        t = param.get('t', 1)

        # Adam
        # 1st moment <- momentum as usual
        # 2nd moment <- scale factor
        # unbiased moments <- step corrected moments
        # dw = lr * 1st / sqrt(2nd)
        if optimizer.lower() == 'adam':
            m1 = beta1 * delta.m1 + (1 - beta1) * delta.dw
            m2 = beta2 * delta.m2 + (1 - beta2) * np.square(delta.dw)
            u1 = m1 / (1 - beta1 ** t)
            u2 = m2 / (1 - beta2 ** t)
            return (lr * u1 / (np.sqrt(u2) + eps)), m1, m2

        # Momentum
        # dw = lr * velocity
        elif optimizer.lower() == 'momentum':
            m1 = momentum * delta.m1 + (1-momentum) * delta.dw
            return (lr * m1), m1, delta.m2

        # SGD
        # dw = lr * dw
        elif optimizer.lower() == 'sgd':
            return (lr * delta.dw), delta.m1, delta.m2
        
    def print_parameters(self):
        print(f"Printing {self.type} layer:")
        pprint(vars(self))