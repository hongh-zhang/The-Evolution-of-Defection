# This file contains Layer objects to be used in neural networks

import numpy as np
from pprint import pprint
from collections import namedtuple

class Layer:
    """This is a blank template for layers."""
    
    def __init__(self):
        self.type = "blank"
        self.freeze = False
        
    def reset(self):
        pass
        
    def forward(self, X, param):
        return X
    
    def backward(self, dout, param):
        return dout
    
    # backward2 allows a layer to be freezed
    def backward2(self, *args, **kwargs):
        if not self.freeze:
            return self.backward(*args, **kwargs)
        return np.nan
    
    def set_optimizer(self, param):
        if self.type in ['linear', 'maxout', 'batch_norm', 'conv1d']:
            self.optimizer = Optimizer(param)
        
    def print_parameters(self):
        print(f"Printing {self.type} layer:")
        pprint(vars(self))
        
    # function to calculate ridge cost
    # could be overwritten by layers with weights
    def sum_weights(self):
        return 0
    
    
    
class Optimizer:
    """
    Optimizer + regularizer for layers,
    provides a optimize function for the job:
        
    optimize :: (w, dw, m1, m2, param) -> (w, m1, m2)"""
    
    def __init__(self, param):
        """
        Initialize the optimizer,
           
        Parameters
        ----------
        param : dict
            hyperparameters, should contain
            
            optimizer : tuple (string, float, ...)
                name for the optimizer, one of ['adam', 'momentum', 'sgd']
                + hyperparameters for the optimizer e.g. beta1, beta2 for adam
            
            regularizer : tuple (string, float), or None
                name for regularizer, one of ['l1', 'l2'] + decay/lambda hyperparameter
                inside a tuple
            
            lr : float
                learning rate
            
            batch : float
                batch size
            
            
        """
        
        # unpack hyperparameters
        self.lr = param.get('lr', 1e-3)
        self.eps = param.get('eps', 1e-16)
        self.batch = param.get('batch', 16)
        self.optimizer = param.get('optimizer', ('adam', 0.9, 0.999)) 
        
        # set optimizer function according to name given
        # these functions will be curried with the hyperparameters above
        # to save runtime
        if self.optimizer[0].lower() == 'adam':
            try:
                beta1 = self.optimizer[1]
                beta2 = self.optimizer[2]
            except IndexError:
                beta1, beta2 = (0.9, 0.999)
                print("WARNING incomplete optimizer hyperparameter, using default values")
                
            def optimizer_inner(dw, m1, m2, param):
                t = param.get('t', 1)
                m1 = beta1 * m1 + (1 - beta1) * dw
                m2 = beta2 * m2 + (1 - beta2) * np.square(dw)
                u1 = m1 / (1 - beta1 ** t)
                u2 = m2 / (1 - beta2 ** t)
                return u1 / (np.sqrt(u2) + self.eps), m1, m2

        elif self.optimizer[0].lower() == 'momentum':
            try:
                momentum = self.optimizer[1]
            except IndexError:
                momentum = 0.9
                print("WARNING incomplete optimizer hyperparameter, using default values")
                
            def optimizer_inner(dw, m1, m2, param):
                m1 = momentum * m1 + (1 - momentum) * dw
                return m1, m1, m2

        elif self.optimizer[0].lower() == 'sgd':
            def optimizer_inner(dw, m1, m2, param):
                return dw, m1, m2
        
        else:
            raise ValueError("Invalid optimizer")
            
        # decorate regularizer to optimizers
        self.regularizer = param.get("regularizer", None)
        if self.regularizer:    
            method, decay = self.regularizer
            
            if method.lower() == 'l1':
                def optimize(w, dw, m1, m2, param):
                    dw, m1, m2 = optimizer_inner(dw, m1, m2, param)
                    return (1 - self.lr * decay) * w - self.lr * dw, m1, m2
        
            elif method.lower() == 'l2':
                def optimize(w, dw, m1, m2, param):
                    dw, m1, m2 = optimizer_inner(dw, m1, m2, param)
                    return w - self.lr * (dw - (decay/(2 * self.batch) * w)), m1, m2
            
            else:
                raise ValueError("Invalid regularizer")
        
        else:
            def optimize(w, dw, m1, m2, param):
                dw, m1, m2 = optimizer_inner(dw, m1, m2, param)
                return w - self.lr * dw, m1, m2
                
        self.optimize = optimize