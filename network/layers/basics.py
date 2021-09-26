# This file contains Layer objects to be used in neural networks

import numpy as np
from collections import namedtuple
from network.layers.functions import *
from network.layers.layer import Layer

class Linear_layer(Layer):

    def __init__(self, input_nodes, output_nodes, bias=0):
        """
        Arguments:
        (int) input_nodes = number of input nodes,
        (int) output_nodes = number of output nodes,
        (bool) bias: enable or disable bias,
        """
        super().__init__()
        
        self.type = 'linear'
        
        # number of inputs & outputs
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.bias = bias
        self.with_bias = True
        self.reset()
    
    def reset(self):
        """The actual init function, seperate from __init__ to allow NeuralNetworks to be re-initialized"""
        # initialize weights & bias
        # this implementation treats bias as an additional weight parameter
        # Xavier initialization
        self.weights = np.random.randn(self.input_nodes, self.output_nodes) / np.sqrt(self.input_nodes/2)
        if self.bias:
            # concat zeros to weights as additional row
            self.weights = np.concatenate((self.weights, np.ones((1,self.output_nodes))*self.bias), axis=0)
        
        # initialize moments
        self.m1 = np.zeros(self.weights.shape)
        self.m2 = np.zeros(self.weights.shape)
    
    def forward(self, X, param):
        """
        Forward inputs
        Arguments:
        (2d array) X: input, in the form of 2d numpy array #of instances * #of attributes
        """
        if self.bias:
            if self.with_bias:
                # concat ones to x as additional column
                X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
            else:
                # concat 0 instead of 1
                X = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)
        
        # calculate output
        output = np.dot(X, self.weights)
        
        # record inputs & outputs for weight update later
        self.input = X
        self.output = output
        
        return output
    
    def backward(self, dout, param):
        """
        Error backpropogation function,
        Calls self.update to update weights,
        Returns this layer's error for the preceding error to propogate
        
        Arguments:
        (2d array) dout: error from the superior layer,
        (dict) param
        """
        
        lr = param.get("lr", 1e-3)
        decay = param.get("decay", 0.01)
        
        # calculate error to pass
        if self.bias:
            dx = np.dot(dout, self.weights.T[:,:-1])  # bias is not passed
        else:
            dx = np.dot(dout, self.weights.T)
        
        # update self
        dw = np.dot(self.input.T, dout)
        dw, self.m1, self.m2 = self.optimize(self.delta(dw, self.m1, self.m2), param)
        self.weights = (1 - lr*decay) * self.weights - dw
        
        return dx
    

    
class Activation_layer(Layer):
    """An activation layer maps non-linearity to inputs."""
    
    # caches input & output for backpropagation
    Cache = namedtuple('cache', ('x', 'y'))
    
    def __init__(self, function, custom=None):
        """
        Available function: ['sigmoid', 'ReLU', 'LReLU', 'ELU' 'softmax', 'tanh'],
        also accepts custom functions, input in the kwarg custom=(func_forward, func_backward).
        """
        super().__init__()
        self.type = 'activation'
        function = function.lower()
        self.func_name = function
        
        # define activation functions
        if custom:
            self.func_forward  = custom[0]
            self.func_backward = custom[1]
            
        else:
            # functions imported from utils.py
            if function == 'relu':
                self.func_forward  = relu
                self.func_backward = relu_

            elif function == 'lrelu':
                self.func_forward  = lrelu
                self.func_backward = lrelu_

            elif function == 'elu':
                self.func_forward  = elu
                self.func_backward = elu_

            elif function == 'tanh':
                self.func_forward  = tanh
                self.func_backward = tanh_

            elif function == 'softmax':
                self.func_forward  = softmax
                self.func_backward = softmax_
                
            # simplified softmax that does not calculate partial derivative
            # to be used with fast_cross_entropy loss
            elif function == 'fast_softmax':
                self.func_forward  = softmax
                self.func_backward = lambda x, y, dy: dy

            elif function in ['sigmoid', 'logistic']:
                self.func_forward  = expit
                self.func_backward = expit_

            else:
                raise ValueError('Invalid activation function')
    
    def forward(self, X, param):
        output = self.func_forward(X)
        self.cache = Activation_layer.Cache(X, output)
        return output
    
    def backward(self, dout, param):
        return self.func_backward(self.cache.x, self.cache.y, dout)