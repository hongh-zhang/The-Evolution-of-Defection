# This file contains Layer objects to be used in neural networks

import numpy as np
from network.layers.utils import *
from network.layers.layer import Layer

class Linear_layer(Layer):

    def __init__(self, input_nodes, output_nodes, bias=0):
        """
        Arguments:
        (int) input_nodes = number of input nodes,
        (int) output_nodes = number of output nodes,
        (bool) bias: enable or disable bias,
        """
        
        self.type = 'linear'
        
        # number of inputs & outputs
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.bias = bias
        
        super().__init__()
        #self.reset()
    
    def reset(self):
        """The actual init function, seperate from __init__ to allow NeuralNetworks to be re-initialized"""
        # initialize weights & bias
        # this implementation treats bias as an additional weight parameter
        # Xavier initialization
        self.weights = np.random.randn(self.input_nodes, self.output_nodes) / np.sqrt(self.input_nodes/2)
        if self.bias:
            # concat zeros to weights as additional row
            self.weights = np.concatenate((self.weights, np.ones((1,self.output_nodes))*bias), axis=0)
        
        # initialize moments
        self.m1 = np.zeros(self.weights.shape)
        self.m2 = np.zeros(self.weights.shape)
        
        self.parameters = [self.weights]
        
    
    def forward(self, X, param):
        """
        Forward inputs
        Arguments:
        (2d array) X: input, in the form of 2d numpy array #of instances * #of attributes
        """
        if self.bias:
            # concat ones to x as additional column
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        
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
    
    def __init__(self, function='sigmoid'):
        """
        An activation layer,
        available function: ['sigmoid', 'ReLU', 'LReLU', 'ELU' 'softmax', 'tanh']
        """
        super().__init__()
        self.type = 'activation'
        function = function.lower()
        self.func_name = function
        
        # define activation functions
        # func_forward is the activation function
        # func_backward is the backpropagation formulas, usually error * f'(x)
        # lambda arguments: x=inputs, e=errors
        
        if function == 'relu':
            self.func_forward = lambda x: x * (x>=0)
            self.func_backward = lambda e, x: e * (x>=0).astype(int)
            
        elif function == 'lrelu':
            self.func_forward = lambda x: np.maximum(x, 0.001*x)
            self.func_backward = lambda e, x: e * ((x<0) * 0.001 + 0.999)
            
        elif function == 'elu':
            self.func_forward = lambda x: elu(x, 1.0)
            self.func_backward = lambda e, x: e * elu_prime(x, 1.0)
        
        elif function == 'tanh':
            self.func_forward = lambda x: tanh(x)
            self.func_backward = lambda e, x: e * sech(x)
            
        elif function == 'softmax':
            self.func_forward = lambda x: stable_softmax(x)
            self.func_backward = lambda e, x: e
            # note this is a oversimplified version
            # only valid on the output layer with cross entropy loss
            
        elif function in ['sigmoid', 'logistic']:
            self.func_forward = lambda x: expit(x)
            self.func_backward = lambda e, x: e * expit(x) * (1-expit(x))

        else:
            raise ValueError('Invalid activation function')
    
    def forward(self, X, param):
        
        output = self.func_forward(X)
        
        # cache inputs & outputs for backward calculation
        self.input = X
        self.output = output
        
        return output
    
    def backward(self, error, param):
        return self.func_backward(error, self.input)
