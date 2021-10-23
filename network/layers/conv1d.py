# file for 1d convolutional layers, contains Conv1d_layer & Flatten_layer
# very poorly optimized... didn't get to use this in the actual project

import numpy as np
from network.layers.layer import Layer
from network.layers.functions import *

class Conv1d_layer(Layer):
    
    def __init__(self, in_channels, out_channels, in_width, kernel_width, stride=1, bias=0):
        
        self.type = 'conv1d'
        self.ic = in_channels  # height
        self.oc = out_channels  # number of kernels
        self.iw = in_width
        self.kw = kernel_width  # width
        self.ow = int((self.iw - self.kw) / stride + 1)
        self.bias = bias
        self.stride = 1
        self.reset()
        
    def reset(self):
        self.kernels = np.random.randn(self.oc, self.ic, self.kw) / np.sqrt(self.oc/2)  # what is the best initialization for kernels?
        # oc is used here since it's analogous to input_nodes in linear layers (i.e., the number of units in a layer)
        
        self.m1 = np.zeros(self.kernels.shape)
        self.m2 = np.zeros(self.kernels.shape)
        
        self.bias = np.ones((self.oc, self.ow)) * self.bias
        self.db1 = np.zeros(self.bias.shape)
        self.db2 = np.zeros(self.bias.shape)
    
    def forward(self, X, param):
        self.inputs = X
        return correlate3(X, self.kernels) + self.bias
    
    def backward(self, douts, param):
        
        assert douts[0].shape == (self.oc, self.ow)
        
        lr = param.get('lr', 1e-5)
        clip = param.get('clip', 1.0)       
        
        # update bias
        db = np.sum(douts, axis=0)
        
        # calculate dx to pass to next layer
        dxs = []
        for dy in douts:
            dx = [[convolve1(d, k) for k in ks] for d, ks in zip(dy, self.kernels)]
            dx = np.sum(np.array(dx), axis=0)
            dxs.append(dx)
        
        # calculate dw to update
        dws = []
        for x, dout in zip(self.inputs, douts):
            for error in dout:  # each error come from one kernel
                dws.append([convolve1(row, error) for row in x])
                
        dws = np.array(dws)
        num = len(self.kernels)  # number of kernels
        
        # sum all dw to each kernel
        dks = np.array([np.sum(dws[i::num, :, :], axis=0) for i in range(num)])
        
        # update
        self.kernels, self.m1, self.m2  = self.optimizer.optimize(self.kernels, dks, self.m1, self.m2, param)
        self.bias, self.db1, self.db2 = self.optimizer.optimize(self.bias, db, self.db1, self.db2, param)
        
        return np.array(dxs)
        

class Flatten_layer(Layer):
    
    def __init__(self):
        self.type = "flatten"
    
    def forward(self, X, param):
        self.shape = X.shape
        return np.reshape(X, (self.shape[0],-1))
    
    def backward(self, dout, param):
        return dout.reshape(self.shape)