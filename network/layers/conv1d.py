# file for 1d convolutional layers, contains Conv1d_layer & Flatten_layer

import numpy as np
from network.layers.layer import Layer
from network.layers.functions import *

class Conv1d_layer(Layer):
    
    def __init__(self, in_channels, out_channels, kernel_wid):
        
        self.type = 'conv1d'
        self.ic = in_channels  # height
        self.oc = out_channels  # number of kernels
        self.kw = kernel_wid  # width
        self.reset()
        
    def reset(self):
        self.kernels = np.random.randn(self.oc, self.ic, self.kw) / np.sqrt(self.oc/2)  # what is the best initialization for kernels?
        # oc is used here since it's analogous to input_nodes in linear layers (i.e., the number of units in a layer)
    
    def forward(self, X, param):
        
        self.inputs = X
        return correlate3(X, self.kernels)
    
    def backward(self, douts, param):
        
        lr = param.get('lr', 1e-5)
        clip = param.get('clip', 1.0)       
        
        # calculate dx to pass to next layer
        dxs = []
        for dy in douts:
            dx = [[convolve1(d, row) for row in k] for d, k in zip(dy, self.kernels)]
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
        dks = []
        for i in range(num):
            dks.append(np.sum(dws[i::num, :, :], axis=0))
        
        # update
        for i in range(num):
            dk = dks[i] * lr
            magnitude = np.linalg.norm(dk)
            if magnitude > clip:
                dk = dk / magnitude * clip
            self.kernels[i] -= dk
        
        return dxs
        

class Flatten_layer(Layer):
    
    def __init__(self):
        self.type = "flatten"
    
    def forward(self, X, param):
        self.shape = X.shape
        return np.reshape(X, (self.shape[0],-1))
    
    def backward(self, dout, param):
        return dout.reshape(self.shape)