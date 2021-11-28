# maxout layer

import numpy as np
from network.layers.layer import Layer


class Maxout_layer(Layer):
    """
    A maxout2-1 layer,
    has 2 sets of weight & bias for non-linearity,
    output = max(w1X+b1, w2X+b2)
    """
    

    def __init__(self, input_nodes, output_nodes, bias=0):
        super().__init__()
        self.type = 'maxout'
        
        # number of inputs & outputs
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.bias = bias
        self.reset()
    
    def reset(self):
        # initialize weights & bias
        self.w1 = np.random.randn(self.input_nodes, self.output_nodes) / np.sqrt(self.input_nodes/2)
        self.w2 = np.random.randn(self.input_nodes, self.output_nodes) / np.sqrt(self.input_nodes/2)
        
        b1 = np.ones((1,self.output_nodes)) * self.bias
        b2 = np.ones((1,self.output_nodes)) * self.bias
        
        self.w1 = np.concatenate((self.w1, b1), axis=0)
        self.w2 = np.concatenate((self.w2, b2), axis=0)
        
        # initialize moments
        self.m11 = np.zeros(self.w1.shape)
        self.m12 = np.zeros(self.w1.shape)
        
        self.m21 = np.zeros(self.w2.shape)
        self.m22 = np.zeros(self.w2.shape)
        
        self.parameters = [self.w1, self.w2]
        
    
    def forward(self, X, param):
        # add bias
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        
        y1 = np.dot(X, self.w1)
        y2 = np.dot(X, self.w2)
        pos1 = (y1>y2)
        pos2 = np.logical_not(pos1)
        out = y1 * pos1 + y2 * pos2
        
        # record inputs & outputs for weight update later
        self.input = X
        self.pos1 = pos1
        self.pos2 = pos2
        
        return out
    
    def backward(self, dout, param):
        
        dout1 = dout * self.pos1
        dout2 = dout * self.pos2
        
        # calculate error to pass
        # by multiplying by pos each entry will either be 0 or dx
        dx1 = np.dot(dout1, self.w1.T[:,:-1])
        dx2 = np.dot(dout2, self.w2.T[:,:-1])
        
        # combine the 2 dx
        dx = dx1 + dx2
        
        # update
        dw1 = np.dot(self.input.T, dout1)
        dw2 = np.dot(self.input.T, dout2)

        self.w1, self.m11, self.m12 = self.optimizer.optimize(self.w1, dw1, self.m11, self.m12, param)
        self.w2, self.m21, self.m22 = self.optimizer.optimize(self.w2, dw2, self.m21, self.m22, param)
        
        return dx
    
    def reset_moments(self):
        self.m11 = self.m12 = self.m21 = self.m22 = np.zeros(self.w1.shape)
        
    def get_weights(self):
        return self.w1, self.w2