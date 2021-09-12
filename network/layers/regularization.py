import numpy as np
from network.layers.layer import Layer

class BatchNorm_layer(Layer):
    
    def __init__(self, nodes, verbosity=0):
        self.nodes = nodes
        self.verbosity = verbosity
        self.type = 'batch_norm'
        super().__init__()
    
    def reset(self):
        # initialize mean & std
        self.mean = self.std = 0
        
        # initialize scale & shift
        self.gamma = np.expand_dims(np.ones(self.nodes), axis=0)
        self.beta = np.expand_dims(np.zeros(self.nodes), axis=0)
        
        # initialize momentum
        self.gamma1 = np.zeros(self.gamma.shape)
        self.gamma2 = np.zeros(self.gamma.shape)
        self.beta1 = np.zeros(self.beta.shape)
        self.beta2 = np.zeros(self.beta.shape)
        
        self.parameters = [self.mean, self.std, self.gamma, self.beta]
    
    def forward(self, X, param):
        
        # set parameters
        mode = param.get("mode", 'test')
        momentum = param.get("momentum", 0.9)
        
        # compute col-wise mean & std
        sample_mean = np.mean(X, axis=0)
        sample_std = np.std(X, axis=0)
        
        # update mean & std
        if mode=='train':
            self.mean = momentum * self.mean + (1 - momentum) * sample_mean
            self.std = momentum * self.std + (1 - momentum) * sample_std
        
        # normalize
        X = (X - self.mean)/self.std
        
        # apply scale & shift
        X = X * self.gamma + self.beta
        
        self.cache = (sample_mean, (1/(sample_std)), X)
        
        if self.verbosity:
            print(f"Sample: {sample_mean[0]}.")
            print(f"Self: {self.mean[0]}.")
        
        return X
    
    def backward(self, error, param):
        
        lr = param.get("lr", 1e-3)
        decay = param.get("decay", 0.01)
        
        # adapted from https://kevinzakka.github.io/2016/09/14/batch_normalization/
        N, D = error.shape
        x_mu, inv_var, x_hat = self.cache

        # intermediate partial derivatives
        dxhat = error * self.gamma

        # final partial derivatives
        dx = (1. / N) * inv_var * (N*dxhat - np.sum(dxhat, axis=0)
            - x_hat*np.sum(dxhat*x_hat, axis=0))
        dbeta = np.sum(error, axis=0)
        dgamma = np.sum(x_hat*error, axis=0)

        dgamma, self.gamma1, self.gamma2 = self.optimize(self.delta(dgamma, self.gamma1, self.gamma2), param)
        dbeta, self.beta1, self.beta2 = self.optimize(self.delta(dbeta, self.beta1, self.beta2), param)
        
        self.gamma = (1 - lr*decay) * self.gamma - dgamma
        self.beta = (1 - lr*decay) * self.beta - dbeta
        
        return dx
    
class Dropout_layer(Layer):
    def __init__(self, rate=0.5):
        self.rate = rate
        self.type = 'dropout'
        super().__init__()
    
    def forward(self, X, param):

        mode = param.get("mode", 'test')
        
        if mode == 'train':
            # create a random Bool matrix same shape as X
            # then apply dropout & scaling
            scale = (np.random.rand(X.shape[0], X.shape[1]) > self.rate) * (1/(1-self.rate))
            
            # record scale for backpropagation
            self.scale = scale
            return X * scale
        
        else:
            return X
    
    def backward(self, error, param):
        return error * self.scale