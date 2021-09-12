gpu = True
if gpu:  # enable gpu training, requires cupy library
    import cupy as np
    expit = np.ElementwiseKernel(
        'float64 x',
        'float64 y',
        'y = 1 / (1 + exp(-x))',
        'expit')
    
    tanh = lambda x: np.tanh(x)
    sech = lambda x: 1 / np.cosh(x)
    
    # this version of softmax prevents overflow
    sub_max = lambda x: x - np.expand_dims(np.max(x, axis=1), axis=1)
    softmax = lambda x: np.exp(x) / np.array([np.sum(np.exp(x), axis=1)]).T
    stable_softmax = lambda x: softmax(sub_max(x))
    
    print("Cupy: Training on GPU.")
    
else:
    import numpy as np
    from scipy.special import expit as expit
    import scipy.special
    tanh = lambda x: np.tanh(x)
    sech = lambda x: 1 / np.cosh(x)
    stable_softmax = lambda x: scipy.special.softmax(x, axis=1)
    
    print("Numpy: Training on CPU.")

    

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from decorator import decorator

@decorator
def cupy_dec(func, *args, **kwargs):
    """
    This decorator function unsures the input is consistent with the module mode (CPU/GPU). Under GPU training, 
    any numpy array is converted into cupy array before passing into the actual functions.
    For efficiency, only functions under the NeuralNetwork class is decorated.
    """
    if gpu:
        args = map(lambda x: np.array(x) if type(x)==ndarray else x, args)
    return func(*args, **kwargs)



def update_weights(weights, param):
    """
    Optimizers for any parameter,
    processe gradient "dw" into actual weight change,
    
    (list) weights: a list in the form [delta_weight, moment1, moment2],
                    moment1 is shared between 'Momentum' & 'Adam'
    (dict) param: hyperparameters, should include learning_rate, momentum, epsilon, beta, epoch, method,
    
    Returns adjusted delta_p (value to be updated)
    """
    
    dw, m1, m2 = weights
    
    method = param.get('method', 'momentum')
    lr = param.get('lr', 1e-3)
    batch = param.get('batch', 16)
    
    momentum = param.get('momentum', 0.9)
    beta1, beta2 = param.get('beta', (0.9, 0.999))
    
    eps = param.get('eps', 1e-9)
    t = param.get('t', 1)
    
    # Adam
    # 1st moment <- momentum as usual
    # 2nd moment <- scale factor
    # unbiased moments <- step corrected moments
    # dw = lr * 1st / sqrt(2nd)
    if method.lower() == 'adam':
        m1 = beta1 * m1 + (1 - beta1) * dw
        m2 = beta2 * m2 + (1 - beta2) * np.square(dw)
        weights[1] = m1
        weights[2] = m2
        u1 = m1 / (1 - beta1 ** t)
        u2 = m2 / (1 - beta2 ** t)
        return (lr * u1 / (np.sqrt(u2) + eps)), m1, m2
    
    # Momentum
    # dw = lr * velocity
    elif method.lower() == 'momentum':
        m1 = momentum * m1 + (1-momentum) * dw
        weights[1] = m1
        return (lr * m1), m1, m2
    
    # SGD
    # dw = lr * dw
    elif method.lower() == 'sgd':
        return (lr * dw), m1, m2
    
    # Nestrov Momentum
    elif method.lower() == 'nesterov':
        pass


# In[7]:

class Linear_layer:
    """
    A linear layer class,
    Note that every array input is in the form [#instance x #attributes]
    """
    def __init__(self, input_nodes, output_nodes, bias=True):
        """
        Arguments:
        (int) input_nodes = number of input nodes,
        (int) output_nodes = number of output nodes,
        (bool) bias: enable or disable bias,
        """
        # number of inputs & outputs
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        
        # initialize weights & bias
        # this implementation treats bias as an additional weight parameter
        # Xavier initialization
        self.weights = np.random.randn(self.input_nodes, self.output_nodes) / np.sqrt(self.input_nodes/2)
        self.bias = bias
        if self.bias:
            # concat zeros to weights as additional row
            self.weights = np.concatenate((self.weights, np.zeros((1,self.output_nodes))), axis=0)
        
        # initialize moments
        self.m1 = np.zeros(self.weights.shape)
        self.m2 = np.zeros(self.weights.shape)
        
        self.type = 'linear'
    
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
        dw, self.m1, self.m2 = update_weights([dw, self.m1, self.m2], param)
        self.weights = (1 - lr*decay) * self.weights - dw
        
        return dx
    
    # -----
    # debugging tools
    # ----
    def print_parameters(self):
        print("Printing linear layer")
        print(f"Max = {np.max(self.weights)}")
        print(f"FCL weights = {self.weights}")
        print(f"FCL momentum = {self.m1, self.m2}")
        
    def save_weights(self, folder='models', prefix='Untitled'):
        file_name = f"{folder}/{prefix}_layer_{self.input_nodes}x{self.output_nodes}.npy"
        np.save(file_name, self.weights)
        print(f"Model saved to {file_name}")
    
    def load_weights(self, file):
        self.weights = np.load(file)


# In[8]:


class Activation_layer:
    def __init__(self, function='sigmoid'):
        """Available function: ['sigmoid', 'ReLU']"""
        
        # initialize activation function
        # func_forward is the activation function
        # func_backward is the backpropagation formulas, usually error * f'(x)
        # lambda arguments: x=inputs, e=errors
        
        if function == 'ReLU':
            self.func_forward = lambda x: x * (x>=0)
            self.func_backward = lambda e, x: e * (x>=0).astype(int)
        
        elif function == 'tanh':
            self.func_forward = lambda x: tanh(x)
            self.func_backward = lambda e, x: e * sech(x)
            
        elif function == 'softmax':
            self.func_forward = lambda x: stable_softmax(x)
            self.func_backward = lambda e, x: e
            # note this is a oversimplified version
            # only valid on the output layer with cross entropy loss
            
        elif function == 'sigmoid':
            self.func_forward = lambda x: expit(x)
            self.func_backward = lambda e, x: e * expit(x) * (1-expit(x))

        else:
            raise ValueError('Invalid activation function')
        
        self.func_name = function
        self.type = 'activation'
    
    def forward(self, X, param):
        """Apply activation function & forward"""
        
        output = self.func_forward(X)
        
        # record inputs & outputs for backward calculation
        self.input = X
        self.output = output
        return output
    
    def backward(self, error, param):
        return self.func_backward(error, self.input)


# In[9]:


class Dropout_layer:
    def __init__(self, rate=0.5):
        self.rate = rate
        self.type = 'dropout'
    
    def forward(self, X, param):

        mode = param.get("mode", 'test')
        
        if mode == 'train':
            # create a random matrix same shape as X
            # then apply dropout & scaling
            scale = (np.random.rand(X.shape[0], X.shape[1]) > self.rate) * (1/(1-self.rate))
            
            # record scale for backpropagation
            self.scale = scale
            return X * scale
        
        else:
            return X
    
    def backward(self, error, param):
        return error * self.scale


# In[10]:


class BatchNorm_layer:
    
    def __init__(self, output_nodes, verbosity=0):
        
        # initialize scale & shift
        self.gamma = np.expand_dims(np.ones(output_nodes), axis=0) # [[1, 1, ...]]
        self.beta = np.expand_dims(np.zeros(output_nodes), axis=0) # [[0, 0, ...]]
        
        # initialize momentum
        self.gamma1 = np.zeros(self.gamma.shape)
        self.gamma2 = np.zeros(self.gamma.shape)
        self.beta1 = np.zeros(self.beta.shape)
        self.beta2 = np.zeros(self.beta.shape)
        
        # initialize mean & std
        self.mean = self.std = 0
        
        # debug stuff
        self.verbosity = verbosity
        self.type = 'batch_norm'
    
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
        
#         if np.isnan(np.sum(dx)):
#             print(f"dx min max = {np.min(dx), np.max(dx)}")
#             print(f"dx = {dx}")
#             print(f"dxhat = {dxhat}")
#             print(f"x_mu, inv_var, x_hat = {self.cache}")
#             assert False, "NaN in Batch Norm dx!"
        
#         if np.isnan(np.sum(dbeta)):
#             print(f"dbeta min max = {np.min(dbeta), np.max(dbeta)}")
#             print(f"dbeta = {dbeta}")
#             assert False, "NaN in Batch Norm dbeta!"
            
#         if np.isnan(np.sum(dgamma)):
#             print(f"dgamma min max = {np.min(dgamma), np.max(dgamma)}")
#             print(f"dgamma = {dgamma}")
#             assert False, "NaN in Batch Norm dgamma!"

        dgamma, self.gamma1, self.gamma2 = update_weights([dgamma, self.gamma1, self.gamma2], param)
        dbeta, self.beta1, self.beta2 = update_weights([dbeta, self.beta1, self.beta2], param)
        
        self.gamma = (1 - lr*decay) * self.gamma - dgamma
        self.beta = (1 - lr*decay) * self.beta - dbeta
        
        return dx
    
    def print_parameters(self):
        print("Printing Batch Norm layer")
        print(f"Max gamma = {np.max(self.gamma)}")
        print(f"gamma: {self.gamma}")
        print(f"Max beta = {np.max(self.beta)}")
        print(f"beta: {self.beta}")


# # Neural network class

# In[37]:


# neural network class definition
class NeuralNetwork:
    """
    A simple ANN class,
    Note that every array input is in the form [#instance x #attributes]
    """
    def __init__(self, layers):
        # define structure
        self.layers = layers
        
        self.train_loss = []  # for storing loss value in each epoch
        self.test_loss = []
        
        self.dummy_param = {"lr": None, 'batch': None, "momentum": None, "mode": "test", "eps": None, "beta":None, 
         "epoch": None, 'method': None, 't': None, 'clip': None, 'decay': None}
    
    def __call__(self, X):
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        return self.query(X, self.dummy_param)
    
    def forward(self, X, param):
        """
        Forward signals,
        Arguments:
        (numpy 2d array) X: Inputs
        """
        output = X  # temp output from each layer
        for layer in self.layers:
            output = layer.forward(output, param)
        return output
    
    @cupy_dec
    def train(self, X, y, param, loss_func="mse"):
        """
        Train the ANN,
        Arguments:
        (numpy 2d array) X: Inputs,
        (numpy 2d array) y: True labels,
        (dict) param: Dictionary of hyperparameters,
        (int) batch_size,
        (str) loss_func: ["mse", "cross_entropy"]
        """
        param["mode"] = 'train'
        param["epoch"] += 1
        clip = param.get("clip", 1.0)
        batch_size = param.get("batch", 32)
        error_ls = []
        
        # get random batches then iterate
        X_split, y_split = self.split_data(X, y, batch_size)
        for X_batch, y_batch in zip(X_split, y_split):
            
            # forward
            yhat = self.forward(X_batch, param)
            dout, batch_loss = self.calculate_loss(y_batch, yhat, function=loss_func)
            error_ls.append(batch_loss)
            
            # backward
            self.backprop(dout, param)
        
        # record & report
        avg_loss = sum(error_ls)/len(error_ls)
        self.train_loss.append((param['epoch'], float(avg_loss)))
        print(f"Average loss = {avg_loss:.6f}.")
    
    @cupy_dec
    def backprop(self, dout, param):
        param['t'] += 1
        clip = param.get('clip', 1.0)
        for layer in self.layers[::-1]:
            # gradient clipping
            magnitude = np.linalg.norm(dout)
            if magnitude > clip:
                dout = dout / magnitude * clip
            
            dout = layer.backward(dout, param)
                   
    @cupy_dec
    def query(self, X, mode='classification'):
        """
        Query the neural network with input X,
        If mode is set to 'classification',
        Returns the index with highest conditional probability
        """
        output = self.forward(X, self.dummy_param)
        if mode == 'classification':
            output = np.argmax(output, axis=1)
        return output
    
    @cupy_dec
    def validate(self, X_t, y_t, param, loss_func = "cross_entropy"):
        param["mode"] = 'test'
        yhat = self.forward(X_t, param)
        _, test_loss = self.calculate_loss(y_t, yhat, function=loss_func)
        self.test_loss.append((param['epoch'], float(test_loss)))
    
    # -----
    # help functions
    # ----
    @cupy_dec
    def calculate_loss(self, y_true, yhat, function="cross_entropy"):
        assert function in ["mse", "cross_entropy"], "Invalid loss function!"
        errors = yhat - y_true
        if function == "cross_entropy":
            yhat = np.clip(yhat, 1e-12, 1. - 1e-12)  # avoid zero
            loss = -np.sum(np.log(yhat+1e-10) * y_true) / yhat.shape[0]
        elif function == "mse":
            erros = loss = np.sum(errors**2)/(errors.size)
        return errors, loss
    
    def split_data(self, X, y, batch_size):
        """
        Shuffle and split training data into random batches
        """
        # shuffle
        X, y = shuffle(X, y)
        # split
        sections = X.shape[0] // batch_size
        X_split = np.array_split(X, sections, axis=0)
        y_split = np.array_split(y, sections, axis=0)
        return X_split, y_split
    
    def plot_loss(self, mode='both'):
        """Plots the train & test loss stored"""
        figure(figsize=(10, 8), dpi=80)
        plt.scatter(*zip(*self.train_loss), c='orange', marker='x', label='train')
        if mode!='both':
            plt.scatter(*zip(*self.test_loss), c='chartreuse', marker='+', label='test')
        plt.yscale("log")
        plt.legend(loc='upper right')
        plt.show()
    
    # -----
    # debugging tools
    # ----
    def print_parameters(self):
        i = 0
        for layer in self.layers:
            print(f"--{i}--")
            if layer.type in ['linear', 'batch_norm']:
                layer.print_parameters()
            i += 1