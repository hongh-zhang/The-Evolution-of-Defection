import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class NeuralNetwork:
    """A Neural Network."""
    
    def __init__(self, layers):
        """Initialize NeuralNetwork.

        Parameters
        ----------
        layers :  list of Layer
            The structure of this NeuralNetwork

        
        Example
        -------
        A 3-layer network for MNIST dataset can be created via:
        
        nn = network.NeuralNetwork([
                    network.Linear_layer(784, 200, bias=0),
                    network.Activation_layer(function='ReLU'),

                    network.Linear_layer(200, 10, bias=0),
                    network.Activation_layer(function='softmax')
                    ])

        """

        self.layers = layers
        
        self.train_loss = []  # for storing loss value in each epoch
        self.test_loss = []
        
        self.dummy_param = {"lr": None, 'batch': None, "momentum": None, "mode": "test", 
                            "eps": None, "beta":None, "epoch": None, 'method': None, 
                            't': None, 'clip': None, 'decay': None}
    
    def forward(self, X, param):
        """Forward signal"""
        output = X  # temp output from each layer
        for layer in self.layers:
            output = layer.forward(output, param)
        return output
    
    def query(self, X, param=None):
        """Query the network, return argmax(y)
        
        Parameters
        -------------
        X: ndarray
            Inputs
        """

        if not param:
            param = self.dummy_param
        output = self.forward(X, param)
        return np.argmax(output, axis=1)
    
    def train(self, X, y, param, loss_func="mse", rand=True):
        """
        Train the ANN with given dataset
        
        Parameters
        ----------
        X :  ndarray
            Attributes of training data
            
        y :  ndarray
            Labels of training data
            
        param :  Dict
            Dictionary containing hyperparameters
            
        loss_func :  str, default='mse'
            The structure of this NeuralNetwork
            
        rand :  Bool, default=True
            Enable random shuffle or not
            
        """
        
        start = time()
        
        param["mode"] = 'train'
        param["epoch"] += 1
        batch_size = param.get("batch", 16)
        error_ls = []
        self.set_loss_func(loss_func)
        
        # get random batches then iterate
        X_split, y_split = self.split_data(X, y, batch_size, rand=rand)
        for X_batch, y_batch in zip(X_split, y_split):
            
            # forward
            yhat = self.forward(X_batch, param)
            
            # backward
            dout, batch_loss = self.loss_fn(y_batch, yhat)
            error_ls.append(batch_loss)
            
            self.backprop(dout/batch_size, param)
        
        # record & report
        avg_loss = sum(error_ls)/len(error_ls)
        self.train_loss.append((param['epoch'], float(avg_loss)))
        print(f"Average loss = {avg_loss:.6f}, elapsed time = {time()-start}.")
    
    def backprop(self, dout, param):
        param['t'] += 1
        clip = param.get('clip', 1.0)
        for layer in self.layers[::-1]:
            # gradient clipping
            magnitude = np.linalg.norm(dout)
            if magnitude > clip:
                dout = dout / magnitude * clip
            dout = layer.backward(dout, param)
    
    def validate(self, X_t, y_t, param, loss_func = "mse"):
        param["mode"] = 'test'
        yhat = self.forward(X_t, param)
        _, test_loss = self.loss_fn(y_t, yhat)
        self.test_loss.append((param['epoch'], test_loss))
    
    def set_loss_func(self, function):
        """Return a function that calculate output loss,
        loss_func :: (ytrue, yhat) -> (dout, loss)"""

        if function == "cross_entropy":
            def loss_func(ytrue, yhat):
                yhat = np.clip(yhat, 1e-16, 1. - 1e-16)  # numerical stability
                dout = -(ytrue/yhat) + (1-ytrue)/(1-yhat)
                loss = -np.sum(ytrue * np.log(yhat))
                return dout, loss
            
        elif function == "fast_cross_entropy":  # simplified cross entropy, to be used with fast_softmax
            def loss_func(ytrue, yhat):
                dout = yhat - ytrue
                yhat = np.clip(yhat, 1e-16, 1. - 1e-16)
                loss = -np.sum(ytrue * np.log(yhat))
                return dout, loss
            
        elif function == "mse":
            def loss_func(ytrue, yhat):
                dout = yhat - ytrue
                loss = np.sum(dout**2)/(dout.size)
                return dout, loss
            
        else:
            raise ValueError('Invalid loss function')
            
        self.loss_fn = loss_func
    
    # -----
    # help functions
    # ----
    
    @staticmethod
    def split_data(X, y, batch_size, rand=True):
        """Shuffle and split training data into random batches."""
        # shuffle
        if rand:
            X, y = shuffle(X, y)
        # split
        sections = X.shape[0] // batch_size
        X_split = np.array_split(X, sections, axis=0)
        y_split = np.array_split(y, sections, axis=0)
        return X_split, y_split
    
    def plot_loss(self, mode='both'):
        """Plots the train & test loss stored"""
        plt.figure(figsize=(10, 8), dpi=80)
        plt.scatter(*zip(*self.train_loss), c='orange', marker='x', label='train')
        if mode!='both':
            try:
                plt.scatter(*zip(*self.test_loss), c='chartreuse', marker='+', label='test')
            except:
                pass
        plt.yscale("log")
        plt.legend(loc='upper right')
        plt.show()
    
    def print_parameters(self):
        i = 0
        for layer in self.layers:
            print(f"--{i}--")
            layer.print_parameters()
            i += 1
    
    def __call__(self, X, param=None):
        return self.forward(X, param) if param else self.forward(X, self.dummy_param)
    
    def reset(self):
        self.train_loss = []
        self.test_loss = []
        for layer in self.layers:
            layer.reset()
        print("Network reinitialized.")