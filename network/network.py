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
                            't': None, 'clip': None, 'decay': None, "loss_fn":"mse"}
        
        # new param should look like this
        # but i'm not sure if changing this will break saved models
#         param = {"lr": None, 'batch': None, "mode": "test", "eps": 1e-16, "epoch": None, 't': None, 'clip': None,
#          'optimizer': None, 'regularizer': None, "loss_fn":"mse"}
    
    def forward(self, X, param):
        """Forward signal"""
        output = X  # temp output from each layer
        for layer in self.layers:
            output = layer.forward(output, param)
        return output
    
    def query(self, X, param=None):
        """Query for classification,
        
        Parameters
        -------------
        X: ndarray
            Inputs
        """

        if not param:
            param = self.dummy_param
        output = self.forward(X, param)
        return np.argmax(output, axis=1)
    
    def __call__(self, X, param=None):
        """Query for regression"""
        return self.forward(X, param) if param else self.forward(X, self.dummy_param)
    
    def train(self, X, y, param, rand=True):
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
        error_ls = []
        
        param["epoch"] += 1
        param["mode"] = 'train'
        batch_size = param.get("batch", 16)
        
        self.set_up(param)
        
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
        print(f"Average loss = {avg_loss:.6f}, elapsed time = {time()-start:.2f}.")
    
    def backprop(self, dout, param):
        """Backpropagate errors,
        
        Parameters
        ----------
        dout : ndarray
            output error, produced by loss_fn
            
        param : dict
            hyperparameters
        """
        
        param['t'] += 1
        clip = param.get('clip', 1.0)
        for layer in self.layers[::-1]:
            # gradient clipping
            magnitude = np.linalg.norm(dout)
            if magnitude > clip:
                dout = dout / magnitude * clip
            # pass to layer and compute the error to the next layer
            dout = layer.backward2(dout, param)
    
    def validate(self, X_t, y_t, param):
        """Test performaance against validation set"""
        param["mode"] = 'test'
        yhat = self.forward(X_t, param)
        _, test_loss = self.loss_fn(y_t, yhat)
        self.test_loss.append((param['epoch'], test_loss))
    
    
    def set_up(self, param):
        """Set up optimizer and loss function."""
        self.set_optimizer(param)
        self.set_loss_func(param)
    
    def set_optimizer(self, param):
        """Set optimizer for each layer, specified in param dict"""
        for l in self.layers:
            l.set_optimizer(param)
            
    def set_loss_func(self, param):
        """Return a function that calculate output loss,
        loss_fn :: (ytrue, yhat) -> (dout, loss)"""
        
        function = param.get("loss_fn", "mse").lower()
        
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
            
#         # add cost for l2 regularization
#         if regularizer:
#             if regularizer[0].lower() == 'l2':
#                 decay = regularizer[1]
#                 batch = param.get("batch", 16)
#                 def loss_fn(ytrue, yhat):
#                     dout, loss = loss_func_inner(ytrue, yhat)
#                     ridge_cost = self.ridge_cost() * decay / (2 * batch)
#                     return (dout + ridge_cost), loss
#                 self.loss_fn = loss_fn

        self.loss_fn = loss_func
    
    # additional cost for L2 regularization
    def ridge_cost(self):
        return sum([l.sum_weights() for l in self.layers])
    
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
        """Print out parameters of the layers"""
        i = 0
        for layer in self.layers:
            print(f"--{i}--")
            layer.print_parameters()
            i += 1
    
    def reset(self):
        """Re-initialize the parameters"""
        self.train_loss = []
        self.test_loss = []
        for layer in self.layers:
            layer.reset()
        print("Network reinitialized.")
    
    def reset_moments(self):
        """Reset moments for adam & momentum optimizers"""
        for l in self.layers:
            try:
                l.reset_moments()
            except AttributeError:
                pass