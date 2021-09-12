import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

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
        
        self.dummy_param = {"lr": None, 'batch': None, "momentum": None, "mode": "test", 
                            "eps": None, "beta":None, "epoch": None, 'method': None, 
                            't': None, 'clip': None, 'decay': None}
    
    def __call__(self, X):
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        return self.query(X)
    
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
    
    def train(self, X, y, param, loss_func="mse", rand=True):
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
        X_split, y_split = self.split_data(X, y, batch_size, rand=rand)
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
    
    def backprop(self, dout, param):
        param['t'] += 1
        clip = param.get('clip', 1.0)
        for layer in self.layers[::-1]:
            # gradient clipping
            magnitude = np.linalg.norm(dout)
            if magnitude > clip:
                dout = dout / magnitude * clip
            
            dout = layer.backward(dout, param)
                   
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
    
    def validate(self, X_t, y_t, param, loss_func = "cross_entropy"):
        param["mode"] = 'test'
        yhat = self.forward(X_t, param)
        _, test_loss = self.calculate_loss(y_t, yhat, function=loss_func)
        self.test_loss.append((param['epoch'], float(test_loss)))
    
    # -----
    # help functions
    # ----
    def calculate_loss(self, y_true, yhat, function="cross_entropy"):
        assert function in ["mse", "cross_entropy"], "Invalid loss function!"
        errors = yhat - y_true
        if function == "cross_entropy":
            yhat = np.clip(yhat, 1e-12, 1. - 1e-12)  # avoid zero
            loss = -np.sum(np.log(yhat+1e-10) * y_true) / yhat.shape[0]
        elif function == "mse":
            erros = loss = np.sum(errors**2)/(errors.size)
        return errors, loss
    
    def split_data(self, X, y, batch_size, rand=True):
        """
        Shuffle and split training data into random batches
        """
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
            plt.scatter(*zip(*self.test_loss), c='chartreuse', marker='+', label='test')
        plt.yscale("log")
        plt.legend(loc='upper right')
        plt.show()
    
    def print_parameters(self):
        i = 0
        for layer in self.layers:
            print(f"--{i}--")
            layer.print_parameters()
            i += 1
            
    def reset(self):
        self.train_loss = []
        self.test_loss = []
        for layer in self.layers:
            layer.reset()
        print("Network reinitialized.")