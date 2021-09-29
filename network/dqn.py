# deep q learner

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from network import NeuralNetwork
from collections import namedtuple

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class DQN():
    
    def __init__(self, layers):
        
        # define networks
        self.policy_net = NeuralNetwork(layers)
        self.target_net = deepcopy(self.policy_net)
        self.loss = np.zeros(self.policy_net.layers[-1].output_nodes)
        self.loss_ls = []
    
    def query(self, state):
        """make decision from given state"""
        d = self.policy_net(state)
        return d
    
    def update_target(self):
        self.target_net = deepcopy(self.policy_net)
        
    def learn(self, data, param, gamma):
        """train one epoch on the given ReplayMemory"""
        
        batch_size = param['batch']
        sections = len(data[0]) // batch_size
        
        param['epoch'] += 1
        param['mode'] = 'train'
        self.policy_net.set_loss_func('mse')

        # split training data into batches
        ss, ss_, ats, rs = map(lambda x: np.array_split(x, sections), data)
        
        # train
        for s, s_, at, r in zip(ss, ss_, ats, rs):
            
            # estimated value of current state
            Q_values = self.policy_net(s, param=param) * at
            
            # estimated value of next state
            Q_values_ = np.max(self.target_net(s_), axis=1, keepdims=True)
            
            # expected value of current state = E(next) + reward
            E_values = gamma*Q_values_ + r
            
            # hard code the value of last state to 0.0
            np.nan_to_num(E_values, copy=False, nan=0.0)
            
            # feedback
            loss, _ = self.policy_net.loss_fn(E_values, Q_values)
            loss = loss * at  # relocate loss to action taken
            self.policy_net.backprop(loss, param)
            
            # track training loss
            loss = np.sum(np.abs(loss),axis=0) / np.clip(np.sum(loss!=0, axis=0), 1, None)
            self.loss = 0.9*self.loss + 0.1*loss
        self.loss_ls.append((param['epoch'], self.loss))
        
    def plot(self, min_ran=0, max_ran=-1):
        
        max_ran = max_ran if max_ran!=-1 else len(self.loss_ls)
        
        plt.figure(figsize=(10, 8), dpi=80)
        plt.scatter(*zip(*[(i[0], i[1][0]) for i in self.loss_ls[min_ran:max_ran]]), c='chartreuse', marker='x', label='Cooperation')
        plt.scatter(*zip(*[(i[0], i[1][1]) for i in self.loss_ls[min_ran:max_ran]]), c='orange', marker='+', label='Defection')
        #plt.yscale("log")
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='upper right')
        plt.show()