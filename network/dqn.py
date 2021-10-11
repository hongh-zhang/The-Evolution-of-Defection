# deep q learner

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from network import NeuralNetwork
from collections import namedtuple

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class DQN():
    
    def __init__(self, layers, greedy=0.1):
        
        # define networks
        self.policy_net = NeuralNetwork(layers)
        self.target_net = deepcopy(self.policy_net)
        self.loss = np.zeros(self.policy_net.layers[-1].output_nodes)
        self.loss_ls = []
        self.epoch = 0
        
        self.greedy = greedy
        self.verbosity = False
    
    def query(self, state):
        """Calculate Q value for each action, or make aa random choice"""
        
        # make random choice to explore
        if random.random() < self.greedy:
            if self.verbosity:
                print('randomed')
            return random.choice(self.decision)
        
        # or query the network to exploit
        else:
            qvalues = self.policy_net(state)
            if self.verbosity:
                print(qvalues)
            return np.argmax(qvalues)
    
    def update_target(self):
        self.target_net = deepcopy(self.policy_net)
        
    def learn(self, data, param, gamma):
        """train one epoch on the given ReplayMemory"""
        
        batch_size = param['batch']
        sections = len(data[0]) // batch_size
        
        param['epoch'] += 1
        self.epoch += 1
        param['mode'] = 'train'
        self.policy_net.set_loss_func('mse')

        # split training data into batches
        ss, ss_, ats, rs = map(lambda x: np.array_split(x, sections), data)
        
        # train
        for s, s_, at, r in zip(ss, ss_, ats, rs):
            
            # estimate value of current state
            Q_values = self.policy_net(s, param=param) * at
            
            # estimate value of next state
            Q_values_ = np.max(self.target_net(s_), axis=1, keepdims=True)
            
            # expected value of current state = discounted E(next) + reward
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
        self.loss_ls.append((self.epoch, self.loss))
        
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
        
    def reset(self):
        self.policy_net.reset()
        self.update_target()
        
    def test_mode(self, on):
        if on:
            self.verbosity = True
            self.temp = self.greedy
            self.greedy = 0
        else:
            self.verbosity = False
            self.greedy = self.temp