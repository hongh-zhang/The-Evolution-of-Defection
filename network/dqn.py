# deep q learner

import numpy as np
from copy import deepcopy
from network import NeuralNetwork
from collections import namedtuple

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class DQN():
    
    def __init__(self, layers):
        
        # define networks
        self.policy_net = NeuralNetwork(layers)
        self.target_net = deepcopy(self.policy_net)
        self.loss = None
    
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
            
            # value of current state
            Q_values = self.policy_net(s, param=param) * at
            
            # value of next state
            Q_values_ = np.max(self.target_net(s_), axis=1, keepdims=True)
            
            # expected value of current state
            E_values = gamma*Q_values_ + r
            
            # hard code the value of last state to 0.0
            np.nan_to_num(E_values, copy=False, nan=0.0)
            
            # feedback
            loss, _ = self.policy_net.loss_fn(E_values, Q_values)
            loss = loss * at  # relocate loss to action taken
            self.policy_net.backprop(loss, param)
            
            # track training loss
            if not self.loss:
                self.loss = np.mean(np.max(np.abs(loss),axis=1))
            else:
                self.loss = 0.9*self.loss + 0.1*np.mean(np.max(np.abs(loss),axis=1))