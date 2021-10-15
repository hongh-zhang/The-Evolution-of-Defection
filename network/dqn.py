# deep q learner

import gc
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.utils import shuffle
from collections import namedtuple


from network import NeuralNetwork

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class DQN():
    
    def __init__(self, layers, memory, gamma=0.9, greedy=0.1):
        """Initialize DQN,
        
        Parameters
        ----------
        layers : list of Layer object
            architecture of the network
        
        memory : ReplayMemory object
            memory to record experience (see network/replay.py)
        
        (optional) gamma : float
            discount rate of future reward
        
        (optional) greedy : float
            epsilon-greedy, controls exploring behaviour

        """
        
        # define networks
        self.memory = memory
        self.policy_net = NeuralNetwork(layers)
        self.target_net = deepcopy(self.policy_net)
        self.loss = np.zeros(self.policy_net.layers[-1].output_nodes)
        self.loss_ls = []
        self.epoch = 0
        
        self.gamma = gamma
        self.greedy = greedy
        self.verbosity = False
    
    def push(self, transition):
        """Push one transition into replay memory"""
        self.memory.push(*transition)
        
    def query(self, state):
        """Calculate Q value for each action, or make a random choice
        
        Parameters
        ----------
        state: ndarray
            numeric representation of state, to be input to the network
            
        """
        
        # make random choice to explore
        if random.random() < self.greedy:
            if self.verbosity:
                print('randomed')
            return random.choice([0, 1])  # Hardcoded, fix this later
        
        # or query the network to exploit
        else:
            qvalues = self.policy_net(state)
            if self.verbosity:
                print(qvalues)
            return np.argmax(qvalues)
    
    def __call__(self, state):
        return self.policy_net(state)
    
    def update_target(self):
        self.target_net = deepcopy(self.policy_net)
        gc.collect()
        
    def plot(self, min_ran=0, max_ran=-1, log=False):
        """Plot training loss,
        
        Parameters
        ----------
        (optional) min_ran : Int
            lower bound of the plot
        
        (optional) max_ran : Int
            higher bound of the plot
        
        (optional) log : Bool
            enable log scale for y axis or not
        
        """
        max_ran = max_ran if max_ran!=-1 else len(self.loss_ls)
        
        plt.figure(figsize=(10, 8), dpi=80)
        plt.scatter(*zip(*[(i[0], i[1][0]) for i in self.loss_ls[min_ran:max_ran]]), c='chartreuse', marker='x', label='Cooperation')
        plt.scatter(*zip(*[(i[0], i[1][1]) for i in self.loss_ls[min_ran:max_ran]]), c='orange', marker='+', label='Defection')
        if log:
            plt.yscale("log")
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='upper right')
        plt.show()
        
    def reset(self):
        """Re-initlize the networks"""
        self.policy_net.reset()
        self.update_target()
        
    def test_mode(self, on):
        """Enter/exit test mode, which eliminate random behaviour and print out estimated Q-values"""
        if on:
            self.verbosity = True
            self.temp = self.greedy
            self.greedy = 0
        else:
            self.verbosity = False
            self.greedy = self.temp
            
         
        
    def learn(self, data, param):
        """Train one epoch on the given ReplayMemory
        
        Parameters
        ----------
        data:
        
        param: dict
            hyperparameters
        
        gamma: float
            discount rate of future rewards
        
        """
        
        batch_size = param['batch']
        sections = len(data[0]) // batch_size
        
        self.epoch += 1
        param['epoch'] += 1
        param['mode'] = 'train'
        self.policy_net.set_up(param)

        # split training data into batches
        ss, ss_, ats, rs = map(lambda x: np.array_split(x, sections), data)
        
        # train
        for s, s_, at, r in zip(ss, ss_, ats, rs):
            
            # estimate value of current state
            Q_values = self.policy_net(s, param=param) * at
            
            # estimate value of next state
            Q_values_ = np.max(self.target_net(s_), axis=1, keepdims=True)
            
            # expected value of current state = discounted E(next) + reward
            E_values = self.gamma*Q_values_ + r
            
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
        
        
    def train(self, epochs, param):
        param['t'] = 1
        
        # organize data
        ts = Transition(*zip(*self.memory.values()))
        ss  = np.vstack(ts.state)
        ss_ = np.vstack(ts.next_state)
        ats = np.array(ts.action, ndmin=2)
        rs  = np.array(ts.reward, ndmin=2).T
        del ts
        gc.collect()
        
        #print(ss[:10], ss_[:10], ats[:10], rs[:10])
        
        for _ in range(epochs):
            
            ss, ss_, ats, rs = shuffle(ss, ss_, ats, rs)
            
            # pass to network
            self.learn((ss, ss_, ats, rs), param)
        
        self.update_target()