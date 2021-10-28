# deep q learner
# design adapted from
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gc
import random
import numpy as np
from copy import deepcopy
from pickle import dump, load
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from collections import namedtuple, deque

from network import NeuralNetwork

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))



class ReplayMemory():
    """A memory for recording Transition tuples, use a queue (deque) to
    control memory depth and throw away outdated records."""
    
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Push single transition into memory, 
        arguments will be automatically cast into Transition tuple
        
        Parameters
        ----------
        state : State object
            numeric representation of current state (see axl_utils/nnplayer.py)
        
        action : axl.Action
            action taken
        
        next_state : State object
            numeric representation of the next state
        
        reward : Num
            immediate reward from the environment

        """
        self.memory.append(Transition(*args))

    def sample(self, n):
        """Randomly select n transitions"""
        return random.sample(self.memory, n)
    
    def values(self):
        """Return all values in the ReplayMemory, in list"""
        return list(self.memory)

    def __len__(self):
        return len(self.memory)
    
    def __repr__(self):
        if len(self) >= 100:
            out = list(self.memory)[:100]
        else:
            out = self.memory
        return str(out).replace("), ", "),\n")
    
    def save(self, path):
        with open(path, "wb") as file:
            dump(self, file)
    
    def load(self, path, mode='overwrite'):
        with open(path, "rb") as file:
            if mode == 'overwrite':
                self = load(file)
            elif mode == 'add':
                for i in load(file).memory:
                    self.memory.append(i)




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
        """Update the target_net by deepcopying policy_net"""
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
        plt.scatter(*zip(*[(i[0], i[1][0]) for i in self.loss_ls[min_ran:max_ran]]), c='tab:blue', marker='x', label='Cooperation')
        plt.scatter(*zip(*[(i[0], i[1][1]) for i in self.loss_ls[min_ran:max_ran]]), c='tab:orange', marker='+', label='Defection')
        if log:
            plt.yscale("log")
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='upper right')
        plt.show()
        
    def reset(self):
        """Re-initliaze the networks (randomize the weights etc.)"""
        self.policy_net.reset()
        self.update_target()
        
    def test_mode(self, on):
        """Enter/exit test mode, which eliminate random behaviour and print out estimated Q-values.
        This is called when I use 'with' statement in the notebook."""
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
            
            # (1)
            # estimate value of current state
            Q_values = self.policy_net(s, param=param) * at
            
            # (2)
            # estimate value of next state
            Q_values_ = np.max(self.target_net(s_), axis=1, keepdims=True)
            
            # (3)
            # (the regression target)
            # expected value of current state = discounted E(next) + immediate reward
            E_values = self.gamma*Q_values_ + r
            
            # hard code the value of last state to 0.0
            np.nan_to_num(E_values, copy=False, nan=0.0)

            # (4)
            # feedback
            # backpropagate the error between (3) - (1)
            loss, _ = self.policy_net.loss_fn(E_values, Q_values)
            loss = loss * at  # relocate loss to the action taken, only
            self.policy_net.backprop(loss, param)
            
            # track training loss
            loss = np.sum(np.abs(loss),axis=0) / np.clip(np.sum(loss!=0, axis=0), 1, None)
            self.loss = 0.9*self.loss + 0.1*loss
        
        self.loss_ls.append((self.epoch, self.loss))
        
        
    def train(self, epochs, param, loss_targ=None):
        """
        Organize raw data from ReplayMemory then pass them to the learn function, 
        repeatedly for {epoch} iterations,
        after the learning finished, update target network to finish this cycle of training.
        
        Parameters
        ----------
        epochs : int
            number of epochs to train
        
        param : dict
            hyperparameters
        
        (optional) loss_targ : float
            minimum percentage in loss before terminate one iteration
            * this is a disaster don't use it
            * I'm looking into statistical tests to correctly implement auto-termination
        """
        
        # reset adam optimizer by
        # 1. reset the adagrad counter
        # 2. clear the momentum from last iteration
        param['t'] = 1
        self.policy_net.reset_moments()
        
        # queue for recoding running loss, for auto termination
        temp_loss = deque([], maxlen=4)
        
        # organize data
        ts = Transition(*zip(*self.memory.values()))
        ss  = np.vstack(ts.state)
        ss_ = np.vstack(ts.next_state)
        ats = np.array(ts.action, ndmin=2)
        rs  = np.array(ts.reward, ndmin=2).T
        del ts
        gc.collect()
        
        for i in range(epochs):
            
            ss, ss_, ats, rs = shuffle(ss, ss_, ats, rs)
            
            # pass to learn
            self.learn((ss, ss_, ats, rs), param)
            
            # terminate training loss change is small enough
            if loss_targ and (len(temp_loss)==4):
                
                # calculate lower/higher bound by taking the average of the last 4 loss
                low = np.array(temp_loss).mean(axis=0) * (1-loss_targ)
                high = np.array(temp_loss).mean(axis=0) * (1+loss_targ)
                              
                if np.all(self.loss <= high) and np.all(self.loss >= low) and i > (epochs/3):
                    # print(f"terminated at {i} epochs")
                    break
            temp_loss.append(self.loss)
        
        self.update_target()
        gc.collect()
