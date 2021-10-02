# this file provide a nnplayer subclass for the axelrod library
# it organize states and replay memory to cooperate with DQN

import random
import numpy as np
import axelrod as axl
from time import time
from collections import deque, namedtuple

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class State():
    """State container with configurable encoding"""
    
    def __init__(self, depth, C=1, D=0, N=-1):
        self.C = C
        self.D = D
        self.N = N
        self.depth = depth
        self.reset()
        
    def reset(self):
        self.state = [deque([self.N for _ in range(self.depth)], maxlen=self.depth) for _ in range(2)]
    
    def __repr__(self):
        return str(self.state).replace("),", "),\n")
    
    def values(self):
        return np.array(self.state, ndmin=3)
    
    def push(self, *args):
        play, coplay = map(self.encode, args)
        self.state[0].append(play)
        self.state[1].append(coplay)
        return self.values()
    
    def encode(self, play):
        if play == axl.Action.C:
            return self.C
        else:
            return self.D
        
        
class NNplayer(axl.Player):
    """
    """
    
    name = 'NNplayer'
    classifier = {
        'memory_depth': -1,
        'stochastic': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }
    
    decision = (axl.Action.C, axl.Action.D)
    
    def __init__(self, network, memory, state, greedy=0.2, gamma=0.999, mode="dense", N=-1):
        super().__init__()
        
        self.network = network
        self.memory  = memory
        self.state   = state
        
        self.greedy  = greedy
        self.gamma   = gamma
        
        self.verbosity = False
        
        self.mode = 1 if mode=="dense" else 0
        self.N = -1
        self.reset()
        
    def reset(self):
        self.state.reset()
        self.reward = 0
        
    def strategy(self, opponent):
        """Make decision"""
        
        # make random choice to explore
        if random.random() < self.greedy:
            return random.choice(self.decision)
        
        # or query the network to exploit
        else:
            d = self.network.query(self.state.values())
            if self.verbosity:
                print(d)
            return self.decision[np.argmax(d)]
    
    # overwrite update_history to update self state
    # this function is automatically called by axelrod library
    def update_history(self, *args):
        self.history.append(*args)
        self.update_state(*args)
        
    def update_state(self, play, coplay):
        """update current game state & record transition into replay memory"""
        s  = self.state.values()
        s_ = self.state.push(play, coplay)
        
        # reward
        r  = axl.interaction_utils.compute_scores([(play, coplay)])[0][0]
        
        # dense reward
        if self.mode:
            r  = r if s[0,0,1]==-1 else np.NaN  # set last turn reward to NaN
            self.memory.push(s, play, s_, r)
        
        # sparse reward
        else:
            if s[0,0,1]==self.N:
                self.memory.push(s, play, s_, 0)
                self.reward += r
            else:
                self.memory.push(s, play, s_, r+self.reward)
                self.reward = 0
        
    def train(self, epoch, param):
        param['t'] = 1
        length = len(self.memory)
        for _ in range(epoch):
            # organize data
            ts = Transition(*zip(*self.memory.sample(length)))
            ss  = np.vstack(ts.state)
            ss_ = np.vstack(ts.next_state)
            ats = np.array([[True, False] if a==axl.Action.C else [False, True] for a in ts.action])
            rs  = np.array(ts.reward, ndmin=2).T
            
            # pass to network
            self.network.learn((ss, ss_, ats, rs), param, self.gamma)
        
        self.network.update_target()
        self.loss = self.network.loss
    
    def plot(self, **kwargs):
        self.network.plot(**kwargs)
        
    # test mode using "with" statement
    def __enter__(self, *args):
        self.verbosity = True
        self.temp = self.greedy
        self.greedy = 0.0
    
    def __exit__(self, *args):
        self.verbosity = False
        self.greedy = self.temp
    
    
    
# function handling training
def train(nnplayer, epoch, param):
    for _ in range(epoch):
        start = time()
        nnplayer.train(60, param)
        print(f'loss: {nnplayer.loss},            time: +{time()-start:.2f} sec')