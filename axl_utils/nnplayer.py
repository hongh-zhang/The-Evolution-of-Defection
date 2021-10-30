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
    """
    State container with configurable encoding
    will represent state in 2d arrays
    """
    
    def __init__(self, depth, C=1, D=0.1, N=-1):
        """
        Parameters
        ----------
        depth : Int
            deque length (which should be # of turns in the game)
            
        (optional) C : Num
            encoding for cooperation
        
        (optional) D : Num
            encoding for defection
            
        (optional) N : Num
            encoding for not-yet-happened turns
            
        """
        self.C = C
        self.D = D
        self.N = N
        self.depth = depth
        self.reset()
        
    def reset(self):
        """Clear memory for new game"""
        self.state = [deque([self.N for _ in range(self.depth)], maxlen=self.depth) for _ in range(2)]
    
    def __repr__(self):
        return str(self.state).replace("),", "),\n")
    
    def values(self):
        """Return the state, in 3d array (with only 1 2d element)"""
        return np.array(self.state, ndmin=3)
    
    def push(self, *args):
        """Push interaction into record"""
        play, coplay = map(self.encode, args)
        self.state[0].append(play)
        self.state[1].append(coplay)
        return self.values()
    
    def encode(self, play):
        """Encode axl.Action object into numerical representation"""
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
    
    def __init__(self, network, state, greedy=0.2, mode="dense", N=-1, learner="off", name='DQN'):
        super().__init__()
        
        self.name = name
        self.state   = state
        self.network = network
        
        self.mode = 1 if mode=="dense" else 0      # dense reward = 1, sparse reward = 0
        self.learner = 1 if learner=="off" else 0  # off-policy learner = 1, on-policy = 0
        self.N = -1                                # how not-yet-happened turn is encoded
        self.transitions = []
        self.reset()
    
    def __str__(self):
        return self.name
    
    # the following 3 functions overload the orginal implementation in axelrod library
    # they are automatically called by axl during each game
    def reset(self):
        """Reset the states to start a new game"""
        self.reward = 0
        self.state.reset()
        self._history = axl.history.History()
        
        # push buffer into memory then reset it
        if self.learner:
            self.push_transitions()
        self.transitions = []
        
    def strategy(self, opponent):
        """Query the network (each turn) to make decision"""
        idx = self.network.query(self.state.values())
        return self.decision[idx]
    
    # overwrite update_history to update our state
    def update_history(self, *args):
        self.history.append(*args)
        self.update_state(*args)
    # --------------------------------------------------------------------------------
        
    def update_state(self, play, coplay):
        """Update current game state & record transition into replay memory"""
        s  = self.state.values()
        s_ = self.state.push(play, coplay)
        
        # compute reward
        r  = axl.interaction_utils.compute_scores([(play, coplay)])[0][0]
        
        # rewrite action
        action = [True, False] if play==axl.Action.C else [False, True]
        
        # dense reward
        if self.mode:
            r  = r if s[0,0,1]==-1 else np.NaN  # set last turn reward to NaN
            transition = (s, action, s_, r)
        
        # sparse reward
        else:
            if s[0,0,1] == self.N:  # not last turn
                transition = (s, action, s_, 0)
                self.reward += r
            else:                 # last turn
                transition = (s, action, s_, r+self.reward)
                self.reward = 0
        
        # buffer the transition for on-policy learners (DQN)
        self.transitions.append(transition)
        
        # push into the replay memory when match ends
        if (s[0,0,1] != self.N) and self.learner:
            self.push_transitions()
        
    def push_transitions(self):
        """Push all the buffered transitions into the network's memory"""
        for t in self.transitions:
            self.network.push(t)
        self.transitions = []
    
    def train(self, *args, **kwargs):
        self.network.train(*args, **kwargs)
    
    def plot(self, **kwargs):
        """Let the network plot its training loss"""
        self.network.plot(**kwargs)

    # test mode using "with" statement
    def __enter__(self, *args):
        self.network.test_mode(True)
    
    def __exit__(self, *args):
        self.network.test_mode(False)
    
    def set_greedy(self, value):
        self.network.greedy = value