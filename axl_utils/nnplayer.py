# this file provide a nnplayer subclass for the axelrod library
# it organize states and replay memory to cooperate with DQN

import random
import numpy as np
import axelrod as axl
from time import time
from collections import deque, namedtuple

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

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
    
    def __init__(self, network, state, reward='dense', policy='off', name='DQN'):
        super().__init__()
        
        self.name    = name
        self.state   = state
        self.network = network
        
        self.policy_mode = True if policy=="off" else False      # off-policy = 1, on-policy = 0
        self.reward_mode = True if reward=="dense" else False    # dense reward = 1, sparse reward = 0
        self.N = self.state.N                                    # how not-yet-happened turn is encoded
        self.reset()
    
    def __str__(self):
        return self.name
    
    # the following 3 functions override the orginal implementation in axelrod library
    # they are automatically called by axl during each game
    def reset(self):
        """Reset the attributes to start a new game"""
        self.reward = 0
        self.state.reset()
        self.transitions = []
        self.network.reset_state()
        self._history = axl.history.History()
        
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
        """Update current game state & record transition into replay memory
        
        Parameters
        ----------
        play : axl.Action
            action from last turn, (C or D)
        coplay: axl.Action
        
        """
        
        # update game state
        s  = self.state.values()
        s_ = self.state.push(play, coplay)
        last_turn = s[0,0,1]!=self.N
        
        # compute reward
        r  = axl.interaction_utils.compute_scores([(play, coplay)])[0][0]
        
        # rewrite action
        action = [True, False] if play==axl.Action.C else [False, True]
        
        # dense reward
        if self.reward_mode:
            transition = Transition(s, action, s_, r, last_turn)
        
        # sparse reward
        else:
            if not last_turn:
                transition = Transition(s, action, s_, 0, last_turn)
                self.reward += r
            else:
                transition = Transition(s, action, s_, r+self.reward, last_turn)
                self.reward = 0
        
        # record transitions for training
        self.transitions.append(transition)
        
        # last turn operations
        if last_turn:
            self.end_episode()
    
    def end_episode(self):
        # for off-policy learner,
        # push all transitions into replay memory
        if self.policy_mode:
            for t in self.transitions:
                self.network.push(t)
            self.transitions = []

        # for on-policy learner,
        # push all rewards,
        # then call train function
        else:
            for t in self.transitions:
                self.network.push(t.reward)
            self.transitions = []
            self.network.train()
    
    
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