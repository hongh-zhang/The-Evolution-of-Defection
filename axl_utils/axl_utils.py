# Author: Honghao Zhang
# This file is intended to provide support functions to train RL algorithms with axelrod library

import random
import pickle
import numpy as np
import axelrod as axl
from itertools import permutations
from collections import namedtuple, deque
C = axl.Action.C
D = axl.Action.D
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def __repr__(self):
        if len(self) >= 100:
            out = self.memory[:100]
        else:
            out = self.memory
        return str(out).replace("), ", "),\n")
    
    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)
    
    def load(self, path, mode='overwrite'):
        with open(path, "rb") as file:
            if mode == 'overwrite':
                self = pickle.load(file)
            elif mode == 'add':
                for i in pickle.load(file).memory:
                    self.memory.append(i)
                    
def extract_states(history, size, memory, N=-1, mode='int'):
    temp = deque([N for i in range(size)], maxlen=size)
    memory.append(list(temp))
    for state in history:
        temp.append(state)
        memory.append(list(temp))
        
def extract_transitions(actions, scores, size, memory):
    """
    Extract transitions from a game, and push them into a given replay memory,
    player should be in the 1st position of tuples,
    
    Arguments:
    -------
    (list) actions: action history of the game e.g. [(C,C), (D,C), ...]
    (list) scores: score history of the game e.g. [(3,3), (5,0), ...]
    (Maybe int) size: desired player memory size, could be 'all'
    (ReplayMemory) memory: replay memory to save the transitions, must support a push(*args) method
    """
    # format inputs
    assert len(actions) == len(scores), "Length not matching!"
    actions, scores = map(lambda x: list(list(zip(*x))[0])+[0], (actions, scores))  # extract column then pad for iterator

    # extract states from history
    states = []
    extract_states(scores, size, states)
    
    # save transitions(state, action, next_state, reward) into replay memory
    iterator = iter(zip(states, actions, scores))
    s, a, r = next(iterator)
    while True:
        try:
            s_, a_, r_ = next(iterator)
            memory.push(s, a, s_, r)
            s, a, r = (s_, a_, r_)
        except StopIteration:
            break
            
def collect_exp(players, memory):
    old = len(memory)
    for pair in players:
        game = axl.Match(pair, turns=GAME_LEN)
        actions = game.play()
        scores = game.scores()
        extract_transitions(actions, scores, GAME_LEN, memory)
    new = len(memory)
    print(f"Collected {new-old} experience.")