# replay memory for the nnplayer

from random import sample
from pickle import dump, load
from collections import namedtuple, deque

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
        return sample(self.memory, n)
    
    def values(self):
        """All values of the ReplayMemory, in list"""
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