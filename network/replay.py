from random import sample
from pickle import dump, load
from collections import namedtuple, deque

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """push single transition into memory"""
        self.memory.append(Transition(*args))

    def sample(self, n):
        """randomly select n transitions"""
        return sample(self.memory, n)

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