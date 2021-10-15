# functions to extract replay experience, to be used with off-policy algorithm like DQN
"""DEPRECATED as nnplayer now records memory while playing"""

import numpy as np
import axelrod as axl
from axl_utils.nnplayer import State

GAME_LEN = 20 + 1

def extract(game, memory, depth=GAME_LEN):
    """
    extract all transitions from a full game
    game = axl.Match object, with a finished game,
    memory = ReplayMemory
    """
    actions = game.result
    rewards = game.scores()
    state = State(depth)
    
    s = state.values()
    iterator = iter(zip(actions, rewards))
    while True:
        a_, r_ = next(iterator)
        s_ = state.push(*a_)
        a_ = [True, False] if a_[0]==axl.Action.C else [False, True]
        memory.push(s, a_, s_, r_[0])
        s, a, r = (s_, a_, r_)

        # hardcoding the last state
        if s[0,0,1] != -1:
            a_, r_ = next(iterator)
            s_ = state.push(*a_)
            a_ = [True, False] if a_[0]==axl.Action.C else [False, True]
            memory.push(s, a_, s_, np.NaN)
            break
            
# memory = ReplayMemory(1000)
# print(len(memory))
# extract(game, memory, GAME_LEN)
# print(len(memory))

def collect_exp(players, memory):
    old = len(memory)
    for pair in players:
        game = Match(pair, turns=GAME_LEN)
        game.play()
        extract(game, memory, GAME_LEN)
    new = len(memory)
    print(f"Collected {new-old} experience.")
    
# players = permutations([axl.TitForTat(), axl.TitForTat(), axl.Random(), axl.Alternator()], 2)
# collect_exp(players, memory)