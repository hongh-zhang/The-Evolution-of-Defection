# a copy of the training loop in stage 2
# each iteration is ~15 seconds faster in the terminal than in jupyter notebook

import gc
import dill
import network
import numpy as np
import pandas as pd
import axelrod as axl
from time import time
from pprint import pprint
from random import shuffle
import matplotlib.pyplot as plt
from axl_utils import NNplayer, State, set_match, set_play, Tournament

np.set_printoptions(precision=3)
pd.options.display.float_format = "{:,.2f}".format

C = axl.Action.C
D = axl.Action.D
GAME_LEN = 20 + 1
GAME = axl.Game(r=3, s=0, t=5, p=1)
Match = set_match(game=GAME, turns=GAME_LEN)
play = set_play(Match)

cooperative = [axl.TitFor2Tats(), axl.Random()]
provocative = (axl.Prober(), axl.Prober4(), axl.RemorsefulProber())
retaliative = (axl.TitForTat(), axl.Grudger(), axl.Punisher())


def train_against(player, opponents, iterations=40):
    for _  in range(iterations):
        shuffle(opponents)
        for opponent in opponents:
            play(player, opponent, show=False)

            
if __name__ == "__main__":
    
    players = [*cooperative, *provocative, *retaliative]
    tournament = Tournament(players, game=GAME, turns=GAME_LEN)
    results = tournament.play()

    summary = results.summarise()
    headers = "Rank,Name,Median_score,Cooperation_rating,Wins,Initial_C_rate,CC_rate,CD_rate,DC_rate,DD_rate,CC_to_C_rate,CD_to_C_rate,DC_to_C_rate,DD_to_C_rate".split(',')
    pd.DataFrame(map(list, summary), columns=headers)


    dqn = network.DQN([
                        network.Flatten_layer(),
                        network.Linear_layer(GAME_LEN*2, 300),
                        network.Activation_layer('ReLU'),
                        network.Linear_layer(300, 150),
                        network.Activation_layer('ReLU'),
                        network.Linear_layer(150, 80),
                        network.Activation_layer('ReLU'),
                        network.Linear_layer(80, 40),
                        network.Activation_layer('ReLU'),
                        network.Linear_layer(40, 2),
                        ],
                network.ReplayMemory(8000), gamma=0.9, greedy=0.2)
    p1 = NNplayer(dqn, State(GAME_LEN))
    del dqn
    gc.collect()

    param = {"lr": 7e-6, 'batch': 256, "mode": "train", "eps": 1e-16, "epoch": 0, 't': 1, 'clip': 1.0,
             'optimizer': ('Adam', 0.9, 0.999), 'regularizer': ('l2', 1e-3), "loss_fn":"mse"}
    
    train_against(p1, players)
    len(p1.network.memory)

    ls = []
    for i in range(60):
        
        print(f"-- Iter{i} --")
        
        start = time()
        p1.train(200, param)
        train_against(p1, players)
        print(f'loss: {p1.network.loss},            time: +{time()-start:.2f} sec')

        if i % 2 == 0:
            with p1:
                p1.network.verbosity = False
                tournament = Tournament([p1, *players], game=GAME, turns=GAME_LEN)
                results = tournament.play()
                summary = pd.DataFrame(map(list, results.summarise()), columns=headers).set_index('Name')
                ls.append(summary)

            print(summary.loc['DQN', ['Rank', 'Median_score']])

            with open('data/s2/tn_results.pkl', "wb") as file:
                dill.dump(ls, file)

            with open(f'data/s2/p1_{i}.pkl', "wb") as file:
                dill.dump(p1, file)
