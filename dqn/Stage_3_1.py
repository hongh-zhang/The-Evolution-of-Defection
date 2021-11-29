# training loop for stage 3
# takes serveral hours to run

import gc
import dill
import numpy as np
import pandas as pd
import axelrod as axl
from time import time
from copy import deepcopy
from pprint import pprint

np.set_printoptions(precision=3)
pd.options.display.float_format = "{:,.2f}".format

import network
from axl_utils import NNplayer, State, set_match, set_play, Tournament

GAME_LEN = 20 + 1
C = axl.Action.C
D = axl.Action.D
GAME = axl.Game(r=3, s=0, t=5, p=1)
Match = set_match(game=GAME, turns=GAME_LEN)
play = set_play(Match)
greedy = (0.35, 0.98, 0.05)
headers = "Rank,Name,Median_score,Cooperation_rating,Wins,Initial_C_rate,CC_rate,CD_rate,DC_rate,DD_rate,CC_to_C_rate,CD_to_C_rate,DC_to_C_rate,DD_to_C_rate".split(',')


if __name__ == '__main__':
    # initializing
    dqn = network.DQN([
                        network.Flatten_layer(),
                        network.Linear_layer(GAME_LEN*2, 200),
                        network.Activation_layer('ReLU'),
                        network.Linear_layer(200, 100),
                        network.Activation_layer('ReLU'),
                        network.Linear_layer(100, 40),
                        network.Activation_layer('ReLU'),
                        network.Linear_layer(40, 2),
                        ],
                network.ReplayMemory(4000), gamma=0.9, greedy=0.2)
    p1 = NNplayer(dqn, State(GAME_LEN), name='DQN1')

    dqn = network.DQN([
                        network.Flatten_layer(),
                        network.Linear_layer(GAME_LEN*2, 200),
                        network.Activation_layer('ReLU'),
                        network.Linear_layer(200, 100),
                        network.Activation_layer('ReLU'),
                        network.Linear_layer(100, 40),
                        network.Activation_layer('ReLU'),
                        network.Linear_layer(40, 2),
                        ],
                network.ReplayMemory(4000), gamma=0.9, greedy=0.2)
    p2 = NNplayer(dqn, State(GAME_LEN), name='DQN2')
    del dqn
    gc.collect()

    param1 = {"lr": 1e-4, 'batch': 128, "mode": "train", "eps": 1e-16, "epoch": 0, 't': 1, 'clip': 1.0,
          'optimizer': ('Adam', 0.9, 0.999), 'regularizer': ('l2', 1e-3), "loss_fn":"mse"}
    param2 = deepcopy(param1)   
    
    # start training
    [Match((p1, p2)).play() for _ in range(200)]
    ls = []
    for i in range(40):

        # test
        if i % 2 == 0:
            with p1:
                with p2:
                    p1.network.verbosity = p2.network.verbosity = False
                    tournament = Tournament((p1, p2), game=GAME, turns=GAME_LEN)
                    results = tournament.play()
                    summary = pd.DataFrame(map(list, results.summarise()), columns=headers).set_index('Name')
                    ls.append(summary)

            print(summary.loc[['DQN1','DQN2'], ['Median_score', 'Cooperation_rating']])

            # save
            with open('data/s3/tn_results_1.pkl', "wb") as file:
                dill.dump(ls, file)
            with open(f'data/s3/p1_1_{i}.pkl', "wb") as file:
                dill.dump(p1, file)
            with open(f'data/s3/p2_1_{i}.pkl', "wb") as file:
                dill.dump(p2, file)

        print(f'--------Iter {i}--------')
        start = time()

        # get new experience
        [Match((p1, p2)).play() for _ in range(200)]

        # train
        for _ in range(8):
            p1.train(100, param1)
            p2.train(100, param2)
            param1['lr'] = param2['lr'] = param1['lr'] * 0.9
        param1['lr'] = param2['lr'] = 1e-4
        print(p1.network.loss)
        print(p2.network.loss)

        if i >= 5:
            g = (greedy[0] * greedy[1] ** i) + greedy[2]
            p1.network.greedy = p2.network.greedy = g
        print(f"+ {time() - start:.2f} sec")
