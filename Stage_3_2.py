# similar to stage_3_2.py but we import trained DQNs from stage 2
# rather than build it from scratch

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
    with open("data/s2/backup/p1_54.pkl", "rb") as file:
        p1 = dill.load(file)
        p1.name = 'DQN1'

    with open("data/s2/backup/p1_58.pkl", "rb") as file:
        p2 = dill.load(file)
        p2.name = 'DQN2'

    # reset loss ls
    p1.network.epoch = 0
    p1.network.loss_ls = []
    p2.network.epoch = 0
    p2.network.loss_ls = []
    
    # reset replay memory
    p1.network.memory = network.ReplayMemory(4000)
    p2.network.memory = network.ReplayMemory(4000)

    # freeze layers
    p1.network.policy_net.layers[0].freeze = True
    p1.network.policy_net.layers[1].freeze = True
    p1.network.policy_net.layers[2].freeze = True
    p1.network.policy_net.layers[3].freeze = True
    p2.network.policy_net.layers[0].freeze = True
    p2.network.policy_net.layers[1].freeze = True
    p2.network.policy_net.layers[2].freeze = True
    p2.network.policy_net.layers[3].freeze = True
    
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
            with open('data/s3/tn_results_2.pkl', "wb") as file:
                dill.dump(ls, file)
            with open(f'data/s3/p1_2_{i}.pkl', "wb") as file:
                dill.dump(p1, file)
            with open(f'data/s3/p2__2{i}.pkl', "wb") as file:
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
        param1['lr'] = param2['lr'] = 3e-4
        print(p1.network.loss)
        print(p2.network.loss)

        g = (greedy[0] * greedy[1] ** i) + greedy[2]
        p1.network.greedy = p2.network.greedy = g
        print(f"+ {time() - start:.2f} sec")
