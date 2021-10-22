# modified axl.tournament that ignores the last turn
# code copied from axelrod library

import axelrod as axl
from collections import defaultdict
from pprint import pprint

class Tournament(axl.Tournament):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _play_matches(self, chunk, build_results=True):
        """
        Play matches in a given chunk.

        Parameters
        ----------
        chunk : tuple (index pair, match_parameters, repetitions)
            match_parameters are also a tuple: (turns, game, noise)
        build_results : bool
            whether or not to build a results set

        Returns
        -------
        interactions : dictionary
            Mapping player index pairs to results of matches:

                (0, 1) -> [(C, D), (D, C),...]
        """
        interactions = defaultdict(list)
        index_pair, match_params, repetitions, seed = chunk
        p1_index, p2_index = index_pair
        player1 = self.players[p1_index].clone()
        player2 = self.players[p2_index].clone()
        match_params["players"] = (player1, player2)
        match_params["seed"] = seed
        match = axl.Match(**match_params)
        for _ in range(repetitions):
            match.play()

            if build_results:
                results = self._calculate_results(match.result[:-1])  # drop last turn
            else:
                results = None

            interactions[index_pair].append([match.result, results])
        return interactions