import axelrod as axl

GAME_LEN = 20 + 1
GAME = axl.Game(r=3, s=0, t=5, p=1)

def set_match(game=GAME, turns=GAME_LEN):
    def match(players):
        return axl.Match(players, turns=turns, reset=True, game=GAME)
    return match

def set_play(match):
    def play(player1, player2):
        game = match((player1, player2))
        actions = game.play()
        scores = game.scores()[:-1]
        print(scores)
        print(f"Player 1 score = {sum(list(zip(*scores))[0])}")
        print(f"Player 2 score = {sum(list(zip(*scores))[1])}")
        return game
    return play