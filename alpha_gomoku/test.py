"""
@创建日期 ：2021/10/28
@修改日期 ：2021/10/28
@作者 ：jzj
@功能 ：测试mcts模拟
"""


import random
import numpy as np
from collections import defaultdict, deque
from alpha_gomoku.game import Board, Game
from alpha_gomoku.mcts_pure import MCTSPlayer as MCTS_Pure_Player
from alpha_gomoku.mcts_model import MCTSPlayer as MCTS_Model_Player
from alpha_gomoku.policy_value_net import Model


def demo():
    board = Board(width=6, height=6, n_in_row=4)
    game = Game(board)
    model = Model(6, 6)
    player = MCTS_Model_Player(policy_value_fn=model)
    winner, data = game.start_self_play(player=player)
    print(winner)
    print(data)






if __name__ == '__main__':
    demo()
