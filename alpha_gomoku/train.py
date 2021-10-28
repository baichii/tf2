"""
@创建日期 ：2021/10/25
@修改日期 ：2021/10/25
@作者 ：jzj
@功能 ：
"""

import random
import numpy as np
from collections import defaultdict, deque
from alpha_gomoku.game import Board, Game
from alpha_gomoku.mcts_pure import MCTSPlayer as MCTS_Pure_Player
from alpha_gomoku.mcts_model import MCTSPlayer as MCTS_Model_PLAYER
from alpha_gomoku.policy_value_net import Model as PolicyValueNet


class TrainPipeline:
    def __init__(self, init_model=None):
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)

        self.game = Game(board=self.board)
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000
        if init_model:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        self.mcts_player = MCTS_Model_PLAYER(self.policy_value_net,
                                             c_puct=self.c_puct,
                                             n_playout=self.n_playout,
                                             is_self_play=True)

    def get_equi_data(self, play_data):
        pass

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            print(winner, play_data)

    def train_step(self, bath):
        pass

    def run(self):
        for i in range(self.game_batch_num):
            pass


def demo():
    pipe = TrainPipeline()
    pipe.collect_selfplay_data()


if __name__ == '__main__':
    demo()








