"""
@创建日期 ：2021/10/25
@修改日期 ：2021/10/28
@作者 ：jzj
@功能 ：track
       1、资格迹？重复采样
       2、kl-divergence
"""

import time
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict, deque
from alpha_gomoku.game import Board, Game
from alpha_gomoku.mcts_pure import MCTSPlayer as MCTS_Pure_Player
from alpha_gomoku.mcts_model import MCTSPlayer as MCTS_Model_PLAYER
from alpha_gomoku.policy_value_net import PolicyValueNet


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
        self.n_playout = 100  # fixme: 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 32  # fixme 512
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
        self.mcts_player = MCTS_Model_PLAYER(self.policy_value_net.policy_value_fn,
                                             c_puct=self.c_puct,
                                             n_playout=self.n_playout,
                                             is_self_play=True)

    def get_equi_data(self, play_data):
        """
        数据增强，包含对角线翻转和水平翻转
        """

        extend_data = []
        for state, mcts_probs, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_probs = np.rot90(np.flipud(mcts_probs.reshape(self.board_width, self.board_height)), i)
            extend_data.append((equi_state, np.flipud(equi_mcts_probs).flatten(), winner))

            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_probs = np.fliplr(equi_mcts_probs)
            extend_data.append((equi_state, np.flipud(equi_mcts_probs).flatten(), winner))
        return extend_data

    def collect_self_play_data(self, n_games=1):
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)

        play_data = list(play_data)[:]
        self.episode_len = len(play_data)
        play_data = self.get_equi_data(play_data)
        self.data_buffer.extend(play_data)

    def policy_update(self):
        """
        在data buffer 采样，像资格季
        """
        mini_batch = random.sample(self.data_buffer, self.batch_size)

        state_batch = [data[0] for data in mini_batch]
        mcts_prob_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        state_batch = np.stack(state_batch)

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            self.policy_value_net.train_step(state_batch, mcts_prob_batch, winner_batch)

    def policy_evaluate(self, n_games=10):
        # current_mcts_player = MCTS_Model_PLAYER(self.policy_value_net.policy_value_fn,
        #                                         c_puct=self.c_puct,
        #                                         n_playout=self.n_playout)

        current_mcts_player = MCTS_Pure_Player(c_puct=5, n_playout=100)

        pure_mcts_player = MCTS_Pure_Player(c_puct=5, n_playout=100)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player, pure_mcts_player, start_player=i % 2, is_shown=False)
            win_cnt[winner] += 1
        win_ratio = (1.0 * win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("mcts win_ratio: ", win_ratio)

    def run(self):
        for i in range(self.game_batch_num):
            self.collect_self_play_data(1)
            print(f"batch: {i+1}, episode_len: {self.episode_len}, cum_buffer: {len(self.data_buffer)}")
            if len(self.data_buffer) > self.batch_size:
                self.policy_update()


def demo():
    pipe = TrainPipeline()
    # pipe.run()
    for i in range(20):
        t1 = time.time()
        pipe.policy_evaluate(10)
        print(f"epoch {i+1}, run time: {time.time() - t1:.4f}s")


if __name__ == '__main__':
    demo()
