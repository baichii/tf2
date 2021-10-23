"""
@创建日期 ：2021/10/23
@修改日期 ：2021/10/23
@作者 ：jzj
@功能 ：https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/game.py
"""

import numpy as np


class Board:
    def __init__(self, **kwargs):
        self.width = int(kwargs.get("width", 8))
        self.height = int(kwargs.get("height", 8))

        self.states = {}
        self.n_in_row = int(kwargs.get("n_in_row", 5))
        self.players = [1, 2]

    def reset(self, start_player=0):
        self.cur_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def cur_state(self):
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.cur_player]
            move_oppo = moves[players != self.cur_player]

            square_state[0][move_curr // self.width, move_curr % self.width] = 1.
            square_state[1][move_oppo // self.width, move_oppo % self.width] = 1.
            square_state[2][self.last_move // self.width, self.last_move % self.width] = 1.
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.cur_player
        self.availables.remove(move)
        self.cur_player = (self.players[0] if self.cur_player == self.players[1] else self.players[1])
        self.last_move = move

    # def has_a_winner(self):
    #     moved = list(set(range(self.width * self.height))) - set(self.availables)
    #
    #     if len(moved) < self.n_in_row * 2 - 1:
    #         return False, -1
    #
    #     for m in moved:
    #         h = m // self.width
    #         w = m % self.width
    #         player = self.states[m]
    #
    #         if (w in range())

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.cur_player


class Game:
    def __init__(self, board, **kwargs):
        self.board = board

    def start_play(self, player1, player2, start_player=0, is_shown=0):
        self.board.reset()

        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        if is_shown:
            raise NotImplementedError

        while True:
            current_player = self.board.cur_player
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.do_move(move)
            if is_shown:
                raise NotImplementedError
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is ", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """
        使用MCTS player
        """
        self.board.reset()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []

        while True:
            move, move_probs = player.get_action(self.board, temp, return_prob=True)
            # 状态存储
            states.append(self.board.cur_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.cur_player)

            self.board.do_move(move)

            if is_shown:
                raise NotImplementedError
            end, winner = self.board.game_end()

            if end:
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.
                    winner_z[np.array(current_players) != winner] = -1.
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is ", winner)
                    else:
                        print("Game end. Tie")
            return winner, zip(states, mcts_probs, winner_z)