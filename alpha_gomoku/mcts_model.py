"""
@创建日期 ：2021/10/25
@修改日期 ：2021/10/25
@作者 ：jzj
@功能 ：
"""

import copy
import numpy as np


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs = probs / np.sum(probs)
    return probs


class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_prior):
        for action, prob in enumerate(action_prior):
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1. * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """
        在使用ucb的时候，需要知道父节点的调用次数，所有要先递归更新父节的信息
        """
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = TreeNode(None, 1.)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        node = self._root
        while 1:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)
        action_probs, leaf_value = self._policy(state.cur_state())
        end, winner = state.game_end()

        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.
            else:
                leaf_value = (1. if winner == state.get_current_player() else -1.)

        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        for i in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node.n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, is_self_play=0):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self._is_self_play = is_self_play

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_probs=False):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)

        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs

            if self._is_self_play:
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            if return_probs:
                return move, probs
            else:
                return move
        else:
            print("WARNING: board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)



