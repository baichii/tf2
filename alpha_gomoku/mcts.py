"""
@创建日期 ：2021/10/23
@修改日期 ：2021/10/23
@作者 ：jzj
@功能 ：MCTS ALPHA
       https://github.com/junxiaosong/AlphaZero_Gomoku
"""

import copy
import numpy as np


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs = probs / np.sum(probs)
    return probs


def rollout_policy_fn(board):
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_prior):
        for action, prob in action_prior:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1. * (leaf_value - self._Q) / self._n_visits





a = TreeNode(parent=None, prior_p=None)
print(a._parent)





