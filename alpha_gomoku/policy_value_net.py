"""
@创建日期 ：2021/10/25
@修改日期 ：2021/10/25
@作者 ：jzj
@功能 ：
"""

import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Model
from


class Model(Model):
    def __init__(self, board_width, board_height):
        super(Model, self).__init__()
        # common layers
        self.conv1 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.conv2 = layers.Conv2D(64, 3, padding="same", activation="relu")
        self.conv3 = layers.Conv2D(128, 3, padding="same", activation="relu")

        # action policy layers
        self.act_conv1 = layers.Conv2D(4, 1, activation="relu")
        self.act_fn = layers.Flatten()
        self.act_fc1 = layers.Dense(board_width * board_height, activation="log_softmax")

        # state value layers
        self.val_conv1 = layers.Conv2D(2, 1, activation="relu")
        self.val_fn = layers.Flatten()
        self.val_fc1 = layers.Dense(64, activation="relu")
        self.val_fc2 = layers.Dense(1, activation="tanh")

    def call(self, x):
        common = self.conv1(x)
        common = self.conv2(common)
        common = self.conv3(common)

        x_act = self.act_conv1(common)
        x_act = self.act_fn(x_act)
        x_act = self.act_fc1(x_act)

        x_val = self.val_conv1(common)
        x_val = self.val_fn(x_val)
        x_val = self.val_fc1(x_val)
        x_val = self.val_fc2(x_val)
        return x_act, x_val


class PolicyValueNet:
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.policy_value_net = Model(board_width, board_height)
        self.optimizer = optimizers.Adam(lr=1e-3)

        if model_file:
            self.policy_value_net.load_weights(model_file)


    def policy_value_fn(self, board):
        pass

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        pass

    def get_policy_param(self):
        pass

    def save_model(self, model_file):
        self.policy_value_net.save(model_file)


if __name__ == '__main__':
    test_tenor = tf.random.uniform(shape=(1, 6, 6, 4))
    model = Model(6, 6)
    print(model(test_tenor))
