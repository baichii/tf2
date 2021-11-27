"""
@创建日期 ：2021/11/27
@修改日期 ：2021/11/27
@作者 ：jzj
@功能 ：tf.Module
"""

import tensorflow as tf
import tensorflow
from tensorflow.keras import layers


def add(a, b):
    return a+ b


a1 = tf.Tensor(1, dtype=tf.int64)
a2 = tf.Tensor(1, dtype=tf.int64)


class Dense(tf.Module):
    def __init__(self, input_dim, output_size, name=None):
        super(Dense, self).__init__()
        self.w = tf.Variable(
            tf.random.normal(shape=[input_dim, output_size], name="w")
        )
        self.b = tf.Variable(tf.zeros([output_size]), name="b")

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


class Mlp(tf.Module):
    def __init__(self, input_size, sizes, name=None):
        super(Mlp, self).__init__(name=name)
        self.layers = []
        with self.name_scope:
            for size in sizes:
                self.layers.append(layers.Dense(input_dim=input_size, output_size=size))
                input_size = size

    @tf.Module.with_name_scope
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
