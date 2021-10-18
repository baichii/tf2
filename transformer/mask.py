"""
@创建日期 ：2021/10/16
@修改日期 ：2021/10/16
@作者 ：jzj
@功能 ：
"""

import tensorflow as tf


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :]  # [batch, 1, 1, seq_len]


def create_look_ahead_mask(size):
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


if __name__ == '__main__':
    # x = tf.constant([[7, 6, 0, 0, 1], [2, 3, 1, 2, 2]])
    # print("mask: ", creat_padding_mask(x))
    print(create_look_ahead_mask(10))
