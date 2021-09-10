"""
@创建日期 ：2021/9/10
@修改日期 ：2021/9/10
@作者 ：jzj
@功能 ：https://www.tensorflow.org/text/tutorials/transformer
"""

import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert self.d_model % self.num_heads == 0
