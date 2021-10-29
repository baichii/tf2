"""
@创建日期 ：2021/10/28
@修改日期 ：2021/10/28
@作者 ：jzj
@功能 ：mmoe
"""

import tensorflow as tf
from tensorflow.keras import layers, models


class MMOE(models.Model):
    def __init__(self):
        pass

    def build(self, input_shape):
        input_dim = input_shape[-1]


a = tf.random.uniform(shape=(2, 512))
b = tf.random.uniform(shape=(512, 64, 4))

print(tf.tensordot(a, b, axes=1))


