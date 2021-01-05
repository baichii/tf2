"""
@创建日期 ：2020/11/27
@修改日期 ：2020/11/27
@作者 ：jzj
@功能 ：
"""

from __future__ import absolute_import
from __future__ import division


import sys
import tensorflow as tf


a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

c = tf.add(a, b)


print(c)

# x = tf.keras.backend.placeholder(tf.float32)
# y = tf.keras.backend.placeholder(tf.float32)
x = tf.placeholder(tf.int16)
print(x)
# print(sys.getsizeof(x))