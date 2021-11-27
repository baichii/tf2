"""
@创建日期 ：2021/11/20
@修改日期 ：2021/11/20
@作者 ：jzj
@功能 ：
"""


import tensorflow as tf


data = tf.constant([1, 2, 3, 4, 5, 6])
indices = tf.constant([2, 3])

print(tf.gather(data, indices))

# print(tf.test.is_gpu_available())