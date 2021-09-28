"""
@创建日期 ：2020/11/27
@修改日期 ：2020/11/27
@作者 ：jzj
@功能 ：
"""

import tensorflow as tf


# @tf.function
def fib(n):
    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    ta = ta.unstack([0, 1])
    print(ta.stack())
    print(ta.read(1))
    # return ta.stack()


print(fib(2))

