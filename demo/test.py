"""
@创建日期 ：2021/9/27
@修改日期 ：2021/9/27
@作者 ：jzj
@功能 ：
"""


import tensorflow as tf
tf.random.set_seed(42)


probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
returns = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

for i in tf.range(10):
    probs = probs.write(i, tf.math.log([tf.random.uniform(shape=(1, )), tf.random.uniform(shape=(1,))]))
    returns = returns.write(i, tf.random.uniform(shape=(1,)))
    values = values.write(i, tf.random.uniform(shape=(1,)))

probs = probs.stack()
advantage = returns.stack() - values.stack()

# print(probs)
actor_loss = tf.math.reduce_sum(tf.expand_dims(probs, 1) * tf.expand_dims(advantage, 1))
print(actor_loss)
# print(tf.(t(probs)) * tf.squeeze(advantage))

# print(probs * advantage)


# actor_loss = -tf.math.reduce_sum()
