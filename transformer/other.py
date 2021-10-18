"""
@创建日期 ：2021/10/18
@修改日期 ：2021/10/18
@作者 ：jzj
@功能 ：学习率、
"""

import tensorflow as tf
from tensorflow.keras import optimizers, losses


class CustomSchedule(optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_step=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_step

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred, loss):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


# loss = losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
