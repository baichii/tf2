"""
@创建日期 ：2021/9/27
@修改日期 ：2021/9/27
@作者 ：jzj
@功能 ：测试、https://www.tensorflow.org/api_docs/python/tf/GradientTape
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, losses, optimizers

tf.random.set_seed(21)


def demo1():
    """
    偏导，多次调用persistent=True
    """
    x1 = tf.constant(3.0)
    x2 = tf.constant(4.)

    with tf.GradientTape(persistent=True) as g:
        g.watch(x1)
        g.watch(x2)
        y = x1 ** 2 + x2 ** 3
    grads1 = g.gradient(y, x1)
    grads2 = g.gradient(y, x2)
    print(grads1)
    print(grads2)


def demo2():
    """
    通过嵌套求高阶导数
    """
    x = tf.constant(5.)
    with tf.GradientTape() as g:
        g.watch(x)
        with tf.GradientTape() as gg:
            gg.watch(x)
            y = x * x
        dy_dx = gg.gradient(y, x)
    d2y_dx2 = g.gradient(dy_dx, x)
    print(dy_dx)
    print(d2y_dx2)


def demo3():
    """
    多个
    """
    x = tf.constant(3.)
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        y = x * x
        z = y * y
    dz_dx = g.gradient(z, x)
    print(dz_dx)
    dy_dx = g.gradient(y, x)
    print(dy_dx)


def demo4():
    """
    不watch,watch_accessed_variables会自动watch上下文的trainable变量
    """
    x = tf.Variable(2.)
    w = tf.Variable(5.)
    with tf.GradientTape(watch_accessed_variables=True, persistent=True) as g:
        g.watch(x)
        y = x ** 2
        z = w ** 3
    dy_dx = g.gradient(y, x)
    dz_dw = g.gradient(z, w)
    print(dy_dx)
    print(dz_dw)


def demo5():
    """
    layer测试
    """
    a = tf.keras.layers.Dense(32, use_bias=False)
    b = tf.keras.layers.Dense(32)
    inputs = tf.random.uniform(shape=(1, 10))

    with tf.GradientTape(watch_accessed_variables=True) as g:
        g.watch(a.variables)

        result = b(a(inputs))

        # print(g.gradient(result, a.variables))
        print(g.gradient(result, b.variables))


class TModel(Model):
    def __init__(self):
        super(TModel, self).__init__()
        self.dense1 = layers.Dense(units=10, use_bias=False)
        self.target1 = layers.Dense(2, use_bias=False)
        self.target2 = layers.Dense(1, use_bias=False, activation=tf.nn.sigmoid)

    def call(self, x):
        common = self.dense1(x)
        t1 = self.target1(common)
        t2 = self.target2(common)
        return t1, t2


def demo6():
    """
    模型测试
    """
    model = TModel()
    optimizer = optimizers.Adam(learning_rate=1e-2)
    test_data = {"input": tf.random.uniform(shape=(1, 10)),
                 "t1": tf.random.uniform(shape=(1, 2)),
                 "t2": tf.constant(3.)}

    with tf.GradientTape() as g:
        res = model(test_data["input"])
        l1 = losses.CategoricalCrossentropy()
        l2 = losses.MSE
        print(test_data["t1"], res[0])
        l1_loss = l1(test_data["t1"], res[0])
        l2_loss = l2(test_data["t2"], res[1])
        print(l1_loss, l2_loss)
        loss = l1_loss + l2_loss

    print(g.gradient(l1_loss, model.trainable_variables))

    # optimizer.apply_gradients(grads, model.trainable_variables)


if __name__ == '__main__':
    demo6()

