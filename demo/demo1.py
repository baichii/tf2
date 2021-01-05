"""
@创建日期 ：2020/11/23
@修改日期 ：2020/11/23
@作者 ：jzj
@功能 ：tensorflow 2 测试
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class Net(keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = keras.layers.Dense(1)

    def call(self, x):
        return self.l1(x)


def toy_dataset():
    inputs = tf.range(10.)[:, None]
    labels = inputs * 5. + tf.range(5.)[None, :]
    dataset = tf.data.Dataset.from_tensor_slices(dict(x=inputs, y=labels))
    return dataset.repeat().batch(2)


def train_step(net, example, optimizer):
    with tf.GradientTape() as tape:
        output = net(example["x"])
        loss = tf.reduce_mean(tf.abs(output - example["y"]))
    variables = net.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


def demo():
    model = keras.applications.MobileNetV2(weights=None)
    # model.summary()
    test_image = np.random.randn(1, 224, 224, 3)
    res = model(test_image)
    save_path = "/Users/jiangzhenjie/Desktop/out"
    tf.saved_model.save(model, save_path)


def load(save_model):
    model = tf.saved_model.load(save_model)
    print(model)


def ckpt():
    net = Net()
    net.save_weights("/Users/jiangzhenjie/Desktop/ckpt/net")



if __name__ == '__main__':
    # load("/Users/jiangzhenjie/Desktop/out")

    # ckpt
    ckpt()