"""
@创建日期 ：2020/11/27
@修改日期 ：2020/11/27
@作者 ：jzj
@功能 ：
"""

import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds


IMAGE_SIZE = 160


def create_model():
    base_model = keras.applications.MobileNetV2(weights=None, input_shape=(160, 160, 3), include_top=False)
    model = keras.models.Sequential([base_model,
                                     keras.layers.GlobalAveragePooling2D(),
                                     keras.layers.Dense(1)])
    model.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    est_model = keras.estimator.model_to_estimator(keras_model=model)
    return est_model


def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image, label


def train_input_fn(batch_size):
    data = tfds.load("cats_vs_dogs", data_dir="/Users/jiangzhenjie/Downloads", as_supervised=True)
    train_data = data["train"]
    train_data = train_data.map(preprocess).shuffle(500).batch_size(batch_size)
    return train_data


def test():
    # tf.data.Dataset()
    est = create_model()
    est.train(input_fn=lambda: train_input_fn(32), steps=500)
    est.evaluate(input_fn=lambda: train_input_fn(32), steps=10)


if __name__ == '__main__':
    # create_model()
    # test()
    train_input_fn(batch_size=2)