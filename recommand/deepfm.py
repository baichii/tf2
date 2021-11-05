"""
@创建日期 ：2021/10/30
@修改日期 ：2021/10/30
@作者 ：jzj
@功能 ：deepfm
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


class FMlayer(layers.Layer):
    def __init__(self):
        super(FMlayer, self).__init__()

    def build(self, input_shape):
        self.b = self.add_weight(name="bias", initializer="zeros", trainable=True)
        super(FMlayer, self).build(input_shape)

    def call(self, outputs):
        sum1 = tf.reduce_sum()

        # todo: what fuck


class DeepLayer(layers.Layer):
    def __init__(self, hidden_dim, layer_nums, out_dim):
        super(DeepLayer, self).__init__()
        self.layers = []
        for i in range(layer_nums):
            self.layers.append(layers.Dense(hidden_dim, activation="relu", name="deem_layer"+str(i+1)))
            self.layers.append(layers.Dropout(0.5))
            self.layers.append(out_dim)

    def call(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Model(models.Model):
    def __init__(self):
        pass

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            fm_var_indexes, deep_var_indexes = [], []
            for varibales in self.model.trainable_variables:
                if varibales.name.startwith("fm_layer"):
                    fm_var_indexes.append(varibales)
                else:
                    deep_var_indexes.append(varibales)
            gradients = tape.gradient(loss, self.model.trainable_variables)
        self.deep_optimizer.appy_gradients(zip([gradients[i], self.model.trainable_variables[i]] for i in deep_var_indexes))
        self.fm_optimizer.appy_gradients(zip([gradients[i], self.model.trainable_variables[i]] for i in fm_var_indexes))
        self.compiled_metrics.update(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
