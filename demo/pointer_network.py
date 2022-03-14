"""
@创建日期 ：2021/10/11
@修改日期 ：2021/10/15
@作者 ：jzj
@功能 ：https://github.com/keon/pointer-networks/blob/master/PointerLSTM.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, activations


class Attention(layers.Layer):
    def __init__(self, hidden_unit, name="attention"):
        super(Attention, self).__init__(name=name)
        self.W1 = layers.Dense(hidden_unit, use_bias=True)
        self.W2 = layers.Dense(hidden_unit, use_bias=True)
        self.V = layers.Dense(hidden_unit, use_bias=True)

    def call(self, encoder_outputs, decoder_outputs, mask=None):
        w1_e = self.W1(encoder_outputs)
        w2_d = self.W2(decoder_outputs)
        tanh_output = activations.tanh(w1_e + w2_d)
        v_dot_tanh = self.V(tanh_output)
        if mask is not None:
            v_dot_tanh += (mask * -1e-9)
        attention_weights = activations.softmax(v_dot_tanh, axis=1)
        att_shape = tf.shape(attention_weights)
        return tf.reshape(attention_weights, (att_shape[0], att_shape[1]))


class Decoder(layers.Layer):
    def __init__(self, hidden_unit, name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.lstm = layers.LSTM(hidden_unit, return_state=True)

    def call(self, x, hidden_states):
        decode_output, state_h, state_c = self.lstm(x, initial_state=hidden_states)
        return decode_output, [state_h, state_c]

    def get_initial_state(self, inputs):
        return self.lstm.get_initial_state(inputs)

    def process_inputs(self, x_input, initial_states, constants):
        return self.lstm._process_inputs(x_input, initial_states, constants)
        

class PointerLSTM(layers.Layer):
    def __init__(self, hidden_unit, name="pointer", **kwargs):
        super(PointerLSTM, self).__init__()
        self.hidden_unit = hidden_unit
        self.attention = Attention(hidden_unit)
        self.decoder = Decoder(hidden_unit)

    def build(self, input_shape):
        super(PointerLSTM, self).build(input_shape)
        self.input_spec = [layers.InputSpec(shape=input_shape)]

    def call(self, x, training=None, mask=None, states=None):
        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1]-1, :]
        x_input = tf.repeat(x_input, input_shape[1])

        if states:
            initial_states = states
        else:
            initial_states = self.decoder.get_initial_state(x_input)

        constants = []
        pre_processed_input, _, constants = self.decoder.process_inputs(x_input,
                                                                        initial_states,
                                                                        constants)

        constants.append(en_seq)
        last_output, outputs, states = layers.RNN

