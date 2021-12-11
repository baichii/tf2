"""
@创建日期 ：2021/10/16
@修改日期 ：2021/10/16
@作者 ：jzj
@功能 ：https://www.tensorflow.org/text/tutorials/transformer#run_inference
"""

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from transformer.mask import create_look_ahead_mask, create_padding_mask
from transformer.positional_encoder import positional_encoding

tf.random.set_seed(42)


def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)

    scaled_attention_logits = matmul_qk / dk

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print(temp_out.shape, temp_attn.shape)


def demo1():
    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)

    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 5]], dtype=tf.float32)

    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)

    print_out(temp_q, temp_k, temp_v)


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


def demo2():
    mha = MultiHeadAttention(d_model=256, num_heads=4)
    test_data = tf.random.uniform(shape=(1, 60, 256))
    output, attention_weights = mha(test_data, test_data, test_data, mask=None)
    print(output.shape, attention_weights.shape)


def point_wise_feed_forward_network(d_model, dff):
    return Sequential([
        layers.Dense(dff, activation="relu"),
        layers.Dense(d_model)
    ])


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads,  dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        atten_out, _ = self.mha(x, x, x, mask)
        atten_out = self.dropout1(atten_out, training=training)
        output1 = self.norm1(x + atten_out)

        ffn_out = self.ffn(output1)
        ffn_out = self.dropout2(ffn_out, training=training)
        output2 = self.norm2(ffn_out)

        return output2


def demo3():
    el = EncoderLayer(d_model=512, num_heads=4, dff=2048)
    test_data = tf.random.uniform(shape=(1, 60, 512))
    out = el(test_data, training=True, mask=None)
    print(out.shape)


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate=rate)
        self.dropout2 = layers.Dropout(rate=rate)
        self.dropout3 = layers.Dropout(rate=rate)

    def call(self, x, enc_out, training, look_ahead_mask, padding_mask):
        atten_out1, atten_weights1 = self.mha1(x, x, x, look_ahead_mask)
        atten_out1 = self.dropout1(atten_out1, training=training)
        out1 = self.norm1(x + atten_out1)

        atten_out2, atten_weights2 = self.mha2(out1, enc_out, enc_out, padding_mask)
        atten_out2 = self.dropout2(atten_out2, training=training)
        out2 = self.norm2(out1 + atten_out2)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out)
        out3 = self.norm3(out2 + ffn_out)

        return out3, atten_weights1, atten_weights2


def demo4():

    dl = DecoderLayer(512, 8, 2048)

    x = tf.random.uniform(shape=(1, 60, 512))
    enc_out = tf.random.uniform((1, 60, 512))
    out, w1, w2 = dl(x, enc_out, training=True, look_ahead_mask=None, padding_mask=None)
    print(out.shape)
    print(w1.shape)
    print(w2.shape)


class Encoder(layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        """
        Args:
            num_layers: Num of encoder block
            d_model:
            num_heads:
            dff:
            input_vocab_size:
            maximum_position_encoding:
            rate:
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


def demo5():
    en = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=500, maximum_position_encoding=1000)
    test_data = tf.random.uniform((64, 50), dtype=tf.int64, minval=0, maxval=200)
    print(en(test_data, training=True, mask=None).shape)


class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        sel_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x += tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :sel_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, b1, b2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f"decoder_layer{i+1}_b1"] = b1
            attention_weights[f"decoder_layer{i+1}_b2"] = b2

        return x, attention_weights


def demo6():
    de = Decoder(num_layers=2, d_model=512, num_heads=8, dff=2048, target_vocab_size=500, maximum_position_encoding=1000)
    test_data = tf.random.uniform((64, 50), dtype=tf.int64, minval=0, maxval=200)
    enc_out = tf.random.uniform((64, 50, 512))
    out, _ = de(test_data, enc_out, False, None, None)
    print(out.shape)


class Transformer(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = layers.Dense(target_vocab_size)

    def call(self, input, training):
        inp, tar = input

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_mask(inp, tar)

        enc_out = self.encoder(inp, training, enc_padding_mask)

        dec_out, attention_weights = self.decoder(tar, enc_out, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_out)
        return final_output, attention_weights

    def create_mask(self, inp, tar):
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = create_padding_mask(inp)

        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask


def demo7():
    transformer = Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048,
                              input_vocab_size=1100, target_vocab_size=1000,
                              pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((64, 38), minval=0, maxval=200, dtype=tf.int64)
    temp_target = tf.random.uniform((64, 36), minval=0, maxval=200, dtype=tf.int64)

    fn_out, _ = transformer([temp_input, temp_target])
    print(fn_out.shape)


if __name__ == '__main__':
    # demo1()
    demo2()
    # demo7()
    # demo5()
    # demo6()
