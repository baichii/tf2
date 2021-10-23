"""
@创建日期 ：2021/10/19
@修改日期 ：2021/10/19
@作者 ：jzj
@功能 ：transformer 推理
"""


import tensorflow as tf


class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length=20):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()
        encoder_input = sentence

        start_end = self.tokenizers.en.tokenize([""])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer([encoder_input, output], training=False)

            predictions = predictions[:, -1:, :]

            prediction_id = tf.argmax(predictions, axis=-1)

            output_array = output_array.write(i+1, prediction_id[0])

            if prediction_id == end:
                break

        output = tf.transpose(output_array.stack())
        text = self.tokenizers.en.detokenize(output)[0]

        tokens = self.tokenizers.en.lookup(output)[0]

        _, attention_weights = self.transformer([encoder_input, output[:, :-1]], training=False)

        return text, tokens, attention_weights


class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (res, tokens, attention_weights) = self.translator(sentence, max_length=100)
        return res

