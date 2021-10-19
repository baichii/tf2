"""
@创建日期 ：2021/10/18
@修改日期 ：2021/10/18
@作者 ：jzj
@功能 ：训练
"""

import re
import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text


# 加载数据
# examples, metadata = tfds.load("ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True)
# train_examples, val_examples = examples["train"], examples["validation"]
# train_en = train_examples.map(lambda pt, en: en)
# train_pt = train_examples.map(lambda pt, en: pt)


# 建立词汇表
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    vocab_size=8000,
    reserved_tokens=reserved_tokens,
    bert_tokenizer_params=bert_tokenizer_params,
    learn_params={}
)


def write_vocab_file(file_path, vocab):
    with open(file_path, "w") as f:
        for token in vocab:
            print(token, file=f)


# pt_vocab = bert_vocab.bert_vocab_from_dataset(train_pt.batch(1000).prefetch(2), **bert_vocab_args)
# en_vocab = bert_vocab.bert_vocab_from_dataset(train_en.batch(1000).prefetch(2), **bert_vocab_args)
# write_vocab_file("pt_vocab.txt", vocab=pt_vocab)
# write_vocab_file("en_vocab.txt", vocab=en_vocab)


# 预处理

START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")


def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)


def clean_text(reserved_tokens, token_txt):
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_tokens_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_tokens_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    result = tf.strings.reduce_join(result, separator=" ", axis=-1)
    return result


class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        self.tokenize.get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.string))

        self.detokenize.get_concrete_function(tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.get_reserved_tokens.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_vocab_size.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return clean_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)

# 分词
# pt_tokenizer = text.BertTokenizer("pt_vocab.txt", **bert_tokenizer_params)
# en_tokenizer = text.BertTokenizer("en_vocab.txt", **bert_tokenizer_params)
#
#
# for pt_examples, en_examples in train_examples.batch(3).take(1):
#     print(en_examples)
#     token_batch = en_tokenizer.tokenize(en_examples)
#     for ex in token_batch.to_list():
#         print(ex)
#     break


if __name__ == '__main__':
    tokenizers = tf.Module()
    tokenizers.pt = CustomTokenizer(reserved_tokens, "pt_vocab.txt")
    tokenizers.en = CustomTokenizer(reserved_tokens, "en_vocab.txt")
    model_name = "ted_hrlr_translate_pt_en_converter"
    tf.saved_model.save(tokenizers, model_name)


