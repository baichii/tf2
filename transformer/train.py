"""
@创建日期 ：2021/10/18
@修改日期 ：2021/10/18
@作者 ：jzj
@功能 ：
"""

import time
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
from tensorflow.keras import losses, optimizers, metrics
from transformer.attention import Transformer
from transformer.other import CustomSchedule
from transformer.other import loss_function, accuracy_function


def data_pipeline():
    examples, metadata = tfds.load("ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True)
    train_examples, val_examples = examples["train"], examples["validation"]
    tokenizers = tf.saved_model.load("ted_hrlr_translate_pt_en_converter")
    print(tokenizers)

    def token_pairs(pt, en):
        pt = tokenizers.pt.tokenize(pt)
        pt = pt.to_tensor()

        en = tokenizers.en.tokenize(en)
        en = en.to_tensor()
        return pt, en

    def make_batches(ds):
        return ds.cache().shuffle(20000).batch(32).map(token_pairs, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)
    return train_batches, val_batches, tokenizers


def main():
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    train_batches, val_batches, tokenizers = data_pipeline()

    model = Transformer(num_layers=num_layers,
                        d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
                        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
                        pe_input=1000,
                        pe_target=1000,
                        rate=dropout_rate)

    lr = CustomSchedule(d_model=d_model)
    loss_object = losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    train_loss = metrics.Mean(name="train_loss")
    train_accuracy = metrics.Mean(name="train_accuracy")

    checkpoint_path = "./checkpoint/train"

    ckpt = tf.train.Checkpoint(transformer=model,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("last restored")

    EPOCHS = 20

    train_step_signature = [tf.TensorSpec(shape=[None, None], dtype=tf.int64),
                            tf.TensorSpec(shape=[None, None], dtype=tf.int64)]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = model([inp, tar_inp], training=True)
            loss = loss_function(tar_real, predictions, loss_object)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(f"Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f}, Accuracy {train_accuracy.result():.4f}")

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

            print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


if __name__ == '__main__':
    # data_pipeline()
    main()

