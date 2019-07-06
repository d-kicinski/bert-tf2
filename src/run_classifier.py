import collections
import os
import argparse
import sys
import logging

import tensorflow as tf
from tensorflow import keras
from dataclasses import dataclass
from pathlib import Path

from bert import modeling
from bert import optimization
from bert import tokenization
from utils import CheckpointLoader
import datautils

# logging.getLogger().setLevel(logging.DEBUG)

EXAMPLES_NUM_MRPC = 3668

@dataclass
class Params:
    data_dir: str = "resources/dataset/MRPC"
    bert_config_file: str = "resources/models/uncased_L-12_H-768_A-12/bert_config.json"
    task_name: str = "mrpc"
    vocab_file: str = "resources/models/uncased_L-12_H-768_A-12/vocab.txt"
    output_dir: str = "output/dev_mrpc"
    init_checkpoint: str = "resources/models/uncased_L-12_H-768_A-12"
    max_seq_length: int = 128
    learning_rate: float = 2e-5
    train_epochs: int = 3
    batch_size: int = 32
    warmup_proportion: float = 0.1
    save_checkpoints_steps: int = 1000
    iterations_per_loop: int = 1000
    do_lower_case: bool = True


def parse_args():
    parser = argparse.ArgumentParser(description='Tran Bert')
    parser.add_argument('--data_dir', type=str, default=Params.data_dir)
    parser.add_argument('--bert_config_file', type=str, default=Params.bert_config_file)
    parser.add_argument('--task_name', type=str, default=Params.task_name)
    parser.add_argument('--vocab_file', type=str, default=Params.vocab_file)
    parser.add_argument('--output_dir', type=str, default=Params.output_dir)
    parser.add_argument('--init_checkpoint', type=str, default=Params.init_checkpoint)
    parser.add_argument('--max_seq_length', type=int, default=Params.max_seq_length)
    parser.add_argument('--batch_size', type=int, default=Params.batch_size)
    parser.add_argument('--learning_rate', type=float, default=Params.learning_rate)
    parser.add_argument('--train_epochs', type=int, default=Params.train_epochs)
    parser.add_argument('--warmup_proportion', type=float, default=Params.warmup_proportion)
    parser.add_argument('--save_checkpoints_steps', type=int, default=Params.save_checkpoints_steps)
    parser.add_argument('--iterations_per_loop', type=int, default=Params.iterations_per_loop)

    parser.add_argument('--do_lower_case', default=Params.do_lower_case, type=lambda x: (str(x).lower() == 'true'))

    return parser.parse_args()


FLAGS = Params(**parse_args().__dict__)


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        feature = datautils.convert_single_example(ex_index, example, label_list,
                                                   max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["position_ids"] = create_int_feature(feature.positional_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "position_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([], tf.int64),
        "is_real_example": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            example[name] = t

        return example

    def input_fn():
        """The actual input function."""
        # batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.map(lambda record: _decode_record(record, name_to_features)).batch(batch_size=FLAGS.batch_size)

        return d

    return input_fn


class BertTextClassifier(keras.Model):
    """Creates a classification model."""

    def __init__(self, bert_config, is_training, num_labels, use_one_hot_embeddings,
                 batch_size=16, seq_length=30, dropout_prob=0.1, dtype=tf.float32):
        super().__init__()

        self.bert_model = modeling.BertModel(config=bert_config,
                                             is_training=is_training,
                                             use_one_hot_embeddings=use_one_hot_embeddings,
                                             batch_size=batch_size,
                                             seq_length=seq_length,
                                             dtype=dtype)

        self.dropout = keras.layers.Dropout(dropout_prob)
        self.dense = keras.layers.Dense(num_labels,
                                        activation=keras.activations.softmax,
                                        name="dense",
                                        dtype=dtype)

    def call(self, inputs):
        # input_ids, input_mask, segment_ids = inputs[0], inputs[1], inputs[2]
        x = self.bert_model(inputs)
        x = self.dropout(x)
        x = self.dense(x)
        return x

    def call(self, inputs):
        # input_ids, input_mask, segment_ids = inputs[0], inputs[1], inputs[2]
        x = self.bert_model(inputs)
        x = self.dropout(x)
        x = self.dense(x)
        return x


def cross_entropy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True))


def main():
    sanity_check()

    processors = {
        "cola": datautils.ColaProcessor,
        "mnli": datautils.MnliProcessor,
        "mrpc": datautils.MrpcProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    Path(FLAGS.output_dir).mkdir(parents=True, exist_ok=True)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_examples = processor.get_train_examples(FLAGS.data_dir)

    logging.debug(f"train_examples_num: {len(train_examples)}")
    assert len(train_examples) == EXAMPLES_NUM_MRPC
    num_train_steps = int(len(train_examples) / FLAGS.batch_size * FLAGS.train_epochs)


    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(train_examples,
                                            label_list,
                                            FLAGS.max_seq_length,
                                            tokenizer,
                                            train_file)

    train_input_fn = file_based_input_fn_builder(input_file=train_file,
                                                 seq_length=FLAGS.max_seq_length,
                                                 is_training=True,
                                                 drop_remainder=True)

    batch_i, batch = enumerate(train_input_fn()).__next__()
    input_tensors = [batch['input_ids'], batch['input_mask'], batch['position_ids'], batch['segment_ids']]

    with tf.device("gpu:0"):
        bert = BertTextClassifier(bert_config=bert_config,
                                  is_training=True,
                                  num_labels=len(label_list),
                                  use_one_hot_embeddings=False,
                                  dtype=tf.float32)
        _ = bert(input_tensors)

        CheckpointLoader.load_google_bert(model=bert,
                                          max_seq_len=FLAGS.max_seq_length,
                                          init_checkpoint=os.path.join(FLAGS.init_checkpoint, "bert_model.ckpt"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        train(model=bert, dataset=train_input_fn(), optimizer=optimizer, output_dim=len(label_list))


def sanity_check():
    print("GPU Available: ", tf.test.is_gpu_available())


# @tf.function
def train(model: keras.Model, dataset: tf.data.Dataset, optimizer: keras.optimizers.Optimizer, output_dim: int,
          dtype=tf.float32):
    for batch_i, batch in enumerate(dataset):
        input_tensors = [
            batch['input_ids'],
            batch['input_mask'],
            batch['position_ids'],
            batch['segment_ids'],
        ]

        with tf.GradientTape(persistent=True) as tape:
            result = model(input_tensors)
            labels = tf.keras.utils.to_categorical(batch['label_ids'], num_classes=output_dim)
            loss = cross_entropy_loss(y_pred=tf.cast(result, dtype=dtype), y_true=tf.cast(labels, dtype=dtype))

        trainable_variables = model.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)
        grad_global_norm = tf.linalg.global_norm(gradients)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        print(f"batch_i: {batch_i}, loss: {loss}, grad_global_norm: {grad_global_norm}")


if __name__ == "__main__":
    main()
