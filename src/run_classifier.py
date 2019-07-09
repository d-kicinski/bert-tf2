import argparse
import collections
import logging
import os
from pathlib import Path

import tensorflow as tf
from dataclasses import dataclass
from sklearn.metrics import f1_score
from tensorflow import keras

import datautils
from bert import modeling
from bert import tokenization
from optimizers import AdamWeightDecayOptimizer, PolynomialDecayWithWarmup
from utils import CheckpointLoader

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
    learning_rate: float = 1e-6
    train_epochs: int = 6
    batch_size: int = 16
    warmup_proportion: float = 0.1
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

    def call(self, inputs, **kwargs):
        # input_ids, input_mask, segment_ids = inputs[0], inputs[1], inputs[2]
        x = self.bert_model(inputs)
        x = self.dropout(x)
        x = self.dense(x)
        return x


def cross_entropy_loss(y_true, y_pred):
    # return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False))

    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)


def setup_tensorflow():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def main():
    sanity_check()
    setup_tensorflow()

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

    test_examples = processor.get_test_examples(FLAGS.data_dir)
    test_file = os.path.join(FLAGS.output_dir, "test.tf_record")
    file_based_convert_examples_to_features(test_examples,
                                            label_list,
                                            FLAGS.max_seq_length,
                                            tokenizer,
                                            test_file)
    test_input_fn = file_based_input_fn_builder(input_file=test_file,
                                                seq_length=FLAGS.max_seq_length,
                                                is_training=False,
                                                drop_remainder=True)

    batch_i, batch = enumerate(train_input_fn()).__next__()
    input_tensors = [batch['input_ids'], batch['input_mask'], batch['position_ids'], batch['segment_ids']]

    bert = BertTextClassifier(bert_config=bert_config,
                              is_training=True,
                              num_labels=len(label_list),
                              use_one_hot_embeddings=False,
                              dtype=tf.float32)
    _ = bert(input_tensors)

    bert = CheckpointLoader.load_google_bert(model=bert,
                                             max_seq_len=FLAGS.max_seq_length,
                                             init_checkpoint=os.path.join(FLAGS.init_checkpoint, "bert_model.ckpt"),
                                             verbose=False)

    optimizer = AdamWeightDecayOptimizer(
        learning_rate=PolynomialDecayWithWarmup(initial_learning_rate=Params.learning_rate,
                                                end_learning_rate=0.0,
                                                grow_steps=int(num_train_steps * Params.warmup_proportion),
                                                decay_steps=num_train_steps,
                                                name="PolynomialDecayWithWarmup"),
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        name="AdamWeightDecayOptimizer")

    train(model=bert, dataset_train=train_input_fn(), optimizer=optimizer, output_dim=len(label_list),
          epochs=Params.train_epochs)


def sanity_check():
    print("Sanity checks:")
    print("\tGPU Available: ", tf.test.is_gpu_available())
    print("\tLD_LIBRARY_PATH: ", os.environ['LD_LIBRARY_PATH'])


def evaluate(model: keras.Model, dataset: tf.data.Dataset) -> float:
    results = []
    labels = []
    for batch_i, batch in enumerate(dataset):
        input_tensors = [batch['input_ids'],
                         batch['input_mask'],
                         batch['position_ids'],
                         batch['segment_ids']]

        result = model(input_tensors)
        max_probs = tf.math.argmax(result, axis=-1)

        results.extend(max_probs)
        labels.extend(batch['label_ids'])
    return f1_score(y_true=labels, y_pred=results)


@tf.function
def train_step(model, optimizer, loss_fn, input_tensors, labels, dtype):
    with tf.GradientTape(persistent=False) as tape:
        prediction = model(input_tensors)
        loss = loss_fn(labels, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    grad_global_norm = tf.linalg.global_norm(gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, grad_global_norm


def train(model: keras.Model,
          dataset_train: tf.data.Dataset,
          optimizer: keras.optimizers.Optimizer,
          output_dim: int,
          epochs: int,
          dtype=tf.float32):
    avg_loss = tf.keras.metrics.Mean(name='loss', dtype=dtype)
    avg_grad_norm = tf.keras.metrics.Mean(name='grad_norm', dtype=dtype)
    log_frequency = 100

    writer = tf.summary.create_file_writer(Params.output_dir)
    compute_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    with writer.as_default():
        for batch_i, batch in enumerate(dataset_train.repeat(epochs)):
            input_tensors = [batch['input_ids'],
                             batch['input_mask'],
                             batch['position_ids'],
                             batch['segment_ids']]
            labels = tf.one_hot(indices=batch['label_ids'], depth=output_dim)
            loss, grad_norm = train_step(model, optimizer, compute_loss, input_tensors, labels, dtype)
            avg_loss.update_state(loss)
            avg_grad_norm.update_state(grad_norm)

            if tf.equal(optimizer.iterations % log_frequency, 0):
                tf.print(f"batch_i: {batch_i + 1}, loss: {loss}")
                tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
                tf.summary.scalar('grad_norm', avg_grad_norm.result(), step=optimizer.iterations)
                avg_loss.reset_states()
                avg_grad_norm.reset_states()
                model.save_weights(os.path.join(Params.output_dir, f"model.cpkt-{batch_i}"))

        tf.print(f"batch_i: {batch_i + 1}, loss: {loss}")
        tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
        tf.summary.scalar('grad_norm', avg_grad_norm.result(), step=optimizer.iterations)
        avg_loss.reset_states()
        avg_grad_norm.reset_states()
        model.save_weights(os.path.join(Params.output_dir, f"model.cpkt-{batch_i}"))


if __name__ == "__main__":
    main()
