import collections
import copy
import json
import math
import re
import numpy as np
import six

import tensorflow as tf
# from tensorflow.python import keras
from tensorflow import keras
from pathlib import Path


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with Path(json_file).open("r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertEmbeddings(keras.Model):
    """Perform embedding lookup on the word ids."""

    def __init__(self, vocab_size, hidden_size, initializer_range, use_one_hot_embeddings, type_vocab_size,
                 hidden_dropout_prob, max_position_embeddings, dtype, *args, **kwargs):
        super().__init__()

        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._initializer_range = initializer_range
        self._use_one_hot_embeddings = use_one_hot_embeddings
        self._type_vocab_size = type_vocab_size
        self._max_position_embeddings = max_position_embeddings

        self._word_embeddings = keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_size,
                                                       embeddings_initializer=create_initializer(
                                                           self._initializer_range),
                                                       dtype=dtype)

        self._position_embeddings = keras.layers.Embedding(input_dim=max_position_embeddings,
                                                           output_dim=hidden_size,
                                                           embeddings_initializer=create_initializer(
                                                               self._initializer_range),
                                                           dtype=dtype)

        self._segment_embeddings = keras.layers.Embedding(input_dim=type_vocab_size, output_dim=hidden_size,
                                                          embeddings_initializer=create_initializer(
                                                              self._initializer_range),
                                                          dtype=dtype)
        self._dropout = dropout(hidden_dropout_prob)

    def call(self, inputs, training=None, mask=None):
        input_ids = inputs[0]
        position_ids = inputs[1]
        token_type_ids = inputs[2]

        embeddings = self._word_embeddings(input_ids) \
                     + self._position_embeddings(position_ids) \
                     + self._segment_embeddings(token_type_ids)

        return self._dropout(embeddings, training)


class BertEncoder(keras.Model):
    def __init__(self,
                 hidden_size,
                 initializer_range,
                 hidden_dropout_prob,
                 num_hidden_layers,
                 num_attention_heads,
                 intermediate_size,
                 hidden_act,
                 attention_probs_dropout_prob,
                 dtype,
                 *args, **kwargs):
        super().__init__()
        self._dtype = dtype

        self.multilayer_transformer = MultilayerTransformer(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            intermediate_act_fn=get_activation(hidden_act),
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_all_layers=True,
            dtype=dtype)

    def call(self, inputs, training=None, mask=None):
        input_ids = inputs[0]
        input_mask = inputs[1]

        # attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask, dtype=self._dtype)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        return self.multilayer_transformer(input_ids)


class BertPooler(keras.Model):
    def __init__(self, hidden_size, initializer_range, dtype, *args, **kwargs):
        super().__init__()
        self._hidden_size = hidden_size
        self._initializer_range = initializer_range

        self.pooled_dense = keras.layers.Dense(units=hidden_size,
                                               activation=tf.tanh,
                                               kernel_initializer=create_initializer(initializer_range),
                                               dtype=dtype)

    def call(self, inputs, training=None, mask=None):
        first_token_tensor = tf.squeeze(inputs[:, 0:1, :], axis=1)
        return self.pooled_dense(first_token_tensor)


class BertModel(keras.Model):
    """BERT model ("Bidirectional Encoder Representations from Transformers")."""

    def __init__(self,
                 config,
                 is_training,
                 batch_size,
                 seq_length,
                 dtype,
                 use_one_hot_embeddings=False,
                 get_pooled_output=True,
                 *args,
                 **kwargs):
        super().__init__()
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self._batch_size = batch_size
        self._seq_length = seq_length
        self._dtype = dtype

        self.get_pooled_output = get_pooled_output

        self.embeddings = BertEmbeddings(use_one_hot_embeddings=use_one_hot_embeddings, dtype=dtype, **config.__dict__)
        self.encoder = BertEncoder(use_one_hot_embeddings=use_one_hot_embeddings, dtype=dtype, **config.__dict__)
        self.pooler = BertPooler(dtype=dtype, **config.__dict__)

    def call(self, inputs, training=None, mask=None):
        input_ids = inputs[0]
        input_mask = inputs[1]
        position_ids = inputs[2]
        token_type_ids = inputs[3]

        int_dtype = tf.int32 if self._dtype is tf.float32 else tf.int16

        if input_mask is None:
            input_mask = tf.ones(shape=[self._batch_size, self._seq_length], dtype=int_dtype)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[self._batch_size, self._seq_length], dtype=int_dtype)

        x_embds = self.embeddings([input_ids, position_ids, token_type_ids])
        x_encoder = self.encoder([x_embds, input_mask])
        if self.get_pooled_output:
            return self.pooler(x_encoder[-1])
        else:
            return x_encoder


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
      activation_string: String name of the activation function.

    Returns:
      A Python function corresponding to the activation function. If
      `activation_string` is None, empty, or "linear", this will return None.
      If `activation_string` is not a string, it will return `activation_string`.

    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names


def dropout(dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return keras.layers.Lambda(lambda x: x)

    return keras.layers.Dropout(dropout_prob)


def layer_norm():
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    return x


def norm_and_dropout(dropout_prob):
    #     """Runs layer normalization followed by dropout."""
    return keras.Sequential(layers=[layer_norm, dropout_prob(dropout_prob)])


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return keras.initializers.TruncatedNormal(mean=0.0, stddev=initializer_range, seed=None)

    # def embedding_lookup(input_ids,
    #                      vocab_size,
    #                      embedding_size=128,
    #                      initializer_range=0.02,
    #                      word_embedding_name="word_embeddings",
    #                      use_one_hot_embeddings=False):
    #     """Looks up words embeddings for id tensor.
    #
    #     Args:
    #       input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
    #         ids.
    #       vocab_size: int. Size of the embedding vocabulary.
    #       embedding_size: int. Width of the word embeddings.
    #       initializer_range: float. Embedding initialization range.
    #       word_embedding_name: string. Name of the embedding table.
    #       use_one_hot_embeddings: bool. If True, use one-hot method for word
    #     embeddings. If False, use `tf.gather()`.
    #
    #     Returns:
    #       float Tensor of shape [batch_size, seq_length, embedding_size].
    #     """
    #     This function assumes that the input is of shape [batch_size, seq_length,
    #     num_inputs].
    #
    #     If the input is a 2D tensor of shape [batch_size, seq_length], we
    #     reshape to [batch_size, seq_length, 1].
    # if input_ids.shape.ndims == 2:
    #     input_ids = tf.expand_dims(input_ids, axis=[-1])
    #
    # embedding_table = tf.get_variable(
    #     name=word_embedding_name,
    #     shape=[vocab_size, embedding_size],
    #     initializer=create_initializer(initializer_range))
    #
    # flat_input_ids = tf.reshape(input_ids, [-1])
    # if use_one_hot_embeddings:
    #     one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    #     output = tf.matmul(one_hot_input_ids, embedding_table)
    # else:
    #     output = tf.gather(embedding_table, flat_input_ids)
    #
    # input_shape = get_shape_list(input_ids)
    #
    # output = tf.reshape(output,
    #                     input_shape[0:-1] + [input_shape[-1] * embedding_size])
    # return output, embedding_table

    # def embedding_postprocessor(input_tensor,
    #                             use_token_type=False,
    #                             token_type_ids=None,
    #                             token_type_vocab_size=16,
    #                             token_type_embedding_name="token_type_embeddings",
    #                             use_position_embeddings=True,
    #                             position_embedding_name="position_embeddings",
    #                             initializer_range=0.02,
    #                             max_position_embeddings=512,
    #                             dropout_prob=0.1):
    #     """Performs various post-processing on a word embedding tensor.
    #
    #     Args:
    #       input_tensor: float Tensor of shape [batch_size, seq_length,
    #         embedding_size].
    #       use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    #       token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
    #         Must be specified if `use_token_type` is True.
    #       token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    #       token_type_embedding_name: string. The name of the embedding table variable
    #         for token type ids.
    #       use_position_embeddings: bool. Whether to add position embeddings for the
    #         position of each token in the sequence.
    #       position_embedding_name: string. The name of the embedding table variable
    #         for positional embeddings.
    #       initializer_range: float. Range of the weight initialization.
    #       max_position_embeddings: int. Maximum sequence length that might ever be
    #         used with this model. This can be longer than the sequence length of
    #         input_tensor, but cannot be shorter.
    #       dropout_prob: float. Dropout probability applied to the final output tensor.
    #
    #     Returns:
    #       float tensor with same shape as `input_tensor`.
    #
    #     Raises:
    #       ValueError: One of the tensor shapes or input values is invalid.
    #     """
    #     input_shape = get_shape_list(input_tensor, expected_rank=3)
    #     batch_size = input_shape[0]
    #     seq_length = input_shape[1]
    #     width = input_shape[2]
    #
    #     output = input_tensor
    #
    #     if use_token_type:
    #         if token_type_ids is None:
    #             raise ValueError("`token_type_ids` must be specified if"
    #                              "`use_token_type` is True.")
    #         token_type_table = tf.get_variable(
    #             name=token_type_embedding_name,
    #             shape=[token_type_vocab_size, width],
    #             initializer=create_initializer(initializer_range))
    #         This vocab will be small so we always do one-hot here, since it is always
    #         faster for a small vocabulary. # flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    # one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    # token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    # token_type_embeddings = tf.reshape(token_type_embeddings,
    #                                    [batch_size, seq_length, width])
    # output += token_type_embeddings
    #
    # if use_position_embeddings:
    #     assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    #     with tf.control_dependencies([assert_op]):
    #         full_position_embeddings = tf.get_variable(
    #             name=position_embedding_name,
    #             shape=[max_position_embeddings, width],
    #             initializer=create_initializer(initializer_range))
    #         Since the position embedding table is a learned variable, we create it
    #         using a (long) sequence length `max_position_embeddings`. The actual
    #         sequence length might be shorter than this, for faster training of
    #         tasks that do not have long sequences.
    #
    #         So `full_position_embeddings` is effectively an embedding table
    #         for position [0, 1, 2, ..., max_position_embeddings-1], and the current
    #         sequence has positions [0, 1, 2, ... seq_length-1], so we can just
    #         perform a slice.
    # position_embeddings = tf.slice(full_position_embeddings, [0, 0],
    #                                [seq_length, -1])
    # num_dims = len(output.shape.as_list())

    # Only the last two dimensions are relevant (`seq_length` and `width`), so
    # we broadcast among the first dimensions, which is typically just
    # the batch size.
    # position_broadcast_shape = []
    # for _ in range(num_dims - 2):
    #     position_broadcast_shape.append(1)
    # position_broadcast_shape.extend([seq_length, width])
    # position_embeddings = tf.reshape(position_embeddings,
    #                                  position_broadcast_shape)
    # output += position_embeddings


#
# output = norm_and_dropout(output, dropout_prob)
# return output


def create_attention_mask_from_input_mask(from_tensor, to_mask, dtype):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), dtype=dtype)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=dtype)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


class Attention(keras.Model):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width].
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      size_per_head: int. Size of each attention head.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      initializer_range: float. Range of the weight initializer.
      do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
        * from_seq_length, num_attention_heads * size_per_head]. If False, the
        output will be of shape [batch_size, from_seq_length, num_attention_heads
        * size_per_head].
      batch_size: (Optional) int. If the input is 2D, this might be the batch size
        of the 3D version of the `from_tensor` and `to_tensor`.
      from_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `from_tensor`.
      to_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `to_tensor`.

    Returns:
      float Tensor of shape [batch_size, from_seq_length,
        num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        true, this will be of shape [batch_size * from_seq_length,
        num_attention_heads * size_per_head]).

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def __init__(self,
                 attention_mask=None,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02,
                 do_return_2d_tensor=False,
                 batch_size=None,
                 from_seq_length=None,
                 to_seq_length=None,
                 dtype=tf.float32,
                 *args, **kwargs):
        super().__init__()
        self._attention_mask = attention_mask
        self._num_attention_heads = num_attention_heads
        self._size_per_head = size_per_head
        self._attention_probs_dropout_prob = attention_probs_dropout_prob
        self._do_return_2d_tensor = do_return_2d_tensor
        self._batch_size = batch_size
        self._from_seq_length = from_seq_length
        self._to_seq_length = to_seq_length
        self._initializer_range = initializer_range
        self._dtype = dtype

        dense_units = num_attention_heads * size_per_head

        self.query_dense = keras.layers.Dense(units=dense_units,
                                              activation=query_act,
                                              kernel_initializer=create_initializer(initializer_range),
                                              dtype=dtype)

        self.key_dense = keras.layers.Dense(units=dense_units,
                                            activation=key_act,
                                            kernel_initializer=create_initializer(initializer_range),
                                            dtype=dtype)

        self.value_dense = keras.layers.Dense(units=dense_units,
                                              activation=value_act,
                                              kernel_initializer=create_initializer(initializer_range),
                                              dtype=dtype)

        self.dropout_attention_probs = dropout(attention_probs_dropout_prob)

    def call(self, inputs, training=None, mask=None):
        from_tensor = inputs[0]
        to_tensor = inputs[1]

        from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
        to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

        if len(from_shape) != len(to_shape):
            raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")

        batch_size = self._batch_size
        from_seq_length = self._from_seq_length
        to_seq_length = self._to_seq_length

        if len(from_shape) == 3:
            batch_size = from_shape[0]
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if batch_size is None or from_seq_length is None or to_seq_length is None:
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified.")

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)

        # `query_layer` = [B*F, N*H]
        query_layer = self.query_dense(from_tensor_2d)

        # `key_layer` = [B*T, N*H]
        key_layer = self.key_dense(to_tensor_2d)

        # `value_layer` = [B*T, N*H]
        value_layer = self.value_dense(to_tensor_2d)

        def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
            output_tensor = tf.reshape(
                input_tensor, [batch_size, seq_length, num_attention_heads, width])

            output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
            return output_tensor

        # `query_layer` = [B, N, F, H]
        query_layer = transpose_for_scores(query_layer,
                                           batch_size, self._num_attention_heads, from_seq_length,
                                           self._size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = transpose_for_scores(key_layer,
                                         batch_size, self._num_attention_heads, to_seq_length,
                                         self._size_per_head)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(self._size_per_head)))

        if self._attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(self._attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, self._dtype)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout_attention_probs(attention_probs, training)

        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, self._num_attention_heads, self._size_per_head])

        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if self._do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, self._num_attention_heads * self._size_per_head])
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, self._num_attention_heads * self._size_per_head])

        return context_layer


class TransformerNormalizedSelfAttention(Attention):
    def __init__(self, hidden_size, hidden_dropout_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_output = keras.layers.Dense(units=hidden_size, dtype=self._dtype,
                                               kernel_initializer=create_initializer(self._initializer_range))
        self.dropout_output = dropout(hidden_dropout_prob)
        self.normalize = layer_norm()

    def call(self, inputs, training=None, mask=None):
        layer_input = inputs

        attention_output = super().call([layer_input, layer_input], training)
        attention_output = self.dense_output(attention_output)
        attention_output = self.dropout_output(attention_output, training)
        attention_output = self.normalize(attention_output + layer_input)  # note that residual connection here
        return attention_output


class TransformerOutputDense(keras.Model):
    def __init__(self,
                 hidden_size_intermediate,
                 hidden_size,
                 hidden_dropout_prob,
                 initializer_range,
                 intermediate_act_fn=gelu,
                 dtype=tf.float32,
                 *args, **kwargs):
        super().__init__()

        # The activation is only applied to the "intermediate" hidden layer.
        self.dense_intermediate = keras.layers.Dense(units=hidden_size_intermediate,
                                                     kernel_initializer=create_initializer(initializer_range),
                                                     activation=intermediate_act_fn,
                                                     dtype=dtype)

        self.dense_output = keras.layers.Dense(units=hidden_size,
                                               kernel_initializer=create_initializer(initializer_range),
                                               dtype=dtype)

        self.dropout = dropout(hidden_dropout_prob)
        self.normalize = layer_norm()

    def call(self, inputs, training=None, mask=None):
        layer_input = inputs

        intermediate_output = self.dense_intermediate(layer_input)

        # Down-project back to `hidden_size` then add the residual.
        layer_output = self.dense_output(intermediate_output)
        layer_output = self.dropout(layer_output, training)
        layer_output = self.normalize(layer_output + layer_input)

        return layer_output


class Transformer(keras.Model):
    def __init__(self,
                 attention_mask=None,
                 hidden_size=768,
                 intermediate_size=3072,
                 intermediate_act_fn=gelu,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 dtype=tf.float32,
                 *args, **kwargs):
        super().__init__()

        self.attention = TransformerNormalizedSelfAttention(
            attention_mask=attention_mask,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            dtype=dtype)

        self.dense = TransformerOutputDense(
            hidden_size_intermediate=intermediate_size,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            intermediate_act_fn=intermediate_act_fn,
            initializer_range=initializer_range,
            dtype=dtype)

    def call(self, inputs, training=None, mask=None):
        input_tensor = inputs
        x = self.attention(input_tensor)
        x = self.dense(x)
        return x


class MultilayerTransformer(keras.Model):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers or just the final
        layer.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.

    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """

    def __init__(self,
                 attention_mask=None,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 intermediate_act_fn=gelu,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 do_return_all_layers=False,
                 *args, **kwargs):
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.hidden_size = hidden_size
        self.do_return_all_layers = do_return_all_layers

        self.transformer_layers = [Transformer(attention_mask=attention_mask,
                                               hidden_size=hidden_size,
                                               intermediate_size=intermediate_size,
                                               intermediate_act_fn=intermediate_act_fn,
                                               hidden_dropout_prob=hidden_dropout_prob,
                                               attention_probs_dropout_prob=attention_probs_dropout_prob,
                                               initializer_range=initializer_range,
                                               dtype=tf.float32) for _ in range(num_hidden_layers)]

    def call(self, inputs, training=None, mask=None):
        input_tensor = inputs

        input_shape = get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]

        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        if input_width != self.hidden_size:
            raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                             (input_width, self.hidden_size))

        # We keep the representation as a 2D tensor to avoid re-shaping it back and
        # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        # help the optimizer.
        # prev_output = reshape_to_matrix(input_tensor)
        prev_output = input_tensor

        all_layer_outputs = []

        for transformer_layer in self.transformer_layers:
            transformer_output = transformer_layer(prev_output)
            all_layer_outputs.append(transformer_output)
            prev_output = transformer_output

        if self.do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = reshape_from_matrix(prev_output, input_shape)
            return final_output


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """

    # if name is None:
    #     name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    # if name is None:
    #     name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        # scope_name = tf.get_variable_scope().name
        raise ValueError(
            f"For the tensor, the actual rank "
            f"{actual_rank} (shape = {str(tensor.shape)}) is not equal to the expected rank {str(expected_rank)}")
