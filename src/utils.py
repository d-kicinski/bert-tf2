import logging
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np

from bert.modeling import BertConfig
from bert.modeling import BertModel

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class CheckpointLoader:

    @staticmethod
    def _get_embedding_index(parts):
        n = parts[-1]
        if n == 'token_type_embeddings':
            w_id = 1
        elif n == 'position_embeddings':
            w_id = 2
        elif n == 'word_embeddings':
            w_id = 0
        elif n == 'gamma':
            w_id = 3
        elif n == 'beta':
            w_id = 4
        else:
            raise ValueError()
        return w_id

    @staticmethod
    def load_google_bert(model: BertModel,
                         init_checkpoint: str,
                         max_seq_len: int,
                         layer_num: int = 12,
                         verbose: bool = False):

        var_names = tf.train.list_variables(init_checkpoint)
        checkpoint = tf.train.load_checkpoint(init_checkpoint)

        weights = [np.zeros(w.shape) for w in model.weights]

        # print("Keras model weights:")
        with Path("our.txt").open('w') as file_out:
            for i_weight, weight in enumerate(model.weights):
                file_out.write(f"{i_weight}\t{weight.name}\n")
        #
        # print("Google checkpoint weights:")
        # for name, _ in var_names:
        #     print(f"\t{name}")

        for var_name, _ in var_names:
            w_id = None
            qkv = None
            unsqueeze = False
            parts = var_name.split('/')
            first_vars_size = 5
            variables_in_layer_num = 16
            if parts[1] == 'embeddings':
                w_id = CheckpointLoader._get_embedding_index(parts)

            elif parts[2].startswith('layer_'):
                layer_number = int(parts[2].split('_')[1])
                if parts[3] == 'attention':
                    if parts[-1] == 'beta':
                        w_id = first_vars_size + layer_number * variables_in_layer_num + 9
                    elif parts[-1] == 'gamma':
                        w_id = first_vars_size + layer_number * variables_in_layer_num + 8
                    elif parts[-2] == 'dense':
                        if parts[-1] == 'bias':
                            w_id = first_vars_size + layer_number * variables_in_layer_num + 7
                        elif parts[-1] == 'kernel':
                            w_id = first_vars_size + layer_number * variables_in_layer_num + 6
                            unsqueeze = True
                        else:
                            raise ValueError()
                    elif parts[-2] == 'key':
                        w_id = first_vars_size + layer_number * variables_in_layer_num + (
                            0 if parts[-1] == 'kernel' else 1) + 2
                        n = model.weights[w_id].name
                        unsqueeze = parts[-1] == 'kernel'
                        qkv = parts[-2][0]
                    elif parts[-2] == 'value':
                        w_id = first_vars_size + layer_number * variables_in_layer_num + (
                            0 if parts[-1] == 'kernel' else 1) + 4
                        n = model.weights[w_id].name
                        unsqueeze = parts[-1] == 'kernel'
                        qkv = parts[-2][0]
                    elif parts[-2] == 'query':
                        w_id = first_vars_size + layer_number * variables_in_layer_num + (
                            0 if parts[-1] == 'kernel' else 1) + 0
                        n = model.weights[w_id].name
                        unsqueeze = parts[-1] == 'kernel'
                        qkv = parts[-2][0]
                    else:
                        raise ValueError()
                elif parts[3] == 'intermediate':
                    if parts[-1] == 'bias':
                        w_id = first_vars_size + layer_number * variables_in_layer_num + 11
                    elif parts[-1] == 'kernel':
                        w_id = first_vars_size + layer_number * variables_in_layer_num + 10
                        unsqueeze = True
                    else:
                        raise ValueError()
                elif parts[3] == 'output':
                    if parts[-1] == 'beta':
                        w_id = first_vars_size + layer_number * variables_in_layer_num + 15
                    elif parts[-1] == 'gamma':
                        w_id = first_vars_size + layer_number * variables_in_layer_num + 14
                    elif parts[-1] == 'bias':
                        w_id = first_vars_size + layer_number * variables_in_layer_num + 13
                    elif parts[-1] == 'kernel':
                        w_id = first_vars_size + layer_number * variables_in_layer_num + 12
                        unsqueeze = True
                    else:
                        raise ValueError()
            elif parts[1] == 'pooler':
                if parts[-1] == 'bias':
                    w_id = first_vars_size + (layer_num - 1) * variables_in_layer_num + 17
                elif parts[-1] == 'kernel':
                    w_id = first_vars_size + (layer_number - 1) * variables_in_layer_num + 16
                    unsqueeze = True
                else:
                    raise ValueError()

            if w_id is not None and qkv is None:
                try:
                    if verbose:
                        print(f"w_id: {w_id}\t{w_id - first_vars_size}", var_name, ' -> ', model.weights[w_id].name)
                except IndexError:
                    pass
                if w_id in [0, 1, 2]:  # embeddings
                    if unsqueeze:
                        weights[w_id][:max_seq_len, :] = checkpoint.get_tensor(var_name)[None, :max_seq_len, :]
                    else:
                        weights[w_id][:max_seq_len, :] = checkpoint.get_tensor(var_name)[:max_seq_len, :]
                else:
                    if unsqueeze:
                        weights[w_id][:] = checkpoint.get_tensor(var_name)[None, ...]
                    else:
                        weights[w_id][:] = checkpoint.get_tensor(var_name)
            elif w_id is not None:
                if verbose:
                    print(f"w_id: {w_id}\t{w_id - first_vars_size}", var_name, ' -> ', model.weights[w_id].name)
                p = {'q': 0, 'k': 1, 'v': 2}[qkv]
                if weights[w_id].ndim == 3:
                    dim_size = weights[w_id].shape[1]
                    weights[w_id][0, :, p * dim_size:(p + 1) * dim_size] = checkpoint.get_tensor(
                        var_name) if not unsqueeze else \
                        checkpoint.get_tensor(var_name)[
                            None, ...]
                else:
                    # dim_size = weights[w_id].shape[0] // 3
                    # weights[w_id][p * dim_size:(p + 1) * dim_size] = checkpoint.get_tensor(var_name)
                    weights[w_id] = checkpoint.get_tensor(var_name)
            else:
                if verbose:
                    print('not mapped: ', var_name)
        model.set_weights(weights)
        return model
#
