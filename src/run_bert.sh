#!/usr/bin/bash

root=$(git rev-parse --show-toplevel)

export BERT_BASE_DIR=${root}/data/uncased_L-12_H-768_A-12
export GLUE_DIR=${root}/data/glue_data


python ${root}/src/run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=${GLUE_DIR}/MRPC \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=${BERT_BASE_DIR}/bert_config_dev.json \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --max_seq_length=35 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --output_dir=/tmp/mrpc_output/

