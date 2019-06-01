#!/usr/bin/env bash

root=$(git rev-parse --show-toplevel)

bert_dir=${root}/resources/models/uncased_L-12_H-768_A-12
dataset_dir=${root}/resources/dataset

output_dir=${root}/output && mkdir -p ${output_dir}

python ${root}/src/run_classifier.py \
  --task_name=MRPC \
  --data_dir=${dataset_dir}/MRPC \
  --vocab_file=${bert_dir}/vocab.txt \
  --bert_config_file=${bert_dir}/bert_config.json \
  --init_checkpoint=${bert_dir}/bert_model.ckpt \
  --max_seq_length=128 \
  --batch_size=32 \
  --learning_rate=2e-5 \
  --train_epochs=3 \
  --output_dir=${output_dir}


