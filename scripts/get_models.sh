#!/usr/bin/env bash

root=$(git rev-parse --show-toplevel)
models_dir=${root}/resources/models
mkdir -p ${models_dir}

# BERT model checkpoint
model_name=uncased_L-12_H-768_A-12
wget -c https://storage.googleapis.com/bert_models/2018_10_18/${model_name}.zip -P ${models_dir}
unzip ${models_dir}/${model_name}.zip -d ${models_dir}
rm  ${models_dir}/${model_name}.zip
