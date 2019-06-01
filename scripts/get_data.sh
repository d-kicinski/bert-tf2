#!/usr/bin/env bash

root=$(git rev-parse --show-toplevel)
data_dir=${root}/resources/dataset
mkdir -p ${data_dir}

python ${root}/scripts/get_glue_data.py --data_dir ${data_dir} --tasks all

