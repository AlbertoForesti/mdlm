#!/bin/bash
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
python main.py -m \
  mode=train \
  train_marginal=True,False \
  data.random_variable.seq_length=1,4,16,64 \
  backbone=unetmlp \
  strategy=single_device \
  wandb.group=categorical_long_vector