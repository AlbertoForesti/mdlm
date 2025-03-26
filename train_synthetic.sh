#!/bin/bash
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="4,5,6,7"
python main.py -m \
  --config-name=config_synthetic \
  mode=train \
  parameterization=sedd \
  train_marginal=True,False \
  data.random_variable.seq_length=256 \
  backbone=unetmlp \
  strategy=single_device \
  strategy.device=0 \
  wandb.group=categorical_long_vector \