#!/bin/bash
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="4,5,6,7"
python main.py -m \
  --config-name=config_synthetic \
  mode=train \
  model=unetmlp007k \
  +model.regression_type=attention \
  parameterization=subs \
  train_marginal=False \
  time_conditioning=False \
  training.compute_entropy=False \
  eval.compute_entropy=False \
  train_marginal=False \
  data.random_variable.seq_length=1 \
  backbone=unetmlp \
  strategy=single_device \
  strategy.device=0 \
  wandb.group=categorical_long_vector_mine \
  eval.disable_ema=True \