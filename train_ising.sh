#!/bin/bash
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
python main.py --config-name=config_entropy\
  mode=train \
  model.length=400 \
  backbone=unetmlp \
  strategy=single_device \
  strategy.device=2