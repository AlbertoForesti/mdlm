#!/bin/bash
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="4,5,6,7"
python main.py --config-name=config_entropy\
  mode=train \
  model.length=400 \
  backbone=unetmlp \
  strategy=single_device \
  strategy.device=2