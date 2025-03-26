#!/bin/bash
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="4,5,6,7"
python main.py \
  mode=train \
  backbone=hf_dit \
  strategy=single_device \
  data=summeval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  strategy.device=0