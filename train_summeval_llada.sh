#!/bin/bash
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="4,5,6,7"
python main.py \
  mode=train \
  noise=linear \
  model=llada8B \
  backbone=llada \
  strategy=single_device \
  data=summeval \
  data.tokenizer_name_or_path=GSAI-ML/LLaDA-8B-Base \
  eval.checkpoint_path=GSAI-ML/LLaDA-8B-Base \
  loader.global_batch_size=2 \
  loader.eval_batch_size=1 \
  strategy.device=0