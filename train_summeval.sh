#!/bin/bash
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python main.py \
  mode=train \
  backbone=hf_dit \
  strategy=single_device \
  data=summeval \
  data.train=/home/foresti/mdlm/model_summaries_cleaned/M9/outputs_extabs+rl+rerank.aligned.paired.jsonl \
  data.valid=/home/foresti/mdlm/model_summaries_cleaned/M9/outputs_extabs+rl+rerank.aligned.paired.jsonl \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  loader.global_batch_size=128 \
  strategy.device=4 \