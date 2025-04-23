#!/bin/bash
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python main.py \
  mode=train \
  +model.regression_type=attention \
  backbone=hf_dit \
  strategy=single_device \
  data=summeval \
  parameterization=mine \
  training.compute_entropy=False \
  eval.compute_entropy=False \
  data.p_random=0.9 \
  train_marginal=False \
  callbacks.checkpoint_monitor.monitor=trainer/loss \
  data.train=/home/foresti/mdlm/model_summaries_cleaned/M22/outputs_cnndm.aligned.paired.jsonl \
  data.valid=/home/foresti/mdlm/model_summaries_cleaned/M22/outputs_cnndm.aligned.paired.jsonl \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  loader.global_batch_size=128 \
  strategy.device=4 \
  eval.generate_samples=False \
  wandb.notes="Mine experiments p decaying" \
  time_conditioning=False