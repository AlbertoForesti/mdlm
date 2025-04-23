#!/bin/bash

# Force decimal point to be a period, not a comma
export LC_NUMERIC="C"

# Loop through p_random values from 0.0 to 1.0 with 0.1 increments
for p in $(seq 0.0 0.1 1.0); do
  echo "Starting run with p_random = $p"
  
  # Run the command with the current p_random value
  export HF_HOME="/home/foresti/.cache/huggingface"
  export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
  export HYDRA_FULL_ERROR=1
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  
  python main.py \
    mode=train \
    backbone=hf_dit \
    strategy=single_device \
    data=summeval \
    data.train=/home/foresti/mdlm/model_summaries_cleaned/M22/outputs_cnndm.aligned.paired.jsonl \
    data.valid=/home/foresti/mdlm/model_summaries_cleaned/M22/outputs_cnndm.aligned.paired.jsonl \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    loader.global_batch_size=128 \
    strategy.device=4 \
    data.p_random=$p \
    dora.enabled=False \
    parameterization=mine \
    +model.regression_type=attention \
    training.compute_entropy=False \
    eval.compute_entropy=False \
    time_conditioning=False \
    wandb.group=consistency_summaries_mine \
    eval.generate_samples=False \
  
  echo "Completed run with p_random = $p"
  echo "------------------------"
done

echo "All runs completed!"