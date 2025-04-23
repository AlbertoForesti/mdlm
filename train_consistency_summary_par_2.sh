#!/bin/bash

# Export common environment variables
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Launch jobs in parallel
p_values=(0.8 0.9 1.0)
for i in {0..2}; do
  p=${p_values[$i]}
  device=$i
  
  # Run in background with & to enable parallelism
  (
    echo "Starting run with p_random = $p on device $device"
    python main.py \
      mode=train \
      +model.regression_type=attention \
      backbone=hf_dit \
      strategy=single_device \
      data=summeval \
      parameterization=mine \
      training.compute_entropy=False \
      eval.compute_entropy=False \
      eval.disable_ema=True \
      data.p_random=$p \
      train_marginal=False \
      callbacks.checkpoint_monitor.monitor=trainer/loss \
      data.train=/home/foresti/mdlm/model_summaries_cleaned/M22/outputs_cnndm.aligned.paired.jsonl \
      data.valid=/home/foresti/mdlm/model_summaries_cleaned/M22/outputs_cnndm.aligned.paired.jsonl \
      eval.checkpoint_path=kuleshov-group/mdlm-owt \
      loader.global_batch_size=128 \
      strategy.device=$device \
      eval.generate_samples=False \
      time_conditioning=False > "log_p_random_${p}_device_${device}.txt" 2>&1
    echo "Completed run with p_random = $p on device $device"
  ) &
  
  # Short delay to avoid race conditions
  sleep 2
done

# Wait for all background jobs to finish
wait
echo "All runs completed!"