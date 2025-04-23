#!/bin/bash

# Get current date in YYYY_MM_DD format
current_date=$(date +%Y_%m_%d)
echo "Running script with date: $current_date"

# Export common environment variables
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Create output directory for logs
mkdir -p "output_logs/${current_date}"

# Define p_values from 0.0 to 1.0
p_values=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Function to find inactive GPUs - UPDATED VERSION
get_inactive_gpus() {
  # Get GPU utilization and memory usage
  # Consider a GPU inactive if it has less than 500MB memory used
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | 
  awk '{if ($2 < 500) print $1}' |
  tr -d ',' | # Remove any commas
  tr '\n' ' '
}

echo "Starting parallel runs for p_values from 0.0 to 1.0"

# Process all p_values
for p in "${p_values[@]}"; do
  # Wait until we have an available GPU
  while true; do
    # Get list of inactive GPUs
    available_gpus=($(get_inactive_gpus))
    
    if [ ${#available_gpus[@]} -gt 0 ]; then
      # Take the first available GPU
      device=${available_gpus[0]}
      echo "Found inactive GPU $device, assigning p_random = $p"
      break
    else
      echo "No inactive GPUs available, waiting 300 seconds..."
      sleep 300
    fi
  done
  
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
      mine_loss=infonce \
      training.compute_entropy=False \
      eval.compute_entropy=False \
      eval.disable_ema=True \
      data.p_random=$p \
      train_marginal=False \
      callbacks.checkpoint_monitor.monitor=trainer/loss \
      data.train=/home/foresti/mdlm/model_summaries_cleaned/M22/outputs_cnndm.aligned.paired.jsonl \
      data.valid=/home/foresti/mdlm/model_summaries_cleaned/M22/outputs_cnndm.aligned.paired.jsonl \
      eval.checkpoint_path=kuleshov-group/mdlm-owt \
      loader.global_batch_size=32 \
      trainer.accumulate_grad_batches=16 \
      strategy.device=$device \
      eval.generate_samples=False \
      wandb.notes="infonce_experiments_p$p" \
      wandb.group=consistency_summaries_infonce_${current_date}_v2 \
      time_conditioning=False > "output_logs/${current_date}/log_infonce_p_random_${p}_device_${device}.txt" 2>&1
    echo "Completed run with p_random = $p on device $device"
  ) &
  
  # Short delay to ensure nvidia-smi has time to update
  sleep 20
done

# Wait for all background jobs to finish
wait
echo "All runs completed!"