#!/bin/bash

# Force decimal point to be a period, not a comma
export LC_NUMERIC="C"

# Set environment variables
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Define p_values array
p_values=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Function to find inactive GPUs
get_inactive_gpus() {
  # Get GPU utilization and memory usage
  # Consider a GPU inactive if it has less than 500MB memory used
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | 
  awk '{if ($2 < 500) print $1}' |
  tr -d ',' | # Remove any commas
  tr '\n' ' '
}

# Get current date in YYYY_MM_DD format
current_date=$(date +%Y_%m_%d)
echo "Running script with date: $current_date"
echo "Starting parallel runs for p_values from 0.0 to 1.0"

# Create output directory for logs
mkdir -p "output_logs/${current_date}"

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
      echo "No inactive GPUs available, waiting 30 seconds..."
      sleep 300
    fi
  done
  
  # Run in background with & to enable parallelism
  (
    echo "Starting run with p_random = $p on device $device"
    python main.py \
      mode=train \
      backbone=hf_dit \
      strategy=single_device \
      data=summeval \
      +data.seq_to_seq_exp=False \
      variant=c \
      variant_c_target=1 \
      data.train=/home/foresti/mdlm/model_summaries_cleaned/M22/outputs_cnndm.aligned.paired.jsonl \
      data.valid=/home/foresti/mdlm/model_summaries_cleaned/M22/outputs_cnndm.aligned.paired.jsonl \
      eval.checkpoint_path=kuleshov-group/mdlm-owt \
      loader.global_batch_size=128 \
      strategy.device=$device \
      data.p_random=$p \
      dora.enabled=False \
      parameterization=subs \
      +model.regression_type=attention \
      training.compute_entropy=False \
      eval.compute_entropy=False \
      time_conditioning=False \
      wandb.group=consistency_summaries_infosedd_c_${current_date} \
      eval.generate_samples=False > "output_logs/${current_date}/log_consistency_summaries_infosedd_c_prandom=${p}_device_${device}.txt" 2>&1
    echo "Completed run with p_random = $p on device $device"
  ) &
  
  # Short delay to ensure nvidia-smi has time to update
  sleep 300
done

# Wait for all background processes to finish
wait

echo "All runs completed!"