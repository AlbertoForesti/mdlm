#!/bin/bash

# Export common environment variables
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

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
      sleep 30
    fi
  done
  
  # Run in background with & to enable parallelism
  (
    echo "Starting run with p_random = $p on device $device"
    python main.py --config-name=config_genomics \
      mode=train \
      noise=loglinear \
      model=caduceus1k \
      lr_scheduler=constant \
      +model.hidden_size=256 \
      +model.regression_type=attention \
      backbone=caduceus \
      parameterization=subs \
      mine_loss=nishiyama \
      strategy=single_device \
      data=genomic \
      eval.generate_samples=False \
      eval.checkpoint_path=kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
      training.compute_entropy=False \
      eval.compute_entropy=False \
      loader.global_batch_size=128 \
      loader.batch_size=128 \
      loader.eval_batch_size=128 \
      dora.enabled=False \
      optim.lr=1e-3 \
      strategy.device=$device \
      data.p_random=$p \
      wandb.notes="Caduceus 1k consistency infonce" \
      wandb.group="consistency_dna_infosedd_debug_accuracy_23_04_2025" > "output_logs/${current_date}/log_consistency_dna_infosedd_debug_accuracy_prandom=${p}_device_${device}.txt" 2>&1
    echo "Completed run with p_random = $p on device $device"
  ) &
  
  # Short delay to ensure nvidia-smi has time to update
  sleep 30
done

# Wait for all background jobs to finish
wait
echo "All runs completed!"