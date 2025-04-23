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
  
  python main.py --config-name=config_genomics \
    mode=train \
    noise=loglinear \
    model=nano-dimamba \
    backbone=dimamba \
    strategy=single_device \
    data=genomic \
    eval.checkpoint_path=kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
    loader.global_batch_size=128 \
    loader.batch_size=128 \
    loader.eval_batch_size=128 \
    dora.enabled=False \
    optim.lr=1e-3 \
    strategy.device=3 \
    data.p_random=$p \
    wandb.notes="Caduceus 1k consistency infoseed" \
    wandb.group=consistency_dna_subs_caduceus_2025_04_10
  
  echo "Completed run with p_random = $p"
  echo "------------------------"
done

echo "All runs completed!"