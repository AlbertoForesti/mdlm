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
    noise=geometric \
    mode=train \
    model=unetmlp070k \
    model.length=512 \
    backbone=unetmlp \
    strategy=single_device \
    data=genomic \
    loader.batch_size=1024 \
    strategy.device=1 \
    data.p_random=$p \
    dora.enabled=False \
    parameterization=mine \
    +model.regression_type=attention \
    training.compute_entropy=False \
    trainer.accumulate_grad_batches=1 \
    eval.compute_entropy=False \
    eval.generate_samples=False \
    time_conditioning=False \
    wandb.group=consistency_dna_mine_2025_04_10 \
    optim.lr=1e-2 \
    training.ema=0.999 \
    time_conditioning=False \
    trainer.max_epochs=500 \
    lr_scheduler.num_warmup_steps=1000 \
  
  echo "Completed run with p_random = $p"
  echo "------------------------"
done

echo "All runs completed!"