#!/bin/bash
export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="4,5,6,7"
python main.py --config-name=config_genomics \
  mode=train \
  noise=loglinear \
  model=nano-dimamba \
  backbone=dimamba \
  strategy=single_device \
  data=genomic \
  loader.global_batch_size=128 \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  dora.enabled=False \
  data.p_random=0.01 \
  strategy.device=2