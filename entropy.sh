export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
python main.py \
  mode=info_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-no_flashattn-fp32-owt \
  eval.mc_estimates=100 \
  compute_entropy=True \
  data=openwebtext  \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=1000 \
  loader.eval_batch_size=2 \
  sampling.num_sample_batches=1 \
  data.cache_dir=$HF_DATASET_CACHE \
  backbone=hf_dit