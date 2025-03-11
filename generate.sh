python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-no_flashattn-fp32-owt \
  eval.compute_entropy=True \
  data=openwebtext-split  \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=1 \
  backbone=hf_dit \ 
  data.cache_dir=$HF_DATASET_CACHE \