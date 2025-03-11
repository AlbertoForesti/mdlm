export HF_HOME="/home/foresti/.cache/huggingface"
export HF_DATASET_CACHE="/home/foresti/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
python main.py \
  mode=train \
  eval.compute_entropy=True \
  eval.compute_mutinfo=True \
  training.compute_entropy=True \
  training.compute_mutinfo=True \
  data=multi_news  \
  model=nano  \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=100 \
  sampling.num_sample_batches=1 \
  backbone=unetmlp \
  strategy=single_device \