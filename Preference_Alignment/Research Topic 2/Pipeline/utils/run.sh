#!/bin/bash


# 设置环境变量以最大化内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:16
export CUDA_LAUNCH_BLOCKING=1
export WANDB_MODE=offline
export TORCH_USE_CUDA_DSA=1

# 激活conda环境
source /root/miniconda3/bin/activate myenv

# 运行训练命令
python -u train.py \
  model=mistra \
  model.name_or_path=models/模型路径 \
  datasets=[datasets] \  
  loss=sft \
  gradient_accumulation_steps=1 \
  batch_size=1 \
  eval_batch_size=1 \
  trainer=BasicTrainer \
  sample_during_eval=false \
  base_data_dir=datasets/ \
  reverse_dataset=false \
  transform.method=rank_based \
  max_length=512 \
  max_prompt_length=128 \
  activation_checkpointing=true \
  eval_every=5000 \
  n_eval_examples=8 \
  n_epochs=1
