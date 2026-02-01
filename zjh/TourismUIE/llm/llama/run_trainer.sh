# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -x
unset CUDA_VISIBLE_DEVICES
task_name="llama_hybrid"
rm -rf output/$task_name/
rm -rf "output/$task_name""_log"


PYTHONPATH=../../:$PYTHONPATH  \
python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name""_log" \
    run_pretrain.py \
    --model_type "llama" \
    --model_name_or_path "facebook/llama-7b" \
    --tokenizer_name_or_path "facebook/llama-7b" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --use_flash_attention 1 \
    --use_fused_rms_norm 0 \
    --fp16  \
    --fp16_opt_level "O2"  \
    --scale_loss 1024 \
    --learning_rate 0.0001 \
    --min_learning_rate 0.00001 \
    --max_steps 10000 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 20\
    --dataloader_num_workers 1 \
    --sharding "stage2" \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --continue_training 1\
    --recompute 1 \
    --do_train \
    --do_eval \
    --device "gpu" \
    --data_impl "mmap"

    python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3" \
    finetune_generation.py \
    --model_type "llama" \
    --model_name_or_path "/home/ubuntu/PaddleNLP-develop/llm/checkpoints/llama_lora_ckpts/checkpoint-12000" \
    --tokenizer_name_or_path "/home/ubuntu/PaddleNLP-develop/llm/checkpoints/llama_lora_ckpts/checkpoint-12000" \
    --input_dir "/home/ubuntu/PaddleNLP-develop/llm/llama/data" \
    --output_dir "output/ie" \
    --max_seq_length 2048 \
    --per_device_eval_batch_size 1 \
    --use_flash_attention 1 \
    --use_fused_rms_norm 0 \
    --fp16  \
    --do_eval \
    --device "gpu" \

    python merge_lora_params.py \
    --model_name_or_path /home/ubuntu/.paddlenlp/models/meta-llama/Llama-2-7b-chat \
    --lora_path /home/ubuntu/PaddleNLP-develop/llm/checkpoints/llama_lora_ckpts \
    --merge_model_path /home/ubuntu/PaddleNLP-develop/checkpoints/lora_merge \
    --device "gpu"


python  -u  -m paddle.distributed.launch --gpus "0,1,2,3"  finetune_generation.py ./llama/eval.json