#!/bin/bash
###
 # @Author: flying-lby 2230232178@qq.com
 # @Date: 2025-04-10 14:17:31
 # @LastEditors: flying-lby 2230232178@qq.com
 # @LastEditTime: 2025-05-15 02:47:26
 # @FilePath: /qwen_radz/Qwen2.5-VL/qwen-vl-finetune/scripts/sft_7b.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# Distributed training configuration

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
# llm=/srv/lby/qwen_vl_7b/Qwen2.5-VL-7B-Instruct  # Using HuggingFace model ID
llm=/mnt/nlp-ali/usr/huangwenxuan/home/official_llava_med/Qwen2.5-VL-7B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-5
batch_size=32
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=mimic_classify_clip

# Output configuration
run_name="qwen2vl-baseline"
output_dir=/mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/qwen2.5_radz_v5_5_14
NPROC_PER_NODE=4
# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

# python -m pdb ${entry_file} ${args}
