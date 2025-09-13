#!/bin/bash 

# 改进的CLIP风格Qwen2.5-VL训练脚本

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
echo "当前工作目录: $(pwd)"
echo "项目根目录: ${PROJECT_ROOT}"
echo "训练脚本路径: ${PROJECT_ROOT}/src/train/clip_train_improved.py"

# 设置Python路径
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
echo "PYTHONPATH设置为: $PYTHONPATH"

# 设置环境变量与激活环境


export WANDB_PROJECT="clip-qwen2vl-improved"
export CUDA_LAUNCH_BLOCKING=1

# 模型和数据路径配置
MODEL_NAME_OR_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/official_llava_med/Qwen2.5-VL-7B-Instruct"
DATA_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/data/best_classify_mimic_file_clip.json"
IMAGE_FOLDER="/mnt/nlp-ali/usr/huangwenxuan/home/dataset/srv/lby/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
DISEASE_DESC_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/code/llava_test/llava/run/data/disease_desc.json"
OUTPUT_DIR="/mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/2025-09-13/qwen_lora_new_clip_version10"


# 创建输出目录
mkdir -p $OUTPUT_DIR

# 训练参数  
NUM_TRAIN_EPOCHS=2
LEARNING_RATE=2e-5
GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=64
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
MAX_LENGTH=8192

# CLIP特定参数
IMGCLS_COUNT=4
TXTCLS_COUNT=8
HIDDEN_DIM=1024
OUTPUT_DIM=3584
TEMPERATURE=0.05
CLIP_TRAINING_RATIO=0.8

# LoRA参数
USE_LORA=True
LORA_R=128
LORA_ALPHA=256  
LORA_DROPOUT=0.05
USE_BNB=False

# DeepSpeed配置
DEEPSPEED_CONFIG="scripts/zero3.json"

# GPU配置

echo "=========================================="
echo "开始改进的CLIP风格Qwen2.5-VL训练"
echo "模型: $MODEL_NAME_OR_PATH"
echo "数据: $DATA_PATH"
echo "输出: $OUTPUT_DIR"
echo "=========================================="

# 训练命令 - 使用DeepSpeed
deepspeed --master_port=12345 \
    src/train/clip_train_improved.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --model_max_length $MAX_LENGTH \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.1 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --bf16 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to "wandb" \
    --do_train True \
    --img_cls_token_count $IMGCLS_COUNT \
    --txt_cls_token_count $TXTCLS_COUNT \
    --hidden_dim $HIDDEN_DIM \
    --output_dim $OUTPUT_DIM \
    --temperature $TEMPERATURE \
    --clip_training_ratio $CLIP_TRAINING_RATIO \
    --use_disease_desc True \
    --disease_desc_path $DISEASE_DESC_PATH \
    --img_mlp_type 0 \
    --txt_mlp_type 0 \
    --use_local_loss True \
    --use_cross_attention_loss True \
    --pooling_strategy "mean" \
    --feature_extraction_layer -1 \
    --use_data_augmentation False \
    --use_lora $USE_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --use_bnb $USE_BNB \
    --image_min_pixels $((16 * 28 * 28)) \
    --image_max_pixels $((576 * 28 * 28)) \
    --deepspeed $DEEPSPEED_CONFIG \
    --seed 42 \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "训练完成！模型保存在: $OUTPUT_DIR"

# 是否合并LoRA权重（CLIP专用：合并LoRA并覆盖非LoRA可训练模块）
MERGE_LORA=true
if [ "$MERGE_LORA" = "true" ]; then
  echo "开始合并LoRA权重..."
  SAVE_MERGED_DIR="$OUTPUT_DIR/merged"
  python src/clip_merge_lora.py \
    --model-path "$OUTPUT_DIR" \
    --model-base "$MODEL_NAME_OR_PATH" \
    --save-model-path "$SAVE_MERGED_DIR" \
    --safe-serialization \
    --export-with-clip-head
  echo "LoRA合并完成，保存至: $SAVE_MERGED_DIR"
fi
