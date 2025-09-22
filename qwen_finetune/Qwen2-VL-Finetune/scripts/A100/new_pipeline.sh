#!/bin/bash 

# 简化版CLIP风格Qwen2.5-VL训练脚本
# 基于LLaVA RadZ设计理念，修复关键性能问题

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
echo "当前工作目录: $(pwd)"
echo "项目根目录: ${PROJECT_ROOT}"

# 设置Python路径
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

# 设置环境变量
export WANDB_PROJECT="simplified-clip-qwen2vl"
export CUDA_LAUNCH_BLOCKING=1

echo "=========================================="
echo "简化版CLIP风格Qwen2.5-VL训练配置"
echo "基于LLaVA RadZ设计理念"
echo "=========================================="

# 模型和数据路径配置
MODEL_NAME_OR_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/official_llava_med/Qwen2.5-VL-7B-Instruct"
DATA_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/data/best_classify_mimic_file_clip.json"
IMAGE_FOLDER="/mnt/nlp-ali/usr/huangwenxuan/home/dataset/srv/lby/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
DISEASE_DESC_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/code/llava_test/llava/run/data/disease_desc.json"
OUTPUT_DIR="/mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/2025-09-21/simplified_qwen_clip_v1"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# ============ 关键修复：训练参数优化 ============
NUM_TRAIN_EPOCHS=2

# 修复1：学习率策略优化
LEARNING_RATE=2e-5
WARMUP_RATIO=0.1          # 增加热身比例：0.01 → 0.1
WEIGHT_DECAY=0.01         # 减少权重衰减：0.1 → 0.01

# 修复2：批次大小优化 
GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=64       # 增加批次：32 → 64 (与LLaVA RadZ一致)
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# 修复3：序列长度优化
MAX_LENGTH=4096           # 减少序列长度：8192 → 4096 (与LLaVA RadZ一致)

# ============ 关键修复：CLIP参数优化 ============
# 修复4：Token数量平衡
IMGCLS_COUNT=4
TXTCLS_COUNT=4            # 平衡设置：8 → 4 (与图像token一致)

# 修复5：MLP配置修复（最关键）
IMG_MLP_TYPE=1            # 启用GELU MLP：0 → 1
TXT_MLP_TYPE=1            # 启用GELU MLP：0 → 1

# 简化的特征配置
HIDDEN_DIM=1024
OUTPUT_DIM=512            # 简化输出维度：3584 → 512
TEMPERATURE=0.05
FEATURE_LAYER=-2          # 使用倒数第二层（与LLaVA RadZ一致）

# 修复6：CLIP训练比例优化
CLIP_TRAINING_RATIO=0.3   # 降低比例：0.8 → 0.3 (提高训练稳定性)

# ============ LoRA参数 ============
USE_LORA=True
LORA_R=128
LORA_ALPHA=256  
LORA_DROPOUT=0.05

# ============ 简化的损失配置 ============
# 修复7：简化损失函数
USE_LOCAL_LOSS=false           # 禁用局部损失
USE_CROSS_ATTENTION_LOSS=false # 禁用交叉注意力损失

# ============ 优化的数据加载配置 ============
# 修复8：数据加载优化
DATALOADER_NUM_WORKERS=8       # 增加worker：4 → 8
USE_DATA_AUGMENTATION=true     # 启用数据增强

# DeepSpeed配置
DEEPSPEED_CONFIG="scripts/zero3.json"

echo "=========================================="
echo "关键修复点总结:"
echo "1. 学习率策略: warmup_ratio=$WARMUP_RATIO, weight_decay=$WEIGHT_DECAY"
echo "2. 批次大小: per_device=$BATCH_PER_DEVICE (增加一倍)"
echo "3. 序列长度: max_length=$MAX_LENGTH (减少一半)"
echo "4. Token平衡: img_cls=$IMGCLS_COUNT, txt_cls=$TXTCLS_COUNT"
echo "5. MLP修复: img_mlp_type=$IMG_MLP_TYPE, txt_mlp_type=$TXT_MLP_TYPE (关键!)"
echo "6. CLIP比例: clip_training_ratio=$CLIP_TRAINING_RATIO (降低)"
echo "7. 损失简化: 禁用局部损失和交叉注意力损失"
echo "8. 数据优化: workers=$DATALOADER_NUM_WORKERS, augmentation=$USE_DATA_AUGMENTATION"
echo "=========================================="
echo "开始训练..."

# 训练命令 - 使用简化的训练脚本
deepspeed --master_port=29500 \
    src/train/new_clip_train_improved.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --disease_desc_path $DISEASE_DESC_PATH \
    --output_dir $OUTPUT_DIR \
    --model_max_length $MAX_LENGTH \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --bf16 True \
    --gradient_checkpointing True \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --report_to "wandb" \
    --do_train True \
    --seed 42 \
    --img_cls_token_count $IMGCLS_COUNT \
    --txt_cls_token_count $TXTCLS_COUNT \
    --hidden_dim $HIDDEN_DIM \
    --output_dim $OUTPUT_DIM \
    --temperature $TEMPERATURE \
    --feature_extraction_layer $FEATURE_LAYER \
    --pooling_strategy "mean" \
    --img_mlp_type $IMG_MLP_TYPE \
    --txt_mlp_type $TXT_MLP_TYPE \
    --clip_training_ratio $CLIP_TRAINING_RATIO \
    --use_lora $USE_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --image_min_pixels $((16 * 28 * 28)) \
    --image_max_pixels $((256 * 28 * 28)) \
    --deepspeed $DEEPSPEED_CONFIG \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "=========================================="
echo "训练完成！模型保存在: $OUTPUT_DIR"
echo "=========================================="

# 是否合并LoRA权重
MERGE_LORA=true
if [ "$MERGE_LORA" = "true" ]; then
    echo "开始合并LoRA权重..."
    SAVE_MERGED_DIR="$OUTPUT_DIR/merged"
    
    # 使用简化的合并脚本
    python src/clip_merge_lora.py \
        --model-path "$OUTPUT_DIR" \
        --model-base "$MODEL_NAME_OR_PATH" \
        --save-model-path "$SAVE_MERGED_DIR" \
        --safe-serialization \
        --export-with-clip-head
        
    echo "LoRA合并完成，保存至: $SAVE_MERGED_DIR"
    
    echo "=========================================="
    echo "简化版训练Pipeline完成!"
    echo "主要改进:"
    echo "- 参考LLaVA RadZ的简单高效设计"
    echo "- 修复MLP缺失问题 (最关键)"
    echo "- 优化学习率和批次配置"
    echo "- 简化损失函数和pipeline架构"
    echo "- 平衡Token配置和降低CLIP比例"
    echo "=========================================="
fi
