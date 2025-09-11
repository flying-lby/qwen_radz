#!/bin/bash
###
 # @Author: AI Assistant
 # @Date: 2025-01-27
 # @FilePath: /qwen_radz/qwen_finetune/Qwen2-VL-Finetune/scripts/clip_eval_script.sh
 # @Description: CLIP风格Qwen2.5-VL胸部X光分类评估脚本
### 

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# 设置基本路径
BASE_DIR="/mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune"
cd $BASE_DIR


# 设置Python路径
export PYTHONPATH=src:$PYTHONPATH

# 设置CUDA调试环境变量
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 模型路径配置
# MODEL_PATH="/srv/lby/qwen_radz/checkpoints/qwen_new_clip_v2"  # 训练输出的CLIP模型路径
MODEL_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/qwen_lora_new_clip_version5/merged"

# MODEL_BASE="/srv/lby/qwen_vl_7b/Qwen2.5-VL-7B-Instruct"  # 基础模型路径

# 如果需要使用合并后的模型，可以使用下面的路径（并注释掉MODEL_BASE）
# MODEL_PATH="/srv/lby/qwen_radz/checkpoints/qwen_new_clip_v1"
# MODEL_BASE=""

# 数据路径配置
IMAGE_FOLDER="/mnt/nlp-ali/usr/zhaizijie/huangwx_ali/zijie_ali/libangyan/dataset/"

# 疾病描述配置
USE_DISEASE_DESCRIPTIONS=true  # 是否使用疾病描述：true/false
DISEASE_DESC_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/code/llava_test/llava/run/data/disease_desc.json"  # 疾病描述文件路径
DESCRIPTION_SOURCE="file"  # 描述来源："file"（详细描述）或"template"（简单模板）

# MedKLIP对齐配置
MEDKLIP_STYLE_OUTPUT=true     # 是否使用MedKLIP风格的输出格式：true/false

# 评估参数配置
BATCH_SIZE=1
MAX_SAMPLES=-1  # 最大样本数量，-1表示全部样本

# 结果输出路径
RESULT_DIR="$BASE_DIR/results/clip_eval_experiments_v5"
mkdir -p $RESULT_DIR

echo "============================================"
echo "开始CLIP风格胸部X光分类评估"
echo "模型路径: $MODEL_PATH"
echo "图像文件夹: $IMAGE_FOLDER"
echo "使用疾病描述: $USE_DISEASE_DESCRIPTIONS"
if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
    echo "疾病描述文件: $DISEASE_DESC_PATH"
    echo "描述来源: $DESCRIPTION_SOURCE"
fi
echo "结果保存到: $RESULT_DIR"
echo "============================================"

# echo "评估ChestX-ray14数据集..."
# if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
#     python -m src.eval.clip_eval \
#         --model_path $MODEL_PATH \
#         --data_path "/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/chest_xray/chest_xray_llava_val.jsonl" \
#         --image_folder $IMAGE_FOLDER \
#         --dataset "chestxray" \
#         --batch_size $BATCH_SIZE \
#         --output_path $RESULT_DIR/chestxray14_clip_results.json \
#         --num_chunks 1 \
#         --chunk_idx 0 \
#         --max_samples $MAX_SAMPLES \
#         --use_disease_descriptions \
#         --disease_desc_path $DISEASE_DESC_PATH \
#         --description_source $DESCRIPTION_SOURCE
# else
#     python -m src.eval.clip_eval \
#         --model_path $MODEL_PATH \
#         --data_path "/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/chest_xray/chest_xray_llava_val.jsonl" \
#         --image_folder $IMAGE_FOLDER \
#         --dataset "chestxray" \
#         --batch_size $BATCH_SIZE \
#         --output_path $RESULT_DIR/chestxray14_clip_results.json \
#         --num_chunks 1 \
#         --chunk_idx 0 \
#         --max_samples $MAX_SAMPLES
# fi

# echo "评估CheXpert数据集..."
# if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
#     python -m src.eval.clip_eval \
#         --model_path $MODEL_PATH \
#         --data_path "/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/chexpert/chexpert_llava_val.jsonl" \
#         --image_folder $IMAGE_FOLDER \

#         --dataset "chexpert" \
#         --batch_size $BATCH_SIZE \
#         --output_path $RESULT_DIR/chexpert_clip_results.json \
#         --num_chunks 1 \
#         --chunk_idx 0 \
#         --max_samples $MAX_SAMPLES \
#         --use_disease_descriptions \
#         --disease_desc_path $DISEASE_DESC_PATH \
#         --description_source $DESCRIPTION_SOURCE
# else
#     python -m src.eval.clip_eval \
#         --model_path $MODEL_PATH \
#         --data_path "/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/chexpert/chexpert_llava_val.jsonl" \
#         --image_folder $IMAGE_FOLDER \
#         --dataset "chexpert" \
#         --batch_size $BATCH_SIZE \
#         --output_path $RESULT_DIR/chexpert_clip_results.json \
#         --num_chunks 1 \
#         --chunk_idx 0 \
#         --max_samples $MAX_SAMPLES
# fi



echo "评估COVIDx_CXR数据集..."
# 构建基础命令
BASE_CMD="python -m src.eval.clip_eval \
    --model_path $MODEL_PATH \
    --data_path /mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/COVIDx_CXR/COVIDx_CXR_llava_origin_val.jsonl \
    --image_folder $IMAGE_FOLDER \
    --dataset COVIDx_CXR \
    --batch_size $BATCH_SIZE \
    --output_path $RESULT_DIR/COVIDx_CXR_clip_results.json \
    --num_chunks 1 \
    --chunk_idx 0 \
    --max_samples $MAX_SAMPLES"
    

# 添加疾病描述参数
if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
    BASE_CMD="$BASE_CMD --use_disease_descriptions --disease_desc_path \"$DISEASE_DESC_PATH\" --description_source $DESCRIPTION_SOURCE"
fi

# 添加MedKLIP对齐参数
if [ "$MEDKLIP_STYLE_OUTPUT" = "true" ]; then
    BASE_CMD="$BASE_CMD --medklip_style_output"
fi

# 执行命令
eval $BASE_CMD

echo "评估SIIM_Pneumothorax数据集..."
# 构建基础命令
BASE_CMD="python -m src.eval.clip_eval \
    --model_path $MODEL_PATH \
    --data_path /mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/SIIM_Pneumothorax/SIIM_Pneumothorax_llava_val.jsonl \
    --image_folder $IMAGE_FOLDER \
    --dataset SIIM_Pneumothorax \
    --batch_size $BATCH_SIZE \
    --output_path $RESULT_DIR/SIIM_Pneumothorax_clip_results.json \
    --num_chunks 1 \
    --chunk_idx 0 \
    --max_samples $MAX_SAMPLES"

# 添加疾病描述参数
if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
    BASE_CMD="$BASE_CMD --use_disease_descriptions --disease_desc_path \"$DISEASE_DESC_PATH\" --description_source $DESCRIPTION_SOURCE"
fi

# 添加MedKLIP对齐参数
if [ "$MEDKLIP_STYLE_OUTPUT" = "true" ]; then
    BASE_CMD="$BASE_CMD --medklip_style_output"
fi

# 执行命令
eval $BASE_CMD

echo "评估RSNA数据集..."
# 构建基础命令
BASE_CMD="python -m src.eval.clip_eval \
    --model_path $MODEL_PATH \
    --data_path /mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/rsna/rsna_pneumonia_llava_origin_val.jsonl \
    --image_folder $IMAGE_FOLDER \
    --dataset rsna \
    --batch_size $BATCH_SIZE \
    --output_path $RESULT_DIR/rsna_pneumonia_clip_results.json \
    --num_chunks 1 \
    --chunk_idx 0 \
    --max_samples $MAX_SAMPLES"

# 添加疾病描述参数
if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
    BASE_CMD="$BASE_CMD --use_disease_descriptions --disease_desc_path \"$DISEASE_DESC_PATH\" --description_source $DESCRIPTION_SOURCE"
fi

# 添加MedKLIP对齐参数
if [ "$MEDKLIP_STYLE_OUTPUT" = "true" ]; then
    BASE_CMD="$BASE_CMD --medklip_style_output"
fi

# 执行命令
eval $BASE_CMD

# ===== 评估ChestX-ray14数据集 =====
echo "评估ChestX-ray14数据集..."
BASE_CMD="python -m src.eval.clip_eval \
    --model_path $MODEL_PATH \
    --data_path /mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/chest_xray/chest_xray_llava_val.jsonl \
    --image_folder $IMAGE_FOLDER \
    --dataset chestxray \
    --batch_size $BATCH_SIZE \
    --output_path $RESULT_DIR/chestxray14_clip_results.json \
    --num_chunks 1 \
    --chunk_idx 0 \
    --max_samples $MAX_SAMPLES"

# 添加疾病描述参数
if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
    BASE_CMD="$BASE_CMD --use_disease_descriptions --disease_desc_path \"$DISEASE_DESC_PATH\" --description_source $DESCRIPTION_SOURCE"
fi

# 添加MedKLIP对齐参数
if [ "$MEDKLIP_STYLE_OUTPUT" = "true" ]; then
    BASE_CMD="$BASE_CMD --medklip_style_output"
fi

# 执行命令
eval $BASE_CMD

# ===== 评估CheXpert数据集 =====
echo "评估CheXpert数据集..."
BASE_CMD="python -m src.eval.clip_eval \
    --model_path $MODEL_PATH \
    --data_path /mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/chexpert/chexpert_llava_val.jsonl \
    --image_folder $IMAGE_FOLDER \
    --dataset chexpert \
    --batch_size $BATCH_SIZE \
    --output_path $RESULT_DIR/chexpert_clip_results.json \
    --num_chunks 1 \
    --chunk_idx 0 \
    --max_samples $MAX_SAMPLES"

# 添加疾病描述参数
if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
    BASE_CMD="$BASE_CMD --use_disease_descriptions --disease_desc_path \"$DISEASE_DESC_PATH\" --description_source $DESCRIPTION_SOURCE"
fi

# 添加MedKLIP对齐参数
if [ "$MEDKLIP_STYLE_OUTPUT" = "true" ]; then
    BASE_CMD="$BASE_CMD --medklip_style_output"
fi

# 执行命令
eval $BASE_CMD

echo "============================================"
echo "评估完成！结果保存在: $RESULT_DIR"
echo "============================================"





