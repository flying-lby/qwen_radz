#!/bin/bash
###
 # @Author: AI Assistant
 # @Date: 2025-09-20
 # @FilePath: /qwen_radz/qwen_finetune/Qwen2-VL-Finetune/scripts/llava_med_eval_script.sh
 # @Description: LLaVA-Med对齐的Qwen2.5-VL胸部X光分类评估脚本
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
MODEL_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/qwen_lora_new_clip_version13/merged"
# 数据路径配置
IMAGE_FOLDER="/mnt/nlp-ali/usr/zhaizijie/huangwx_ali/zijie_ali/libangyan/dataset/"

# LLaVA-Med风格疾病描述配置
USE_DISEASE_DESCRIPTIONS=true  # 是否使用疾病描述：true/false
DISEASE_DESC_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/code/llava_test/llava/run/data/disease_desc.json"  # 疾病描述文件路径
DESCRIPTION_SOURCE="file"  # 描述来源："file"（详细描述）或"template"（简单模板）

# LLaVA-Med对齐配置参数
IMGCLS_COUNT=4          # 图像分类token数量
TXTCLS_COUNT=4          # 文本分类token数量
FEATURE_LAYER=1         # 特征提取层级
TEMPERATURE=0.05        # 相似度计算温度参数
BOOK_CHOICE=1           # 疾病描述来源选择

# 评估参数配置
BATCH_SIZE=4
MAX_SAMPLES=-1  # 使用最小样本数进行测试（模型权重有问题）

# 结果输出路径
RESULT_DIR="$BASE_DIR/results/llava_med_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULT_DIR

echo "============================================"
echo "开始LLaVA-Med对齐的胸部X光分类评估"
echo "模型路径: $MODEL_PATH"
echo "图像文件夹: $IMAGE_FOLDER"
echo "使用疾病描述: $USE_DISEASE_DESCRIPTIONS"
if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
    echo "疾病描述文件: $DISEASE_DESC_PATH"
    echo "描述来源: $DESCRIPTION_SOURCE"
fi
echo "LLaVA-Med参数: Imgcls=$IMGCLS_COUNT, Txtcls=$TXTCLS_COUNT, Layer=$FEATURE_LAYER, Temp=$TEMPERATURE"
echo "结果保存到: $RESULT_DIR"
echo "============================================"


echo "评估COVIDx_CXR数据集..."
# 构建基础命令
BASE_CMD="python -m src.eval.clip_eval_original \
    --model_path $MODEL_PATH \
    --data_path /mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/COVIDx_CXR/COVIDx_CXR_llava_origin_val.jsonl \
    --image_folder $IMAGE_FOLDER \
    --dataset COVIDx_CXR \
    --batch_size $BATCH_SIZE \
    --output_path $RESULT_DIR/COVIDx_CXR_clip_results.json \
    --num_chunks 1 \
    --chunk_idx 0 \
    --max_samples $MAX_SAMPLES \
    --Imgcls_count $IMGCLS_COUNT \
    --Txtcls_count $TXTCLS_COUNT \
    --feature_layer $FEATURE_LAYER \
    --temperature $TEMPERATURE \
    --Book_choice $BOOK_CHOICE"
    
# 添加疾病描述参数
if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
    BASE_CMD="$BASE_CMD --use_disease_descriptions --disease_desc_path \"$DISEASE_DESC_PATH\" --description_source $DESCRIPTION_SOURCE"
fi


echo "评估SIIM_Pneumothorax数据集..."
BASE_CMD="python -m src.eval.clip_eval_original \
    --model_path $MODEL_PATH \
    --data_path /mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/rsna/rsna_pneumonia_llava_origin_val.jsonl \
    --image_folder $IMAGE_FOLDER \
    --dataset SIIM_Pneumothorax \
    --batch_size $BATCH_SIZE \
    --output_path $RESULT_DIR/SIIM_Pneumothorax_results.json \
    --num_chunks 1 \
    --chunk_idx 0 \
    --max_samples $MAX_SAMPLES \
    --Imgcls_count $IMGCLS_COUNT \
    --Txtcls_count $TXTCLS_COUNT \
    --feature_layer $FEATURE_LAYER \
    --temperature $TEMPERATURE \
    --Book_choice $BOOK_CHOICE"

if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
    BASE_CMD="$BASE_CMD --use_disease_descriptions --disease_desc_path \"$DISEASE_DESC_PATH\" --description_source $DESCRIPTION_SOURCE"
fi


echo "评估RSNA数据集..."
BASE_CMD="python -m src.eval.clip_eval_original \
    --model_path $MODEL_PATH \
    --data_path /mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/rsna/rsna_pneumonia_llava_origin_val.jsonl \
    --image_folder $IMAGE_FOLDER \
    --dataset rsna \
    --batch_size $BATCH_SIZE \
    --output_path $RESULT_DIR/rsna_pneumonia_results.json \
    --num_chunks 1 \
    --chunk_idx 0 \
    --max_samples $MAX_SAMPLES \
    --Imgcls_count $IMGCLS_COUNT \
    --Txtcls_count $TXTCLS_COUNT \
    --feature_layer $FEATURE_LAYER \
    --temperature $TEMPERATURE \
    --Book_choice $BOOK_CHOICE"


if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
    BASE_CMD="$BASE_CMD --use_disease_descriptions --disease_desc_path \"$DISEASE_DESC_PATH\" --description_source $DESCRIPTION_SOURCE"
fi

eval $BASE_CMD

# ===== 评估ChestX-ray14数据集 =====
echo "评估ChestX-ray14数据集..."
BASE_CMD="python -m src.eval.clip_eval_original \
    --model_path $MODEL_PATH \
    --data_path /mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/chest_xray/chest_xray_llava_val.jsonl \
    --image_folder $IMAGE_FOLDER \
    --dataset chestxray \
    --batch_size $BATCH_SIZE \
    --output_path $RESULT_DIR/chestxray14_results.json \
    --num_chunks 1 \
    --chunk_idx 0 \
    --max_samples $MAX_SAMPLES \
    --Imgcls_count $IMGCLS_COUNT \
    --Txtcls_count $TXTCLS_COUNT \
    --feature_layer $FEATURE_LAYER \
    --temperature $TEMPERATURE \
    --Book_choice $BOOK_CHOICE"

if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
    BASE_CMD="$BASE_CMD --use_disease_descriptions --disease_desc_path \"$DISEASE_DESC_PATH\" --description_source $DESCRIPTION_SOURCE"
fi

eval $BASE_CMD

# ===== 评估CheXpert数据集 =====
echo "评估CheXpert数据集..."
BASE_CMD="python -m src.eval.clip_eval_original \
    --model_path $MODEL_PATH \
    --data_path /mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/chexpert/chexpert_llava_val.jsonl \
    --image_folder $IMAGE_FOLDER \
    --dataset chexpert \
    --batch_size $BATCH_SIZE \
    --output_path $RESULT_DIR/chexpert_results.json \
    --num_chunks 1 \
    --chunk_idx 0 \
    --max_samples $MAX_SAMPLES \
    --Imgcls_count $IMGCLS_COUNT \
    --Txtcls_count $TXTCLS_COUNT \
    --feature_layer $FEATURE_LAYER \
    --temperature $TEMPERATURE \
    --Book_choice $BOOK_CHOICE"

if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
    BASE_CMD="$BASE_CMD --use_disease_descriptions --disease_desc_path \"$DISEASE_DESC_PATH\" --description_source $DESCRIPTION_SOURCE"
fi

eval $BASE_CMD

echo "============================================"
echo "评估完成！结果保存在: $RESULT_DIR"
echo "============================================"
