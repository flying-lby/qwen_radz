#!/bin/bash
###
 # @Description: Qwen2.5-VL Grounding Evaluation Script for RSNA and SIIM datasets
###

# 设置基本路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd $PROJECT_ROOT

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate qwen_vl

# 设置Python路径
export PYTHONPATH=src:$PYTHONPATH

# 设置CUDA环境变量
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
 
# 可选：设置日志级别减少警告输出（如果需要）
# export PYTHONPATH=src:$PYTHONPATH
# export PYTHONWARNINGS="ignore::UserWarning"
  
# 模型和数据路径配置
# MODEL_PATH="/srv/lby/qwen_radz/checkpoints/qwen_new_clip_v2"  # 当前模型有NaN参数问题
MODEL_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/2025-09-13/qwen_lora_new_clip_version13/merged"  # 尝试使用备用模型
IMAGE_FOLDER="/mnt/nlp-ali/usr/zhaizijie/huangwx_ali/zijie_ali/libangyan/dataset/"
DISEASE_DESC_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/code/llava_test/llava/run/data/observation_explanation.json"

# 执行参数配置
BATCH_SIZE=4
MAX_SAMPLES=100
TARGET_SIZE=224

# 输出配置
RESULT_DIR="$PROJECT_ROOT/results"
mkdir -p "$RESULT_DIR/rsna_grounding"
mkdir -p "$RESULT_DIR/siim_grounding"

echo "============================================"
echo "开始Grounding评估 (使用重构后的代码)"
echo "模型路径: $MODEL_PATH"
echo "图像文件夹: $IMAGE_FOLDER"
echo "结果保存到: $RESULT_DIR"
echo "批次大小: $BATCH_SIZE"
echo "最大样本数: $MAX_SAMPLES"
echo "============================================"

# 检查重构后的评估脚本是否存在
if [ ! -f "src/eval/eval_grounding.py" ]; then
    echo "❌ 错误: eval_grounding.py不存在"
    echo "请确保重构后的代码已正确放置"
    exit 1
fi
echo "✅ 重构后的评估脚本存在"

# ===== 评估RSNA数据集 =====
echo ""
echo "📊 开始评估RSNA数据集..."
RSNA_OUTPUT_FILE="$RESULT_DIR/rsna_grounding/rsna_grounding_results_$(date +%Y%m%d_%H%M%S).json"

if python src/eval/eval_grounding.py \
    --model_path "$MODEL_PATH" \
    --jsonl_path "/mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/rsna/rsna_pneumonia_llava_origin_val.jsonl" \
    --image_folder "$IMAGE_FOLDER" \
    --disease_desc_path "$DISEASE_DESC_PATH" \
    --dataset_name "RSNA_Pneumonia" \
    --batch_size $BATCH_SIZE \
    --max_samples $MAX_SAMPLES \
    --output_path "$RSNA_OUTPUT_FILE" \
    --save_visualizations \
    --viz_dir "$RESULT_DIR/rsna_grounding/visualizations" \
    --target_size $TARGET_SIZE; then
    echo "✅ RSNA数据集评估完成"
    echo "📁 结果保存至: $RSNA_OUTPUT_FILE"
else
    echo "❌ RSNA数据集评估失败"
    exit 1
fi

# ===== 评估SIIM数据集 =====
echo ""
echo "📊 开始评估SIIM Pneumothorax数据集..."
SIIM_OUTPUT_FILE="$RESULT_DIR/siim_grounding/siim_grounding_results_$(date +%Y%m%d_%H%M%S).json"

if python src/eval/eval_grounding.py \
    --model_path "$MODEL_PATH" \
    --jsonl_path "/mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/SIIM_Pneumothorax/SIIM_Pneumothorax_llava_origin_val.jsonl" \
    --dataset_name "SIIM_Pneumothorax" \
    --image_folder "$IMAGE_FOLDER" \
    --disease_desc_path "$DISEASE_DESC_PATH" \
    --batch_size $BATCH_SIZE \
    --max_samples $MAX_SAMPLES \
    --output_path "$SIIM_OUTPUT_FILE" \
    --save_visualizations \
    --viz_dir "$RESULT_DIR/siim_grounding/visualizations" \
    --target_size $TARGET_SIZE; then
    echo "✅ SIIM数据集评估完成"
    echo "📁 结果保存至: $SIIM_OUTPUT_FILE"
else
    echo "❌ SIIM数据集评估失败"
    exit 1
fi

echo ""
echo "🎉 所有数据集grounding评估完成！"
echo "============================================"
echo "📁 评估结果:"
echo "   RSNA结果: $RSNA_OUTPUT_FILE"
echo "   SIIM结果: $SIIM_OUTPUT_FILE"
echo "📁 可视化文件:"
echo "   RSNA可视化: $RESULT_DIR/rsna_grounding/visualizations/"
echo "   SIIM可视化: $RESULT_DIR/siim_grounding/visualizations/"
echo "============================================"
echo "✨ 使用重构后的eval_utils模块评估完成！"