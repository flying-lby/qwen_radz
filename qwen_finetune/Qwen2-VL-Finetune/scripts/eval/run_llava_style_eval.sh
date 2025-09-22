#!/bin/bash

# LLaVA RadZ风格的Qwen2.5-VL分类评估脚本
# 与LLaVA RadZ保持一致的评估方法

echo "=========================================="
echo "LLaVA RadZ风格Qwen2.5-VL分类评估"
echo "评估方法: 简单池化 + embedding均值 + 概率决策"
echo "=========================================="

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
echo "当前工作目录: $(pwd)"

# 设置Python路径
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

# ============ 评估配置 ============
# 模型路径 - 请根据实际情况修改
MODEL_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/2025-09-21/simplified_qwen_clip_v1/merged"

# 数据配置
DATA_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/data/best_classify_mimic_file_clip.json"
IMAGE_FOLDER="/mnt/nlp-ali/usr/huangwenxuan/home/dataset/srv/lby/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
DATASET="mimic"

# 疾病描述文件（可选）
USE_DISEASE_DESC=false
DISEASE_DESC_PATH="/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/new_full_disease.json"

# 评估参数
BATCH_SIZE=16
MAX_SAMPLES=-1  # -1表示全部样本

# LLaVA RadZ风格配置
FEATURE_LAYER=-2           # 使用倒数第二层
TEMPERATURE=0.05           # 相似度计算温度
CLASSIFICATION_THRESHOLD=0.5  # 简单阈值，不使用F1优化

# 输出配置
OUTPUT_DIR="./eval_results/llava_style"
mkdir -p $OUTPUT_DIR
OUTPUT_PATH="$OUTPUT_DIR/llava_style_${DATASET}_results.json"

echo "=========================================="
echo "评估配置:"
echo "模型路径: $MODEL_PATH"
echo "数据集: $DATASET"
echo "图像路径: $IMAGE_FOLDER"
echo "批次大小: $BATCH_SIZE"
echo "特征层: $FEATURE_LAYER"
echo "温度参数: $TEMPERATURE"
echo "分类阈值: $CLASSIFICATION_THRESHOLD"
echo "使用疾病描述: $USE_DISEASE_DESC"
echo "输出路径: $OUTPUT_PATH"
echo "=========================================="

# 构建命令
CMD="python src/eval/new_clip_eval_llava_style.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --dataset $DATASET \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_PATH \
    --feature_layer $FEATURE_LAYER \
    --temperature $TEMPERATURE \
    --classification_threshold $CLASSIFICATION_THRESHOLD"

# 添加疾病描述参数（如果启用）
if [ "$USE_DISEASE_DESC" = "true" ]; then
    CMD="$CMD --use_disease_descriptions --disease_desc_path $DISEASE_DESC_PATH"
fi

# 添加样本限制（如果设置）
if [ "$MAX_SAMPLES" -gt 0 ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

echo "开始LLaVA RadZ风格评估..."
echo "执行命令: $CMD"
echo ""

# 执行评估
eval $CMD

echo ""
echo "=========================================="
echo "LLaVA RadZ风格评估完成！"
echo "结果保存在: $OUTPUT_PATH"
echo "=========================================="

# 快速显示关键结果（如果文件存在）
if [ -f "$OUTPUT_PATH" ]; then
    echo ""
    echo "关键指标预览:"
    python -c "
import json
try:
    with open('$OUTPUT_PATH', 'r') as f:
        data = json.load(f)
    metrics = data.get('metrics', {})
    print(f\"宏平均F1分数: {metrics.get('macro_f1', 0):.3f}\")
    print(f\"宏平均平衡准确率: {metrics.get('macro_balanced_accuracy', 0):.3f}\")
    print(f\"平均AUC-ROC: {metrics.get('mean_auc', 0):.3f}\")
    print(f\"总体准确率: {metrics.get('overall_accuracy', 0):.3f}\")
    print(f\"处理样本数: {data.get('num_samples', 0)}\")
except Exception as e:
    print(f\"结果读取失败: {e}\")
"
fi

echo ""
echo "注意:"
echo "1. 此评估使用LLaVA RadZ风格方法: 简单池化 + embedding均值"
echo "2. 分类决策基于softmax概率和固定阈值"
echo "3. 不使用F1优化阈值，与原始Qwen RadZ评估不同"
echo "4. 可与原始评估结果对比，分析一致性"
