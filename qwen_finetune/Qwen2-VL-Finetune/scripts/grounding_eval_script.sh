#!/bin/bash

# grounding_eval_script.sh
# RSNA Pneumonia Grounding评估脚本
# 基于优化后的grounding_eval_rsna.py，集成了clip_eval_original.py的先进特性

set -e  # 遇到错误立即退出

echo "=========================================="
echo "RSNA Pneumonia Grounding Evaluation Script"
echo "基于Qwen2.5-VL CLIP模型的grounding评估"
echo "=========================================="

# 环境配置
echo "🔧 正在配置环境..."

# 激活conda环境
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate qwen_vl

# CUDA配置
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 打印环境信息
echo "✅ Conda环境: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "✅ Python版本: $(python --version)"
echo "✅ PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "✅ CUDA可用性: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "✅ GPU数量: $(python -c 'import torch; print(torch.cuda.device_count())')"

# 检查关键Python依赖
echo ""
echo "🔍 检查Python依赖..."
python -c "
try:
    import torch, transformers, numpy, pandas, json
    print('✅ 核心依赖检查通过')
except ImportError as e:
    print(f'❌ 依赖缺失: {e}')
    exit(1)
"

# 基础路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

echo "✅ 项目根目录: $PROJECT_ROOT"
echo "✅ 源代码目录: $SRC_DIR"

# 模型和数据路径配置
MODEL_PATH="/srv/lby/qwen_radz/qwen_lora_new_clip_version1/merged"
CSV_PATH="/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/process_data/rsna/test.csv"
IMAGE_FOLDER="/srv/lby/"
DISEASE_DESC_PATH="/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/observation_explanation.json"

# 执行参数配置
BATCH_SIZE=4                    # 批次大小，考虑内存限制
MAX_SAMPLES=-1                  # 最大样本数(-1表示全部)
TARGET_SIZE=224                 # 目标图像尺寸

# 输出配置
OUTPUT_DIR="$PROJECT_ROOT/grounding_results"
OUTPUT_PATH="$OUTPUT_DIR/rsna_grounding_results_$(date +%Y%m%d_%H%M%S).json"
VIZ_DIR="$OUTPUT_DIR/visualizations"

echo ""
echo "📁 路径配置检查:"
echo "   模型路径: $MODEL_PATH"
echo "   数据CSV: $CSV_PATH"
echo "   图像目录: $IMAGE_FOLDER"
echo "   疾病描述: $DISEASE_DESC_PATH"
echo "   输出目录: $OUTPUT_DIR"

# 检查关键路径是否存在
echo ""
echo "🔍 正在检查关键路径..."

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi
echo "✅ 模型路径存在"

if [ ! -f "$CSV_PATH" ]; then
    echo "❌ 错误: CSV文件不存在: $CSV_PATH"
    exit 1
fi
echo "✅ CSV文件存在"

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "❌ 错误: 图像目录不存在: $IMAGE_FOLDER"
    exit 1
fi
echo "✅ 图像目录存在"

if [ ! -f "$DISEASE_DESC_PATH" ]; then
    echo "⚠️  警告: 疾病描述文件不存在: $DISEASE_DESC_PATH"
    echo "   将使用默认描述"
else
    echo "✅ 疾病描述文件存在 (MedKLIP官方)"
fi

# 创建输出目录
echo ""
echo "📁 正在创建输出目录..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "$VIZ_DIR"
echo "✅ 输出目录创建完成"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 设置Python路径，确保能够导入自定义模块
export PYTHONPATH="$SRC_DIR:$PYTHONPATH"

# 验证Python脚本存在
echo ""
echo "🔍 验证Python脚本..."
if [ ! -f "src/eval/grounding_eval_rsna.py" ]; then
    echo "❌ 错误: grounding_eval_rsna.py不存在"
    exit 1
fi
echo "✅ grounding_eval_rsna.py存在"

# 验证训练模块存在
if [ ! -f "src/train/clip_modeling_qwen2_5_vl.py" ]; then
    echo "⚠️  警告: clip_modeling_qwen2_5_vl.py不存在，可能导致导入错误"
else
    echo "✅ clip_modeling_qwen2_5_vl.py存在"
fi

# 构建评估命令
EVAL_CMD="python src/eval/grounding_eval_rsna.py \
    --model_path \"$MODEL_PATH\" \
    --csv_path \"$CSV_PATH\" \
    --image_folder \"$IMAGE_FOLDER\" \
    --disease_desc_path \"$DISEASE_DESC_PATH\" \
    --batch_size $BATCH_SIZE \
    --max_samples $MAX_SAMPLES \
    --output_path \"$OUTPUT_PATH\" \
    --save_visualizations \
    --viz_dir \"$VIZ_DIR\" \
    --target_size $TARGET_SIZE"

echo ""
echo "🚀 开始执行grounding评估..."
echo "================================================"
echo "执行命令:"
echo "$EVAL_CMD"
echo "================================================"

# 记录开始时间
START_TIME=$(date +%s)

# 执行评估命令
if eval "$EVAL_CMD"; then
    # 计算执行时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    echo ""
    echo "🎉 评估完成!"
    echo "================================================"
    echo "✅ 总执行时间: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "✅ 结果保存至: $OUTPUT_PATH"
    echo "✅ 可视化保存至: $VIZ_DIR"
    echo ""
    
    # 显示结果文件大小
    if [ -f "$OUTPUT_PATH" ]; then
        RESULT_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
        echo "📊 结果文件大小: $RESULT_SIZE"
    fi
    
    # 显示可视化文件数量
    if [ -d "$VIZ_DIR" ]; then
        VIZ_COUNT=$(find "$VIZ_DIR" -name "*.png" | wc -l)
        echo "🖼️  可视化文件数量: $VIZ_COUNT"
    fi
    
    echo ""
    echo "📋 接下来你可以:"
    echo "   1. 查看结果文件: cat \"$OUTPUT_PATH\""
    echo "   2. 浏览可视化: ls \"$VIZ_DIR\""
    echo "   3. 分析处理统计和grounding指标"
    echo "================================================"
    
else
    echo ""
    echo "❌ 评估执行失败!"
    echo "================================================"
    echo "🔍 调试信息:"
    echo "   工作目录: $(pwd)"
    echo "   Python路径: $PYTHONPATH"
    echo "   模型路径: $MODEL_PATH"
    echo "   CSV路径: $CSV_PATH"
    echo "   图像目录: $IMAGE_FOLDER"
    echo ""
    echo "🔧 请检查:"
    echo "   1. 模型路径是否正确并包含所有必需文件"
    echo "   2. CSV数据文件格式是否正确"
    echo "   3. GPU内存是否充足"
    echo "   4. Python依赖是否完整安装"
    echo "   5. 导入模块是否存在错误"
    echo ""
    echo "💡 建议:"
    echo "   1. 手动运行: cd $PROJECT_ROOT && python src/eval/grounding_eval_rsna.py --help"
    echo "   2. 检查错误日志中的具体错误信息"
    echo "   3. 验证模型文件完整性"
    echo "================================================"
    exit 1
fi

echo ""
echo "✨ grounding_eval_script.sh 执行完成 ✨"