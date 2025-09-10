#!/bin/bash

# siim_grounding_eval_script.sh
# SIIM Pneumothorax Grounding评估脚本
# 基于grounding_eval_rsna.py，适配SIIM数据集

set -e  # 遇到错误立即退出

echo "=========================================="
echo "SIIM Pneumothorax Grounding Evaluation Script"
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

# SIIM模型和数据路径配置
MODEL_PATH="/srv/lby/qwen_radz/checkpoints/qwen_new_clip_v2" 
JSONL_PATH="/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/SIIM_Pneumothorax/SIIM_Pneumothorax_llava_origin_val.jsonl"
IMAGE_FOLDER="/srv/lby/"
DISEASE_DESC_PATH="/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/observation_explanation.json"

# 执行参数配置
BATCH_SIZE=4                    # 批次大小，考虑内存限制
MAX_SAMPLES=100                 # 最大样本数(-1表示全部)
TARGET_SIZE=224                 # 目标图像尺寸

# 输出配置
OUTPUT_DIR="$PROJECT_ROOT/results/siim_grounding"
OUTPUT_PATH="$OUTPUT_DIR/siim_grounding_results_$(date +%Y%m%d_%H%M%S).json"
VIZ_DIR="$OUTPUT_DIR/visualizations"

echo ""
echo "📁 路径配置检查:"
echo "   模型路径: $MODEL_PATH"
echo "   数据JSONL: $JSONL_PATH"
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

if [ ! -f "$JSONL_PATH" ]; then
    echo "❌ 错误: JSONL文件不存在: $JSONL_PATH"
    exit 1
fi
echo "✅ JSONL文件存在"

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "❌ 错误: 图像目录不存在: $IMAGE_FOLDER"
    exit 1
fi
echo "✅ 图像目录存在"

if [ ! -f "$DISEASE_DESC_PATH" ]; then
    echo "❌ 错误: 疾病描述文件不存在: $DISEASE_DESC_PATH"
    exit 1
fi
echo "✅ 疾病描述文件存在"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$VIZ_DIR"
echo "✅ 输出目录已创建"

# 设置Python路径
export PYTHONPATH="$SRC_DIR:$PYTHONPATH"

# 验证关键文件存在
echo ""
echo "🔍 验证关键Python文件..."

if [ ! -f "$SRC_DIR/eval/grounding_eval_rsna.py" ]; then
    echo "❌ 错误: grounding_eval_rsna.py不存在"
    exit 1
fi
echo "✅ grounding_eval_rsna.py存在"

if [ ! -f "$SRC_DIR/train/clip_modeling_qwen2_5_vl.py" ]; then
    echo "❌ 错误: clip_modeling_qwen2_5_vl.py不存在"
    exit 1
else
    echo "✅ clip_modeling_qwen2_5_vl.py存在"
fi

# 构建评估命令
EVAL_CMD="python src/eval/grounding_eval_rsna.py \
    --model_path \"$MODEL_PATH\" \
    --jsonl_path \"$JSONL_PATH\" \
    --dataset_name \"SIIM_Pneumothorax\" \
    --image_folder \"$IMAGE_FOLDER\" \
    --disease_desc_path \"$DISEASE_DESC_PATH\" \
    --batch_size $BATCH_SIZE \
    --max_samples $MAX_SAMPLES \
    --output_path \"$OUTPUT_PATH\" \
    --save_visualizations \
    --viz_dir \"$VIZ_DIR\" \
    --target_size $TARGET_SIZE"


echo ""
echo "🚀 开始执行SIIM grounding评估..."
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
    echo "🎉 SIIM Pneumothorax grounding评估完成!"
    echo "================================================"
    echo "⏱️  执行时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    echo "📊 结果文件: $OUTPUT_PATH"
    
    # 检查结果文件是否生成
    if [ -f "$OUTPUT_PATH" ]; then
        echo "✅ 结果文件已生成"
        echo "📁 结果目录: $OUTPUT_DIR"
        if [ -d "$VIZ_DIR" ] && [ "$(ls -A $VIZ_DIR)" ]; then
            echo "🖼️  可视化文件已保存到: $VIZ_DIR"
        fi
    else
        echo "⚠️ 结果文件未生成，请检查评估过程"
    fi
    
    echo ""
    echo "📈 评估总结:"
    echo "   数据集: SIIM Pneumothorax"
    echo "   模型: $(basename $MODEL_PATH)"
    echo "   样本数: $MAX_SAMPLES"
    echo "   批次大小: $BATCH_SIZE"
    echo "   图像尺寸: ${TARGET_SIZE}x${TARGET_SIZE}"
    echo ""
    echo "✨ 您可以查看结果文件获取详细的评估指标！"
    
else
    echo ""
    echo "❌ 评估执行失败!"
    echo "================================================"
    echo "🔍 调试信息:"
    echo "   工作目录: $(pwd)"
    echo "   Python路径: $PYTHONPATH"
    echo "   模型路径: $MODEL_PATH"
    echo "   JSONL路径: $JSONL_PATH"
    echo "   图像目录: $IMAGE_FOLDER"
    echo ""
    echo "🔧 请检查:"
    echo "   1. 模型路径是否正确并包含所有必需文件"
    echo "   2. JSONL数据文件格式是否正确"
    echo "   3. GPU内存是否充足"
    echo "   4. Python依赖是否完整安装"
    echo "   5. 导入模块是否存在错误"
    echo ""
    echo "💡 建议:"
    echo "   1. 手动运行: cd $PROJECT_ROOT && python src/eval/grounding_eval_rsna.py --help"
    echo "   2. 检查错误日志中的具体错误信息"
    echo "   3. 验证模型文件完整性"
    echo "   4. 检查SIIM图像文件路径是否正确"
    echo ""
    echo "================================================"
    exit 1
fi

echo ""
echo "✨ siim_grounding_eval_script.sh 执行完成 ✨"
