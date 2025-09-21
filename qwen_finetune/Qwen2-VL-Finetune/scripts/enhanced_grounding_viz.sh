#!/bin/bash
###
# @Description: Enhanced Grounding Visualization Script - GT vs Prediction对比可视化
# @Usage: ./enhanced_grounding_viz.sh [STRATEGY] [NUM_SAMPLES]
# @Examples:
#   ./enhanced_grounding_viz.sh balanced 15
#   ./enhanced_grounding_viz.sh quality 20 
#   ./enhanced_grounding_viz.sh diverse 12
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

# 参数配置
VIZ_STRATEGY=${1:-"balanced"}  # 可视化策略：balanced, quality, diverse, challenging
NUM_VIZ_SAMPLES=${2:-15}      # 可视化样本数量
BATCH_SIZE=4
MAX_SAMPLES=100               # 限制评估样本数以加快生成速度

# 模型和数据路径配置
MODEL_PATH="/srv/lby/qwen_radz/qwen_lora_new_clip_version1/merged"
IMAGE_FOLDER="/srv/lby/"
DISEASE_DESC_PATH="/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/observation_explanation.json"

# 输出配置
RESULT_DIR="$PROJECT_ROOT/enhanced_visualizations"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VIZ_OUTPUT_DIR="$RESULT_DIR/gt_vs_prediction_${VIZ_STRATEGY}_${NUM_VIZ_SAMPLES}samples_${TIMESTAMP}"

mkdir -p "$RESULT_DIR"

echo "============================================"
echo "🎨 增强Grounding可视化生成器"
echo "============================================"
echo "模型路径: $MODEL_PATH"
echo "图像文件夹: $IMAGE_FOLDER"
echo "可视化策略: $VIZ_STRATEGY"
echo "样本数量: $NUM_VIZ_SAMPLES"
echo "输出目录: $VIZ_OUTPUT_DIR"
echo "============================================"

# 检查评估脚本是否存在
if [ ! -f "src/eval/eval_grounding.py" ]; then
    echo "❌ 错误: eval_grounding.py不存在"
    exit 1
fi

# 检查可视化策略参数
case $VIZ_STRATEGY in
    balanced|quality|diverse|challenging)
        echo "✅ 使用可视化策略: $VIZ_STRATEGY"
        ;;
    *)
        echo "❌ 错误: 无效的可视化策略: $VIZ_STRATEGY"
        echo "   支持的策略: balanced, quality, diverse, challenging"
        exit 1
        ;;
esac

# ===== 生成RSNA数据集的增强可视化 =====
echo ""
echo "🎨 生成RSNA数据集的GT vs Prediction可视化..."
RSNA_VIZ_DIR="$VIZ_OUTPUT_DIR/rsna"
RSNA_OUTPUT_FILE="$RSNA_VIZ_DIR/evaluation_results.json"

if python src/eval/eval_grounding.py \
    --model_path "$MODEL_PATH" \
    --jsonl_path "/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/rsna/rsna_pneumonia_llava_origin_val.jsonl" \
    --image_folder "$IMAGE_FOLDER" \
    --disease_desc_path "$DISEASE_DESC_PATH" \
    --dataset_name "RSNA_Pneumonia" \
    --batch_size $BATCH_SIZE \
    --max_samples $MAX_SAMPLES \
    --output_path "$RSNA_OUTPUT_FILE" \
    --save_visualizations \
    --enhanced_viz \
    --num_viz_samples $NUM_VIZ_SAMPLES \
    --viz_strategy $VIZ_STRATEGY \
    --viz_dir "$RSNA_VIZ_DIR" \
    --target_size 224; then
    echo "✅ RSNA可视化生成完成"
    echo "📁 可视化保存至: $RSNA_VIZ_DIR/gt_vs_prediction/"
else
    echo "❌ RSNA可视化生成失败"
    exit 1
fi

# ===== 生成SIIM数据集的增强可视化 =====
echo ""
echo "🎨 生成SIIM数据集的GT vs Prediction可视化..."
SIIM_VIZ_DIR="$VIZ_OUTPUT_DIR/siim"
SIIM_OUTPUT_FILE="$SIIM_VIZ_DIR/evaluation_results.json"

if python src/eval/eval_grounding.py \
    --model_path "$MODEL_PATH" \
    --jsonl_path "/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/SIIM_Pneumothorax/SIIM_Pneumothorax_llava_origin_val.jsonl" \
    --dataset_name "SIIM_Pneumothorax" \
    --image_folder "$IMAGE_FOLDER" \
    --disease_desc_path "$DISEASE_DESC_PATH" \
    --batch_size $BATCH_SIZE \
    --max_samples $MAX_SAMPLES \
    --output_path "$SIIM_OUTPUT_FILE" \
    --save_visualizations \
    --enhanced_viz \
    --num_viz_samples $NUM_VIZ_SAMPLES \
    --viz_strategy $VIZ_STRATEGY \
    --viz_dir "$SIIM_VIZ_DIR" \
    --target_size 224; then
    echo "✅ SIIM可视化生成完成"
    echo "📁 可视化保存至: $SIIM_VIZ_DIR/gt_vs_prediction/"
else
    echo "❌ SIIM可视化生成失败"
    exit 1
fi

echo ""
echo "🎉 所有增强可视化生成完成！"
echo "============================================"
echo "📊 生成统计:"
echo "   策略: $VIZ_STRATEGY"
echo "   每个数据集样本数: $NUM_VIZ_SAMPLES"
echo "   总生成图片数: ~$((NUM_VIZ_SAMPLES * 2 + 2)) 张"
echo ""
echo "📁 输出文件："
echo "   RSNA GT vs Prediction: $RSNA_VIZ_DIR/gt_vs_prediction/"
echo "   RSNA 网格展示: $RSNA_VIZ_DIR/gt_vs_prediction/samples_grid.png"
echo "   SIIM GT vs Prediction: $SIIM_VIZ_DIR/gt_vs_prediction/"
echo "   SIIM 网格展示: $SIIM_VIZ_DIR/gt_vs_prediction/samples_grid.png"
echo ""
echo "🎨 可视化特性："
echo "   - 左右对比：GT (左) vs Prediction (右)"
echo "   - 红色高亮显示病理区域"
echo "   - 显示Dice和IoU分数"
echo "   - 高分辨率输出 (300 DPI)"
echo "============================================"

# 生成查看脚本
VIEW_SCRIPT="$VIZ_OUTPUT_DIR/view_results.sh"
cat > "$VIEW_SCRIPT" << 'EOF'
#!/bin/bash
# 快速查看生成的可视化结果

echo "🖼️  查看增强可视化结果"
echo "============================================"

if command -v eog >/dev/null 2>&1; then
    VIEWER="eog"
elif command -v display >/dev/null 2>&1; then
    VIEWER="display" 
elif command -v feh >/dev/null 2>&1; then
    VIEWER="feh"
else
    echo "❌ 未找到图像查看器 (eog, display, feh)"
    echo "请手动打开以下目录查看："
    echo "$(dirname "$0")/rsna/gt_vs_prediction/"
    echo "$(dirname "$0")/siim/gt_vs_prediction/"
    exit 1
fi

echo "📁 使用 $VIEWER 查看可视化结果..."
echo ""

# 查看RSNA网格图
RSNA_GRID="$(dirname "$0")/rsna/gt_vs_prediction/samples_grid.png"
if [ -f "$RSNA_GRID" ]; then
    echo "🔍 打开RSNA样本网格图..."
    $VIEWER "$RSNA_GRID" &
fi

# 查看SIIM网格图  
SIIM_GRID="$(dirname "$0")/siim/gt_vs_prediction/samples_grid.png"
if [ -f "$SIIM_GRID" ]; then
    echo "🔍 打开SIIM样本网格图..."
    $VIEWER "$SIIM_GRID" &
fi

echo ""
echo "💡 提示："
echo "   - 网格图显示所有选定样本的概览"
echo "   - 单个样本图在各自的子目录中"
echo "   - 左侧是GT(真实标注)，右侧是Prediction(模型预测)"
echo "   - 红色高亮区域表示病理位置"

EOF

chmod +x "$VIEW_SCRIPT"

echo ""
echo "💡 快捷查看脚本已生成: $VIEW_SCRIPT"
echo "   运行 '$VIEW_SCRIPT' 可快速查看网格图"
echo ""
echo "✨ 增强可视化生成完成！"
