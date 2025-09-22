#!/bin/bash
###
# 运行Grounding可视化脚本
###

# 进入项目目录
cd /home/lby/iclr2026/qwen_radz/qwen_finetune/Qwen2-VL-Finetune

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate qwen_vl

# 设置Python路径
export PYTHONPATH=src:$PYTHONPATH

# 设置CUDA环境
export CUDA_VISIBLE_DEVICES=0

echo "🎨 Starting Grounding Visualization..."
echo "=================================="

# 运行可视化脚本
python src/eval/grounding_visualization.py

echo ""
echo "🎉 Visualization completed!"
echo "📁 Check the 'visualizations' folder for results"
