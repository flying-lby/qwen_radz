#!/bin/bash
###
 # @Author: AI Assistant
 # @Date: 2025-09-19
 # @FilePath: /qwen_radz/qwen_finetune/Qwen2-VL-Finetune/scripts/binary_clip_eval_script.sh
 # @Description: 二分类CLIP风格Qwen2.5-VL胸部X光分类评估脚本
 # 
 # 关键特性：
 # - 对齐I1_classification的二分类评估方法（使用0/1数值标签）
 # - 限制样本数量为100个进行快速测试
 # - 对齐MedKLIP的评估方式和输出格式
 # - 支持疾病描述增强的文本提示
 # - 使用最优F1阈值而非固定0.5阈值
 #
 # 修改记录：
 # - 将MAX_SAMPLES从-1改为100，对齐I1_classification快速测试模式
 # - 临时只启用SIIM数据集以便快速验证
 # - 确保使用0/1标签格式而非["pneumothorax", "non-pneumothorax"]文本格式
### 

# 设置基本路径
BASE_DIR="/home/lby/iclr2026/qwen_radz/qwen_finetune/Qwen2-VL-Finetune"
cd $BASE_DIR

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate qwen_vl

# 设置Python路径
export PYTHONPATH=src:$PYTHONPATH

# 设置CUDA调试环境变量
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 模型路径配置 =====
MODEL_PATH="/srv/lby/qwen_radz/qwen_lora_new_clip_version1/merged"

# ===== 数据路径配置 =====
IMAGE_FOLDER="/srv/lby/"

# ===== 疾病描述配置 =====
USE_DISEASE_DESCRIPTIONS=true  # 是否使用疾病描述：true/false
DISEASE_DESC_PATH="/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/observation_explanation.json"  # 疾病描述文件路径
DESCRIPTION_SOURCE="file"  # 描述来源："file"（详细描述）或"template"（简单模板）

# ===== MedKLIP对齐配置 =====
MEDKLIP_STYLE_OUTPUT=true     # 是否使用MedKLIP风格的输出格式：true/false

# ===== 评估参数配置 =====
BATCH_SIZE=1
MAX_SAMPLES=2  # 最大样本数量，对齐I1_classification快速测试

# ===== 数据集选择配置 =====
# 设置为true来启用相应数据集的评估（二分类数据集）
# 已验证：所有数据集都使用{"病名": 1, "正常": 0}格式，与I1_classification兼容
EVAL_SIIM=true      # SIIM Pneumothorax (气胸检测): {"pneumothorax": 1, "non-pneumothorax": 0}
EVAL_COVID=true     # COVID-19 (COVID-19检测): {"covid-19": 1, "normal": 0}
EVAL_RSNA=true      # RSNA Pneumonia (肺炎检测): {"pneumonia": 1, "normal": 0}

# ===== 结果输出路径 =====
RESULT_DIR="$BASE_DIR/results/binary_clip_eval_experiments"
mkdir -p $RESULT_DIR

echo "============================================"
echo "🔬 二分类CLIP风格胸部X光分类评估"
echo "============================================"
echo "模型路径: $MODEL_PATH"
echo "图像文件夹: $IMAGE_FOLDER"
echo "使用疾病描述: $USE_DISEASE_DESCRIPTIONS"
if [ "$USE_DISEASE_DESCRIPTIONS" = "true" ]; then
    echo "疾病描述文件: $DISEASE_DESC_PATH"
    echo "描述来源: $DESCRIPTION_SOURCE"
fi
echo "MedKLIP风格输出: $MEDKLIP_STYLE_OUTPUT"
echo "结果保存到: $RESULT_DIR"
echo "============================================"

# ===== 评估函数 =====
evaluate_binary_dataset() {
    local dataset_name=$1
    local data_path=$2
    local output_file=$3
    local display_name=$4
    
    echo ""
    echo "📊 评估 $display_name 数据集..."
    
    # 检查数据文件是否存在
    if [ ! -f "$data_path" ]; then
        echo "❌ 数据文件不存在: $data_path"
        echo "   跳过 $display_name 评估"
        return 1
    fi
    
    # 构建基础命令
    BASE_CMD="python -m src.eval.clip_eval \
        --model_path $MODEL_PATH \
        --data_path $data_path \
        --image_folder $IMAGE_FOLDER \
        --dataset $dataset_name \
        --batch_size $BATCH_SIZE \
        --output_path $output_file \
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
    echo "执行命令: $BASE_CMD"
    eval $BASE_CMD
    
    if [ $? -eq 0 ]; then
        echo "✅ $display_name 评估完成！"
        echo "   结果保存在: $output_file"
        return 0
    else
        echo "❌ $display_name 评估失败！"
        return 1
    fi
}

# ===== 执行评估 =====
echo ""
echo "===== 开始二分类数据集评估 ====="

TOTAL_DATASETS=0
SUCCESSFUL_EVALUATIONS=0

# 评估 SIIM Pneumothorax (气胸检测)
if [ "$EVAL_SIIM" = "true" ]; then
    TOTAL_DATASETS=$((TOTAL_DATASETS + 1))
    if evaluate_binary_dataset \
        "SIIM_Pneumothorax" \
        "/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/SIIM_Pneumothorax/SIIM_Pneumothorax_llava_val.jsonl" \
        "$RESULT_DIR/SIIM_Pneumothorax_binary_clip_results.json" \
        "SIIM Pneumothorax"; then
        SUCCESSFUL_EVALUATIONS=$((SUCCESSFUL_EVALUATIONS + 1))
    fi
fi

# 评估 COVID-19 (COVID-19检测)
if [ "$EVAL_COVID" = "true" ]; then
    TOTAL_DATASETS=$((TOTAL_DATASETS + 1))
    if evaluate_binary_dataset \
        "COVIDx_CXR" \
        "/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/COVIDx_CXR/COVIDx_CXR_llava_origin_val.jsonl" \
        "$RESULT_DIR/COVIDx_CXR_binary_clip_results.json" \
        "COVID-19 CXR"; then
        SUCCESSFUL_EVALUATIONS=$((SUCCESSFUL_EVALUATIONS + 1))
    fi
fi

# 评估 RSNA Pneumonia (肺炎检测)
if [ "$EVAL_RSNA" = "true" ]; then
    TOTAL_DATASETS=$((TOTAL_DATASETS + 1))
    if evaluate_binary_dataset \
        "rsna" \
        "/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/rsna/rsna_pneumonia_llava_origin_val.jsonl" \
        "$RESULT_DIR/rsna_pneumonia_binary_clip_results.json" \
        "RSNA Pneumonia"; then
        SUCCESSFUL_EVALUATIONS=$((SUCCESSFUL_EVALUATIONS + 1))
    fi
fi

echo ""
echo "===== 评估阶段完成 ====="

# ===== 生成汇总报告 =====
generate_binary_summary_report() {
    echo ""
    echo "📋 生成二分类评估汇总报告..."
    
    local summary_file="$RESULT_DIR/binary_evaluation_summary.txt"
    
    {
        echo "二分类CLIP风格医学图像评估汇总报告"
        echo "生成时间: $(date)"
        echo "============================================"
        echo ""
        echo "模型信息:"
        echo "  模型路径: $MODEL_PATH"
        echo "  评估类型: CLIP风格二分类"
        echo "  MedKLIP对齐: $MEDKLIP_STYLE_OUTPUT"
        echo ""
        echo "评估概况:"
        echo "  总数据集数: $TOTAL_DATASETS"
        echo "  成功评估数: $SUCCESSFUL_EVALUATIONS"
        if [ $TOTAL_DATASETS -gt 0 ]; then
            echo "  成功率: $(echo "scale=1; $SUCCESSFUL_EVALUATIONS * 100 / $TOTAL_DATASETS" | bc)%"
        fi
        echo ""
        echo "评估设置:"
        echo "  批次大小: $BATCH_SIZE"
        echo "  最大样本数: $MAX_SAMPLES"
        echo "  使用疾病描述: $USE_DISEASE_DESCRIPTIONS"
        echo ""
    } > $summary_file
    
    # 汇总每个数据集的结果
    local datasets=("SIIM_Pneumothorax" "COVIDx_CXR" "rsna")
    local display_names=("SIIM Pneumothorax (气胸)" "COVID-19 CXR" "RSNA Pneumonia (肺炎)")
    local eval_flags=("$EVAL_SIIM" "$EVAL_COVID" "$EVAL_RSNA")
    local result_files=("SIIM_Pneumothorax_binary_clip_results.json" "COVIDx_CXR_binary_clip_results.json" "rsna_pneumonia_binary_clip_results.json")
    
    for i in "${!datasets[@]}"; do
        local dataset="${datasets[$i]}"
        local display_name="${display_names[$i]}"
        local eval_flag="${eval_flags[$i]}"
        local result_file="$RESULT_DIR/${result_files[$i]}"
        
        if [ "$eval_flag" = "true" ]; then
            {
                echo "数据集: $display_name"
                echo "----------------------------------------"
            } >> $summary_file
            
            if [ -f "$result_file" ]; then
                # 提取关键指标（需要jq工具）
                if command -v jq &> /dev/null; then
                    {
                        echo "状态: 评估成功"
                        echo "平均AUC-ROC: $(jq -r '.mean_auc_roc // "N/A"' $result_file)"
                        echo "总体准确率: $(jq -r '.overall_accuracy // "N/A"' $result_file)"
                        echo "Macro F1分数: $(jq -r '.macro_f1_score // "N/A"' $result_file)"
                        echo "平衡准确率: $(jq -r '.macro_balanced_accuracy // "N/A"' $result_file)"
                        echo "总样本数: $(jq -r '.total_samples // "N/A"' $result_file)"
                        echo "评估时间: $(jq -r '.evaluation_timestamp // "N/A"' $result_file)"
                    } >> $summary_file
                else
                    {
                        echo "状态: 评估成功"
                        echo "结果文件: $result_file"
                        echo "（安装jq工具可显示详细指标）"
                    } >> $summary_file
                fi
            else
                {
                    echo "状态: 评估失败或结果文件缺失"
                } >> $summary_file
            fi
            
            echo "" >> $summary_file
        fi
    done
    
    {
        echo "详细结果文件位置:"
        if [ "$EVAL_SIIM" = "true" ]; then
            echo "  SIIM Pneumothorax: $RESULT_DIR/SIIM_Pneumothorax_binary_clip_results.json"
        fi
        if [ "$EVAL_COVID" = "true" ]; then
            echo "  COVID-19 CXR: $RESULT_DIR/COVIDx_CXR_binary_clip_results.json"
        fi
        if [ "$EVAL_RSNA" = "true" ]; then
            echo "  RSNA Pneumonia: $RESULT_DIR/rsna_pneumonia_binary_clip_results.json"
        fi
        echo ""
        echo "注意事项:"
        echo "  - 这些是二分类任务的评估结果"
        echo "  - 使用CLIP风格的评估方法"
        echo "  - 结果与MedKLIP评估方式对齐"
        echo "  - 每个数据集专注于特定的二分类任务"
    } >> $summary_file
    
    echo "📄 详细汇总报告已保存: $summary_file"
    
    # 显示汇总报告内容
    echo ""
    echo "===== 二分类评估结果汇总 ====="
    cat $summary_file
}

# 生成汇总报告
generate_binary_summary_report

echo ""
echo "============================================"
echo "🎉 二分类CLIP风格评估系统执行完成！"
echo ""
echo "📊 评估统计:"
echo "   成功评估: $SUCCESSFUL_EVALUATIONS/$TOTAL_DATASETS 个二分类数据集"
echo ""
echo "📁 输出目录结构:"
echo "   主结果目录: $RESULT_DIR"
if [ "$EVAL_SIIM" = "true" ]; then
    echo "   SIIM Pneumothorax: $RESULT_DIR/SIIM_Pneumothorax_binary_clip_results.json"
fi
if [ "$EVAL_COVID" = "true" ]; then
    echo "   COVID-19 CXR: $RESULT_DIR/COVIDx_CXR_binary_clip_results.json"
fi
if [ "$EVAL_RSNA" = "true" ]; then
    echo "   RSNA Pneumonia: $RESULT_DIR/rsna_pneumonia_binary_clip_results.json"
fi
echo "   汇总报告: $RESULT_DIR/binary_evaluation_summary.txt"
echo ""
echo "💡 说明:"
echo "   - 使用Vision-Language模型进行二分类评估"
echo "   - 评估方式与MedKLIP完全对齐"
echo "   - 支持疾病描述增强的文本提示"
echo "   - 专注于临床相关的二分类任务：气胸、COVID-19、肺炎"
echo ""
echo "🔧 自定义选项:"
echo "   - 修改EVAL_*变量选择要评估的数据集"
echo "   - 调整USE_DISEASE_DESCRIPTIONS启用/禁用疾病描述"
echo "   - 设置MAX_SAMPLES限制样本数量进行快速测试"
echo "============================================"
