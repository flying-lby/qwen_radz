--- START OF FILE llava_qwen_a100.py ---

#!/bin/bash
###
 # @Description: Qwen Radz RSNA数据集评测脚本（适配下划线参数格式）
 # @适配Python代码：llava_qwen2.py（参数全为下划线，如--model_path）
###

# ======================================
# 1. 基础环境配置（根据你的路径修改）
# ======================================
# 项目根目录（llava_qwen2.py所在的src/eval的上级目录）
PROJECT_ROOT="/mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune"
# Conda环境路径
CONDA_ENV_PATH="/mnt/shared-storage-user/gaozhenkun/envs/qwen-radz" # eval_A100.py中未提供conda环境路径，沿用原文件
# 模型路径（合并后的Qwen Radz模型）
MODEL_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/2025-09-13/qwen_lora_new_clip_version13/merged"

# 疾病描述文件路径（参考LLaVA-Med的增强配置）
DISEASE_DESC_PATH="/mnt/nlp-ali/usr/huangwenxuan/home/code/llava_test/llava/run/data/disease_desc.json"

# ======================================
# 2. 环境初始化（解决依赖和警告）- 这部分可以放在所有数据集评估之前
# ======================================
echo "🔧 初始化评测环境..."
# 进入项目根目录
cd "$PROJECT_ROOT" || { echo "❌ 无法进入项目目录 $PROJECT_ROOT"; exit 1; }
# 激活Conda环境
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PATH" || { echo "❌ 无法激活Conda环境 $CONDA_ENV_PATH"; exit 1; }
# 安装DICOM依赖（解决"pydicom not available"警告）
pip install pydicom scikit-image -q || echo "⚠️ DICOM依赖已安装，跳过"
# 设置Python路径（确保能导入src下的模块）
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
# 解决DeepSpeed Triton缓存警告（指定非NFS路径）
export TRITON_CACHE_DIR="$PROJECT_ROOT/triton_cache"
mkdir -p "$TRITON_CACHE_DIR" || echo "⚠️ Triton缓存目录已存在，跳过"


# ======================================
# 3. 评测参数配置（与Python代码对齐）
# ======================================
# 与EvaluationArguments对应的参数（下划线格式）
BATCH_SIZE=1                # 批次大小（根据GPU显存调整，eval_A100.py中为1）
USE_DISEASE_DESCRIPTIONS=true  # 启用疾病描述增强（参考LLaVA-Med）
THRESHOLD=0.5               # 分类阈值（二分类默认0.5）
MAX_NEW_TOKENS=128          # 最大生成token数
DEVICE="cuda"               # 设备（cuda/cpu，默认自动识别）
NUM_CHUNKS=1                # 数据分块数（单卡评测设为1）
CHUNK_IDX=0                 # 分块索引（从0开始）


# =================================== ===
# 4. 执行评测（核心命令，参数全下划线）
# ======================================

# --- RSNA 评测 ---
echo -e "\n📊 开始RSNA数据集评测..."
RSNA_IMAGE_FOLDER="/mnt/nlp-ali/usr/zhaizijie/huangwx_ali/zijie_ali/libangyan/dataset/"
RSNA_QUESTION_FILE="/mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/rsna/rsna_pneumonia_llava_origin_val.jsonl"
RSNA_RESULT_DIR="$PROJECT_ROOT/results/rsna_$(date +%Y%m%d_%H%M%S)"
RSNA_RESULT_FILE="$RSNA_RESULT_DIR/rsna_eval_results.json"
mkdir -p "$RSNA_RESULT_DIR" || { echo "❌ 无法创建结果目录 $RSNA_RESULT_DIR"; exit 1; }

echo "📌 模型路径：$MODEL_PATH"
echo "📌 图像路径：$RSNA_IMAGE_FOLDER"
echo "📌 标注文件：$RSNA_QUESTION_FILE"
echo "📌 结果路径：$RSNA_RESULT_FILE"

python -m src.eval.llava_qwen2 \
    --model_path "$MODEL_PATH" \
    --model_base None \
    --image_folder "$RSNA_IMAGE_FOLDER" \
    --question_file "$RSNA_QUESTION_FILE" \
    --result_file "$RSNA_RESULT_FILE" \
    --dataset "rsna" \
    --num_chunks "$NUM_CHUNKS" \
    --chunk_idx "$CHUNK_IDX" \
    --batch_size "$BATCH_SIZE" \
    --use_disease_descriptions "$USE_DISEASE_DESCRIPTIONS" \
    --disease_desc_path "$DISEASE_DESC_PATH" \
    --threshold "$THRESHOLD" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo -e "\n✅ RSNA数据集评测完成！"
    echo "📁 评测结果文件：$RSNA_RESULT_FILE"
    echo "💡 可查看JSON文件获取准确率、F1、AUC等完整指标"
else
    echo -e "\n❌ RSNA数据集评测失败！"
    echo "💡 请检查：1. 参数路径是否正确 2. GPU显存是否充足 3. 图像文件格式是否正常"
    # exit 1 # 如果一个失败就退出，取消注释
fi


# --- SIIM 评测 ---
echo -e "\n📊 开始SIIM数据集评测..."
SIIM_IMAGE_FOLDER="/mnt/nlp-ali/usr/zhaizijie/huangwx_ali/zijie_ali/libangyan/dataset/"
SIIM_QUESTION_FILE="/mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/SIIM_Pneumothorax/SIIM_Pneumothorax_llava_val.jsonl"
SIIM_RESULT_DIR="$PROJECT_ROOT/results/siim_$(date +%Y%m%d_%H%M%S)"
SIIM_RESULT_FILE="$SIIM_RESULT_DIR/siim_eval_results.json"
mkdir -p "$SIIM_RESULT_DIR" || { echo "❌ 无法创建结果目录 $SIIM_RESULT_DIR"; exit 1; }

echo "📌 模型路径：$MODEL_PATH"
echo "📌 图像路径：$SIIM_IMAGE_FOLDER"
echo "📌 标注文件：$SIIM_QUESTION_FILE"
echo "📌 结果路径：$SIIM_RESULT_FILE"

python -m src.eval.llava_qwen2 \
    --model_path "$MODEL_PATH" \
    --model_base None \
    --image_folder "$SIIM_IMAGE_FOLDER" \
    --question_file "$SIIM_QUESTION_FILE" \
    --result_file "$SIIM_RESULT_FILE" \
    --dataset "SIIM_Pneumothorax" \
    --num_chunks "$NUM_CHUNKS" \
    --chunk_idx "$CHUNK_IDX" \
    --batch_size "$BATCH_SIZE" \
    --use_disease_descriptions "$USE_DISEASE_DESCRIPTIONS" \
    --disease_desc_path "$DISEASE_DESC_PATH" \
    --threshold "$THRESHOLD" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo -e "\n✅ SIIM数据集评测完成！"
    echo "📁 评测结果文件：$SIIM_RESULT_FILE"
    echo "💡 可查看JSON文件获取准确率、F1、AUC等完整指标"
else
    echo -e "\n❌ SIIM数据集评测失败！"
    echo "💡 请检查：1. 参数路径是否正确 2. GPU显存是否充足 3. 图像文件格式是否正常"
    # exit 1
fi

# --- COVIDx_CXR 评测 ---
echo -e "\n📊 开始COVIDx_CXR数据集评测..."
COVIDx_IMAGE_FOLDER="/mnt/nlp-ali/usr/zhaizijie/huangwx_ali/zijie_ali/libangyan/dataset/"
COVIDx_QUESTION_FILE="/mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/COVIDx_CXR/COVIDx_CXR_llava_origin_val.jsonl"
COVIDx_RESULT_DIR="$PROJECT_ROOT/results/covid_$(date +%Y%m%d_%H%M%S)"
COVIDx_RESULT_FILE="$COVIDx_RESULT_DIR/covid_eval_results.json"
mkdir -p "$COVIDx_RESULT_DIR" || { echo "❌ 无法创建结果目录 $COVIDx_RESULT_DIR"; exit 1; }

echo "📌 模型路径：$MODEL_PATH"
echo "📌 图像路径：$COVIDx_IMAGE_FOLDER"
echo "📌 标注文件：$COVIDx_QUESTION_FILE"
echo "📌 结果路径：$COVIDx_RESULT_FILE"

python -m src.eval.llava_qwen2 \
    --model_path "$MODEL_PATH" \
    --model_base None \
    --image_folder "$COVIDx_IMAGE_FOLDER" \
    --question_file "$COVIDx_QUESTION_FILE" \
    --result_file "$COVIDx_RESULT_FILE" \
    --dataset "COVIDx_CXR" \
    --num_chunks "$NUM_CHUNKS" \
    --chunk_idx "$CHUNK_IDX" \
    --batch_size "$BATCH_SIZE" \
    --use_disease_descriptions "$USE_DISEASE_DESCRIPTIONS" \
    --disease_desc_path "$DISEASE_DESC_PATH" \
    --threshold "$THRESHOLD" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo -e "\n✅ COVIDx_CXR数据集评测完成！"
    echo "📁 评测结果文件：$COVIDx_RESULT_FILE"
    echo "💡 可查看JSON文件获取准确率、F1、AUC等完整指标"
else
    echo -e "\n❌ COVIDx_CXR数据集评测失败！"
    echo "💡 请检查：1. 参数路径是否正确 2. GPU显存是否充足 3. 图像文件格式是否正常"
    # exit 1
fi

# --- ChestX-ray14 评测 ---
echo -e "\n📊 开始ChestX-ray14数据集评测..."
ChestX_IMAGE_FOLDER="/mnt/nlp-ali/usr/zhaizijie/huangwx_ali/zijie_ali/libangyan/dataset/"
ChestX_QUESTION_FILE="/mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/chest_xray/chest_xray_llava_val.jsonl"
ChestX_RESULT_DIR="$PROJECT_ROOT/results/chestx_$(date +%Y%m%d_%H%M%S)"
ChestX_RESULT_FILE="$ChestX_RESULT_DIR/chestx_eval_results.json"
mkdir -p "$ChestX_RESULT_DIR" || { echo "❌ 无法创建结果目录 $ChestX_RESULT_DIR"; exit 1; }

echo "📌 模型路径：$MODEL_PATH"
echo "📌 图像路径：$ChestX_IMAGE_FOLDER"
echo "📌 标注文件：$ChestX_QUESTION_FILE"
echo "📌 结果路径：$ChestX_RESULT_FILE"

python -m src.eval.llava_qwen2 \
    --model_path "$MODEL_PATH" \
    --model_base None \
    --image_folder "$ChestX_IMAGE_FOLDER" \
    --question_file "$ChestX_QUESTION_FILE" \
    --result_file "$ChestX_RESULT_FILE" \
    --dataset "chestxray" \
    --num_chunks "$NUM_CHUNKS" \
    --chunk_idx "$CHUNK_IDX" \
    --batch_size "$BATCH_SIZE" \
    --use_disease_descriptions "$USE_DISEASE_DESCRIPTIONS" \
    --disease_desc_path "$DISEASE_DESC_PATH" \
    --threshold "$THRESHOLD" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo -e "\n✅ ChestX-ray14数据集评测完成！"
    echo "📁 评测结果文件：$ChestX_RESULT_FILE"
    echo "💡 可查看JSON文件获取准确率、F1、AUC等完整指标"
else
    echo -e "\n❌ ChestX-ray14数据集评测失败！"
    echo "💡 请检查：1. 参数路径是否正确 2. GPU显存是否充足 3. 图像文件格式是否正常"
    # exit 1
fi

# --- CheXpert 评测 ---
echo -e "\n📊 开始CheXpert数据集评测..."
CheXpert_IMAGE_FOLDER="/mnt/nlp-ali/usr/zhaizijie/huangwx_ali/zijie_ali/libangyan/dataset/"
CheXpert_QUESTION_FILE="/mnt/nlp-ali/usr/huangwenxuan/home/code/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/new_data/chexpert/chexpert_llava_val.jsonl"
CheXpert_RESULT_DIR="$PROJECT_ROOT/results/CheXpert_$(date +%Y%m%d_%H%M%S)"
CheXpert_RESULT_FILE="$CheXpert_RESULT_DIR/CheXpert_eval_results.json"
mkdir -p "$CheXpert_RESULT_DIR" || { echo "❌ 无法创建结果目录 $CheXpert_RESULT_DIR"; exit 1; }

echo "📌 模型路径：$MODEL_PATH"
echo "📌 图像路径：$CheXpert_IMAGE_FOLDER"
echo "📌 标注文件：$CheXpert_QUESTION_FILE"
echo "📌 结果路径：$CheXpert_RESULT_FILE"

python -m src.eval.llava_qwen2 \
    --model_path "$MODEL_PATH" \
    --model_base None \
    --image_folder "$CheXpert_IMAGE_FOLDER" \
    --question_file "$CheXpert_QUESTION_FILE" \
    --result_file "$CheXpert_RESULT_FILE" \
    --dataset "chexpert" \
    --num_chunks "$NUM_CHUNKS" \
    --chunk_idx "$CHUNK_IDX" \
    --batch_size "$BATCH_SIZE" \
    --use_disease_descriptions "$USE_DISEASE_DESCRIPTIONS" \
    --disease_desc_path "$DISEASE_DESC_PATH" \
    --threshold "$THRESHOLD" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo -e "\n✅ CheXpert数据集评测完成！"
    echo "📁 评测结果文件：$CheXpert_RESULT_FILE"
    echo "💡 可查看JSON文件获取准确率、F1、AUC等完整指标"
else
    echo -e "\n❌ CheXpert数据集评测失败！"
    echo "💡 请检查：1. 参数路径是否正确 2. GPU显存是否充足 3. 图像文件格式是否正常"
    # exit 1
fi

echo -e "\n🎉 所有数据集评测任务完成！"