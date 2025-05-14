###
 # @Author: fly
 # @Date: 2024-12-26 16:55:54
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/scripts_A100/2025-02-09/llava_mistral_clip_A100_v1_img4_txt8_lr2e5_layer2_2epoch_2A100/eval_chestxray_classify.sh
 # @Description: 
### 

# ========================
# Testing/Evaluation
# ========================
echo "Starting evaluation process..."

python -m eval.run_eval \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/qwen_vl_7b/Qwen2.5-VL-7B-Instruct \
    --answer_file ./results/Qwen2.5-VL-7B-Instruct/siim/answers.jsonl \
    --image-folder "/mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/" \
    --dataset "siim" \
    --result-file ./results/Qwen2.5-VL-7B-Instruct/siim/siim_qwen7b_metrics.json

python -m eval.run_eval \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/qwen_vl_7b/Qwen2.5-VL-7B-Instruct \
    --answer_file ./results/Qwen2.5-VL-7B-Instruct/covid/answers.jsonl \
    --image-folder "/mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/" \
    --dataset "covid-cxr2" \
    --result-file ./results/Qwen2.5-VL-7B-Instruct/covid/covid_qwen7b_metrics.json

python -m eval.run_eval \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/qwen_vl_7b/Qwen2.5-VL-7B-Instruct \
    --answer_file ./results/Qwen2.5-VL-7B-Instruct/chexpert/answers.jsonl \
    --image-folder "/mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/" \
    --dataset "chexpert" \
    --result-file ./results/Qwen2.5-VL-7B-Instruct/chexpert/chexpert_qwen7b_metrics.json


python -m eval.run_eval \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/qwen_vl_7b/Qwen2.5-VL-7B-Instruct \
    --answer_file ./results/Qwen2.5-VL-7B-Instruct/chestxray/answers.jsonl \
    --image-folder "/mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/" \
    --dataset "chestxray" \
    --result-file ./results/Qwen2.5-VL-7B-Instruct/chestxray/chestxray_qwen7b_metrics.json

