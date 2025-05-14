import argparse
import json
import math
import os
from torch.nn.utils.rnn import pad_sequence
import shortuuid
import torch
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
import pandas as pd
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    # eval_tokenizer_image_token,
)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from dataclasses import dataclass
import argparse
from dataclasses import asdict
from transformers import HfArgumentParser
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, recall_score, f1_score, roc_auc_score
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from typing import List
import pydicom
from skimage import exposure

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_classes(args):
      
    # 加载类别数据
    chexray14_cls = ["fibrosis","edema","pneumothorax","cardiomegaly","atelectasis","nodule","emphysema","no finding",
                     "mass","pleural_thickening","effusion","infiltration","pneumonia","hernia","consolidation"]  #Fibrosis seldom appears in MIMIC_CXR and is divided into the 'tail_abnorm_obs' entitiy.  
    
    if args.dataset == 'chexpert':
        chexpert_subset = args.chexpert_subset

        if chexpert_subset == 'False':
            chexpert_cls = [
            'no finding', 'enlarged cardiomediastinum', 'cardiomegaly', 
            'lung opacity', 'lung lesion', 'edema', 'consolidation', 
            'pneumonia', 'atelectasis', 'pneumothorax', 'pleural effusion', 
            'pleural other', 'fracture', 'support devices']
        else:
            chexpert_cls = ['cardiomegaly','edema', 'consolidation', 'atelectasis','pleural effusion']

    siim_cls = ['normal','pneumothorax']
    rsna_cls = ['normal','pneumonia']
    covid_cls = ['normal','covid-19']

    
    original_class = [
                'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
                'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
                'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
                'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
                'tail_abnorm_obs', 'excluded_obs'
            ]
    
    if args.dataset == 'chestxray':
        dataset_cls = chexray14_cls
        question_file = './data/chest_xray/Chest-X-ray_llava_origin_val.jsonl'
    elif args.dataset == 'chexpert':
        dataset_cls = chexpert_cls
        question_file = './data/chexpert/chexpert_llava_origin_val.jsonl'
    elif args.dataset == 'siim':
        dataset_cls = siim_cls
        question_file = './data/SIIM_Pneumothorax/SIIM_Pneumothorax_llava_origin_val.jsonl'
    elif args.dataset == 'rsna':
        dataset_cls = rsna_cls
        question_file = './data/rsna/rsna_pneumonia_llava_origin_val.jsonl'
    elif args.dataset == 'covid-cxr2':
        dataset_cls = covid_cls
        question_file = './data/COVIDx_CXR/COVIDx_CXR_llava_origin_val.jsonl'
  
        
    original_class.extend(item for item in dataset_cls if item not in original_class)
    # original_class = dataset_cls
    mapping = []
    for disease in dataset_cls:
        if disease in original_class:
            print(disease)
            mapping.append(original_class.index(disease))
        else:
            mapping.append(-1)
    MIMIC_mapping = [ _ for i,_ in enumerate(mapping) if _ != -1] # valid MIMIC class index
    dataset_mapping = [ i for i,_ in enumerate(mapping) if _ != -1] # valid (exist in MIMIC) chexray class index
    target_class = [dataset_cls[i] for i in dataset_mapping ] # Filter out non-existing class
    
    return target_class,question_file

def eval_model(args, classes,question_file):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map='cuda:0')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/srv/lby/qwen_vl_7b/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # questions = random.sample(questions, min(100, len(questions)))
    
    answers_file = os.path.expanduser(args.output_path)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["label"]
        image_file = os.path.join(args.image_folder,line["image"])
        
        qs = line["text"].replace('<image>', '').strip()
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        cur_prompt = '<image>' + '\n' + cur_prompt
        # qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
        # cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
        if args.use_cot:
            cot = " Let's think step by step."
        else:
            cot=""
        qs = qs + cot
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda(0)

        try:
            if args.dataset == 'rsna':
                img = pydicom.dcmread(image_file).pixel_array  # 读取 DICOM 图像数据
                img = img.astype(float) / 255.0  # 归一化图像
                img = exposure.equalize_hist(img)  # 直方图均衡化

                # 转换为 PIL 图像并应用预处理
                img = (255 * img).astype(np.uint8)  # 转换为 uint8 类型
                image = Image.fromarray(img).convert('RGB') 
                # image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)[0].cuda(0)
            else:
                image = Image.open(image_file).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)[0].to(device)
        except Exception as e:
            print(f"Warning: Skipping image {image_file} due to error: {e}")
            continue 
        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(0),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()



# 通过疾病名字计算指标
def get_metrics1(args, classes, question_file):
    import json
    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score

    # 读取数据
    answers = [json.loads(line) for line in open(args.output_path)]
    disease_list = classes
    print(f"Total number of answers: {len(answers)}")

    # 存储真实标签和预测标签
    y_true = []
    y_pred = []

    # 遍历每个 answer，提取真实标签和预测类别
    for item in answers:
        labels = item["question_id"]
        text = item["text"].lower()

        # 预测类别
        if len(disease_list) == 2:  # 二分类
            predicted_categories = [0] * len(disease_list)
            for disease in disease_list:
                if disease in text:
                    predicted_categories[disease_list.index(disease)] = 1
                    break
        else:  # 多标签分类
            predicted_categories = [1 if disease in text else 0 for disease in disease_list]

        # 生成真实标签向量
        true_labels = torch.zeros(len(disease_list))
        for disease, value in labels.items():
            if value == 1 and disease in disease_list:
                true_labels[disease_list.index(disease)] = 1

        y_true.append(true_labels.numpy())
        y_pred.append(predicted_categories)

    # 转换为 NumPy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算 AUC 指标
    try:
        if len(disease_list) > 2:
            valid_indices = [i for i in range(len(disease_list)) if len(set(y_true[:, i])) > 1]
            if valid_indices:
                auc_micro = roc_auc_score(y_true[:, valid_indices], y_pred[:, valid_indices], average='micro')
                auc_macro = roc_auc_score(y_true[:, valid_indices], y_pred[:, valid_indices], average='macro')
            else:
                auc_micro, auc_macro = 0, 0
        else:
            auc_micro = roc_auc_score(y_true[:, 1], y_pred[:, 1])
            auc_macro = auc_micro
    except ValueError:
        auc_micro, auc_macro = 0, 0

    # -------------------------------
    # 按样本层面计算自定义的平均准确率和 F1 分数
    # -------------------------------

    total_correct = 0
    total_true_labels = 0
    sample_f1_scores = []

    for i in range(y_true.shape[0]):
        true_labels = y_true[i]
        pred_labels = y_pred[i]
        
        # 计算当前样本中正确预测正标签的个数和真实正标签总数
        correct_count = np.sum((true_labels == 1) & (pred_labels == 1))
        true_count = np.sum(true_labels == 1)
        total_correct += correct_count
        total_true_labels += true_count

        # 计算当前样本的 F1 分数
        # 如果当前样本没有任何真实标签或预测标签，则定义 F1 为 0
        pred_count = np.sum(pred_labels == 1)
        if true_count == 0 or pred_count == 0:
            sample_f1 = 0
        else:
            precision = correct_count / pred_count
            recall = correct_count / true_count
            sample_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        sample_f1_scores.append(sample_f1)

    custom_acc = total_correct / total_true_labels if total_true_labels > 0 else 0
    custom_f1 = np.mean(sample_f1_scores)

    # 输出结果
    print(f"Custom Average Accuracy: {custom_acc * 100:.2f}%")
    print(f"Custom Average F1 Score: {custom_f1:.4f}")
    print(f"AUC (Micro): {auc_micro}")
    print(f"AUC (Macro): {auc_macro}")

    result = {
        "average_accuracy": custom_acc,
        "auc_micro": auc_micro,
        "auc_macro": auc_macro,
        "f1": custom_f1
    }

    result_dir = os.path.dirname(args.result_file)

    # 确保目录存在
    os.makedirs(result_dir, exist_ok=True)
    with open(args.result_file, 'w') as f:
        json.dump(result, f, indent=4)

# 通过疾病索引计算指标 A,B,C,D...

def get_metrics2(args,classes,question_file):
    # 读取数据
    answers = [json.loads(line) for line in open(args.output_path)]

    # 疾病类别及其索引 A, B, C, D...
    # disease_list = [
    #     'fibrosis', 'edema', 'pneumothorax', 'cardiomegaly', 'atelectasis', 
    #     'nodule', 'emphysema', 'no finding', 'mass', 'pleural_thickening', 
    #     'effusion', 'infiltration', 'pneumonia', 'hernia', 'consolidation'
    # ]
    disease_indices = [chr(65 + i) for i in range(len(classes))]  # A, B, C, ..., O

    print(f"Total number of answers: {len(answers)}")

    # 存储真实标签和预测标签
    y_true = []
    y_pred = []

    # 遍历每个 answer，提取 labels 和预测类别
    for item in answers:
        # 真实标签（A, B, C, D...）
        labels = item["question_id"]

        # 获取预测的 text
        text = item["text"].upper().strip()  # 转换为大写匹配 A, B, C, D...
        predicted_categories = [1 if disease in text else 0 for disease in disease_indices]

        # 生成真实标签向量
        true_labels = [1 if disease in labels else 0 for disease in disease_indices]

        y_true.append(true_labels)
        y_pred.append(predicted_categories)

    # 转换为 NumPy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算 AUC（先筛选掉全 0 或全 1 的类别）
    valid_indices = [i for i in range(len(disease_indices)) if len(set(y_true[:, i])) > 1]

    if valid_indices:
        auc_micro = roc_auc_score(y_true[:, valid_indices], y_pred[:, valid_indices], average='micro')
        auc_macro = roc_auc_score(y_true[:, valid_indices], y_pred[:, valid_indices], average='macro')
    else:
        auc_micro, auc_macro = 0, 0  # 避免计算错误

    # 计算 F1 分数
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # 计算每个类别的准确率
    category_accuracies = (y_true * y_pred).sum(axis=0) / y_true.sum(axis=0) * 100
    category_accuracies = {
        classes[i]: acc if not np.isnan(acc) else 0 
        for i, acc in enumerate(category_accuracies)
    }

    # 计算类别平均准确率
    average_accuracy = sum(category_accuracies.values()) / len(category_accuracies)

    # 输出结果
    print(f"Category accuracies: {category_accuracies}")
    print(f"Average accuracy: {average_accuracy}%")
    print(f"AUC (Micro): {auc_micro}")
    print(f"AUC (Macro): {auc_macro}")
    print(f"F1 Score (Micro): {f1_micro}")
    print(f"F1 Score (Macro): {f1_macro}")

    result = {
        "category_accuracies": category_accuracies,
        "average_accuracy": average_accuracy,
        "auc_micro": auc_micro,
        "auc_macro": auc_macro,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro
    }

    with open(args.result_file, 'w') as f:
        json.dump(result, f, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/srv/lby/llava_med/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--use-cot", type=int, default=0)
    parser.add_argument("--output-path", type=str, default="./data/chest_xray/Chest-X-ray_llava_origin_val_ans.jsonl")
    parser.add_argument("--dataset", type=str, default="siim")
    parser.add_argument("--chexpert-subset", type=str, default="False")
    parser.add_argument("--result-file", type=str, default="./result/chest_xray/Chest-X-ray_classify.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--inference", type=str, default="origin")
    args, remaining_args = parser.parse_known_args()
    
    # Use HfArgumentParser for SparseArguments
    # hf_parser = HfArgumentParser(SparseArguments)
    # sparse_args, = hf_parser.parse_args_into_dataclasses(remaining_args)
    
  
    # /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_ft2
    # /srv/lby/llava_med/checkpoints/llava-llava-mistral_ft2
    

    classes,question_file = get_classes(args)
    if  args.inference == "clip":
        clip_eval_model(args,classes,question_file)
    else:
        eval_model(args,classes,question_file)
        get_metrics1(args,classes,question_file)
        
        # # 根据 model_path 选择合适的 metrics 计算方式
        # if "llava_med" in args.model_path.lower():
        #     get_metrics1(args,classes,question_file )
        # else:
        #     if "sft" in args.model_path.lower():
        #         get_metrics1(args,classes,question_file)
        #     else:
        #         get_metrics2(args,classes,question_file)  