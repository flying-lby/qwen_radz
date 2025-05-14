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
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    # eval_tokenizer_image_token,
)
from llava.model.clip_llava_builder import load_pretrained_model
from llava.utils import disable_torch_init
from dataclasses import dataclass
import argparse
from dataclasses import asdict
from transformers import HfArgumentParser
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, recall_score, f1_score, roc_auc_score
import pandas as pd
import pydicom
from skimage import exposure

@dataclass
class SparseArguments:
    Imgcls_count: int = 4
    Txtcls_count: int = 4
    hidden_dim: int = 1024
    output_dim: int = 512
    img_mlp_type: int = 1
    txt_mlp_type: int = 1
    knowledge_mlp_type: int = 1
    loss_threshold: float = 0.5
    temperature: float = 0.05
    use_local_loss: bool = False
    feature_layer: int = 1
    special_tokens_mlp_type: int = 1
    use_ca_loss: bool = True
    inference_type: int = 2
    use_cat: bool = True
    use_prompt: bool = True
    Book_choice: int = 1


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



def test(args, sparse_args):
    # Model
    # disable_torch_init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------- 模型加载与全局类别嵌入计算 ---------------------
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, sparse_args # 让 DataParallel 处理 device
    )

    # 多 GPU 推理
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])  # 指定 GPU 0 和 1
    model.to(device)
    model.eval()
    if hasattr(model, "module"):
        model_config = model.module.config
    else:
        model_config = model.config
    
    # 加载类别数据
    chexray14_cls = ["fibrosis","edema","pneumothorax","cardiomegaly","atelectasis","nodule","emphysema","no finding",
                     "mass","pleural_thickening","effusion","infiltration","pneumonia","hernia","consolidation"]  #Fibrosis seldom appears in MIMIC_CXR and is divided into the 'tail_abnorm_obs' entitiy.  
    mura_cls = lera_cls = ['abnormality']
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

    siim_cls = ['pneumothorax', 'non-pneumothorax']
    rsna_cls = ['pneumonia','normal']
    covid_cls = ['covid19','non-covid19']

    padchest_seen_class = ['normal', 'pleural effusion', 'pacemaker', 'atelectasis', 'pneumonia', 'consolidation', 'cardiomegaly', 'emphysema', 
                           'nodule', 'edema', 'pneumothorax', 'fracture', 'mass', 'catheter']


    original_class = [
                'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
                'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
                'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
                'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
                'tail_abnorm_obs', 'excluded_obs'
            ]
    
    padchest_rare = ['suture material', 'sternotomy', 'supra aortic elongation', 'metal', 'abnormal foreign body', 'central venous catheter via jugular vein', 'vertebral anterior compression', 'diaphragmatic eventration', #'consolidation', 
    'calcified densities', 'volume loss', 'single chamber device', 'vertebral compression', 'bullas', 'axial hyperostosis', 'aortic button enlargement', 'calcified granuloma', 'clavicle fracture', 'dual chamber device', 'mediastinic lipomatosis',
                     'esophagic dilatation', 'azygoesophageal recess shift', 'breast mass', 'round atelectasis', 'surgery humeral', 'aortic aneurysm', 'nephrostomy tube', 'sternoclavicular junction hypertrophy', 'pulmonary artery hypertension', 'pleural mass', 'empyema', 'external foreign body', 'respiratory distress', 'total atelectasis', 'ventriculoperitoneal drain tube', 'right sided aortic arch', 'aortic endoprosthesis', 'cyst', 'pulmonary venous hypertension', 'double J stent']
    
    padchest_unseen_class = [
        'hypoexpansion basal', 'non axial articular degenerative changes', 'central venous catheter via jugular vein', 'multiple nodules', 
        'COPD signs', 'calcified densities', 'mediastinal shift', 'hiatal hernia', 
        'volume loss', 'mediastinic lipomatosis', 'central venous catheter', 
        'ground glass pattern', 'surgery lung', 'miliary opacities', 'sclerotic bone lesion', 'pleural plaques', 'osteosynthesis material', 
        'calcified mediastinal adenopathy', 'apical pleural thickening', 'aortic elongation', 'major fissure thickening', 'callus rib fracture', 
        'pulmonary venous hypertension', 'cervical rib', 'loculated pleural effusion', 
        'flattened diaphragm' 
    ]

    padchest_unseen_class = list(set(padchest_unseen_class + padchest_rare))
    if args.dataset == 'chestxray':
        dataset_cls = chexray14_cls
        question_file = './data/chest_xray/chest_xray_llava_val.jsonl'
        result_file = args.result_folder + 'Chest_Xray_classify.txt'
    elif args.dataset == 'chexpert':
        dataset_cls = chexpert_cls
        question_file = './data/chexpert/chexpert_llava_val.jsonl'
        result_file = args.result_folder + 'chexpert_classify.txt'
    elif args.dataset == 'siim':
        dataset_cls = siim_cls
        question_file = './data/SIIM_Pneumothorax/SIIM_Pneumothorax_llava_val.jsonl'
        result_file = args.result_folder + 'siim_classify.txt'
    elif args.dataset == 'rsna':
        dataset_cls = rsna_cls
        question_file = './data/rsna/rsna_pneumonia_llava.jsonl'
        result_file = args.result_folder + 'rsna_classify.txt'
    elif args.dataset == 'covid-cxr2':
        dataset_cls = covid_cls
        question_file = './data/COVIDx_CXR/COVIDx_CXR_llava_val.jsonl'
        result_file = args.result_folder + 'COVIDx_CXR_classify.txt'
    elif args.dataset == 'covid-r':
        dataset_cls = covid_cls
        original_class.append('covid19')
    elif args.dataset == 'padchest':
        # dataset_cls = padchest_seen_class + padchest_unseen_class
        if args.subdata == 'unseen':
            original_class += padchest_unseen_class
            dataset_cls = padchest_unseen_class
            result_file = args.result_folder + 'padchest_unseen_classify.txt'
        elif args.subdata == 'rare':
            dataset_cls = padchest_rare
            result_file = args.result_folder + 'padchest_rare_classify.txt'
        else:
            dataset_cls = padchest_seen_class
            result_file = args.result_folder + 'padchest_seen_classify.txt'
        question_file = './data/padchest/padchest_llava_val.jsonl'
        
        # if 'pleural effusion' in dataset_cls:
        #     dataset_cls[dataset_cls.index('pleural effusion')] = 'effusion'
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
    # target_class = chexray14_cls
    print(MIMIC_mapping)


    # 确保类别数据是一个列表
    categories = [f"This is a chest X-ray showing {category}" for category in target_class]

    # 对类别进行编码
    encoded_categories = [tokenizer(category, return_tensors="pt") for category in categories]
    category_ids = pad_sequence([item.input_ids.squeeze(0) for item in encoded_categories], batch_first=True).to(device)
    category_attention_mask = pad_sequence([item.attention_mask.squeeze(0) for item in encoded_categories], batch_first=True).to(device)
    
    # 类别特征向量存储, 只需要计算一次
    global_category_embeddings_cache = []
    sparse_args_dict = asdict(sparse_args)
    
    if sparse_args_dict["Book_choice"]:
        with open("data/disease_desc.json", "r", encoding="utf-8") as f:
            disease_desc = json.load(f)  # 读取 JSON 文件
    else:
        with open("data/new_full_disease.json", "r", encoding="utf-8") as f:
            disease_desc = json.load(f)  # 读取 JSON 文件
    
    # 预计算疾病描述的 tokenized ID
    tokenized_desc = [
        tokenizer.encode(desc, return_tensors="pt").squeeze(0).clone().detach()
        for desc in disease_desc.values()
    ]

    # 进行 padding，确保形状为 [num_diseases, max_seq_len]
    disease_desc_ids_padded = torch.nn.utils.rnn.pad_sequence(
        tokenized_desc, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    disease_desc_attention_mask = disease_desc_ids_padded.ne(tokenizer.pad_token_id)
    
    with torch.no_grad():
        for i in range(category_ids.size(0)):
            category_input_ids = category_ids[i].unsqueeze(0).to(device)
            category_attention = category_attention_mask[i].unsqueeze(0).to(device)

            category_output = model(
                input_ids=category_input_ids, 
                attention_mask=category_attention,
                output_hidden_states=True,
                return_emb=True,
                return_dict=True
            )

            # 取最后指定层的隐藏状态，并取末尾 Txtcls_count 个 token
            global_category_embedding = category_output.hidden_states[-sparse_args_dict["feature_layer"]][:, -sparse_args_dict["Txtcls_count"]:]
            global_category_embedding = model.module.txt_mlp(global_category_embedding) if hasattr(model, 'module') else model.txt_mlp(global_category_embedding)
            global_category_embedding = global_category_embedding.mean(dim=1)
            global_category_embeddings_cache.append(global_category_embedding)
    
    global_category_embeddings_cache = torch.cat(global_category_embeddings_cache, dim=0).to(device)
    print('Global Category embeddings:', global_category_embeddings_cache)   
    # print('Local Category embeddings:', local_category_embeddings_cache)          

    questions = [
        json.loads(q) for q in open(os.path.expanduser(question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # questions = random.sample(questions, min(100, len(questions)))

    # 存储真实标签和预测结果
    all_labels = []
    all_probs = []
    all_image_embeddings = []
    for line in tqdm(questions):
        img_path = args.image_folder + line["image"]
        qs = line["text"]
        label_dict = line["label"]
        
        if model_config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda(0)
        )
        attention_mask = (input_ids != tokenizer.pad_token_id).long().cuda(0)
        
        # 尝试加载图像，如果遇到异常则跳过该图像
        try:
            if args.dataset == 'rsna':
                img = pydicom.dcmread(img_path).pixel_array  # 读取 DICOM 图像数据
                img = img.astype(float) / 255.0  # 归一化图像
                img = exposure.equalize_hist(img)  # 直方图均衡化

                # 转换为 PIL 图像并应用预处理
                img = (255 * img).astype(np.uint8)  # 转换为 uint8 类型
                image = Image.fromarray(img).convert('RGB') 
                # image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
                image_tensor = process_images([image], image_processor, model_config)[0].cuda(0)
            else:
                image = Image.open(img_path).convert("RGB")
                image_tensor = process_images([image], image_processor, model_config)[0].to(device)
        except Exception as e:
            print(f"Warning: Skipping image {img_path} due to error: {e}")
            continue 
        
        with torch.inference_mode():
            outputs,global_image_embedding = model.module.inference_pipeline(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                global_category_embeddings_cache=global_category_embeddings_cache,
                images=image_tensor.unsqueeze(0).half().to(device),
                image_sizes=[image.size],
                disease_desc_ids=disease_desc_ids_padded,
                disease_desc_attention_mask=disease_desc_attention_mask,
                use_cache=True,
            ) if hasattr(model, 'module') else model.inference_pipeline(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                global_category_embeddings_cache=global_category_embeddings_cache,
                images=image_tensor.unsqueeze(0).half().to(device),
                image_sizes=[image.size],
                disease_desc_ids=disease_desc_ids_padded,
                disease_desc_attention_mask=disease_desc_attention_mask,
                use_cache=True,
            )

            # similarity_probs 是一个 (batch_size, num_classes) 的矩阵
            similarity_probs = outputs  # 已经 softmax 过了，得到每个类别的预测概率
            # 获取真实标签
        
            true_labels = torch.zeros(len(target_class))  # 假设 `classes` 是类别列表
            for disease, value in label_dict.items():
                if value == 1 and disease in target_class:
                    true_labels[target_class.index(disease)] = 1
           
            # 将标签和预测概率存储到全局变量
            all_labels.append(true_labels.cpu().numpy())
            all_probs.append(similarity_probs.cpu().numpy())
            
             # 保存global_image_embedding
            global_image_embedding_numpy = global_image_embedding.cpu().numpy()
            all_image_embeddings.append(global_image_embedding_numpy.flatten())  # Flatten to a single vector for each image


    # 将 all_labels 和 all_probs 转换为 numpy 数组
    all_labels = np.array(all_labels)  # shape: (num_samples, num_classes)
    all_probs = np.array(all_probs).squeeze(1)  # shape: (num_samples, num_classes)

    result_metrics = {}
    embedding_df = pd.DataFrame(all_image_embeddings)
    embedding_df.to_csv("llava_med_rsna.csv", index=False)

    # 保存标签到CSV文件
    labels_df = pd.DataFrame(all_labels, columns=target_class)  # 设置列名为类别名
    labels_df.to_csv("label_rsna.csv", index=False)


    # 计算每个类别的准确率、AUC、AUPRC、F1、精确度、召回率
    accuracies, auc_scores, auprc_scores, f1_scores, recall_scores, precision_scores = [], [], [], [], [], []
    
    for i in range(all_labels.shape[1]):
        # 计算精确度、召回率和阈值
        precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_probs[:, i])

        # 计算 F1 分数并找到最大值
        f1 = 2 * precision * recall / (precision + recall + 1e-8)  # 避免分母为0
        max_f1_idx = np.argmax(f1)  # 最大 F1 对应的索引
        
        # 选择最大 F1 对应的阈值
        best_threshold = thresholds[max_f1_idx]
        
        # 二值化预测并计算准确率
        all_predictions_binary = (all_probs[:, i] >= best_threshold).astype(int)
        accuracy = (all_predictions_binary == all_labels[:, i]).mean()

        # 计算 AUC 和 AUPRC
        try:
            auc_score = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            # pass
            auc_score = np.nan  # 如果该类别标签全为0或1，返回 NaN
        
        # 计算 AUPRC
        auprc_score = auc(recall, precision)

        # 保存每个类别的指标
        accuracies.append(accuracy)
        auc_scores.append(auc_score)
        auprc_scores.append(auprc_score)
        f1_scores.append(np.max(f1))
        recall_scores.append(recall[max_f1_idx])
        precision_scores.append(precision[max_f1_idx])

    # 汇总结果
    result_metrics["mean_accuracy"] = np.mean(accuracies)
    result_metrics["mean_auc"] = np.nanmean(auc_scores)
    result_metrics["mean_f1"] = np.mean(f1_scores)
    result_metrics["mean_auprc"] = np.mean(auprc_scores)
    result_metrics["mean_recall"] = np.mean(recall_scores)
    result_metrics["mean_precision"] = np.mean(precision_scores)
    
    result_metrics["accuracies_per_class"] = accuracies
    result_metrics["auc_scores_per_class"] = auc_scores
    result_metrics["auprc_scores_per_class"] = auprc_scores
    result_metrics["f1_scores_per_class"] = f1_scores
    result_metrics["recall_scores_per_class"] = recall_scores
    result_metrics["precision_scores_per_class"] = precision_scores

    # 打印所有计算的结果
    print(f"\n===== Evaluation Metrics for {args.dataset} =====")
    for key, value in result_metrics.items():
        if isinstance(value, (list, np.ndarray)):  # 打印所有元素
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")

    # 创建结果目录
    result_dir = os.path.dirname(result_file)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    # 写入结果文件
    with open(result_file, 'w') as f:
        for key, value in result_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Results saved to {result_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/srv/lby/llava_med/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--result-folder", type=str, default="./result/R4090/llava-mistral_new_clip_v9/")
    parser.add_argument("--dataset", type=str, default="siim")
    parser.add_argument("--subdata", type=str, default="unseen")
    parser.add_argument("--chexpert-subset", type=str)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args, remaining_args = parser.parse_known_args()
    
    # Use HfArgumentParser for SparseArguments
    hf_parser = HfArgumentParser(SparseArguments)
    sparse_args, = hf_parser.parse_args_into_dataclasses(remaining_args)

    test(args, sparse_args)
    # eval_model_chest_xray(args, sparse_args)
    # eval_model_SIIM(args, sparse_args)
    # eval_model_chexpert(args, sparse_args)
    # eval_model_rsna(args, sparse_args)
    # eval_model_padchest(args, sparse_args)