import json
import torch
import os
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

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
        question_file = './eval_data/chest_xray/Chest-X-ray_llava_origin_val.jsonl'
    elif args.dataset == 'chexpert':
        dataset_cls = chexpert_cls
        question_file = './eval_data/chexpert/chexpert_llava_origin_val.jsonl'
    elif args.dataset == 'siim':
        dataset_cls = siim_cls
        question_file = './eval_data/SIIM_Pneumothorax/SIIM_Pneumothorax_llava_origin_val.jsonl'
    elif args.dataset == 'rsna':
        dataset_cls = rsna_cls
        question_file = './eval_data/rsna/rsna_pneumonia_llava_origin_val.jsonl'
    elif args.dataset == 'covid-cxr2':
        dataset_cls = covid_cls
        question_file = './eval_data/COVIDx_CXR/COVIDx_CXR_llava_origin_val.jsonl'
  
        
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

def infer_and_save(args, answer_file):
    image_folder = args.image_folder
    target_class, question_file = get_classes(args)
    model_path = args.model_path

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 你也可以设置为 torch.bfloat16 或其他
    ).to(device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model.eval()

    with open(question_file, 'r', encoding='utf-8') as _f:
        total_samples = sum(1 for _ in _f)

    with open(answer_file, 'w', encoding='utf-8') as fout, \
         open(question_file, 'r', encoding='utf-8') as fin:

        for idx, line in enumerate(tqdm(fin, total=total_samples, desc="Inference"), start=1):
            try:
                # 1. 读取并包装 messages
                messages = [json.loads(line.strip())]

                # 2. 给 image 路径加前缀
                for msg in messages:
                    if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                        for content in msg["content"]:
                            if content.get("type") == "image":
                                content["image"] = os.path.join(image_folder, content["image"])
                                # 提前检查图像文件是否存在
                                if not os.path.exists(content["image"]):
                                    raise FileNotFoundError(f"Image not found: {content['image']}")

                # 3. 构造输入
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                vision_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=vision_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                # 4. 推理
                with torch.no_grad():
                    gen_ids = model.generate(**inputs, max_new_tokens=128)

                # 5. 解码
                trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
                ]
                output_text = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                 # 6. 写入
                label = None
                for msg in messages:
                    if "label" in msg:
                        label = msg["label"]
                        break
                record = {"id": idx, "prediction": output_text, "label": label}
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                if idx % 100 == 0:
                    fout.flush()

            except FileNotFoundError as fe:
                print(f"[跳过] 找不到文件：{fe}")
                continue
            except Exception as e:
                print(f"[跳过] 第 {idx} 条数据推理异常：{str(e)}")
                continue

    print(f"All results saved to {answer_file}")



import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute_metrics(answer_file: str, result_file: str):
    # 1. Load all predictions
    with open(answer_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 2. Infer class list from the first sample
    classes = list(data[0]['label'].keys())
    num_classes = len(classes)

    # 3. Build y_true and y_pred
    y_true = []
    y_pred = []

    for item in data:
        # ground-truth vector
        true_vec = [item['label'][cls] for cls in classes]
        y_true.append(true_vec)

        # prediction vector: find the first class whose name appears in the prediction string
        pred_vec = [0] * num_classes
        pred_str = item['prediction'].strip().lower()
        if num_classes == 2:
            # binary: assign one positive label if keyword found
            for i, cls in enumerate(classes):
                if cls.lower() in pred_str:
                    pred_vec[i] = 1
                    break
        else:
            # multi-label: all classes mentioned
            for i, cls in enumerate(classes):
                if cls.lower() in pred_str:
                    pred_vec[i] = 1
        y_pred.append(pred_vec)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 4. Compute metrics
    results = {}
    # AUC metrics: only compute for classes with >1 positive sample
    if num_classes > 2:
        valid_indices = [i for i in range(num_classes) if len(set(y_true[:, i])) > 1]
        if valid_indices:
            results['auc_micro'] = roc_auc_score(y_true[:, valid_indices], y_pred[:, valid_indices], average='micro')
            results['auc_macro'] = roc_auc_score(y_true[:, valid_indices], y_pred[:, valid_indices], average='macro')
        else:
            results['auc_micro'] = None
            results['auc_macro'] = None
    else:
        # binary AUC: use the positive class (index=1)
        try:
            results['auc'] = roc_auc_score(y_true[:, 1], y_pred[:, 1])
        except ValueError:
            results['auc'] = None

    if num_classes == 2:
        # binary classification metrics
        y_true_bin = y_true.argmax(axis=1)
        y_pred_bin = y_pred.argmax(axis=1)
        results['accuracy'] = accuracy_score(y_true_bin, y_pred_bin)
        results['f1'] = f1_score(y_true_bin, y_pred_bin)
    else:
        # multi-label metrics
        # micro and macro F1
        results['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        # sample-level custom metrics
        total_correct = 0
        total_true_labels = 0
        sample_f1_scores = []
        for i in range(y_true.shape[0]):
            true_vec = y_true[i]
            pred_vec = y_pred[i]
            correct = np.sum((true_vec == 1) & (pred_vec == 1))
            true_count = np.sum(true_vec == 1)
            pred_count = np.sum(pred_vec == 1)
            total_correct += correct
            total_true_labels += true_count

            if true_count == 0 or pred_count == 0:
                sample_f1 = 0.0
            else:
                precision = correct / pred_count
                recall = correct / true_count
                sample_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            sample_f1_scores.append(sample_f1)

        results['avg_acc'] = total_correct / total_true_labels if total_true_labels > 0 else 0.0
        results['avg_f1'] = float(np.mean(sample_f1_scores))

    # 5. Print and save
    print("Metrics:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, indent=2)

    return results

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="siim",
                        choices=["chestxray","chexpert","siim","rsna","covid-cxr2"])
    parser.add_argument("--chexpert_subset", type=str, default="False")
    parser.add_argument("--answer_file", type=str, default="./results/siim/answers.jsonl")
    parser.add_argument("--image-folder", type=str, default="/srv/lby/")
    parser.add_argument("--model-path", type=str, default="/srv/lby/qwen_vl_7b/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--result-file", type=str, default="./results/siim/siim_qwen7b_metrics.json")
    args = parser.parse_args()

    infer_and_save(args, args.answer_file)
    compute_metrics(args.answer_file, args.result_file)