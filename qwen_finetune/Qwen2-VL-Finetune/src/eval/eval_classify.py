import argparse
import json
import math
import os
import re
import shortuuid
import torch
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, recall_score, f1_score, roc_auc_score, precision_score, balanced_accuracy_score, confusion_matrix
import pandas as pd

# 导入DICOM处理库
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("Warning: pydicom not available. DCM files will not be supported.")

from qwen_vl_utils import process_vision_info
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_image_file(img_path):
    """
    加载图像文件，支持常规格式和DICOM格式
    """
    try:
        # 检查文件扩展名
        file_ext = os.path.splitext(img_path)[1].lower()
        
        if file_ext == '.dcm' and DICOM_AVAILABLE:
            # 处理DICOM文件
            dicom_data = pydicom.dcmread(img_path)
            
            # 获取像素数据
            if hasattr(dicom_data, 'pixel_array'):
                pixel_array = dicom_data.pixel_array.astype(float)
                
                # 应用DICOM窗口调整（如果存在）
                if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                    try:
                        window_center = float(dicom_data.WindowCenter[0] if hasattr(dicom_data.WindowCenter, '__iter__') else dicom_data.WindowCenter)
                        window_width = float(dicom_data.WindowWidth[0] if hasattr(dicom_data.WindowWidth, '__iter__') else dicom_data.WindowWidth)
                        
                        # 应用窗口调整
                        img_min = window_center - window_width // 2
                        img_max = window_center + window_width // 2
                        pixel_array = np.clip(pixel_array, img_min, img_max)
                        pixel_array = (pixel_array - img_min) / (img_max - img_min) * 255
                    except:
                        # 如果窗口调整失败，使用默认归一化
                        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
                else:
                    # 默认归一化到0-255范围
                    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
                
                # 处理不同的像素数据格式
                if len(pixel_array.shape) == 2:
                    # 灰度图像
                    pixel_array = pixel_array.astype(np.uint8)
                    # 转换为PIL图像
                    image = Image.fromarray(pixel_array, mode='L')
                    # 转换为RGB
                    image = image.convert('RGB')
                elif len(pixel_array.shape) == 3:
                    # 彩色图像或多帧图像，取第一帧
                    if pixel_array.shape[0] < pixel_array.shape[2]:
                        # 假设第一个维度是帧数
                        pixel_array = pixel_array[0]
                    pixel_array = pixel_array.astype(np.uint8)
                    image = Image.fromarray(pixel_array)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                else:
                    raise ValueError(f"Unsupported pixel array shape: {pixel_array.shape}")
                
                return image
            else:
                raise ValueError("DICOM file has no pixel_array attribute")
                
        elif file_ext == '.dcm' and not DICOM_AVAILABLE:
            raise ImportError("pydicom is required to read DCM files. Please install: pip install pydicom")
        else:
            # 处理常规图像文件
            image = Image.open(img_path).convert('RGB')
            return image
            
    except Exception as e:
        raise Exception(f"Failed to load image {img_path}: {str(e)}")


def load_image_file_fallback(img_path):
    """
    备用图像加载函数，处理各种图像格式
    """
    try:
        return load_image_file(img_path)
    except Exception:
        # 如果主加载函数失败，尝试直接用PIL加载
        try:
            image = Image.open(img_path).convert('RGB')
            return image
        except Exception as e:
            raise Exception(f"All image loading methods failed for {img_path}: {str(e)}")


def eval_model_chest_xray(args):
    """
    评估Qwen2-VL模型在胸部X光分类任务上的性能
    """
    # 检查必需的依赖
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("错误: 无法导入qwen_vl_utils。请确保安装了相关依赖。")
        return
    # 加载模型和处理器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from: {args.model_path}")
    
    # 使用项目标准的模型加载方式
    model_name = get_model_name_from_path(args.model_path)
    
    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path, 
        args.model_base, 
        model_name,
        device_map="auto"
    )
    
    # 数据集类别定义
    dataset_classes = {
        'chestxray': ["fibrosis", "edema", "pneumothorax", "cardiomegaly", "atelectasis", 
                     "nodule", "emphysema", "no finding", "mass", "pleural_thickening", 
                     "effusion", "infiltration", "pneumonia", "hernia", "consolidation"],
        'chexpert': ['no finding', 'enlarged cardiomediastinum', 'cardiomegaly', 
                    'lung opacity', 'lung lesion', 'edema', 'consolidation', 
                    'pneumonia', 'atelectasis', 'pneumothorax', 'pleural effusion', 
                    'pleural other', 'fracture', 'support devices'],
        'mimic': ["atelectasis", "cardiomegaly", "consolidation", "edema", "enlarged cardiomediastinum",
                 "fracture", "lung lesion", "lung opacity", "no finding", "pleural effusion", 
                 "pleural other", "pneumonia", "pneumothorax", "support devices"],
        'rsna': ["pneumonia", "normal"],  # RSNA Pneumonia Detection Challenge
        'COVIDx_CXR': ["covid-19", "pneumonia", "normal"],  # COVIDx CXR dataset
        'SIIM_Pneumothorax': ["pneumothorax", "no finding"]  # SIIM Pneumothorax dataset
    }
    
    # 根据数据集选择类别
    if args.dataset in dataset_classes:
        target_classes = dataset_classes[args.dataset]
    else:
        target_classes = dataset_classes['mimic']  # 默认使用MIMIC类别
    
    print(f"Evaluating on dataset: {args.dataset}")
    print(f"Target classes: {target_classes}")
    
    # 构建问题文件路径
    if args.question_file:
        question_file = args.question_file
    else:
        question_file = f'./data/{args.dataset}/{args.dataset}_val.jsonl'
    
    # 读取测试数据
    print(f"Loading questions from: {question_file}")
    questions = []
    if os.path.exists(question_file):
        with open(question_file, 'r') as f:
            for line in f:
                questions.append(json.loads(line))
    else:
        print(f"Warning: Question file {question_file} not found!")
        return
    
    # 分块处理（支持多进程评估）
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    print(f"Processing {len(questions)} questions (chunk {args.chunk_idx+1}/{args.num_chunks})")
    
    questions = questions[:100] 
    
    # 存储预测结果和真实标签
    all_labels = []
    all_predictions = []
    all_probs = []
    
    model.eval()
    
    # 初始GPU内存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    processed_count = 0
    success_count = 0
    error_count = 0
    
    with torch.no_grad():
        for line in tqdm(questions, desc="Evaluating"):
            # 构建图像路径
            if 'image' in line:
                img_path = os.path.join(args.image_folder, line['image'])
            elif 'image_path' in line:
                img_path = os.path.join(args.image_folder, line['image_path'])
            else:
                print("Warning: No image path found in question")
                continue
            
            # 检查图像文件是否存在
            if not os.path.exists(img_path):
                print(f"Warning: Image file {img_path} not found")
                continue
            
            try:
                # 加载图像并验证（支持DCM格式）
                image = load_image_file_fallback(img_path)
                
                # 验证图像尺寸，防止过大图像导致内存问题
                # 更严格的尺寸限制以避免GPU内存不足
                max_size = 768  # 降低最大尺寸以节省GPU内存
                if image.size[0] > max_size or image.size[1] > max_size:
                    # 保持宽高比的缩放
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # 构建对话内容 - 根据评估模式选择不同的提示策略
                classes_str = ", ".join(target_classes)
                
                if args.eval_mode == "confidence":
                    # 置信度评分模式 - 改进的严格格式提示
                    if args.conv_mode == "simple":
                        text_prompt = f"<image>\nAnalyze this chest X-ray and rate each disease from 0 to 10 (0=no evidence, 10=strong evidence).\n\nDiseases to evaluate: {classes_str}\n\nIMPORTANT: Use EXACTLY this format for each disease:\ndisease_name: number\n\nExample:\nfibrosis: 3\nedema: 0\npneumothorax: 7\n\nProvide a rating for ALL diseases listed above."
                    else:
                        text_prompt = f"<image>\nCarefully analyze this chest X-ray image. You must evaluate each disease and provide a confidence rating from 0-10.\n\nDiseases to evaluate: {classes_str}\n\nCRITICAL INSTRUCTIONS:\n1. Rate ALL diseases listed above\n2. Use EXACTLY this format: disease_name: number\n3. Scale: 0=no evidence, 5=uncertain, 10=strong evidence\n4. One rating per line\n\nExample format:\nfibrosis: 2\nedema: 0\npneumothorax: 8\ncardiomegaly: 3\n\nStart your response with the ratings:"
                else:
                    # 硬分类模式
                    if args.conv_mode == "simple":
                        text_prompt = f"<image>\nAnalyze this chest X-ray and identify the diseases from the following categories: {classes_str}. List only the applicable disease names separated by commas, or 'no finding' if no abnormalities are detected."
                    else:
                        text_prompt = f"<image>\nPlease carefully analyze this chest X-ray image. Identify any diseases or abnormalities present from these specific categories: {classes_str}. Provide only the disease names that apply, separated by commas. If no abnormalities are found, respond with 'no finding'."
                
                # 构建消息格式 - 使用已加载的PIL图像对象而不是文件路径
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},  # 使用PIL图像对象
                            {"type": "text", "text": text_prompt}
                        ]
                    }
                ]
                
                # 处理输入 - 添加错误处理和验证
                try:
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    # 验证图像输入
                    if image_inputs is None or len(image_inputs) == 0:
                        continue
                    
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    
                    # 验证inputs的完整性
                    if not hasattr(inputs, 'input_ids') or inputs.input_ids.size(0) == 0:
                        continue
                        
                    inputs = inputs.to(device)
                    
                except Exception as e:
                    print(f"Input processing error for {img_path}: {e}")
                    continue
                
                # 生成响应 - 使用Qwen2-VL的标准API
                try:
                    with torch.inference_mode():
                        generated_ids = model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=128,
                            use_cache=True,
                            pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor.tokenizer, 'eos_token_id') else None
                        )
                    
                    # 使用Qwen2-VL官方推荐的安全token分离方法
                    # 参考官方文档中的实现，但添加安全检查
                    generated_ids_trimmed = []
                    generation_failed = False
                    
                    for input_ids, output_ids in zip(inputs.input_ids, generated_ids):
                        input_length = len(input_ids)
                        output_length = len(output_ids)
                        
                        # 安全检查：确保输出长度大于输入长度
                        if output_length > input_length:
                            # 提取新生成的token
                            trimmed_ids = output_ids[input_length:]
                            generated_ids_trimmed.append(trimmed_ids)
                        else:
                            # 如果没有新生成的token，标记失败
                            generation_failed = True
                            break
                    
                    # 检查生成是否成功
                    if generation_failed or not generated_ids_trimmed:
                        continue
                        
                    # 使用Qwen2-VL的标准解码方式
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )[0].strip()
                    
                    # 删除详细调试输出，保持简洁

                        
                except RuntimeError as e:
                    if "CUDA" in str(e) or "out of memory" in str(e):
                        print(f"CUDA/Memory error during generation for {img_path}: {e}")
                        # 清理GPU缓存并跳过这个样本
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                except Exception as e:
                    print(f"Generation error for {img_path}: {e}")
                    continue
                
                # 解析模型输出 - 根据评估模式选择解析方式
                if args.eval_mode == "confidence":
                    # 置信度模式：只解析置信度分数，不需要预测类别
                    confidence_scores = parse_confidence_scores(output_text, target_classes)
                    # 多标签分类：不使用固定阈值，直接基于置信度计算
                    # pred_vector用于可调阈值的分析，prob_vector用于AUC计算
                    pred_vector = np.zeros(len(target_classes))  # 暂时保留，后续可能用于阈值分析
                else:
                    # 硬分类模式：解析预测类别
                    predicted_classes, confidence_scores = parse_hard_classification(output_text, target_classes)
                    pred_vector = np.zeros(len(target_classes))
                    for cls in predicted_classes:
                        if cls in target_classes:
                            pred_vector[target_classes.index(cls)] = 1
                
                # 获取真实标签
                true_vector = np.zeros(len(target_classes))
                if 'label' in line:
                    labels = line['label']
                    if isinstance(labels, dict):
                        for cls, value in labels.items():
                            if cls in target_classes and value == 1:
                                true_vector[target_classes.index(cls)] = 1
                    elif isinstance(labels, list):
                        for cls in labels:
                            if cls in target_classes:
                                true_vector[target_classes.index(cls)] = 1
                
                all_labels.append(true_vector)
                all_predictions.append(pred_vector)
                
                # 计算概率分数 - 根据评估模式选择不同策略
                prob_vector = np.zeros(len(target_classes))
                
                if args.eval_mode == "confidence":
                    # 置信度模式：直接基于置信度分数计算概率和预测
                    # 多标签分类：每个类别独立处理
                    for i, cls in enumerate(target_classes):
                        if cls in confidence_scores:
                            confidence = confidence_scores[cls]
                            # 将0-10的置信度转换为概率 (用于AUC计算)
                            normalized_confidence = confidence / 10.0
                            sigmoid_prob = 1 / (1 + np.exp(-6 * (normalized_confidence - 0.5)))
                            prob_vector[i] = sigmoid_prob
                            
                            # 使用5.0作为默认阈值构建预测向量 (用于ACC、F1计算)
                            # 这个阈值可以后续优化
                            if confidence >= 5.0:
                                pred_vector[i] = 1
                        else:
                            # 如果模型没有为某个类别提供置信度，视为0分
                            prob_vector[i] = 0.001  # 极小值避免AUC计算问题
                            pred_vector[i] = 0
                else:
                    # 硬分类模式：不需要真实概率，用pred_vector代替
                    # （因为硬分类模式不计算AUC，这里的概率向量不会被使用）
                    prob_vector = pred_vector.copy()
                
                all_probs.append(prob_vector)
                
                # 显示前几个样本的结果
                if len(all_probs) <= 5:
                    if args.eval_mode == "confidence":
                        print(f"Sample {len(all_probs)}: {confidence_scores}")
                    else:
                        print(f"Sample {len(all_probs)}: {predicted_classes}")
                
                # 清理GPU缓存以防止内存累积
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                processed_count += 1
                success_count += 1
                
                # 周期性内存管理（静默）
                if processed_count % 50 == 0 and torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / 1024**3
                    if current_memory > 10:
                        torch.cuda.empty_cache()
                
            except Exception as e:
                error_count += 1
                processed_count += 1
                if "CUDA" in str(e):
                    print(f"CUDA error processing {img_path}: {e}")
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
    
    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    print(f"Final Summary: Processed {processed_count} samples, Successfully processed {len(all_labels)} samples, Errors: {error_count}")
    
    if error_count > 0:
        print(f"Error rate: {error_count/processed_count*100:.1f}%")
    if success_count > 0:
        print(f"Success rate: {success_count/processed_count*100:.1f}%")
    
    # 计算评估指标
    if len(all_labels) > 0:
        print(f"\nUsing evaluation mode: {args.eval_mode}")
        calculate_metrics(all_labels, all_predictions, all_probs, target_classes, args)
    else:
        print("No samples were successfully processed!")


def parse_hard_classification(output_text, target_classes):
    """
    解析硬分类模型输出，直接提取疾病类别（无置信度评分）
    """
    predicted_classes = []
    confidence_scores = {}  # 空的置信度字典，保持接口一致
    
    output_lower = output_text.lower().strip()
    
    # 寻找明确提到的疾病名称
    for cls in target_classes:
        cls_lower = cls.lower()
        if cls_lower in output_lower:
            # 检查是否是完整的词匹配
            pattern = r'\b' + re.escape(cls_lower) + r'\b'
            if re.search(pattern, output_lower):
                predicted_classes.append(cls)
    
    # 如果没有找到明确的疾病名称，尝试解析逗号分隔的列表
    if not predicted_classes:
        parts = [part.strip() for part in output_lower.replace(',', '|').replace(';', '|').split('|')]
        for part in parts:
            for cls in target_classes:
                if cls.lower() in part:
                    predicted_classes.append(cls)
                    break
    
    # 如果没有检测到任何类别，标记为"no finding"
    if not predicted_classes and "no finding" in target_classes:
        predicted_classes = ["no finding"]
    
    # 去重并保持顺序
    predicted_classes = list(dict.fromkeys(predicted_classes))
    return predicted_classes, confidence_scores


def parse_confidence_scores(output_text, target_classes):
    """
    解析模型输出文本，提取每个类别的置信度评分（置信度模式专用）
    只返回置信度分数字典，不进行预测类别判断
    """
    confidence_scores = {}
    
    # 解析置信度评分格式 "disease_name: rating"
    rating_pattern = r'([a-zA-Z][a-zA-Z\s_\-]+?):\s*(\d+(?:\.\d+)?)'
    matches = re.findall(rating_pattern, output_text, re.IGNORECASE)
    
    for disease_match, score_str in matches:
        disease_clean = disease_match.strip().lower()
        try:
            score = float(score_str)
            # 寻找最匹配的目标类别
            best_match = None
            best_score_match = 0
            
            for cls in target_classes:
                cls_lower = cls.lower()
                # 精确匹配优先
                if disease_clean == cls_lower:
                    best_match = cls
                    break
                # 包含匹配：计算匹配度
                elif cls_lower in disease_clean or disease_clean in cls_lower:
                    match_score = min(len(cls_lower), len(disease_clean)) / max(len(cls_lower), len(disease_clean))
                    if match_score > best_score_match:
                        best_match = cls
                        best_score_match = match_score
            
            if best_match:
                confidence_scores[best_match] = score
        except ValueError:
            continue
    
    return confidence_scores


def parse_prediction_text(output_text, target_classes):
    """
    解析模型输出文本，提取预测的疾病类别和置信度评分（置信度模式）
    """
    predicted_classes = []
    confidence_scores = {}
    
    output_lower = output_text.lower().strip()
    
    # 方法1: 解析置信度评分格式 "disease_name: rating" - 改进的正则表达式
    # 匹配 "疾病名: 数字" 格式，支持下划线、空格、连字符
    rating_pattern = r'([a-zA-Z][a-zA-Z\s_\-]+?):\s*(\d+(?:\.\d+)?)'
    matches = re.findall(rating_pattern, output_text, re.IGNORECASE)
    
    for disease_match, score_str in matches:
        disease_clean = disease_match.strip().lower()
        try:
            score = float(score_str)
            # 寻找最匹配的目标类别 - 改进匹配逻辑
            best_match = None
            best_score_match = 0
            
            for cls in target_classes:
                cls_lower = cls.lower()
                # 精确匹配优先
                if disease_clean == cls_lower:
                    best_match = cls
                    break
                # 包含匹配：计算匹配度
                elif cls_lower in disease_clean or disease_clean in cls_lower:
                    match_score = min(len(cls_lower), len(disease_clean)) / max(len(cls_lower), len(disease_clean))
                    if match_score > best_score_match:
                        best_match = cls
                        best_score_match = match_score
            
            if best_match:
                confidence_scores[best_match] = score
                # 在置信度模式下，不在解析阶段应用阈值
                # 所有有置信度分数的类别都记录，阈值判断留给后续处理
        except ValueError:
            continue
    
    # 方法2: 如果没有找到评分格式，不分配默认置信度
    # 删除默认置信度逻辑，强制模型按格式输出
    if not confidence_scores:
        # 不再提供回退机制，如果模型没有按格式输出就视为解析失败
        pass
    
    # 删除所有默认置信度分配，包括"no finding"的默认置信度
    # 如果模型没有按格式输出置信度，就不进行任何预测
    # 这样可以更真实地反映模型的性能
    
    # 在置信度模式下，predicted_classes将在后续基于阈值动态构建
    # 这里只返回有置信度分数的类别用于调试显示
    predicted_classes = [cls for cls, score in confidence_scores.items() if score >= 5.0]
    return predicted_classes, confidence_scores


def calculate_metrics(all_labels, all_predictions, all_probs, target_classes, args):
    """计算医学分类任务的合理评估指标"""
    result_metrics = {}
    
    # 计算每个类别的指标
    balanced_accuracies, f1_scores, precision_scores, recall_scores = [], [], [], []
    sensitivities, specificities = [], []  # 医学术语：敏感性、特异性
    auc_scores, auprc_scores = [], []
    supports = []  # 支持度（正样本数量）
    
    for i, class_name in enumerate(target_classes):
        # 医学分类指标计算
        if all_labels[:, i].sum() > 0:  # 确保该类别有正样本
            # 核心指标
            balanced_acc = balanced_accuracy_score(all_labels[:, i], all_predictions[:, i])
            f1 = f1_score(all_labels[:, i], all_predictions[:, i], zero_division=0)
            precision = precision_score(all_labels[:, i], all_predictions[:, i], zero_division=0)
            recall = recall_score(all_labels[:, i], all_predictions[:, i], zero_division=0)
            
            # 计算混淆矩阵统计
            tn, fp, fn, tp = confusion_matrix(all_labels[:, i], all_predictions[:, i]).ravel()
            
            # 医学术语：敏感性（sensitivity）= 召回率，特异性（specificity）
            sensitivity = recall  # TP/(TP+FN) - 能正确识别阳性病例的比例
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TN/(TN+FP) - 能正确识别阴性病例的比例
            support = tp + fn  # 真实正样本数量
            
            # 存储指标
            balanced_accuracies.append(balanced_acc)
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            supports.append(support)
            
            # AUC指标（仅在置信度模式下计算）
            if args.eval_mode == "confidence":
                try:
                    if len(np.unique(all_labels[:, i])) > 1:  # 确保有正负样本
                        auc_score = roc_auc_score(all_labels[:, i], all_probs[:, i])
                        precision_curve, recall_curve, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
                        auprc_score = auc(recall_curve, precision_curve)
                    else:
                        auc_score = 0.0
                        auprc_score = 0.0
                except Exception:
                    auc_score = 0.0
                    auprc_score = 0.0
                
                auc_scores.append(auc_score)
                auprc_scores.append(auprc_score)
                
                # 置信度模式：显示完整指标
                print(f"{class_name}: F1={f1:.3f}, Bal_Acc={balanced_acc:.3f}, Sen={sensitivity:.3f}, Spec={specificity:.3f}, AUC={auc_score:.3f}")
                print(f"           Precision={precision:.3f}, Recall={recall:.3f}, AUPRC={auprc_score:.3f}")
            else:
                # 硬分类模式：关注核心指标
                auc_scores.append(0.0)
                auprc_scores.append(0.0)
                print(f"{class_name}: F1={f1:.3f}, Bal_Acc={balanced_acc:.3f}, Sen={sensitivity:.3f}, Spec={specificity:.3f}")
                print(f"           Precision={precision:.3f}, Recall={recall:.3f}")
            
            print(f"           TP={tp}, FP={fp}, TN={tn}, FN={fn}, Support={support}")
        else:
            print(f"{class_name}: No positive samples")
            balanced_accuracies.append(0.0)
            f1_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            sensitivities.append(0.0)
            specificities.append(1.0)  # 没有正样本时，特异性为1（所有负样本都被正确识别）
            supports.append(0)
            auc_scores.append(0.0)
            auprc_scores.append(0.0)
    
    # 计算宏平均指标（每个类别等权重，不依赖类别分布）
    # 过滤掉没有正样本的类别
    valid_indices = [i for i, support in enumerate(supports) if support > 0]
    
    if valid_indices:
        macro_f1 = np.mean([f1_scores[i] for i in valid_indices])
        macro_balanced_acc = np.mean([balanced_accuracies[i] for i in valid_indices])
        macro_sensitivity = np.mean([sensitivities[i] for i in valid_indices])
        macro_specificity = np.mean([specificities[i] for i in valid_indices])
        macro_precision = np.mean([precision_scores[i] for i in valid_indices])
        macro_recall = np.mean([recall_scores[i] for i in valid_indices])
    else:
        macro_f1 = macro_balanced_acc = macro_sensitivity = macro_specificity = 0.0
        macro_precision = macro_recall = 0.0
    
    result_metrics = {
        "macro_f1": macro_f1,
        "macro_balanced_accuracy": macro_balanced_acc,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "mean_auc": np.mean(auc_scores),
        "mean_auprc": np.mean(auprc_scores),
        "f1_scores_per_class": f1_scores,
        "balanced_accuracies_per_class": balanced_accuracies,
        "sensitivities_per_class": sensitivities,
        "specificities_per_class": specificities,
        "precision_scores_per_class": precision_scores,
        "recall_scores_per_class": recall_scores,
        "supports_per_class": supports,
        "auc_scores_per_class": auc_scores,
        "auprc_scores_per_class": auprc_scores
    }
    
    # 打印总体结果 - 使用更合理的指标
    print(f"\n===== Medical Classification Results for {args.dataset} ({args.eval_mode} mode) =====")
    print("Macro-averaged Metrics (each class has equal weight):")
    print(f"  F1 Score: {result_metrics['macro_f1']:.3f}")
    print(f"  Balanced Accuracy: {result_metrics['macro_balanced_accuracy']:.3f}")
    print(f"  Sensitivity (Recall): {result_metrics['macro_sensitivity']:.3f}")
    print(f"  Specificity: {result_metrics['macro_specificity']:.3f}")
    print(f"  Precision: {result_metrics['macro_precision']:.3f}")
    
    if args.eval_mode == "confidence":
        print(f"\nAUC Metrics:")
        print(f"  Mean AUC-ROC: {result_metrics['mean_auc']:.3f}")
        print(f"  Mean AUC-PR: {result_metrics['mean_auprc']:.3f}")
    
    
    
    
    
    # 保存结果到文件
    if args.result_file:
        result_dir = os.path.dirname(args.result_file)
        if result_dir and not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        
        with open(args.result_file, 'w') as f:
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Model: {args.model_path}\n\n")
            for key, value in result_metrics.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Results saved to {args.result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--model-base", type=str, default=None, help="Path to the base model (for LoRA)")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to the image folder")
    parser.add_argument("--question-file", type=str, default=None, help="Path to the question file")
    parser.add_argument("--result-file", type=str, default="./result/eval_results.txt", help="Path to save results")
    parser.add_argument("--dataset", type=str, default="mimic", choices=["chestxray", "chexpert", "mimic", "rsna", "COVIDx_CXR", "SIIM_Pneumothorax"], help="Dataset name")
    parser.add_argument("--conv-mode", type=str, default="detailed", choices=["simple", "detailed"], help="Conversation mode")
    parser.add_argument("--eval-mode", type=str, default="confidence", choices=["confidence", "hard"], 
                        help="Evaluation mode: 'confidence' for confidence scoring, 'hard' for direct classification")
    parser.add_argument("--num-chunks", type=int, default=1, help="Number of chunks for parallel processing")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Current chunk index")
    
    args = parser.parse_args()
    
    eval_model_chest_xray(args)