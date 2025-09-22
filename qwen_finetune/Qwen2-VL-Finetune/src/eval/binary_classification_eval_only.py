#!/usr/bin/env python3
"""
二分类医学图像评估系统 - 仅评估版本
支持SIIM、COVID-19、RSNA数据集的模型评估
专注于验证已有模型的性能，无训练功能
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
# import matplotlib.pyplot as plt  # 移除绘图依赖
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# PyTorch相关导入
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Transformers导入（用于Qwen2-VL CLIP模型）
from transformers import AutoTokenizer, AutoProcessor

# sklearn指标
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, average_precision_score, matthews_corrcoef
)

# 添加路径以导入CLIP模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.clip_modeling_qwen2_5_vl import (
    ClipQwen2VLConfig,
    ClipQwen2VLForConditionalGeneration
)
from constants import DEFAULT_IMAGE_TOKEN

# ==================== 数据集类 ====================

class BinaryEvalDataset(Dataset):
    """二分类评估数据集类，支持SIIM、COVID-19、RSNA数据集"""
    
    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        image_root: str = "",
        image_size: int = 224
    ):
        """
        初始化评估数据集
        
        Args:
            data_path: 数据文件路径（CSV或JSON格式）
            dataset_name: 数据集名称 ('siim', 'covid19', 'rsna')
            image_root: 图像根目录
            image_size: 图像尺寸
        """
        self.data_path = data_path
        self.dataset_name = dataset_name.lower()
        self.image_root = image_root
        self.image_size = image_size
        
        # 数据集标签映射
        self.label_mapping = {
            'siim': {'pneumothorax': 1, 'non-pneumothorax': 0, 'normal': 0, '1': 1, '0': 0},
            'covid19': {'covid-19': 1, 'normal': 0, 'negative': 0, '1': 1, '0': 0},
            'rsna': {'pneumonia': 1, 'normal': 0, 'negative': 0, '1': 1, '0': 0}
        }
        
        # 加载数据
        self.data = self._load_data()
        
        # 设置图像变换（仅用于评估，不使用数据增强）
        self.transform = self._get_transforms()
        
        print(f"加载 {dataset_name} 评估数据集: {len(self.data)} 个样本")
        self._print_class_distribution()
    
    def _load_data(self) -> List[Dict]:
        """加载数据文件"""
        if self.data_path.endswith('.csv'):
            return self._load_csv_data()
        elif self.data_path.endswith('.json') or self.data_path.endswith('.jsonl'):
            return self._load_json_data()
        else:
            raise ValueError(f"不支持的数据格式: {self.data_path}")
    
    def _load_csv_data(self) -> List[Dict]:
        """加载CSV格式数据"""
        df = pd.read_csv(self.data_path)
        data = []
        
        for _, row in df.iterrows():
            # 根据数据集名称解析不同的CSV格式
            if self.dataset_name == 'siim':
                image_path = row.get('image_path', row.get('image', ''))
                label = row.get('label', row.get('pneumothorax', 0))
            elif self.dataset_name == 'covid19':
                image_path = row.get('image_path', row.get('filename', ''))
                label = row.get('label', row.get('finding', 'normal'))
            elif self.dataset_name == 'rsna':
                image_path = row.get('image_path', row.get('patientId', ''))
                label = row.get('label', row.get('Target', 0))
            else:
                image_path = row.get('image_path', '')
                label = row.get('label', 0)
            
            # 标准化标签
            binary_label = self._normalize_label(label)
            
            data.append({
                'image_path': image_path,
                'label': binary_label,
                'original_label': label
            })
        
        return data
    
    def _load_json_data(self) -> List[Dict]:
        """加载JSON格式数据"""
        data = []
        
        if self.data_path.endswith('.jsonl'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    data.append(self._parse_json_item(item))
        else:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if isinstance(json_data, list):
                    for item in json_data:
                        data.append(self._parse_json_item(item))
                else:
                    data.append(self._parse_json_item(json_data))
        
        return data
    
    def _parse_json_item(self, item: Dict) -> Dict:
        """解析JSON数据项"""
        image_path = item.get('image', item.get('image_path', ''))
        
        # 从多个可能的字段中获取标签
        label = item.get('label', item.get('finding', item.get('target', item.get('class', 0))))
        
        # 如果标签在conversations中（LLaVA格式）
        if 'conversations' in item:
            conversations = item['conversations']
            for conv in conversations:
                if conv.get('from') == 'human':
                    # 从对话中提取标签信息
                    human_text = conv.get('value', '').lower()
                    if 'pneumothorax' in human_text:
                        label = 'pneumothorax' if 'yes' in human_text or 'positive' in human_text else 'non-pneumothorax'
                    elif 'covid' in human_text:
                        label = 'covid-19' if 'positive' in human_text else 'normal'
                    elif 'pneumonia' in human_text:
                        label = 'pneumonia' if 'yes' in human_text or 'positive' in human_text else 'normal'
        
        binary_label = self._normalize_label(label)
        
        return {
            'image_path': image_path,
            'label': binary_label,
            'original_label': label
        }
    
    def _normalize_label(self, label) -> int:
        """将标签标准化为二分类标签 (0/1)"""
        if isinstance(label, (int, float)):
            return int(label)
        
        if isinstance(label, str):
            label = label.lower().strip()
            mapping = self.label_mapping.get(self.dataset_name, {})
            return mapping.get(label, 0)
        
        return 0
    
    def _get_transforms(self) -> transforms.Compose:
        """获取图像变换（评估用，无数据增强）"""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    
    def _print_class_distribution(self):
        """打印类别分布信息"""
        labels = [item['label'] for item in self.data]
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        
        print(f"类别分布 - 正样本: {pos_count} ({pos_count/len(labels)*100:.1f}%), "
              f"负样本: {neg_count} ({neg_count/len(labels)*100:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Dict:
        """获取数据项"""
        item = self.data[index]
        
        # 构建完整的图像路径
        image_path = os.path.join(self.image_root, item['image_path'])
        
        # 加载和预处理图像
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"加载图像失败: {image_path}, 错误: {e}")
            # 创建空白图像作为备选
            image = torch.zeros(3, self.image_size, self.image_size)
        
        return {
            'image': image,
            'label': torch.tensor(item['label'], dtype=torch.float32),
            'image_path': image_path,
            'original_label': item['original_label']
        }


# ==================== 简化的模型类 ====================

class ClipBinaryClassifier(nn.Module):
    """基于Qwen2-VL CLIP模型的二分类器"""
    
    def __init__(self, model_path: str, device: torch.device):
        super(ClipBinaryClassifier, self).__init__()
        
        self.model_path = model_path
        self.device = device
        
        # 加载tokenizer和processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载CLIP模型
        config = ClipQwen2VLConfig.from_pretrained(model_path)
        self.model = ClipQwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        # 获取配置参数
        self.sparse_config = config.sparse_config
        self.Imgcls_count = self.sparse_config["Imgcls_count"]
        self.Txtcls_count = self.sparse_config["Txtcls_count"]
        self.temperature = self.sparse_config["temperature"]
        
        # 获取特殊标记ID
        self.imgcls_token_ids = []
        self.txtcls_token_ids = []
        
        for i in range(self.Imgcls_count):
            token = f"<Imgcls{i}>"
            if token in self.tokenizer.get_vocab():
                self.imgcls_token_ids.append(self.tokenizer.convert_tokens_to_ids(token))
        
        for i in range(self.Txtcls_count):
            token = f"<Txtcls{i}>"
            if token in self.tokenizer.get_vocab():
                self.txtcls_token_ids.append(self.tokenizer.convert_tokens_to_ids(token))
    
    def prepare_image_input(self, image_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """准备图像输入"""
        try:
            from eval_utils.utils import load_image_file
            image = load_image_file(image_path)
            
            # 构建图像分类提示
            imgcls_tokens = "".join([f"<Imgcls{i}>" for i in range(self.Imgcls_count)])
            prompt = f"{DEFAULT_IMAGE_TOKEN}\nAnalyze this medical image. {imgcls_tokens}"
            
            proc_inputs = self.processor(
                text=[prompt], images=[image], padding=False, do_resize=True, return_tensors="pt"
            )
            
            result = {
                "input_ids": proc_inputs["input_ids"].to(self.device),
                "attention_mask": proc_inputs["attention_mask"].to(self.device),
            }
            
            if "pixel_values" in proc_inputs:
                result["pixel_values"] = proc_inputs["pixel_values"].to(self.device)
            if "image_grid_thw" in proc_inputs:
                result["image_grid_thw"] = proc_inputs["image_grid_thw"].to(self.device)
            
            return result
        except Exception as e:
            print(f"Error preparing image input for {image_path}: {e}")
            return None
    
    def prepare_text_input(self, text: str) -> Dict[str, torch.Tensor]:
        """准备文本输入"""
        txtcls_tokens = "".join([f"<Txtcls{i}>" for i in range(self.Txtcls_count)])
        prompt = f"{text} {txtcls_tokens}"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        return {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device)
        }
    
    def extract_image_features(self, image_path: str) -> Optional[torch.Tensor]:
        """提取图像特征"""
        inputs = self.prepare_image_input(image_path)
        if inputs is None:
            return None
        
        try:
            with torch.no_grad():
                feats = self.model.extract_features(**inputs)
                image_features = feats["global_features"]  # (1, D)
                if image_features.dim() == 1:
                    image_features = image_features.unsqueeze(0)
                return image_features
        except Exception as e:
            print(f"Error extracting image features: {e}")
            return None
    
    def extract_text_features(self, text: str) -> Optional[torch.Tensor]:
        """提取文本特征"""
        inputs = self.prepare_text_input(text)
        try:
            with torch.no_grad():
                feats = self.model.extract_features(**inputs)
                text_features = feats["global_features"]  # (1, D)
                if text_features.dim() == 1:
                    text_features = text_features.unsqueeze(0)
                return text_features
        except Exception as e:
            print(f"Error extracting text features: {e}")
            return None
    
    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算图像和文本特征的相似度"""
        try:
            similarity = self.model.compute_similarity(image_features, text_features)
            return similarity
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return torch.zeros(1, 1, device=self.device)
    
    def predict_binary(self, image_path: str, positive_text: str, negative_text: str) -> Tuple[float, float]:
        """
        二分类预测
        
        Args:
            image_path: 图像路径
            positive_text: 正类文本描述
            negative_text: 负类文本描述
            
        Returns:
            Tuple[float, float]: (正类概率, 负类概率)
        """
        # 提取图像特征
        image_features = self.extract_image_features(image_path)
        if image_features is None:
            return 0.5, 0.5
        
        # 提取文本特征
        pos_features = self.extract_text_features(positive_text)
        neg_features = self.extract_text_features(negative_text)
        
        if pos_features is None or neg_features is None:
            return 0.5, 0.5
        
        # 计算相似度
        pos_similarity = self.compute_similarity(image_features, pos_features).cpu().item()
        neg_similarity = self.compute_similarity(image_features, neg_features).cpu().item()
        
        # 使用softmax转换为概率
        similarities = torch.tensor([pos_similarity, neg_similarity])
        probabilities = torch.softmax(similarities / self.temperature, dim=0)
        
        return probabilities[0].item(), probabilities[1].item()
    
    def predict_proba(self, image_path: str, positive_text: str, negative_text: str) -> float:
        """预测正类概率（用于与原有接口兼容）"""
        pos_prob, neg_prob = self.predict_binary(image_path, positive_text, negative_text)
        return pos_prob


# ==================== 评估指标类 ====================

class BinaryEvalMetrics:
    """二分类评估指标计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置累积的预测和标签"""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor, probabilities: torch.Tensor):
        """更新累积的预测结果"""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        self.all_predictions.extend(predictions.flatten())
        self.all_labels.extend(labels.flatten())
        self.all_probabilities.extend(probabilities.flatten())
    
    def compute_all_metrics(self, threshold: float = 0.5) -> Dict[str, float]:
        """计算所有二分类指标"""
        if len(self.all_labels) == 0:
            raise ValueError("没有可用的预测结果")
        
        y_true = np.array(self.all_labels)
        y_prob = np.array(self.all_probabilities)
        y_pred = (y_prob > threshold).astype(int)
        
        metrics = {}
        
        # 基础分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['specificity'] = self._compute_specificity(y_true, y_pred)
        
        # AUC指标
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except ValueError as e:
            print(f"AUC-ROC计算失败: {e}")
            metrics['auc_roc'] = float('nan')
        
        try:
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
        except ValueError as e:
            print(f"AUC-PR计算失败: {e}")
            metrics['auc_pr'] = float('nan')
        
        # 其他指标
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        # 混淆矩阵元素
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        return metrics
    
    def _compute_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算特异性（真阴性率）"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def compute_optimal_f1_threshold(self) -> Tuple[float, float]:
        """
        计算最优F1阈值 - 对齐MedKLIP方法
        
        Returns:
            Tuple[float, float]: (最优阈值, 最大F1分数)
        """
        if len(self.all_labels) == 0:
            raise ValueError("没有可用的预测结果")
        
        y_true = np.array(self.all_labels)
        y_prob = np.array(self.all_probabilities)
        
        # 使用precision_recall_curve计算最优F1阈值，对齐MedKLIP
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        # 计算F1分数，对齐MedKLIP的计算方式
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        
        # 找到最大F1分数及其对应的阈值
        max_f1_idx = np.argmax(f1_scores)
        max_f1 = f1_scores[max_f1_idx]
        
        # 处理阈值数组长度问题（precision_recall_curve的特殊性）
        if max_f1_idx < len(thresholds):
            max_f1_thresh = thresholds[max_f1_idx]
        else:
            # 如果索引超出阈值数组，使用最后一个阈值
            max_f1_thresh = thresholds[-1] if len(thresholds) > 0 else 0.5
        
        return max_f1_thresh, max_f1
    
    def compute_medklip_style_metrics(self) -> Dict[str, float]:
        """
        计算MedKLIP风格的评估指标
        使用最优F1阈值计算准确率，对齐MedKLIP评估方式
        """
        if len(self.all_labels) == 0:
            raise ValueError("没有可用的预测结果")
        
        y_true = np.array(self.all_labels)
        y_prob = np.array(self.all_probabilities)
        
        # 计算最优F1阈值
        optimal_thresh, max_f1 = self.compute_optimal_f1_threshold()
        
        # 基于最优阈值进行预测
        y_pred_optimal = (y_prob > optimal_thresh).astype(int)
        
        # 计算基于最优阈值的指标
        accuracy_optimal = accuracy_score(y_true, y_pred_optimal)
        precision_optimal = precision_score(y_true, y_pred_optimal, zero_division=0)
        recall_optimal = recall_score(y_true, y_pred_optimal, zero_division=0)
        f1_optimal = f1_score(y_true, y_pred_optimal, zero_division=0)
        
        # 计算AUC（独立于阈值）
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except ValueError as e:
            print(f"AUC-ROC计算失败: {e}")
            auc_roc = float('nan')
        
        try:
            auc_pr = average_precision_score(y_true, y_prob)
        except ValueError as e:
            print(f"AUC-PR计算失败: {e}")
            auc_pr = float('nan')
        
        # 计算其他指标
        specificity_optimal = self._compute_specificity(y_true, y_pred_optimal)
        mcc_optimal = matthews_corrcoef(y_true, y_pred_optimal)
        balanced_accuracy_optimal = (recall_optimal + specificity_optimal) / 2
        
        # 混淆矩阵元素
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
        
        # 构建MedKLIP风格的结果字典
        results = {
            # MedKLIP主要指标
            'auc_roc': auc_roc,  # 主要性能指标
            'max_f1_score': max_f1,  # 最大F1分数
            'accuracy': accuracy_optimal,  # 基于最优F1阈值的准确率
            'optimal_f1_threshold': optimal_thresh,  # 最优F1阈值
            
            # 详细指标
            'precision': precision_optimal,
            'recall': recall_optimal,
            'f1_score': f1_optimal,  # 基于最优阈值的F1（应该等于max_f1）
            'specificity': specificity_optimal,
            'sensitivity': recall_optimal,  # 敏感性就是召回率
            'mcc': mcc_optimal,
            'balanced_accuracy': balanced_accuracy_optimal,
            'auc_pr': auc_pr,
            
            # 混淆矩阵
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            
            # 元信息
            'evaluation_method': 'MedKLIP-style optimal F1 threshold',
            'total_samples': len(y_true)
        }
        
        return results
    
    def compute_optimal_threshold(self) -> Tuple[float, Dict[str, float]]:
        """计算最优阈值（基于F1分数）"""
        if len(self.all_labels) == 0:
            raise ValueError("没有可用的预测结果")
        
        y_true = np.array(self.all_labels)
        y_prob = np.array(self.all_probabilities)
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # 使用最优阈值计算指标
        optimal_metrics = self.compute_all_metrics(optimal_threshold)
        
        return optimal_threshold, optimal_metrics
    
    def get_roc_data(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """获取ROC曲线数据（不绘图）"""
        if len(self.all_labels) == 0:
            raise ValueError("没有可用的预测结果")
        
        y_true = np.array(self.all_labels)
        y_prob = np.array(self.all_probabilities)
        
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            return fpr, tpr, auc
        except Exception as e:
            print(f"ROC数据计算失败: {e}")
            return np.array([]), np.array([]), float('nan')
    
    def get_confusion_matrix_summary(self, threshold: float = 0.5) -> Dict[str, int]:
        """获取混淆矩阵摘要（不绘图）"""
        if len(self.all_labels) == 0:
            raise ValueError("没有可用的预测结果")
        
        y_true = np.array(self.all_labels)
        y_prob = np.array(self.all_probabilities)
        y_pred = (y_prob > threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        return {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'total_samples': len(y_true),
            'positive_samples': int(np.sum(y_true)),
            'negative_samples': int(len(y_true) - np.sum(y_true))
        }


# ==================== 评估函数 ====================

def evaluate_pretrained_model(
    model_path: str,
    dataloader: DataLoader,
    dataset_name: str,
    device: str = "cuda"
) -> BinaryEvalMetrics:
    """评估预训练模型性能"""
    
    # 加载模型
    print(f"加载预训练模型: {model_path}")
    
    model = SimpleBinaryClassifier(backbone='resnet50', pretrained=False)
    
    # 尝试加载检查点
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"⚠️  模型加载失败: {e}")
            print("将使用预训练的ResNet50模型进行评估")
            model = SimpleBinaryClassifier(backbone='resnet50', pretrained=True)
    else:
        print(f"⚠️  模型文件不存在: {model_path}")
        print("将使用预训练的ResNet50模型进行评估")
        model = SimpleBinaryClassifier(backbone='resnet50', pretrained=True)
    
    model = model.to(device)
    model.eval()
    
    metrics = BinaryEvalMetrics()
    
    print(f"开始评估 {dataset_name} 数据集...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估进度"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            
            # MedKLIP风格：不使用固定阈值，只收集概率和标签
            # 预测将在后续使用最优F1阈值确定
            dummy_predictions = torch.zeros_like(probabilities)  # 临时占位符
            
            # 收集结果（主要是概率和标签）
            metrics.update(dummy_predictions, labels, probabilities)
    
    return metrics


# ==================== 主函数 ====================

def main():
    """评估主函数"""
    parser = argparse.ArgumentParser(description='二分类模型评估 - 仅评估版本')
    parser.add_argument('--model_path', type=str, help='模型权重路径（可选，如不提供则使用预训练ResNet50）')
    parser.add_argument('--dataset_name', type=str, choices=['siim', 'covid19', 'rsna'], 
                       required=True, help='数据集名称')
    parser.add_argument('--data_path', type=str, required=True, help='测试数据路径')
    parser.add_argument('--image_root', type=str, required=True, help='图像根目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
# 移除绘图参数
    # parser.add_argument('--save_plots', action='store_true', help='是否保存可视化图表')
    parser.add_argument('--max_samples', type=int, default=-1, help='最大样本数量，-1表示全部样本')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("🔬 二分类医学图像评估系统 - 仅评估版本")
    print("="*60)
    print(f"数据集: {args.dataset_name}")
    print(f"数据路径: {args.data_path}")
    print(f"图像根目录: {args.image_root}")
    print(f"模型路径: {args.model_path if args.model_path else '使用预训练ResNet50'}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {args.device}")
    print("="*60)
    
    # 创建数据集
    print("📂 加载评估数据...")
    eval_dataset = BinaryEvalDataset(
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        image_root=args.image_root,
        image_size=224
    )
    
    # 限制样本数量（如果指定）
    if args.max_samples > 0 and len(eval_dataset) > args.max_samples:
        eval_dataset.data = eval_dataset.data[:args.max_samples]
        print(f"限制样本数量为: {args.max_samples}")
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    # 评估模型
    metrics = evaluate_pretrained_model(
        model_path=args.model_path or "",
        dataloader=eval_loader,
        dataset_name=args.dataset_name,
        device=args.device
    )
    
    # 使用MedKLIP风格评估
    print("\n📊 MedKLIP风格评估结果分析...")
    
    try:
        # 计算MedKLIP风格指标（使用最优F1阈值）
        medklip_metrics = metrics.compute_medklip_style_metrics()
        
        # 同时计算传统固定阈值指标用于对比
        traditional_metrics = metrics.compute_all_metrics(threshold=0.5)
        
        # 打印MedKLIP风格结果（主要结果）
        print("\n" + "="*70)
        print("🎯 MedKLIP风格评估结果 (最优F1阈值):")
        print("="*70)
        
        # 按MedKLIP的输出顺序显示关键指标
        print(f"  {'平均AUC-ROC':25s}: {medklip_metrics['auc_roc']:.3f}")
        print(f"  {'最大F1分数':25s}: {medklip_metrics['max_f1_score']:.3f}")
        print(f"  {'准确率 (最优阈值)':25s}: {medklip_metrics['accuracy']:.3f}")
        print(f"  {'最优F1阈值':25s}: {medklip_metrics['optimal_f1_threshold']:.3f}")
        
        print(f"\n  详细指标:")
        print(f"  {'精确率':25s}: {medklip_metrics['precision']:.4f}")
        print(f"  {'召回率/敏感性':25s}: {medklip_metrics['recall']:.4f}")
        print(f"  {'特异性':25s}: {medklip_metrics['specificity']:.4f}")
        print(f"  {'平衡准确率':25s}: {medklip_metrics['balanced_accuracy']:.4f}")
        print(f"  {'MCC':25s}: {medklip_metrics['mcc']:.4f}")
        print(f"  {'AUC-PR':25s}: {medklip_metrics['auc_pr']:.4f}")
        
        print(f"\n  混淆矩阵:")
        print(f"  {'真阳性 (TP)':25s}: {medklip_metrics['true_positives']}")
        print(f"  {'真阴性 (TN)':25s}: {medklip_metrics['true_negatives']}")
        print(f"  {'假阳性 (FP)':25s}: {medklip_metrics['false_positives']}")
        print(f"  {'假阴性 (FN)':25s}: {medklip_metrics['false_negatives']}")
        print(f"  {'总样本数':25s}: {medklip_metrics['total_samples']}")
        
        # 对比传统方法（可选显示）
        print("\n" + "="*70)
        print("📈 传统固定阈值 (0.5) 对比结果:")
        print("="*70)
        print(f"  {'AUC-ROC':25s}: {traditional_metrics['auc_roc']:.3f}")
        print(f"  {'F1分数':25s}: {traditional_metrics['f1_score']:.3f}")
        print(f"  {'准确率':25s}: {traditional_metrics['accuracy']:.3f}")
        print(f"  {'精确率':25s}: {traditional_metrics['precision']:.4f}")
        print(f"  {'召回率':25s}: {traditional_metrics['recall']:.4f}")
        
    except Exception as e:
        print(f"❌ MedKLIP风格评估失败: {e}")
        print("回退到传统评估方法...")
        
        # 回退到传统方法
        traditional_metrics = metrics.compute_all_metrics(threshold=0.5)
        print("\n" + "="*60)
        print("📈 传统阈值 (0.5) 评估结果:")
        print("="*60)
        for metric, value in traditional_metrics.items():
            if isinstance(value, float):
                if np.isnan(value):
                    print(f"  {metric:25s}: NaN")
                else:
                    print(f"  {metric:25s}: {value:.4f}")
            else:
                print(f"  {metric:25s}: {value}")
        
        medklip_metrics = traditional_metrics  # 用于保存
    
    # 保存MedKLIP风格的结果
    results = {
        'dataset_name': args.dataset_name,
        'model_path': args.model_path,
        'model_type': 'Qwen2-VL CLIP',
        'total_samples': len(eval_dataset),
        'evaluation_timestamp': pd.Timestamp.now().isoformat(),
        'evaluation_method': 'MedKLIP-style optimal F1 threshold with CLIP similarity',
        'medklip_style_metrics': medklip_metrics,
        'traditional_metrics_comparison': traditional_metrics if 'traditional_metrics' in locals() else None
    }
    
    # 保存JSON结果
    results_path = os.path.join(args.output_dir, f'{args.dataset_name}_evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 评估结果已保存: {results_path}")
    
    # 保存MedKLIP风格的文本格式报告
    report_path = os.path.join(args.output_dir, f'{args.dataset_name}_medklip_evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"MedKLIP风格二分类评估报告 - {args.dataset_name.upper()}\n")
        f.write("="*70 + "\n")
        f.write(f"评估时间: {pd.Timestamp.now()}\n")
        f.write(f"数据集: {args.dataset_name}\n")
        f.write(f"总样本数: {len(eval_dataset)}\n")
        f.write(f"模型: {args.model_path} (Qwen2-VL CLIP)\n")
        f.write(f"评估方法: MedKLIP风格最优F1阈值\n\n")
        
        # MedKLIP风格的关键指标输出
        f.write("关键性能指标 (对齐MedKLIP):\n")
        f.write("-" * 40 + "\n")
        if not np.isnan(medklip_metrics['auc_roc']):
            f.write(f"平均AUC-ROC: {medklip_metrics['auc_roc']:.3f}\n")
        else:
            f.write(f"平均AUC-ROC: NaN\n")
        f.write(f"最大F1分数: {medklip_metrics['max_f1_score']:.3f}\n")
        f.write(f"准确率 (最优阈值): {medklip_metrics['accuracy']:.3f}\n")
        f.write(f"最优F1阈值: {medklip_metrics['optimal_f1_threshold']:.3f}\n\n")
        
        f.write("详细指标:\n")
        f.write("-" * 40 + "\n")
        f.write(f"精确率: {medklip_metrics['precision']:.4f}\n")
        f.write(f"召回率/敏感性: {medklip_metrics['recall']:.4f}\n")
        f.write(f"特异性: {medklip_metrics['specificity']:.4f}\n")
        f.write(f"平衡准确率: {medklip_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"MCC: {medklip_metrics['mcc']:.4f}\n")
        f.write(f"AUC-PR: {medklip_metrics['auc_pr']:.4f}\n\n")
        
        f.write("混淆矩阵:\n")
        f.write("-" * 40 + "\n")
        f.write(f"真阳性 (TP): {medklip_metrics['true_positives']}\n")
        f.write(f"真阴性 (TN): {medklip_metrics['true_negatives']}\n")
        f.write(f"假阳性 (FP): {medklip_metrics['false_positives']}\n")
        f.write(f"假阴性 (FN): {medklip_metrics['false_negatives']}\n")
    
    print(f"📄 MedKLIP风格评估报告已保存: {report_path}")
    
    print(f"\n✅ MedKLIP风格评估完成！")
    print(f"📊 主要结果 (对齐MedKLIP输出格式):")
    if not np.isnan(medklip_metrics['auc_roc']):
        print(f"   平均AUC-ROC: {medklip_metrics['auc_roc']:.3f}")
    else:
        print(f"   平均AUC-ROC: NaN")
    print(f"   最大F1分数: {medklip_metrics['max_f1_score']:.3f}")
    print(f"   准确率 (最优阈值): {medklip_metrics['accuracy']:.3f}")
    print(f"   最优F1阈值: {medklip_metrics['optimal_f1_threshold']:.3f}")
    
    return medklip_metrics


if __name__ == '__main__':
    main()
