"""         
CLIP风格Qwen2.5-VL模型的分类评估脚本
采用LLaVA RadZ一致的评测方法：简单池化 + embedding均值 + 概率决策
"""

import os
import sys
import json
import argparse
import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)

# 导入自定义模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.clip_modeling_qwen2_5_vl import (
    ClipQwen2VLConfig,
    ClipQwen2VLForConditionalGeneration
)
from constants import DEFAULT_IMAGE_TOKEN

# 导入DICOM处理库
try:
    import pydicom
    from skimage import exposure
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("Warning: pydicom not available. DCM files will not be supported.")

@dataclass
class LLaVAStyleEvalConfig:
    """LLaVA RadZ风格的评估参数配置"""
    Imgcls_count: int = 4
    Txtcls_count: int = 8  # 保持LLaVA RadZ的配置
    hidden_dim: int = 1024
    output_dim: int = 512
    temperature: float = 0.05
    feature_layer: int = -2  # 使用倒数第二层，与LLaVA RadZ一致
    use_simple_pooling: bool = True  # 使用简单均值池化
    use_embedding_mean: bool = True  # 使用embedding均值作为类别特征
    classification_threshold: float = 0.5  # 简单阈值，不使用F1优化

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressTracker:
    """简化的进度跟踪器"""
    
    def __init__(self, total_samples: int, batch_size: int):
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.start_time = time.time()
        self.processed_samples = 0
        self.success_count = 0
        self.error_count = 0
        
    def update_batch(self, batch_success_count: int):
        """更新批次处理结果"""
        self.processed_samples += batch_success_count
        self.success_count += batch_success_count
        self.error_count += (self.batch_size - batch_success_count)
        
    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        elapsed_time = time.time() - self.start_time
        success_rate = (self.success_count / max(1, self.success_count + self.error_count)) * 100
        samples_per_sec = self.processed_samples / max(1, elapsed_time)
        
        return {
            "processed_samples": self.processed_samples,
            "total_samples": self.total_samples,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "samples_per_sec": samples_per_sec,
            "elapsed_time": elapsed_time,
        }


def load_image_file(img_path):
    """加载图像文件，与LLaVA RadZ保持一致的处理方式"""
    try:
        file_ext = os.path.splitext(img_path)[1].lower()
        
        if file_ext == '.dcm' and DICOM_AVAILABLE:
            # DICOM处理保持与原始方式一致
            img = pydicom.dcmread(img_path).pixel_array
            img = img.astype(float) / 255.0
            img = exposure.equalize_hist(img)
            img = (255 * img).astype(np.uint8)
            image = Image.fromarray(img).convert('RGB') 
            return image
        else:
            # 常规图像处理
            image = Image.open(img_path).convert('RGB')
            return image
            
    except Exception as e:
        raise Exception(f"Failed to load image {img_path}: {str(e)}")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class LLaVAStyleClassificationDataset(Dataset):
    """LLaVA RadZ风格的分类评估数据集"""
    
    # 保持与原始一致的数据集配置
    DATASET_CONFIGS = {
        'chestxray': {
            'classes': ["fibrosis", "edema", "pneumothorax", "cardiomegaly", "atelectasis", 
                       "nodule", "emphysema", "no finding", "mass", "pleural_thickening", 
                       "effusion", "infiltration", "pneumonia", "hernia", "consolidation"],
            'task_type': 'multi_label',
            'domain': 'chest_xray_pathology'
        },
        'chexpert': {
            'classes': ['no finding', 'enlarged cardiomediastinum', 'cardiomegaly', 
                       'lung opacity', 'lung lesion', 'edema', 'consolidation', 
                       'pneumonia', 'atelectasis', 'pneumothorax', 'pleural effusion', 
                       'pleural other', 'fracture', 'support devices'],
            'task_type': 'multi_label',
            'domain': 'chest_xray_pathology'
        },
        'mimic': {
            'classes': ["atelectasis", "cardiomegaly", "consolidation", "edema", "enlarged cardiomediastinum",
                       "fracture", "lung lesion", "lung opacity", "no finding", "pleural effusion", 
                       "pleural other", "pneumonia", "pneumothorax", "support devices"],
            'task_type': 'multi_label',
            'domain': 'chest_xray_pathology'
        },
        'rsna': {
            'classes': ["normal", "pneumonia"],
            'task_type': 'binary',
            'domain': 'pneumonia_screening'
        },
        'COVIDx_CXR': {
            'classes': ["normal", "covid-19"],
            'task_type': 'binary',
            'domain': 'covid_detection'
        },
        'SIIM_Pneumothorax': {
            'classes': ["no finding", "pneumothorax"],
            'task_type': 'binary',
            'domain': 'pneumothorax_detection'
        },
        'siim': {
            'classes': ['non-pneumothorax', 'pneumothorax'],
            'task_type': 'binary',
            'domain': 'pneumothorax_detection'
        },
        'covid-cxr2': {
            'classes': ['normal', 'covid-19'],
            'task_type': 'binary',
            'domain': 'covid_detection'
        }
    }
    
    def __init__(
        self,
        data_path: str,
        image_folder: str = "",
        dataset_name: str = "mimic",
        custom_classes: Optional[List[str]] = None
    ):
        self.image_folder = image_folder
        self.dataset_name = dataset_name
        
        # 加载数据
        self.questions = self._load_data(data_path)
        
        # 设置类别信息
        if custom_classes is not None:
            self.target_classes = custom_classes
            self.task_type = 'custom'
            self.domain = 'custom'
        elif dataset_name in self.DATASET_CONFIGS:
            config = self.DATASET_CONFIGS[dataset_name]
            self.target_classes = config['classes']
            self.task_type = config['task_type']
            self.domain = config['domain']
        else:
            self.target_classes = self._infer_classes_from_data()
            self.task_type = 'inferred'
            self.domain = 'unknown'
        
        print(f"加载数据集 '{dataset_name}': {len(self.target_classes)} 个类别, {len(self.questions)} 个样本")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据文件"""
        questions = []
        try:
            if data_path.endswith('.jsonl'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            questions.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"警告: 第{line_num}行JSON解析失败: {e}")
            else:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        questions = data
                    else:
                        raise ValueError("JSON文件应包含一个列表")
            
            return questions
            
        except Exception as e:
            raise ValueError(f"数据加载失败: {e}")
    
    def _infer_classes_from_data(self) -> List[str]:
        """从数据中自动推断类别"""
        all_classes = set()
        
        for question in self.questions[:100]:
            if 'label' in question:
                labels = question['label']
                if isinstance(labels, dict):
                    all_classes.update(labels.keys())
                elif isinstance(labels, list):
                    all_classes.update(labels)
                elif isinstance(labels, str):
                    all_classes.add(labels)
        
        inferred_classes = sorted(list(all_classes))
        if not inferred_classes:
            raise ValueError("无法从数据中推断类别")
        
        return inferred_classes
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        """获取单个样本"""
        question = self.questions[index]
        
        # 构建图像路径
        img_path = self._get_image_path(question)
        
        # 获取真实标签向量
        true_vector = self._get_label_vector(question)
        
        return {
            "image_path": img_path,
            "true_labels": true_vector,
            "question_data": question
        }
    
    def _get_image_path(self, question: Dict) -> str:
        """获取图像路径"""
        image_fields = ['image', 'image_path', 'img_path', 'file_path']
        
        for field in image_fields:
            if field in question:
                return os.path.join(self.image_folder, question[field])
        
        raise ValueError(f"未找到图像路径字段")
    
    def _get_label_vector(self, question: Dict) -> np.ndarray:
        """获取标签向量"""
        true_vector = np.zeros(len(self.target_classes), dtype=np.float32)
        
        if 'label' not in question:
            return true_vector
        
        labels = question['label']
        
        if isinstance(labels, dict):
            for cls, value in labels.items():
                if cls in self.target_classes:
                    try:
                        if value in [1, 1.0, True, "1", "true", "positive"]:
                            true_vector[self.target_classes.index(cls)] = 1.0
                    except (ValueError, TypeError):
                        continue
                        
        elif isinstance(labels, list):
            for cls in labels:
                if isinstance(cls, str) and cls in self.target_classes:
                    true_vector[self.target_classes.index(cls)] = 1.0
                    
        elif isinstance(labels, str):
            if labels in self.target_classes:
                true_vector[self.target_classes.index(labels)] = 1.0
                
        elif isinstance(labels, (int, float)):
            if len(self.target_classes) == 2:
                if labels in [1, 1.0]:
                    true_vector[1] = 1.0
                else:
                    true_vector[0] = 1.0
        
        return true_vector


class LLaVAStyleEvaluator:
    """LLaVA RadZ风格的CLIP评估器"""
    
    def __init__(
        self,
        model_path: str,
        batch_size: int = 16,
        use_disease_descriptions: bool = False,
        disease_desc_path: Optional[str] = None,
        eval_config: Optional[LLaVAStyleEvalConfig] = None
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.use_disease_descriptions = use_disease_descriptions
        self.disease_desc_path = disease_desc_path
        
        # LLaVA RadZ风格的评估配置
        self.eval_config = eval_config if eval_config is not None else LLaVAStyleEvalConfig()
        self.feature_layer = self.eval_config.feature_layer  # -2
        self.temperature = self.eval_config.temperature
        self.classification_threshold = self.eval_config.classification_threshold
        
        # 加载疾病描述文件（可选）
        self.disease_descriptions = {}
        if self.use_disease_descriptions and self.disease_desc_path:
            try:
                with open(self.disease_desc_path, 'r', encoding='utf-8') as f:
                    self.disease_descriptions = json.load(f)
                logger.info(f"加载了 {len(self.disease_descriptions)} 个疾病描述")
            except Exception as e:
                logger.warning(f"疾病描述加载失败: {e}")
                self.use_disease_descriptions = False
        
        # 加载模型、tokenizer与processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        config = ClipQwen2VLConfig.from_pretrained(model_path)
        self.model = ClipQwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # 获取配置参数
        self.sparse_config = config.sparse_config
        
        logger.info(f"LLaVA RadZ风格评估器初始化完成，设备: {self.device}")
    
    def prepare_image_input(self, image_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """准备图像输入 - LLaVA RadZ风格"""
        try:
            image = load_image_file(image_path)
            # 简单的prompt，不添加特殊分类token
            prompt = f"{DEFAULT_IMAGE_TOKEN}\nAnalyze this medical image."

            proc_inputs = self.processor(
                text=[prompt], images=[image], padding=False, do_resize=True, return_tensors="pt"
            )

            result: Dict[str, torch.Tensor] = {
                "input_ids": proc_inputs["input_ids"].to(self.device),
                "attention_mask": proc_inputs["attention_mask"].to(self.device),
            }
            if "pixel_values" in proc_inputs:
                result["pixel_values"] = proc_inputs["pixel_values"].to(self.device)
            if "image_grid_thw" in proc_inputs:
                result["image_grid_thw"] = proc_inputs["image_grid_thw"].to(self.device)
            return result
        except Exception as e:
            logger.error(f"图像处理失败 {image_path}: {e}")
            return None
    
    @torch.no_grad()
    def extract_image_features_llava_style(self, image_path: str) -> Optional[torch.Tensor]:
        """
        LLaVA RadZ风格的图像特征提取：简单均值池化
        """
        inputs = self.prepare_image_input(image_path)
        if inputs is None:
            return None
            
        try:
            # 标准的模型前向传播
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                output_hidden_states=True,
                return_dict=True
            )
            
            # LLaVA RadZ风格：使用倒数第二层的简单均值池化
            hidden_states = outputs.hidden_states[self.feature_layer]  # (-2)
            
            # 简单均值池化，与LLaVA RadZ保持一致
            image_features = hidden_states.mean(dim=1)  # (1, hidden_size)
            
            # 基本的有效性检查
            if torch.isnan(image_features).any():
                logger.warning(f"图像特征包含NaN: {image_path}")
                return None
            
            if torch.norm(image_features, p=2, dim=-1).item() == 0:
                logger.warning(f"图像特征为零向量: {image_path}")
                return None
                
            return image_features
            
        except Exception as e:
            logger.error(f"图像特征提取失败: {image_path}: {e}")
            return None
    
    @torch.no_grad()
    def prepare_class_embeddings_llava_style(self, class_names: List[str]) -> torch.Tensor:
        """
        LLaVA RadZ风格的类别embedding准备：直接embedding均值
        """
        class_embeddings = []
        
        for class_name in class_names:
            # 生成类别描述文本
            if self.use_disease_descriptions and class_name in self.disease_descriptions:
                class_text = self.disease_descriptions[class_name]
            else:
                if class_name == "no finding":
                    class_text = "This is a chest X-ray showing no finding"
                else:
                    class_text = f"This is a chest X-ray showing {class_name}"
            
            # LLaVA RadZ风格：直接使用embedding均值，不进行完整前向传播
            tokens = self.tokenizer.encode(class_text, return_tensors="pt").to(self.device)
            
            # 直接获取embedding并计算均值
            with torch.no_grad():
                token_embeddings = self.model.get_input_embeddings()(tokens)  # (1, seq_len, hidden_size)
                class_embedding = token_embeddings.mean(dim=1)  # (1, hidden_size)
                class_embeddings.append(class_embedding)
        
        # 合并所有类别embedding
        all_class_embeddings = torch.cat(class_embeddings, dim=0)  # (num_classes, hidden_size)
        
        logger.info(f"LLaVA RadZ风格类别embedding准备完成: {all_class_embeddings.shape}")
        return all_class_embeddings
    
    @torch.no_grad()
    def compute_similarity_llava_style(self, image_features: torch.Tensor, 
                                      class_embeddings: torch.Tensor) -> torch.Tensor:
        """
        LLaVA RadZ风格的相似度计算：L2归一化 + 点积 + 温度缩放
        """
        # L2归一化
        norm_image_features = F.normalize(image_features, p=2, dim=-1)
        norm_class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(norm_image_features, norm_class_embeddings.T)
        
        # 温度缩放
        similarity_matrix = similarity_matrix / self.temperature
        
        return similarity_matrix
    
    def evaluate_classification_llava_style(
        self,
        dataset: LLaVAStyleClassificationDataset
    ) -> Dict[str, float]:
        """
        LLaVA RadZ风格的分类评估：概率决策 + 简单阈值
        """
        target_classes = dataset.target_classes
        
        # 预计算类别embedding（LLaVA RadZ风格）
        print("正在预计算类别embedding（LLaVA RadZ风格）...")
        class_embeddings = self.prepare_class_embeddings_llava_style(target_classes)
        
        # 初始化进度跟踪器
        progress_tracker = ProgressTracker(len(dataset), self.batch_size)
        
        # 存储预测结果
        all_similarities = []
        all_predictions = []
        all_labels = []
        all_probs = []
        
        print(f"开始LLaVA RadZ风格评估 {len(dataset)} 个样本")
        print("=" * 80)
        
        with tqdm(total=len(dataset), desc="LLaVA RadZ风格评估", unit="样本") as pbar:
            for i in range(0, len(dataset), self.batch_size):
                batch_samples = [dataset[j] for j in range(i, min(i + self.batch_size, len(dataset)))]
                batch_success_count = 0
                
                for sample in batch_samples:
                    try:
                        # 提取图像特征（LLaVA RadZ风格）
                        image_features = self.extract_image_features_llava_style(sample["image_path"])
                        if image_features is None:
                            continue
                        
                        # 计算相似度（LLaVA RadZ风格）
                        similarities = self.compute_similarity_llava_style(
                            image_features, class_embeddings
                        ).cpu().numpy().flatten()
                        
                        # LLaVA RadZ风格的分类决策：基于softmax概率
                        probs = torch.softmax(torch.tensor(similarities), dim=0).numpy()
                        
                        # 简单阈值决策，与LLaVA RadZ保持一致
                        predictions = (probs > self.classification_threshold).astype(int)
                        
                        # 如果没有类别超过阈值，选择概率最高的
                        if predictions.sum() == 0:
                            predictions[np.argmax(probs)] = 1
                        
                        # 存储结果
                        all_similarities.append(similarities)
                        all_predictions.append(predictions)
                        all_labels.append(sample["true_labels"])
                        all_probs.append(probs)
                        
                        batch_success_count += 1
                        
                    except Exception as e:
                        logger.error(f"样本处理失败: {e}")
                        continue
                
                # 更新进度
                progress_tracker.update_batch(batch_success_count)
                pbar.update(len(batch_samples))
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print("\n" + "=" * 80)
        
        if not all_predictions:
            print("没有样本被成功处理！")
            return {}
        
        # 转换为numpy数组
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probs = np.array(all_probs)
        
        # 输出处理统计
        final_stats = progress_tracker.get_stats()
        print(f"LLaVA RadZ风格评估完成:")
        print(f"  成功处理: {final_stats['success_count']}/{final_stats['total_samples']}")
        print(f"  成功率: {final_stats['success_rate']:.1f}%")
        print(f"  处理速度: {final_stats['samples_per_sec']:.1f} samples/sec")
        
        # 计算分类指标
        results = self.calculate_classification_metrics_llava_style(
            all_labels, all_predictions, all_probs, target_classes
        )
        
        return results
    
    def calculate_classification_metrics_llava_style(
        self,
        all_labels: np.ndarray,
        all_predictions: np.ndarray,
        all_probs: np.ndarray,
        target_classes: List[str]
    ) -> Dict[str, float]:
        """LLaVA RadZ风格的分类指标计算：简单阈值，不使用F1优化"""
        results = {}
        
        # 计算每个类别的指标
        class_metrics = []
        
        for i, class_name in enumerate(target_classes):
            if all_labels[:, i].sum() > 0:  # 确保该类别有正样本
                
                # 使用固定阈值的预测结果（与LLaVA RadZ一致）
                precision = precision_score(all_labels[:, i], all_predictions[:, i], zero_division=0)
                recall = recall_score(all_labels[:, i], all_predictions[:, i], zero_division=0) 
                f1 = f1_score(all_labels[:, i], all_predictions[:, i], zero_division=0)
                balanced_acc = balanced_accuracy_score(all_labels[:, i], all_predictions[:, i])
                
                # 计算准确率
                accuracy = (all_predictions[:, i] == all_labels[:, i]).mean()
                
                # 混淆矩阵
                cm = confusion_matrix(all_labels[:, i], all_predictions[:, i])
                if cm.size == 1:
                    if all_labels[:, i].sum() == 0:
                        tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                    else:
                        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
                else:
                    tn, fp, fn, tp = cm.ravel()
                
                # 敏感性和特异性
                sensitivity = recall
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                support = tp + fn
                
                # AUC指标
                try:
                    if len(np.unique(all_labels[:, i])) > 1:
                        auc_score = roc_auc_score(all_labels[:, i], all_probs[:, i])
                        precision_curve, recall_curve, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
                        auprc_score = auc(recall_curve, precision_curve)
                    else:
                        auc_score = 0.0
                        auprc_score = 0.0
                except Exception:
                    auc_score = 0.0
                    auprc_score = 0.0
                
                class_metrics.append({
                    'class': class_name,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'balanced_accuracy': balanced_acc,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'auc': auc_score,
                    'auprc': auprc_score,
                    'support': support,
                    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
                })
                
                print(f"{class_name}: F1={f1:.3f}, Bal_Acc={balanced_acc:.3f}, Sen={sensitivity:.3f}, Spec={specificity:.3f}, AUC={auc_score:.3f}")
        
        # 计算宏平均指标
        if class_metrics:
            valid_metrics = [m for m in class_metrics if m['support'] > 0]
            
            if valid_metrics:
                results['macro_f1'] = np.mean([m['f1'] for m in valid_metrics])
                results['macro_precision'] = np.mean([m['precision'] for m in valid_metrics])
                results['macro_recall'] = np.mean([m['recall'] for m in valid_metrics])
                results['macro_balanced_accuracy'] = np.mean([m['balanced_accuracy'] for m in valid_metrics])
                results['macro_sensitivity'] = np.mean([m['sensitivity'] for m in valid_metrics])
                results['macro_specificity'] = np.mean([m['specificity'] for m in valid_metrics])
                results['mean_auc'] = np.mean([m['auc'] for m in valid_metrics])
                results['mean_auprc'] = np.mean([m['auprc'] for m in valid_metrics])
                
                # 总体准确率
                total_correct = sum(m['tp'] + m['tn'] for m in class_metrics)
                total_samples = len(all_labels) * len(target_classes)
                results['overall_accuracy'] = total_correct / total_samples
                
                # Hamming Loss
                hamming_loss = np.mean(all_labels != all_predictions)
                results['hamming_loss'] = hamming_loss
        
        # 打印结果
        print(f"\n===== LLaVA RadZ风格分类结果 =====")
        print("宏平均指标:")
        print(f"  F1 Score: {results.get('macro_f1', 0):.3f}")
        print(f"  Balanced Accuracy: {results.get('macro_balanced_accuracy', 0):.3f}")
        print(f"  Sensitivity (Recall): {results.get('macro_sensitivity', 0):.3f}")
        print(f"  Specificity: {results.get('macro_specificity', 0):.3f}")
        print(f"  Precision: {results.get('macro_precision', 0):.3f}")
        print(f"  Mean AUC-ROC: {results.get('mean_auc', 0):.3f}")
        print(f"  Mean AUC-PR: {results.get('mean_auprc', 0):.3f}")
        print(f"  Overall Accuracy: {results.get('overall_accuracy', 0):.3f}")
        print(f"  Hamming Loss: {results.get('hamming_loss', 0):.3f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="LLaVA RadZ风格的Qwen2.5-VL分类评估")
    parser.add_argument("--model_path", type=str, required=True, help="训练模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="评估数据路径")
    parser.add_argument("--image_folder", type=str, required=True, help="图像文件夹路径")
    parser.add_argument("--dataset", type=str, default="mimic", 
                       choices=["chestxray", "chexpert", "mimic", "rsna", "COVIDx_CXR", "SIIM_Pneumothorax", "siim", "covid-cxr2"],
                       help="数据集名称")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--output_path", type=str, default="llava_style_eval_results.json", help="结果保存路径")
    parser.add_argument("--num_chunks", type=int, default=1, help="数据分块数")
    parser.add_argument("--chunk_idx", type=int, default=0, help="当前分块索引")
    parser.add_argument("--max_samples", type=int, default=-1, help="最大样本数 (-1表示全部)")
    
    # 疾病描述相关参数
    parser.add_argument("--use_disease_descriptions", action="store_true", help="是否使用疾病描述")
    parser.add_argument("--disease_desc_path", type=str, 
                       default="/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/new_full_disease.json",
                       help="疾病描述文件路径")
    
    # LLaVA RadZ风格配置参数
    parser.add_argument("--feature_layer", type=int, default=-2, help="特征提取层")
    parser.add_argument("--temperature", type=float, default=0.05, help="相似度计算温度")
    parser.add_argument("--classification_threshold", type=float, default=0.5, help="分类阈值")
    
    args = parser.parse_args()
    
    print(f"开始LLaVA RadZ风格分类评估...")
    print(f"模型路径: {args.model_path}")
    print(f"数据路径: {args.data_path}")
    print(f"图像路径: {args.image_folder}")
    print(f"数据集: {args.dataset}")
    print(f"使用疾病描述: {args.use_disease_descriptions}")
    
    # 创建评估配置
    eval_config = LLaVAStyleEvalConfig(
        feature_layer=args.feature_layer,
        temperature=args.temperature,
        classification_threshold=args.classification_threshold
    )
    
    # 创建评估器
    try:
        evaluator = LLaVAStyleEvaluator(
            model_path=args.model_path,
            batch_size=args.batch_size,
            use_disease_descriptions=args.use_disease_descriptions,
            disease_desc_path=args.disease_desc_path,
            eval_config=eval_config
        )
        print(f"LLaVA RadZ风格评估器加载成功，设备: {evaluator.device}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 创建数据集
    try:
        dataset = LLaVAStyleClassificationDataset(
            data_path=args.data_path,
            image_folder=args.image_folder,
            dataset_name=args.dataset
        )
        
        # 分块处理
        questions = dataset.questions
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        
        # 限制样本数量
        if args.max_samples > 0:
            questions = questions[:args.max_samples]
        
        dataset.questions = questions
        
        print(f"数据集加载完成: {len(dataset)} 个样本")
        print(f"类别: {dataset.target_classes}")
        
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return
    
    # 执行LLaVA RadZ风格评估
    try:
        print(f"使用LLaVA RadZ风格评估方法: 简单池化 + embedding均值 + 概率决策")
            
        results = evaluator.evaluate_classification_llava_style(dataset)
        
        if not results:
            print("评估失败 - 无结果生成")
            return
        
        # 保存结果
        result_data = {
            "model_path": args.model_path,
            "dataset": args.dataset,
            "num_samples": len(dataset),
            "chunk_info": f"{args.chunk_idx+1}/{args.num_chunks}",
            "evaluation_method": "LLaVA RadZ风格 - 简单池化+embedding均值",
            "metrics": results
        }
        
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存至 {args.output_path}")
        
        # 打印关键指标摘要
        print(f"\n===== 关键性能指标摘要 =====")
        print(f"宏平均F1分数: {results.get('macro_f1', 0):.3f}")
        print(f"宏平均平衡准确率: {results.get('macro_balanced_accuracy', 0):.3f}")
        print(f"平均AUC-ROC: {results.get('mean_auc', 0):.3f}")
        print(f"总体准确率: {results.get('overall_accuracy', 0):.3f}")
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
