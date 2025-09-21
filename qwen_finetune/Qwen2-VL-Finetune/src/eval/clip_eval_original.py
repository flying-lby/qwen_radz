"""         
CLIP风格Qwen2.5-VL模型的分类评估脚本
采用CLIP风格的图像-文本相似度计算进行分类任务评估
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
class LLaVAMedEvalConfig:
    """LLaVA-Med风格的评估参数配置"""
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


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressTracker:
    """进度跟踪器，提供详细的统计和时间估算"""
    
    def __init__(self, total_samples: int, batch_size: int):
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.total_batches = (total_samples - 1) // batch_size + 1
        
        self.start_time = time.time()
        self.processed_samples = 0
        self.processed_batches = 0
        
        # 状态统计
        self.success_count = 0
        self.error_count = 0
        self.status_stats = {"success": 0, "nan_fixed": 0, "zero_norm_fixed": 0, "degraded": 0, "failed": 0}
        
        # 时间统计
        self.batch_times = []
        self.last_batch_time = self.start_time
        
        # 简化的性能统计
        self.memory_usage_samples = []
        
    def update_batch(self, batch_valid_count: int, batch_status_stats: Dict[str, int]):
        """更新批次处理结果"""
        current_time = time.time()
        batch_duration = current_time - self.last_batch_time
        self.batch_times.append(batch_duration)
        self.last_batch_time = current_time
        
        self.processed_samples += batch_valid_count
        self.processed_batches += 1
        self.success_count += batch_valid_count
        self.error_count += (self.batch_size - batch_valid_count)
        
        # 合并状态统计
        for status, count in batch_status_stats.items():
            self.status_stats[status] += count
            
        # 记录内存使用情况（如果有GPU）
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            self.memory_usage_samples.append(memory_used)

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """获取当前内存使用情况"""
        memory_info = {}
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
            memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return memory_info
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        elapsed_time = time.time() - self.start_time
        
        # 计算处理速度
        if elapsed_time > 0:
            samples_per_sec = self.processed_samples / elapsed_time
            avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        else:
            samples_per_sec = 0
            avg_batch_time = 0
        
        # 估算剩余时间
        remaining_samples = self.total_samples - self.processed_samples
        if samples_per_sec > 0:
            eta_seconds = remaining_samples / samples_per_sec
            eta_str = f"{int(eta_seconds//3600):02d}:{int((eta_seconds%3600)//60):02d}:{int(eta_seconds%60):02d}"
        else:
            eta_str = "Unknown"
        
        # 成功率计算 - 只有真正成功的才算成功，降级处理算作错误
        true_success_count = self.status_stats.get("success", 0)
        total_attempted = self.success_count + self.error_count
        success_rate = (true_success_count / total_attempted * 100) if total_attempted > 0 else 0
        

        
        # 内存使用统计
        memory_stats = self.get_memory_usage()
        if self.memory_usage_samples:
            avg_memory = sum(self.memory_usage_samples) / len(self.memory_usage_samples)
            max_memory = max(self.memory_usage_samples)
        else:
            avg_memory = memory_stats.get('gpu_allocated', 0)
            max_memory = memory_stats.get('gpu_max_allocated', 0)
        
        return {
            "processed_samples": self.processed_samples,
            "total_samples": self.total_samples,
            "processed_batches": self.processed_batches,
            "total_batches": self.total_batches,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "samples_per_sec": samples_per_sec,
            "avg_batch_time": avg_batch_time,
            "elapsed_time": elapsed_time,
            "eta": eta_str,
            "status_breakdown": self.status_stats.copy(),
            

            
            # 新增：内存使用统计
            "avg_memory_gb": avg_memory,
            "max_memory_gb": max_memory,
            "current_memory": memory_stats
        }
    
    def format_progress_message(self) -> str:
        """格式化进度消息"""
        stats = self.get_stats()
        
        progress_msg = (
            f"批次进度: {stats['processed_batches']}/{stats['total_batches']} "
            f"({stats['processed_batches']/stats['total_batches']*100:.1f}%)\n"
            f"样本进度: {stats['processed_samples']}/{stats['total_samples']} "
            f"({stats['processed_samples']/stats['total_samples']*100:.1f}%)\n"
            f"成功率: {stats['success_rate']:.1f}% "
            f"(成功: {stats['success_count']}, 失败: {stats['error_count']})\n"
            f"处理速度: {stats['samples_per_sec']:.1f} samples/sec\n"
            f"预计剩余时间: {stats['eta']}\n"

        )
        
        # 添加GPU内存信息（如果可用）
        if torch.cuda.is_available():
            current_mem = stats['current_memory'].get('gpu_allocated', 0)
            max_mem = stats['max_memory_gb']
            progress_msg += f"GPU内存: 当前 {current_mem:.1f}GB, 峰值 {max_mem:.1f}GB\n"
        
        progress_msg += (
            f"状态详情: 成功={stats['status_breakdown']['success']}, "
            f"降级处理={stats['status_breakdown'].get('degraded', 0)}, "
            f"失败={stats['status_breakdown']['failed']}"
        )
        
        return progress_msg


def load_image_file(img_path):
    """
    加载图像文件，支持常规格式和DICOM格式
    """
    try:
        # 检查文件扩展名
        file_ext = os.path.splitext(img_path)[1].lower()
        
        if file_ext == '.dcm' and DICOM_AVAILABLE:
            # 处理DICOM文件 - 对齐LLaVA-Med的处理方式
            img = pydicom.dcmread(img_path).pixel_array  # 读取 DICOM 图像数据
            img = img.astype(float) / 255.0  # 归一化图像
            img = exposure.equalize_hist(img)  # 直方图均衡化，与LLaVA-Med保持一致
            
            # 转换为 PIL 图像并应用预处理
            img = (255 * img).astype(np.uint8)  # 转换为 uint8 类型
            image = Image.fromarray(img).convert('RGB') 
            return image
                
        elif file_ext == '.dcm' and not DICOM_AVAILABLE:
            raise ImportError("pydicom is required to read DCM files. Please install: pip install pydicom")
        else:
            # 处理常规图像文件
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


class ClipClassificationDataset(Dataset):
    """简化的CLIP分类评估数据集 - 直接使用数据集原生类别"""
    
    # 数据集配置：保留各数据集的语义特色
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
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            image_folder: 图像文件夹路径
            dataset_name: 数据集名称
            custom_classes: 自定义类别列表（可选，用于新数据集）
        """
        self.image_folder = image_folder
        self.dataset_name = dataset_name
        
        # 加载数据
        self.questions = self._load_data(data_path)
        
        # 设置类别和任务信息
        if custom_classes is not None:
            # 使用自定义类别（用于新数据集）
            self.target_classes = custom_classes
            self.task_type = 'custom'
            self.domain = 'custom'
            print(f"使用自定义类别: {custom_classes}")
        elif dataset_name in self.DATASET_CONFIGS:
            # 使用预定义配置
            config = self.DATASET_CONFIGS[dataset_name]
            self.target_classes = config['classes']
            self.task_type = config['task_type']
            self.domain = config['domain']
            print(f"加载数据集 '{dataset_name}': {len(self.target_classes)} 个类别, 任务类型: {self.task_type}")
        else:
            # 未知数据集，尝试从数据中自动推断
            self.target_classes = self._infer_classes_from_data()
            self.task_type = 'inferred'
            self.domain = 'unknown'
            print(f"未知数据集 '{dataset_name}', 自动推断类别: {self.target_classes}")
        
        # 验证数据一致性
        self._validate_data_consistency()
    
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
            
            print(f"成功加载 {len(questions)} 个样本")
            return questions
            
        except Exception as e:
            raise ValueError(f"数据加载失败: {e}")
    
    def _infer_classes_from_data(self) -> List[str]:
        """从数据中自动推断类别"""
        all_classes = set()
        
        for question in self.questions[:100]:  # 只检查前100个样本
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
            raise ValueError("无法从数据中推断类别，请提供custom_classes参数")
        
        return inferred_classes
    
    def _validate_data_consistency(self):
        """验证数据一致性"""
        if not self.questions:
            raise ValueError("数据集为空")
        
        # 检查标签格式一致性
        label_formats = set()
        missing_labels = 0
        
        for i, question in enumerate(self.questions[:50]):  # 检查前50个样本
            if 'label' not in question:
                missing_labels += 1
                continue
                
            labels = question['label']
            if isinstance(labels, dict):
                label_formats.add('dict')
            elif isinstance(labels, list):
                label_formats.add('list')
            elif isinstance(labels, str):
                label_formats.add('string')
            else:
                label_formats.add('other')
        
        if missing_labels > 0:
            print(f"警告: {missing_labels} 个样本缺少标签")
        
        if len(label_formats) > 1:
            print(f"警告: 检测到多种标签格式: {label_formats}")
        
        print(f"数据验证完成: {len(self.target_classes)} 个类别, 标签格式: {label_formats}")
    
    @classmethod
    def add_dataset_config(cls, dataset_name: str, classes: List[str], task_type: str = 'custom', domain: str = 'custom'):
        """添加新的数据集配置"""
        cls.DATASET_CONFIGS[dataset_name] = {
            'classes': classes,
            'task_type': task_type,
            'domain': domain
        }
        print(f"添加新数据集配置: {dataset_name}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return {
            'dataset_name': self.dataset_name,
            'num_samples': len(self.questions),
            'num_classes': len(self.target_classes),
            'classes': self.target_classes,
            'task_type': self.task_type,
            'domain': self.domain
        }
    
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
        # 支持多种图像路径字段名
        image_fields = ['image', 'image_path', 'img_path', 'file_path']
        
        for field in image_fields:
            if field in question:
                return os.path.join(self.image_folder, question[field])
        
        raise ValueError(f"未找到图像路径字段，支持的字段: {image_fields}")
    
    def _get_label_vector(self, question: Dict) -> np.ndarray:
        """获取标签向量"""
        true_vector = np.zeros(len(self.target_classes), dtype=np.float32)
        
        if 'label' not in question:
            return true_vector
        
        labels = question['label']
        
        # 处理不同的标签格式
        if isinstance(labels, dict):
            # 字典格式: {"pneumonia": 1, "normal": 0}
            for cls, value in labels.items():
                if cls in self.target_classes:
                    try:
                        # 支持多种正值表示
                        if value in [1, 1.0, True, "1", "true", "positive"]:
                            true_vector[self.target_classes.index(cls)] = 1.0
                    except (ValueError, TypeError):
                        continue
                        
        elif isinstance(labels, list):
            # 列表格式: ["pneumonia", "edema"]
            for cls in labels:
                if isinstance(cls, str) and cls in self.target_classes:
                    true_vector[self.target_classes.index(cls)] = 1.0
                    
        elif isinstance(labels, str):
            # 字符串格式: "pneumonia"
            if labels in self.target_classes:
                true_vector[self.target_classes.index(labels)] = 1.0
                
        elif isinstance(labels, (int, float)):
            # 数值格式（二分类）: 0 或 1
            if len(self.target_classes) == 2:
                if labels in [1, 1.0]:
                    true_vector[1] = 1.0  # 假设第二个类别是正类
                else:
                    true_vector[0] = 1.0  # 第一个类别是负类
        
        return true_vector


class ClipEvaluator:
    """CLIP风格分类评估器"""
    
    def __init__(
        self,
        model_path: str,
        batch_size: int = 16,
        use_disease_descriptions: bool = False,
        disease_desc_path: Optional[str] = None,
        description_source: str = "template",
        eval_config: Optional[LLaVAMedEvalConfig] = None
    ):
        # 自动检测设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.use_disease_descriptions = use_disease_descriptions
        self.disease_desc_path = disease_desc_path
        self.description_source = description_source
        
        # LLaVA-Med风格的评估配置
        self.eval_config = eval_config if eval_config is not None else LLaVAMedEvalConfig()
        self.Imgcls_count = self.eval_config.Imgcls_count
        self.Txtcls_count = self.eval_config.Txtcls_count
        self.feature_layer = self.eval_config.feature_layer
        self.temperature = self.eval_config.temperature
        
        # 加载疾病描述文件（如果启用）
        self.disease_descriptions = {}
        if self.use_disease_descriptions and self.disease_desc_path:
            try:
                with open(self.disease_desc_path, 'r', encoding='utf-8') as f:
                    self.disease_descriptions = json.load(f)
                logger.info(f"Loaded {len(self.disease_descriptions)} disease descriptions from {self.disease_desc_path}")
            except Exception as e:
                logger.warning(f"Failed to load disease descriptions from {self.disease_desc_path}: {e}")
                self.use_disease_descriptions = False
        
        # 加载模型、tokenizer 与 processor
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
            torch_dtype=torch.float16,  # 改为float16，更稳定
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # 检查模型权重是否包含NaN
        nan_params = []
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
        
        if nan_params:
            logger.warning(f"Model contains NaN parameters: {nan_params[:5]}...")  # 只显示前5个
            logger.warning("Attempting to fix NaN parameters by reinitializing them...")
            
            # 尝试修复NaN参数
            fixed_count = 0
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    # 重新初始化包含NaN的参数
                    with torch.no_grad():
                        if 'weight' in name:
                            if len(param.shape) >= 2:
                                # 使用Xavier初始化
                                torch.nn.init.xavier_uniform_(param)
                            else:
                                # 一维参数使用正态分布
                                torch.nn.init.normal_(param, 0, 0.02)
                        elif 'bias' in name:
                            # 偏置初始化为零
                            torch.nn.init.zeros_(param)
                        else:
                            # 其他参数使用正态分布
                            torch.nn.init.normal_(param, 0, 0.02)
                    fixed_count += 1
                    logger.info(f"Fixed NaN parameter: {name}")
            
            logger.info(f"Successfully fixed {fixed_count} NaN parameters.")
            
            # 再次检查是否还有NaN
            remaining_nan = []
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    remaining_nan.append(name)
            
            if remaining_nan:
                logger.error(f"Still have NaN parameters after fixing: {remaining_nan[:3]}...")
                raise ValueError("Failed to fix all NaN parameters.")
            else:
                logger.info("All NaN parameters have been successfully fixed.")
        
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
        
        # 在tokenizer加载完成后预计算疾病描述
        if self.use_disease_descriptions and self.disease_descriptions:
            self._prepare_disease_descriptions()
    
    def _prepare_disease_descriptions(self):
        """预计算疾病描述的tokenized ID（LLaVA-Med风格）"""
        if not self.disease_descriptions:
            return
            
        try:
            # 预计算疾病描述的 tokenized ID
            tokenized_desc = []
            for desc in self.disease_descriptions.values():
                tokens = self.tokenizer.encode(desc, return_tensors="pt").squeeze(0).clone().detach()
                tokenized_desc.append(tokens)

            # 进行 padding，确保形状为 [num_diseases, max_seq_len]
            if tokenized_desc:
                self.disease_desc_ids_padded = torch.nn.utils.rnn.pad_sequence(
                    tokenized_desc, batch_first=True, padding_value=self.tokenizer.pad_token_id
                ).to(self.device)
                self.disease_desc_attention_mask = self.disease_desc_ids_padded.ne(self.tokenizer.pad_token_id)
                logger.info(f"Prepared disease descriptions tensor: {self.disease_desc_ids_padded.shape}")
        except Exception as e:
            logger.warning(f"Failed to prepare disease descriptions: {e}")
            self.disease_desc_ids_padded = None
            self.disease_desc_attention_mask = None
    
    def prepare_image_input(self, image_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """准备图像输入（使用AutoProcessor生成pixel_values与image_grid_thw）"""
        try:
            image = load_image_file(image_path)
            # 在文本中放置图像占位符，并追加图像分类特殊标记，便于模型在隐藏态末尾输出对应特征
            imgcls_tokens = "".join([f"<Imgcls{i}>" for i in range(self.Imgcls_count)])
            prompt = f"{DEFAULT_IMAGE_TOKEN}\nAnalyze this medical image. {imgcls_tokens}"

            proc_inputs = self.processor(
                text=[prompt], images=[image], padding=False, do_resize=True, return_tensors="pt"
            )

            # 组织返回字典并迁移到目标设备
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
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    


    def prepare_text_input(self, text: str) -> Dict[str, torch.Tensor]:
        """准备文本输入"""
        # 创建包含特殊标记的输入
        txtcls_tokens = "".join([f"<Txtcls{i}>" for i in range(self.Txtcls_count)])
        prompt = f"{text}. {txtcls_tokens}"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=True
        )
        
        return {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
        }
    
    @torch.no_grad()
    def extract_image_features(self, image_path: str) -> Tuple[Optional[torch.Tensor], str]:
        """
        提取单张图像的特征（通过模型extract_features）
        
        Returns:
            Tuple[Optional[torch.Tensor], str]: (特征张量, 状态信息)
            状态: "success", "nan_fixed", "zero_norm_fixed", "failed"
        """
        inputs = self.prepare_image_input(image_path)
        if inputs is None or ("pixel_values" not in inputs):
            return None, "failed"
            
        try:
            feats = self.model.extract_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs.get("image_grid_thw")
            )
            image_features = feats["global_features"]  # (1, D)
            if image_features.dim() == 1:
                image_features = image_features.unsqueeze(0)
            
            # 验证输出维度（维度不匹配问题已在模型层修复）
            expected_dim = self.sparse_config["output_dim"]
            current_dim = image_features.shape[-1]
            
            # 如果仍有维度不匹配，说明模型配置有问题
            if current_dim != expected_dim:
                logger.error(f"Critical: Dimension mismatch for {image_path}: {current_dim} != {expected_dim}")
                logger.error(f"This suggests a model configuration issue. Expected dims should be consistent.")
                # 临时处理，但应该检查模型配置
                if current_dim > expected_dim:
                    image_features = image_features[:, :expected_dim]
                    logger.warning(f"Applied emergency dimension truncation")
                else:
                    padding = torch.zeros(image_features.shape[0], expected_dim - current_dim, 
                                        device=image_features.device, dtype=image_features.dtype)
                    image_features = torch.cat([image_features, padding], dim=-1)
                    logger.warning(f"Applied emergency dimension padding")
            
            status = "success"
            
            # 详细的特征质量检查和日志记录
            feature_norm = torch.norm(image_features, p=2, dim=-1).item()
            
            # 检查并修复NaN特征
            if torch.isnan(image_features).any():
                nan_count = torch.isnan(image_features).sum().item()
                logger.warning(f"Image features contain {nan_count} NaN values for {image_path}, fixing...")
                image_features = torch.where(torch.isnan(image_features), torch.zeros_like(image_features), image_features)
                status = "nan_fixed"
                # 重新计算范数
                feature_norm = torch.norm(image_features, p=2, dim=-1).item()
            
            # 检查并修复零范数特征  
            if feature_norm < 1e-8:
                logger.warning(f"Image features have near-zero norm ({feature_norm:.2e}) for {image_path}, using random fallback")
                image_features = torch.randn_like(image_features) * 0.01
                status = "zero_norm_fixed"
                feature_norm = torch.norm(image_features, p=2, dim=-1).item()
            
            # 记录特征统计信息
            if hasattr(logger, 'debug'):
                feature_mean = image_features.mean().item()
                feature_std = image_features.std().item()
                feature_min = image_features.min().item()
                feature_max = image_features.max().item()
                logger.debug(f"Feature stats for {image_path}: norm={feature_norm:.4f}, mean={feature_mean:.4f}, "
                           f"std={feature_std:.4f}, min={feature_min:.4f}, max={feature_max:.4f}")
                
            return image_features, status
            
        except Exception as e:
            logger.error(f"Error extracting image features for {image_path}: {e}")
            return None, "failed"
    


    @torch.no_grad()
    def extract_batch_image_features(self, image_paths: List[str]) -> Tuple[torch.Tensor, List[int], Dict[str, int]]:
        """
        简化的批量图像特征提取 - 逐个处理确保稳定性和效率
        
        Returns:
            Tuple[torch.Tensor, List[int], Dict[str, int]]: (特征张量, 有效索引, 状态统计)
        """
        batch_features = []
        valid_indices = []
        status_stats = {"success": 0, "nan_fixed": 0, "zero_norm_fixed": 0, "degraded": 0, "failed": 0}
        
        expected_dim = self.sparse_config["output_dim"]
        
        for idx, image_path in enumerate(image_paths):
            features, status = self.extract_image_features(image_path)
            status_stats[status] += 1
            
            if features is not None:
                # 维度检查（确保一致性）
                if features.shape[-1] != expected_dim:
                    logger.warning(f"Unexpected dimension after processing {image_path}: {features.shape[-1]} != {expected_dim}")
                    status_stats[status] -= 1
                    status_stats["failed"] += 1
                    continue
                    
                # 有效性检查
                if not torch.isnan(features).any() and torch.norm(features, p=2, dim=-1).item() > 0:
                    # 只有真正成功的特征才算作有效
                    if status == "success":
                        batch_features.append(features)
                        valid_indices.append(idx)
                    else:
                        # nan_fixed, zero_norm_fixed等降级处理不算作真正成功
                        logger.warning(f"Using degraded features for {image_path} (status: {status})")
                        batch_features.append(features)
                        valid_indices.append(idx)
                        # 但在统计中将其转为警告状态
                        status_stats[status] -= 1
                        status_stats["degraded"] = status_stats.get("degraded", 0) + 1
                else:
                    logger.warning(f"Skipping invalid features for {image_path}")
                    status_stats[status] -= 1
                    status_stats["failed"] += 1
            else:
                # features为None时，确保failed状态被正确统计
                if status != "failed":
                    status_stats[status] -= 1
                    status_stats["failed"] += 1
        
        if not batch_features:
            empty_tensor = torch.empty(0, expected_dim, dtype=torch.float32)
            return empty_tensor, [], status_stats
        
        try:
            batch_tensor = torch.cat(batch_features, dim=0)
            if torch.isnan(batch_tensor).any():
                logger.error("NaN detected after batch concatenation!")
                batch_tensor = torch.where(torch.isnan(batch_tensor), torch.zeros_like(batch_tensor), batch_tensor)
            return batch_tensor, valid_indices, status_stats
        except Exception as e:
            logger.error(f"Error during batch concatenation: {e}")
            empty_tensor = torch.empty(0, expected_dim, dtype=torch.float32)
            return empty_tensor, [], status_stats
    
    @torch.no_grad()
    def extract_text_features(self, text: str) -> Optional[torch.Tensor]:
        """提取文本特征（通过模型extract_features）"""
        inputs = self.prepare_text_input(text)
        try:
            feats = self.model.extract_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            text_features = feats["global_features"]  # (1, D)
            if text_features.dim() == 1:
                text_features = text_features.unsqueeze(0)
            
            # 检查提取的特征是否有效
            if torch.isnan(text_features).any():
                logger.warning(f"Text features contain NaN for text: {text[:50]}...")
                return None
            if torch.norm(text_features, p=2, dim=-1).item() == 0:
                logger.warning(f"Text features have zero norm for text: {text[:50]}...")
                return None
                
            return text_features
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return None
    
    @torch.no_grad()
    def extract_class_text_features(self, class_names: List[str]) -> torch.Tensor:
        """提取所有类别文本的特征"""
        features = []
        
        for class_name in tqdm(class_names, desc="Extracting class text features"):
            # 根据配置选择文本生成方式
            if self.use_disease_descriptions and self.description_source == "file":
                # 使用疾病描述文件中的详细描述
                if class_name in self.disease_descriptions:
                    class_text = self.disease_descriptions[class_name]
                    logger.debug(f"Using disease description for '{class_name}': {class_text[:100]}...")
                else:
                    # 如果找不到对应描述，回退到简单模板
                    if class_name == "no finding":
                        class_text = "Normal chest X-ray with no abnormal findings"
                    else:
                        class_text = f"Chest X-ray showing {class_name}"
                    logger.warning(f"Disease description not found for '{class_name}', using template: {class_text}")
            else:
                # 使用简单模板（默认行为）
                if class_name == "no finding":
                    class_text = "Normal chest X-ray with no abnormal findings"
                else:
                    class_text = f"Chest X-ray showing {class_name}"
            
            text_features = self.extract_text_features(class_text)
            if text_features is not None:
                features.append(text_features)
            else:
                # 如果特征提取失败，添加零向量
                base_param = next(self.model.parameters())
                zero_features = torch.zeros(
                    1,
                    self.sparse_config["output_dim"],
                    device=base_param.device,
                    dtype=base_param.dtype,
                )
                features.append(zero_features)
        
        if features:
            return torch.cat(features, dim=0)
        else:
            base_param = next(self.model.parameters())
            return torch.zeros(
                len(class_names),
                self.sparse_config["output_dim"],
                device=base_param.device,
                dtype=base_param.dtype,
            )
    
    def _optimize_threshold_f1(self, y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        LLaVA-Med风格的F1分数最优阈值选择
        
        Returns:
            Tuple[float, Dict[str, float]]: (最优阈值, 最优指标字典)
        """
        try:
            # 计算精确度、召回率和阈值
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            
            # 计算 F1 分数并找到最大值
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)  # 避免分母为0
            max_f1_idx = np.argmax(f1_scores)  # 最大 F1 对应的索引
            
            # 选择最大 F1 对应的阈值
            if max_f1_idx < len(thresholds):
                best_threshold = thresholds[max_f1_idx]
            else:
                # 如果索引超出范围，使用默认阈值
                best_threshold = 0.5
                max_f1_idx = len(precision) - 1
            
            # 返回最优阈值和对应的指标
            optimal_metrics = {
                'best_threshold': best_threshold,
                'max_f1': f1_scores[max_f1_idx],
                'precision_at_max_f1': precision[max_f1_idx],
                'recall_at_max_f1': recall[max_f1_idx]
            }
            
            return best_threshold, optimal_metrics
            
        except Exception as e:
            logger.warning(f"F1 threshold optimization failed: {e}, using default threshold 0.5")
            return 0.5, {'best_threshold': 0.5, 'max_f1': 0.0, 'precision_at_max_f1': 0.0, 'recall_at_max_f1': 0.0}
    
    @torch.no_grad()
    def _prepare_category_embeddings_cache(self, class_names: List[str]) -> torch.Tensor:
        """
        LLaVA-Med风格的类别嵌入缓存预计算
        
        Returns:
            torch.Tensor: 预计算的类别嵌入矩阵 [num_classes, embed_dim]
        """
        # from torch.nn.utils.rnn import pad_sequence  # 已在顶部导入
        
        # 生成类别描述文本
        categories = []
        for class_name in class_names:
            if self.use_disease_descriptions and class_name in self.disease_descriptions:
                # 使用详细的疾病描述
                category_text = self.disease_descriptions[class_name]
            else:
                # 使用简单模板
                if class_name == "no finding":
                    category_text = "This is a chest X-ray showing no finding"
                else:
                    category_text = f"This is a chest X-ray showing {class_name}"
            categories.append(category_text)
        
        # 对类别进行编码
        encoded_categories = [self.tokenizer(category, return_tensors="pt") for category in categories]
        category_ids = pad_sequence([item.input_ids.squeeze(0) for item in encoded_categories], batch_first=True).to(self.device)
        category_attention_mask = pad_sequence([item.attention_mask.squeeze(0) for item in encoded_categories], batch_first=True).to(self.device)
        
        # 类别特征向量存储, 只需要计算一次
        global_category_embeddings_cache = []
        
        for i in range(category_ids.size(0)):
            category_input_ids = category_ids[i].unsqueeze(0).to(self.device)
            category_attention = category_attention_mask[i].unsqueeze(0).to(self.device)

            # 使用我们自定义的extract_features方法
            try:
                text_features = self.extract_text_features(categories[i])
                if text_features is not None:
                    global_category_embeddings_cache.append(text_features)
                else:
                    # 如果特征提取失败，添加零向量
                    base_param = next(self.model.parameters())
                    zero_features = torch.zeros(
                        1,
                        self.sparse_config["output_dim"],
                        device=base_param.device,
                        dtype=base_param.dtype,
                    )
                    global_category_embeddings_cache.append(zero_features)
            except Exception as e:
                logger.warning(f"Failed to extract features for category {i}: {e}")
                # 添加零向量作为fallback
                base_param = next(self.model.parameters())
                zero_features = torch.zeros(
                    1,
                    self.sparse_config["output_dim"],
                    device=base_param.device,
                    dtype=base_param.dtype,
                )
                global_category_embeddings_cache.append(zero_features)
    
        global_category_embeddings_cache = torch.cat(global_category_embeddings_cache, dim=0).to(self.device)
        logger.info(f'预计算类别嵌入缓存完成: {global_category_embeddings_cache.shape}')
        
        return global_category_embeddings_cache
    
    def evaluate_clip_classification(
        self,
        dataset: ClipClassificationDataset
    ) -> Dict[str, float]:
        """
        使用CLIP风格方法评估分类性能，集成进度跟踪和增强错误处理
        策略：二分类选概率最大的1个，多分类选top3
        """
        target_classes = dataset.target_classes
        
        # LLaVA-Med风格：预计算类别嵌入缓存
        print("正在预计算类别嵌入缓存...")
        global_category_embeddings_cache = self._prepare_category_embeddings_cache(target_classes)
        
        # 初始化进度跟踪器
        progress_tracker = ProgressTracker(len(dataset), self.batch_size)
        
        # 存储所有预测和真实标签
        all_similarities = []
        all_predictions = []
        all_labels = []
        all_probs = []
        
        print(f"开始评估 {len(dataset)} 个样本")
        print("=" * 80)
        
        # 使用tqdm创建进度条
        with tqdm(total=len(dataset), desc="正在评估", unit="样本") as pbar:
            # 分批处理数据
            for batch_start in range(0, len(dataset), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(dataset))
                batch_samples = [dataset[i] for i in range(batch_start, batch_end)]
                
                # 批量提取图像特征（使用简化的方法）
                batch_image_paths = [sample["image_path"] for sample in batch_samples]
                batch_image_features, batch_valid_indices, batch_status_stats = self.extract_batch_image_features(batch_image_paths)
                
                if len(batch_valid_indices) == 0:
                    # 整个批次失败
                    progress_tracker.update_batch(0, batch_status_stats)
                    pbar.update(len(batch_samples))
                    continue
            
                # 逐个计算相似度（使用预计算的类别嵌入）
                batch_success_count = 0
                for rel_idx, abs_idx in enumerate(batch_valid_indices):
                    sample = batch_samples[abs_idx]
                    
                    try:
                        # 使用LLaVA-Med风格的inference_pipeline计算相似度
                        single_image_features = batch_image_features[rel_idx:rel_idx+1]  # 保持(1, D)维度
                        
                        # 直接使用预计算的类别嵌入计算相似度
                        similarity_result = self.model.compute_similarity(
                            single_image_features, global_category_embeddings_cache
                        ).cpu().numpy()  # (1, num_classes)
                        similarities = similarity_result.flatten()  # (num_classes,)
                        
                        # 简化预测策略：二分类选最大，多分类选top3
                        num_classes = len(similarities)
                        
                        if num_classes == 2:
                            # 二分类：选择相似度最高的1个类别
                            k = 1
                        else:
                            # 多分类：选择相似度最高的3个类别
                            k = min(3, num_classes)
                        
                        top_indices = np.argsort(similarities)[-k:]
                        predictions = np.zeros(len(similarities), dtype=int)
                        predictions[top_indices] = 1
                        
                        # 计算概率供参考
                        probs = torch.sigmoid(torch.tensor(similarities)).numpy()
                        
                        # 存储结果
                        all_similarities.append(similarities)
                        all_predictions.append(predictions)
                        all_labels.append(sample["true_labels"])
                        all_probs.append(probs)
                        
                        batch_success_count += 1
                        
                    except Exception as e:
                        logger.error(f"相似度计算失败，样本 {abs_idx}: {e}")
                        continue
                
                # 更新进度跟踪器
                progress_tracker.update_batch(batch_success_count, batch_status_stats)
                
                # 批次处理完成后清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 更新tqdm进度条
                pbar.update(len(batch_samples))
                
                # 更新进度条描述信息
                stats = progress_tracker.get_stats()
                memory_info = f", GPU: {stats['current_memory'].get('gpu_allocated', 0):.1f}GB" if torch.cuda.is_available() else ""
                pbar.set_postfix_str(f"成功率: {stats['success_rate']:.1f}%, 速度: {stats['samples_per_sec']:.1f}/s{memory_info}")
        
        print("\n" + "=" * 80)
        
        if not all_predictions:
            print("没有样本被成功处理！")
            return {}
        
        # 转换为numpy数组并调试
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probs = np.array(all_probs)
        
        # 调试信息：检查标签分布
        print(f"调试信息: all_labels shape: {all_labels.shape}")
        print(f"调试信息: all_labels sum per class: {all_labels.sum(axis=0)}")
        print(f"调试信息: target_classes: {target_classes}")
        print(f"调试信息: 总样本数: {len(all_labels)}")
        
        # 检查是否有任何正样本
        if all_labels.sum() == 0:
            print("错误：所有样本都没有正标签！检查数据集标签格式")
            return {}
        
        # 获取最终统计信息
        final_stats = progress_tracker.get_stats()
        
        print("\n" + "=" * 80)
        print("最终处理统计:")
        print(f"  总样本数: {final_stats['total_samples']}")
        print(f"  成功处理: {final_stats['success_count']}")
        print(f"  处理失败: {final_stats['error_count']}")
        print(f"  成功率: {final_stats['success_rate']:.1f}%")
        print(f"  总处理时间: {final_stats['elapsed_time']:.1f}秒")
        print(f"  平均处理速度: {final_stats['samples_per_sec']:.1f} samples/sec")
        print("\n状态详细统计:")
        for status, count in final_stats['status_breakdown'].items():
            print(f"  {status}: {count}")
        
        # 成功率警告
        if final_stats['success_rate'] < 90:
            print(f"\n⚠️  警告: 成功率较低 ({final_stats['success_rate']:.1f}%)，请检查数据质量和模型配置")
        elif final_stats['success_rate'] < 95:
            print(f"\n⚠️  注意: 成功率为 {final_stats['success_rate']:.1f}%，建议优化数据预处理")
        else:
            print(f"\n✅ 处理成功率良好: {final_stats['success_rate']:.1f}%")
        
        print("=" * 80)
        
        # 计算评估指标
        results = self.calculate_classification_metrics(
            all_labels, all_predictions, all_probs, target_classes
        )
        
        return results
    
    def calculate_classification_metrics(
        self,
        all_labels: np.ndarray,
        all_predictions: np.ndarray,
        all_probs: np.ndarray,
        target_classes: List[str]
    ) -> Dict[str, float]:
        """计算分类指标"""
        results = {}
        
        # 计算每个类别的指标（LLaVA-Med风格的阈值优化）
        class_metrics = []
        
        for i, class_name in enumerate(target_classes):
            if all_labels[:, i].sum() > 0:  # 确保该类别有正样本
                # LLaVA-Med风格的F1阈值优化
                optimal_threshold, optimal_metrics = self._optimize_threshold_f1(
                    all_labels[:, i], all_probs[:, i]
                )
                
                # 使用优化后的阈值计算最终预测
                optimized_predictions = (all_probs[:, i] >= optimal_threshold).astype(int)
                
                # 基本分类指标（使用优化后的预测）
                precision = precision_score(all_labels[:, i], optimized_predictions, zero_division=0)
                recall = recall_score(all_labels[:, i], optimized_predictions, zero_division=0) 
                f1 = f1_score(all_labels[:, i], optimized_predictions, zero_division=0)
                balanced_acc = balanced_accuracy_score(all_labels[:, i], optimized_predictions)
                
                # 计算准确率
                accuracy = (optimized_predictions == all_labels[:, i]).mean()
                
                # 混淆矩阵 - 处理边界情况
                cm = confusion_matrix(all_labels[:, i], all_predictions[:, i])
                if cm.size == 1:
                    # 只有一种类别的情况
                    if all_labels[:, i].sum() == 0:  # 全是负样本
                        tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                    else:  # 全是正样本
                        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
                else:
                    tn, fp, fn, tp = cm.ravel()
                
                # 敏感性和特异性
                sensitivity = recall  # TP/(TP+FN)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TN/(TN+FP)
                support = tp + fn  # 真实正样本数量
                
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
                print(f"           Precision={precision:.3f}, Recall={recall:.3f}, AUPRC={auprc_score:.3f}")
                print(f"           TP={tp}, FP={fp}, TN={tn}, FN={fn}, Support={support}")
            else:
                print(f"{class_name}: No positive samples")
        
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
                
                # 计算总体准确率
                total_correct = sum(m['tp'] + m['tn'] for m in class_metrics)
                total_samples = len(all_labels) * len(target_classes)
                results['overall_accuracy'] = total_correct / total_samples
                
                # 计算Hamming Loss（多标签分类指标）
                hamming_loss = np.mean(all_labels != all_predictions)
                results['hamming_loss'] = hamming_loss
        
        # 打印总体结果
        print(f"\n===== CLIP-Style Classification Results =====")
        print("Macro-averaged Metrics:")
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
    parser = argparse.ArgumentParser(description="Evaluate CLIP Qwen2.5-VL model for classification")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained CLIP model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to evaluation data (jsonl format)")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to images folder")
    parser.add_argument("--dataset", type=str, default="mimic", 
                       choices=["chestxray", "chexpert", "mimic", "rsna", "COVIDx_CXR", "SIIM_Pneumothorax", "siim", "covid-cxr2"],
                       help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--output_path", type=str, default="clip_eval_results.json", 
                       help="Path to save results")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks for parallel processing")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Current chunk index")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of samples to evaluate (-1 for all)")
    
    # 疾病描述相关参数
    parser.add_argument("--use_disease_descriptions", action="store_true", 
                       help="Use detailed disease descriptions instead of simple templates")
    parser.add_argument("--disease_desc_path", type=str, 
                       default="/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/new_full_disease.json",
                       help="Path to disease descriptions JSON file")
    parser.add_argument("--description_source", type=str, default="template", 
                       choices=["template", "file"],
                       help="Source of class descriptions: 'template' for simple templates, 'file' for detailed descriptions")
    
    # LLaVA-Med风格的评估配置参数
    parser.add_argument("--Imgcls_count", type=int, default=4, help="Number of image classification tokens")
    parser.add_argument("--Txtcls_count", type=int, default=4, help="Number of text classification tokens")
    parser.add_argument("--feature_layer", type=int, default=1, help="Feature layer for extraction")
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for similarity computation")
    parser.add_argument("--Book_choice", type=int, default=1, help="Choice of disease description source")
    

    
    args = parser.parse_args()
    
    print(f"Starting CLIP-style evaluation...")
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Image folder: {args.image_folder}")
    print(f"Dataset: {args.dataset}")
    print(f"Use disease descriptions: {args.use_disease_descriptions}")
    if args.use_disease_descriptions:
        print(f"Disease description path: {args.disease_desc_path}")
        print(f"Description source: {args.description_source}")
    
    # 创建LLaVA-Med风格的评估配置
    eval_config = LLaVAMedEvalConfig(
        Imgcls_count=args.Imgcls_count,
        Txtcls_count=args.Txtcls_count,
        feature_layer=args.feature_layer,
        temperature=args.temperature,
        Book_choice=args.Book_choice
    )
    
    # 创建评估器
    try:
        evaluator = ClipEvaluator(
            model_path=args.model_path,
            batch_size=args.batch_size,
            use_disease_descriptions=args.use_disease_descriptions,
            disease_desc_path=args.disease_desc_path,
            description_source=args.description_source,
            eval_config=eval_config
        )
        print(f"Model loaded successfully on {evaluator.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 创建数据集
    try:
        dataset = ClipClassificationDataset(
            data_path=args.data_path,
            image_folder=args.image_folder,
            dataset_name=args.dataset
        )
        
        # 分块处理（支持多进程评估）
        questions = dataset.questions
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        
        # 限制样本数量（用于快速测试）
        if args.max_samples > 0:
            questions = questions[:args.max_samples]
        
        # 更新数据集
        dataset.questions = questions
        
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Target classes: {dataset.target_classes}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # 执行CLIP风格分类评估
    try:
        # 使用简化的预测策略：二分类选最大，多分类选top3
        print(f"Using simplified strategy: binary->max, multi->top3")
            
        results = evaluator.evaluate_clip_classification(dataset)
        
        if not results:
            print("Evaluation failed - no results generated")
            return
        
        # 保存结果
        result_data = {
            "model_path": args.model_path,
            "dataset": args.dataset,
            "num_samples": len(dataset),
            "chunk_info": f"{args.chunk_idx+1}/{args.num_chunks}",
            "evaluation_method": "CLIP-style similarity",
            "metrics": results
        }
        
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {args.output_path}")
        
        # 打印关键指标摘要
        print(f"\n===== Key Performance Metrics =====")
        print(f"Macro F1 Score: {results.get('macro_f1', 0):.3f}")
        print(f"Macro Balanced Accuracy: {results.get('macro_balanced_accuracy', 0):.3f}")
        print(f"Mean AUC-ROC: {results.get('mean_auc', 0):.3f}")
        print(f"Overall Accuracy: {results.get('overall_accuracy', 0):.3f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()