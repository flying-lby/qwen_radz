"""
CLIP评估器模块
包含ClipEvaluator类的实现
"""

import os
import sys
import json
import logging
import gc
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.clip_modeling_qwen2_5_vl import (
    ClipQwen2VLConfig,
    ClipQwen2VLForConditionalGeneration
)
from constants import DEFAULT_IMAGE_TOKEN
from .utils import load_image_file
from .progress_tracker import ProgressTracker
from .dataset import ClipClassificationDataset

# 设置日志
logger = logging.getLogger(__name__)


class ClipEvaluator:
    """CLIP风格分类评估器 - 对齐MedKLIP评估方式"""
    
    def __init__(
        self,
        model_path: str,
        batch_size: int = 16,
        use_disease_descriptions: bool = False,
        disease_desc_path: Optional[str] = None,
        description_source: str = "template",
    ):
        # 自动检测设备，使用torch.device对象确保精确匹配
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # 明确指定cuda:0
        else:
            self.device = torch.device("cpu")
        self.batch_size = batch_size
        self.use_disease_descriptions = use_disease_descriptions
        self.disease_desc_path = disease_desc_path
        self.description_source = description_source
        
        # 加载疾病描述文件（如果启用）- 采用MedKLIP风格
        self.disease_descriptions = {}
        self.disease_book = None
        if self.use_disease_descriptions and self.disease_desc_path:
            try:
                with open(self.disease_desc_path, 'r', encoding='utf-8') as f:
                    json_book = json.load(f)
                
                # MedKLIP风格处理：转换为列表格式
                if isinstance(json_book, dict):
                    self.disease_book = [json_book[i] for i in json_book]
                    # 同时保持字典格式用于快速查找
                    self.disease_descriptions = json_book
                else:
                    self.disease_book = json_book
                    self.disease_descriptions = {f"desc_{i}": desc for i, desc in enumerate(json_book)}
                
                logger.info(f"Loaded {len(self.disease_book)} disease descriptions from {self.disease_desc_path} (MedKLIP style)")
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
            device_map=None,  # 改为None，避免设备分布不一致
            trust_remote_code=True
        )
        # 明确将整个模型移动到指定设备，确保所有组件在同一设备上
        self.model = self.model.to(self.device)
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
            # 设备一致性检查和修正（减少日志噪音）
            device_warnings_count = 0
            for key, tensor in inputs.items():
                if tensor.device != self.device:
                    inputs[key] = tensor.to(self.device)
                    device_warnings_count += 1
            
            # 只在有设备不匹配时才记录一次警告
            if device_warnings_count > 0:
                logger.debug(f"自动修正了 {device_warnings_count} 个输入张量的设备位置到 {self.device}")
            
            # 检查GPU内存，如果不足则清理
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                if memory_allocated > 15.0:  # 如果已分配内存超过15GB
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            feats = self.model.extract_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs.get("image_grid_thw")
            )
            image_features = feats["global_features"]  # (1, D)
            if image_features.dim() == 1:
                image_features = image_features.unsqueeze(0)
            
            # 确保输出特征在正确的设备上（静默修正）
            if image_features.device != self.device:
                image_features = image_features.to(self.device)
            
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
                
            # 立即清理临时变量和GPU缓存
            del inputs, feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return image_features, status
            
        except Exception as e:
            logger.error(f"Error extracting image features for {image_path}: {e}")
            # 发生错误时也要清理内存
            if 'inputs' in locals():
                del inputs
            if 'feats' in locals():
                del feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
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
    
    def get_medklip_style_tokenizer(self, disease_book: List[str]) -> Dict[str, torch.Tensor]:
        """
        MedKLIP风格的疾病描述预处理
        模仿MedKLIP的get_tokenizer函数
        """
        if not disease_book:
            return None
            
        # 使用与MedKLIP相同的参数
        disease_book_tokenizer = self.tokenizer(
            disease_book, 
            padding='max_length', 
            truncation=True, 
            max_length=64,  # MedKLIP使用64
            return_tensors="pt"
        )
        
        # 移动到设备
        for key in disease_book_tokenizer:
            disease_book_tokenizer[key] = disease_book_tokenizer[key].to(self.device)
            
        return disease_book_tokenizer
    
    @torch.no_grad()
    def extract_class_text_features(self, class_names: List[str]) -> torch.Tensor:
        """提取所有类别文本的特征 - 支持MedKLIP风格的知识库"""
        features = []
        
        # 如果有MedKLIP风格的disease_book，优先使用
        if self.use_disease_descriptions and self.disease_book:
            logger.info("Using MedKLIP-style disease book for text features")
            # 预处理整个disease_book（类似MedKLIP）
            disease_book_tokenizer = self.get_medklip_style_tokenizer(self.disease_book)
            
            # 为每个类别匹配最相关的疾病描述
            for class_name in tqdm(class_names, desc="Extracting class text features (MedKLIP style)"):
                best_desc = self._find_best_disease_description(class_name, self.disease_book)
                text_features = self.extract_text_features(best_desc)
                
                if text_features is not None:
                    features.append(text_features)
                else:
                    # 回退处理
                    fallback_text = f"Chest X-ray showing {class_name}"
                    fallback_features = self.extract_text_features(fallback_text)
                    features.append(fallback_features if fallback_features is not None else self._get_zero_features())
        else:
            # 原有的处理方式
            for class_name in tqdm(class_names, desc="Extracting class text features"):
                # 根据配置选择文本生成方式
                if self.use_disease_descriptions and self.description_source == "file":
                    # 使用疾病描述文件中的详细描述
                    if class_name in self.disease_descriptions:
                        class_text = self.disease_descriptions[class_name]
                        logger.debug(f"Using disease description for '{class_name}': {class_text[:100]}...")
                    else:
                        # 如果找不到对应描述，回退到简单模板
                        class_text = self._get_default_class_text(class_name)
                        logger.warning(f"Disease description not found for '{class_name}', using template: {class_text}")
                else:
                    # 使用简单模板（默认行为）
                    class_text = self._get_default_class_text(class_name)
                
                text_features = self.extract_text_features(class_text)
                if text_features is not None:
                    features.append(text_features)
                else:
                    features.append(self._get_zero_features())
        
        if features:
            return torch.cat(features, dim=0)
        else:
            return torch.zeros(
                len(class_names),
                self.sparse_config["output_dim"],
                device=next(self.model.parameters()).device,
                dtype=next(self.model.parameters()).dtype,
            )
    
    def _find_best_disease_description(self, class_name: str, disease_book: List[str]) -> str:
        """找到与类别名称最匹配的疾病描述（模仿MedKLIP的匹配逻辑）"""
        class_name_lower = class_name.lower().replace('_', ' ')
        
        # 精确匹配
        for desc in disease_book:
            if class_name_lower in desc.lower():
                return desc
        
        # 部分匹配
        for desc in disease_book:
            desc_words = desc.lower().split()
            class_words = class_name_lower.split()
            if any(word in desc_words for word in class_words):
                return desc
        
        # 如果都没匹配到，返回默认描述
        return self._get_default_class_text(class_name)
    
    def _get_default_class_text(self, class_name: str) -> str:
        """获取默认的类别文本描述"""
        if class_name == "no finding":
            return "Normal chest X-ray with no abnormal findings"
        elif class_name == "non-pneumothorax":
            return "Normal chest X-ray with no pneumothorax"
        else:
            return f"Chest X-ray showing {class_name}"
    
    def _get_zero_features(self) -> torch.Tensor:
        """获取零特征向量"""
        base_param = next(self.model.parameters())
        return torch.zeros(
            1,
            self.sparse_config["output_dim"],
            device=base_param.device,
            dtype=base_param.dtype,
        )
    
    def _ensure_same_device(self, image_features: torch.Tensor, class_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        确保两个特征张量在同一设备上
        
        Args:
            image_features: 图像特征张量
            class_features: 类别特征张量
        
        Returns:
            设备同步后的两个张量
        """
        if image_features.device != class_features.device:
            # 优先使用图像特征所在的设备（通常是主GPU）
            target_device = image_features.device
            logger.debug(f"设备同步: 将class_features从 {class_features.device} 移动到 {target_device}")
            class_features = class_features.to(target_device)
        
        return image_features, class_features
    
    def evaluate_clip_classification(
        self,
        dataset: ClipClassificationDataset
    ) -> Dict[str, float]:
        """
        使用CLIP风格方法评估分类性能，对齐MedKLIP的评估策略
        策略：使用最优F1阈值进行分类决策（与MedKLIP一致）
        """
        target_classes = dataset.target_classes
        
        # 提取所有类别的文本特征
        print("正在提取类别文本特征...")
        class_features = self.extract_class_text_features(target_classes)
        
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
            
                # 逐个计算相似度
                batch_success_count = 0
                for rel_idx, abs_idx in enumerate(batch_valid_indices):
                    sample = batch_samples[abs_idx]
                    
                    try:
                        # 单独计算每个样本的相似度
                        single_image_features = batch_image_features[rel_idx:rel_idx+1]  # 保持(1, D)维度
                        
                        # 设备同步：确保两个特征张量在同一设备上
                        single_image_features, class_features_synced = self._ensure_same_device(
                            single_image_features, class_features
                        )
                        
                        # 安全的相似度计算
                        similarity_result = self.model.compute_similarity(
                            single_image_features, class_features_synced
                        ).cpu().numpy()  # (1, num_classes)
                        similarities = similarity_result.flatten()  # (num_classes,)
                        
                        # MedKLIP对齐：直接使用概率，不进行固定预测
                        # 预测将在后续使用最优F1阈值确定
                        probs = torch.sigmoid(torch.tensor(similarities)).numpy()
                        
                        # 临时预测（将被最优F1阈值覆盖）
                        predictions = (probs > 0.5).astype(int)
                        
                        # 存储结果
                        all_similarities.append(similarities)
                        all_predictions.append(predictions)
                        all_labels.append(sample["true_labels"])
                        all_probs.append(probs)
                        
                        batch_success_count += 1
                        
                    except Exception as e:
                        # 详细的设备相关错误处理
                        error_msg = str(e)
                        if "same device" in error_msg or "cuda" in error_msg.lower():
                            logger.error(f"设备不匹配错误，样本 {abs_idx}: {e}")
                            logger.error(f"图像特征设备: {single_image_features.device}, 类别特征设备: {class_features.device}")
                        else:
                            logger.error(f"相似度计算失败，样本 {abs_idx}: {e}")
                        continue
                
                # 更新进度跟踪器
                progress_tracker.update_batch(batch_success_count, batch_status_stats)
                
                # 批次处理完成后清理GPU缓存和内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                
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
    
    def compute_AUCs_medklip_style(self, gt: torch.Tensor, pred: torch.Tensor, n_class: int) -> List[float]:
        """
        MedKLIP风格的AUC计算 - 改进版本
        处理无效类别，使用np.nan标记计算失败的情况
        """
        AUROCs = []
        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
        
        for i in range(n_class):
            try:
                # 检查数据有效性
                gt_class = gt_np[:, i]
                pred_class = pred_np[:, i]
                
                # 检查是否有NaN或inf值
                if np.any(np.isnan(pred_class)) or np.any(np.isinf(pred_class)):
                    logger.warning(f"Class {i}: prediction contains NaN/inf values")
                    AUROCs.append(np.nan)
                    continue
                
                # 检查标签是否有效（需要同时有0和1）
                unique_labels = np.unique(gt_class)
                if len(unique_labels) < 2:
                    logger.warning(f"Class {i}: insufficient label diversity (only {unique_labels})")
                    AUROCs.append(np.nan)
                    continue
                
                # 计算AUC
                auc_score = roc_auc_score(gt_class, pred_class)
                
                # 检查AUC结果是否有效
                if np.isnan(auc_score) or np.isinf(auc_score):
                    logger.warning(f"Class {i}: AUC calculation returned {auc_score}")
                    AUROCs.append(np.nan)
                else:
                    AUROCs.append(auc_score)
                    
            except (ValueError, IndexError) as e:
                # 处理计算异常
                logger.warning(f"Class {i}: AUC calculation failed - {str(e)}")
                AUROCs.append(np.nan)
                
        return AUROCs

    def calculate_classification_metrics(
        self,
        all_labels: np.ndarray,
        all_predictions: np.ndarray,
        all_probs: np.ndarray,
        target_classes: List[str]
    ) -> Dict[str, float]:
        """计算分类指标 - 对齐MedKLIP的评估方式"""
        results = {}
        
        # 转换为torch tensor以兼容MedKLIP的函数
        gt_tensor = torch.FloatTensor(all_labels)
        pred_tensor = torch.FloatTensor(all_probs)
        
        # 1. 计算AUC (MedKLIP风格)
        AUROCs = self.compute_AUCs_medklip_style(gt_tensor, pred_tensor, len(target_classes))
        
        # 计算有效AUC的平均值，排除NaN值
        valid_aucs = [auc for auc in AUROCs if not np.isnan(auc)]
        invalid_count = len(AUROCs) - len(valid_aucs)
        
        if len(valid_aucs) > 0:
            AUROC_avg = np.mean(valid_aucs)
            logger.info(f"AUC计算：{len(valid_aucs)}/{len(AUROCs)} 个类别有效，{invalid_count} 个类别被排除")
        else:
            AUROC_avg = np.nan
            logger.warning(f"AUC计算：所有 {len(AUROCs)} 个类别都无效，平均AUC设为NaN")
        
        # 2. 使用MedKLIP的最优F1阈值策略重新计算预测
        max_f1s = []
        accs = []
        optimized_predictions = np.zeros_like(all_predictions)
        
        for i in range(len(target_classes)):
            gt_np = all_labels[:, i]
            pred_np = all_probs[:, i]
            
            if gt_np.sum() > 0:  # 确保有正样本
                try:
                    # MedKLIP的最优F1阈值计算
                    precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
                    numerator = 2 * recall * precision
                    denom = recall + precision
                    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
                    max_f1 = np.max(f1_scores)
                    max_f1_thresh = thresholds[np.argmax(f1_scores)]
                    max_f1s.append(max_f1)
                    
                    # 使用最优阈值进行预测
                    optimized_predictions[:, i] = (pred_np > max_f1_thresh).astype(int)
                    accs.append(accuracy_score(gt_np, optimized_predictions[:, i]))
                    
                except (ValueError, IndexError):
                    max_f1s.append(0.0)
                    accs.append(0.0)
                    optimized_predictions[:, i] = 0
            else:
                max_f1s.append(0.0)
                accs.append(0.0)
                optimized_predictions[:, i] = 0
        
        # 3. 计算平均指标
        f1_avg = np.array(max_f1s).mean()
        acc_avg = np.array(accs).mean()
        
        # 存储结果（增加AUC有效性信息）
        results['mean_auc'] = AUROC_avg
        results['macro_f1'] = f1_avg
        results['overall_accuracy'] = acc_avg
        results['individual_aucs'] = AUROCs
        results['individual_f1s'] = max_f1s
        results['individual_accs'] = accs
        
        # AUC有效性统计
        results['valid_auc_count'] = len(valid_aucs)
        results['total_class_count'] = len(target_classes)
        results['invalid_auc_count'] = invalid_count
        results['auc_validity_rate'] = len(valid_aucs) / len(target_classes) if len(target_classes) > 0 else 0.0
        
        # 记录无效类别名称
        invalid_classes = [target_classes[i] for i, auc in enumerate(AUROCs) if np.isnan(auc)]
        results['invalid_classes'] = invalid_classes
        
        # 4. MedKLIP风格的输出格式（增强版）
        print(f"\n===== MedKLIP-Style Classification Results =====")
        
        # AUC统计信息
        if np.isnan(AUROC_avg):
            print('The average AUROC is NaN (no valid classes for AUC calculation)')
        else:
            print('The average AUROC is {AUROC_avg:.4f} (based on {valid_count}/{total_count} valid classes)'.format(
                AUROC_avg=AUROC_avg, valid_count=len(valid_aucs), total_count=len(target_classes)))
        
        # 逐个类别AUC结果
        print("\n=== Individual Class AUC Results ===")
        for i in range(len(target_classes)):
            if np.isnan(AUROCs[i]):
                print('The AUROC of {class_name} is NaN (excluded from average)'.format(class_name=target_classes[i]))
            else:
                print('The AUROC of {class_name} is {auc:.4f}'.format(class_name=target_classes[i], auc=AUROCs[i]))
        
        # 排除类别汇总
        if invalid_classes:
            print(f"\n⚠️  Excluded classes ({len(invalid_classes)}): {', '.join(invalid_classes)}")
            print(f"   Reason: Insufficient label diversity or invalid predictions")
        
        print('\nThe average f1 is {F1_avg:.4f}'.format(F1_avg=f1_avg))
        print('The average ACC is {ACC_avg:.4f}'.format(ACC_avg=acc_avg))
        
        # 5. 详细的每类别指标（增强版）
        if logger.level <= logging.DEBUG:
            print(f"\n===== Detailed Per-Class Metrics =====")
            for i, class_name in enumerate(target_classes):
                positive_count = all_labels[:, i].sum()
                total_count = len(all_labels[:, i])
                if positive_count > 0:
                    auc_str = f"{AUROCs[i]:.4f}" if not np.isnan(AUROCs[i]) else "NaN"
                    print(f"{class_name}: AUC={auc_str}, F1={max_f1s[i]:.4f}, ACC={accs[i]:.4f}, "
                          f"Pos={positive_count}/{total_count} ({positive_count/total_count:.1%})")
                else:
                    print(f"{class_name}: No positive samples (excluded from AUC)")
        
        # 6. 最终验证日志
        logger.info(f"AUC计算完成: 平均AUC={AUROC_avg:.4f if not np.isnan(AUROC_avg) else 'NaN'}, "
                   f"有效类别={len(valid_aucs)}/{len(target_classes)}, 有效率={len(valid_aucs)/len(target_classes):.1%}")
        
        return results
