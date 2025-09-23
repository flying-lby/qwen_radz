import argparse
import json
import math
import os
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
import pandas as pd
import logging
from torch.nn.utils.rnn import pad_sequence  # 添加LLaVA风格导入


# 配置 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 导入DICOM处理库
try:
    import pydicom
    from skimage import exposure
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    logger.warning("pydicom not available. DCM files will not be supported.")


# 导入模型相关库
try:
    from train.clip_modeling_qwen2_5_vl import (
        ClipQwen2VLConfig,
        ClipQwen2VLForConditionalGeneration
    )
    CUSTOM_MODEL_AVAILABLE = True
except ImportError:
    logger.warning("Custom model not available, trying standard imports")
    CUSTOM_MODEL_AVAILABLE = False

from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoConfig

# Qwen相关导入
try:
    from qwen_vl_utils import process_vision_info
    QWEN_UTILS_AVAILABLE = True
except ImportError:
    logger.warning("qwen_vl_utils not available")
    QWEN_UTILS_AVAILABLE = False

try:
    from src.utils import get_model_name_from_path, disable_torch_init
    from src.constants import DEFAULT_IMAGE_TOKEN
    SRC_UTILS_AVAILABLE = True
except ImportError:
    logger.warning("src.utils not available, using fallbacks")
    SRC_UTILS_AVAILABLE = False
    DEFAULT_IMAGE_TOKEN = "<image>"
    
    def get_model_name_from_path(path):
        return os.path.basename(path)


@dataclass
class EvaluationArguments:
    # 模型配置参数
    img_feat_dim: int = 1280
    txt_feat_dim: int = 1280
    feature_layer: int = -1
    temperature: float = 0.07
    
    # 评估参数
    batch_size: int = 4
    use_disease_descriptions: bool = True
    disease_desc_path: str = "./data/disease_desc.json"
    threshold: float = 0.5
    
    # 推理配置
    max_new_tokens: int = 128
    use_cache: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 设备管理配置
    force_single_gpu: bool = False  # 强制使用单GPU
    main_device: int = 0  # 主要设备ID
    
    # LLaVA风格配置
    use_optimal_threshold: bool = True  # 使用最优F1阈值


def split_list(lst, n):
    """将列表分成n个大致相等的块"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """获取第k个块（0-based索引）"""
    chunks = split_list(lst, n)
    return chunks[k]


def load_image_file(img_path, dataset_type=None):
    """加载图像文件，支持常规格式和DICOM格式"""
    try:
        # 检查文件扩展名
        file_ext = os.path.splitext(img_path)[1].lower()
        
        # 特殊处理RSNA数据集的DICOM文件
        if (file_ext == '.dcm' and DICOM_AVAILABLE) or (dataset_type == 'rsna' and not file_ext):
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
                        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8) * 255
                else:
                    # 默认归一化到0-255范围
                    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8) * 255
                
                # 处理不同的像素数据格式
                if len(pixel_array.shape) == 2:
                    # 灰度图像转RGB
                    pixel_array = pixel_array.astype(np.uint8)
                    image = Image.fromarray(pixel_array, mode='L').convert('RGB')
                elif len(pixel_array.shape) == 3:
                    # 彩色图像或多帧图像，取第一帧
                    if pixel_array.shape[0] < pixel_array.shape[2]:
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


class QwenRadzMedEvaluator:
    def __init__(self, model_path, args, eval_args):
        self.args = args
        self.eval_args = eval_args
        
        # 设备管理策略
        self._setup_devices()
        
        # 加载模型和处理器
        logger.info(f"Loading model from: {model_path}")
        self.model_name = get_model_name_from_path(model_path)
        
        # 1. 加载Tokenizer和Processor
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            logger.info("Processor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise
        
        # 2. 加载模型配置和预训练模型（关键修改：设备管理）
        try:
            if CUSTOM_MODEL_AVAILABLE:
                self.config = ClipQwen2VLConfig.from_pretrained(model_path)
                
                # 根据GPU策略选择不同的加载方式
                if self.use_multi_gpu and not eval_args.force_single_gpu:
                    # 多GPU模式：不使用device_map，手动管理
                    self.model_device = ClipQwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        config=self.config,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                else:
                    # 单GPU模式：直接加载到主设备
                    self.model_device = ClipQwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        config=self.config,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
            else:
                # 使用标准transformers模型
                self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                if self.use_multi_gpu and not eval_args.force_single_gpu:
                    self.model_device = AutoModel.from_pretrained(
                        model_path,
                        config=self.config,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                else:
                    self.model_device = AutoModel.from_pretrained(
                        model_path,
                        config=self.config,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
            
            self.model_device.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # 3. 设备管理和多GPU配置
        self._configure_model_devices()
        
        # 4. 读取模型配置参数
        self.sparse_config = getattr(self.config, 'sparse_config', {})
        self.Imgcls_count = self.sparse_config.get("Imgcls_count", 10)
        self.Txtcls_count = self.sparse_config.get("Txtcls_count", 10)
        self.temperature = self.sparse_config.get("temperature", eval_args.temperature)
        
        # 5. 修复模型中的NaN参数
        self._fix_nan_parameters()
        
        # 加载数据集类别
        self.dataset_classes = self._get_dataset_classes()
        
        # 加载疾病描述
        self.disease_descriptions = {}
        if eval_args.use_disease_descriptions and os.path.exists(eval_args.disease_desc_path):
            with open(eval_args.disease_desc_path, 'r', encoding='utf-8') as f:
                self.disease_descriptions = json.load(f)
            logger.info(f"Loaded disease descriptions from {eval_args.disease_desc_path}")
        
        # 检查模型方法
        self._check_model_capabilities()
        
        # 预计算类别特征（修改为LLaVA风格）
        self.category_embeddings = self._precompute_category_embeddings_llava_style()
        
        logger.info(f"Successfully initialized evaluator for {args.dataset} dataset")

    def _setup_devices(self):
        """设置设备管理策略"""
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            
            # 主设备选择
            main_device_id = getattr(self.eval_args, 'main_device', 0)
            if main_device_id >= self.num_gpus:
                main_device_id = 0
                logger.warning(f"Specified main_device {self.eval_args.main_device} not available, using device 0")
            
            self.main_device = torch.device(f"cuda:{main_device_id}")
            self.device = self.main_device  # 兼容性
            
            # 多GPU策略
            if self.num_gpus > 1 and not getattr(self.eval_args, 'force_single_gpu', False):
                self.use_multi_gpu = True
                logger.info(f"Multi-GPU setup: {self.num_gpus} GPUs available, main device: {self.main_device}")
            else:
                self.use_multi_gpu = False
                logger.info(f"Single-GPU setup: using device {self.main_device}")
        else:
            self.main_device = torch.device("cpu")
            self.device = self.main_device
            self.use_multi_gpu = False
            self.num_gpus = 0
            logger.info("Using CPU device")

    def _configure_model_devices(self):
        """配置模型设备"""
        # 首先将模型移动到主设备
        self.model_device = self.model_device.to(self.main_device)
        
        # 配置多GPU包装
        if self.use_multi_gpu:
            # 创建DataParallel包装，指定主设备
            self.model = torch.nn.DataParallel(
                self.model_device, 
                device_ids=list(range(self.num_gpus)),
                output_device=self.main_device.index
            )
            self.model = self.model.to(self.main_device)
            logger.info(f"DataParallel configured with main device: {self.main_device}")
        else:
            self.model = self.model_device
        
        # 验证模型设备
        self._verify_model_devices()

    def _verify_model_devices(self):
        """验证模型所有组件都在正确设备上"""
        model_to_check = self.model_device
        
        devices = set()
        for name, param in model_to_check.named_parameters():
            devices.add(param.device)
        
        if len(devices) > 1:
            logger.warning(f"Model parameters are on multiple devices: {devices}")
            # 强制移动所有参数到主设备
            model_to_check = model_to_check.to(self.main_device)
            logger.info(f"Moved all model parameters to {self.main_device}")
        else:
            logger.info(f"All model parameters are on: {list(devices)[0]}")

    def _check_model_capabilities(self):
        """检查模型的可用方法"""
        model_to_check = self.model_device
        
        # 检查可用的方法
        methods = [method for method in dir(model_to_check) if not method.startswith('_')]
        feature_methods = [m for m in methods if 'feature' in m.lower() or 'encode' in m.lower()]
        
        logger.info(f"Available feature-related methods: {feature_methods}")
        
        # 设置特征提取策略
        if hasattr(model_to_check, 'extract_features'):
            self.feature_extraction_method = 'extract_features'
        elif hasattr(model_to_check, 'encode_images'):
            self.feature_extraction_method = 'encode_images'
        elif hasattr(model_to_check, 'get_image_features'):
            self.feature_extraction_method = 'get_image_features'
        else:
            self.feature_extraction_method = 'forward'
            
        logger.info(f"Using feature extraction method: {self.feature_extraction_method}")

    def _fix_nan_parameters(self):
        """修复模型中的NaN参数"""
        nan_params = []
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
        
        if nan_params:
            logger.warning(f"Model contains NaN parameters: {nan_params[:5]}...")
            fixed_count = 0
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    with torch.no_grad():
                        if 'weight' in name:
                            if len(param.shape) >= 2:
                                torch.nn.init.xavier_uniform_(param)
                            else:
                                torch.nn.init.normal_(param, 0, 0.02)
                        elif 'bias' in name:
                            torch.nn.init.zeros_(param)
                    fixed_count += 1
            logger.info(f"Fixed {fixed_count} NaN parameters.")

    def _get_dataset_classes(self):
        """获取数据集对应的类别列表"""
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
            'rsna': ["pneumonia", "normal"],
            'COVIDx_CXR': ["covid-19", "pneumonia", "normal"],
            'SIIM_Pneumothorax': ["pneumothorax", "no finding"]
        }
        
        if self.args.dataset in dataset_classes:
            return dataset_classes[self.args.dataset]
        else:
            logger.warning(f"Unknown dataset {self.args.dataset}, using MIMIC classes as default")
            return dataset_classes['mimic']

    def _ensure_tensor_device(self, tensor, target_device=None):
        """确保张量在正确设备上"""
        if target_device is None:
            target_device = self.main_device
        
        if tensor.device != target_device:
            tensor = tensor.to(target_device)
        return tensor

    def _safe_processor_call(self, text_input, image_input):
        """安全的processor调用，确保设备一致性"""
        try:
            # 尝试标准调用
            inputs = self.processor(
                text=text_input,
                images=image_input,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            return inputs, "success"
        except Exception as e:
            logger.warning(f"Standard processor call failed: {e}")
            
            # 尝试分别处理文本和图像
            try:
                # 处理文本
                text_inputs = self.tokenizer(
                    text_input,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # 处理图像
                if hasattr(self.processor, 'image_processor'):
                    image_inputs = self.processor.image_processor(
                        images=image_input,
                        return_tensors="pt"
                    )
                else:
                    # 简单的图像预处理
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    image_tensor = transform(image_input[0]).unsqueeze(0)
                    image_inputs = {"pixel_values": image_tensor}
                
                # 合并输入
                combined_inputs = {**text_inputs, **image_inputs}
                return combined_inputs, "fallback"
                
            except Exception as e2:
                logger.error(f"Fallback processor call also failed: {e2}")
                return None, "failed"

    def extract_image_features(self, image) -> Tuple[Optional[torch.Tensor], str]:
        """
        提取图像特征（设备安全版本）
        """
        try:
            # 准备输入
            if hasattr(self, 'Imgcls_count') and self.Imgcls_count > 0:
                imgcls_tokens = "".join([f"<Imgcls{i}>" for i in range(self.Imgcls_count)])
                prompt = f"{DEFAULT_IMAGE_TOKEN}\nAnalyze this medical image. {imgcls_tokens}"
            else:
                prompt = f"{DEFAULT_IMAGE_TOKEN}\nAnalyze this medical image."
            
            # 安全的processor调用
            inputs, processor_status = self._safe_processor_call([prompt], [image])
            
            if inputs is None:
                logger.error("Failed to process image input")
                return None, "failed"
            
            # 确保所有输入张量都在主设备上
            device_inputs = {}
            for key, value in inputs.items():
                if value is not None and hasattr(value, 'to'):
                    try:
                        device_inputs[key] = self._ensure_tensor_device(value, self.main_device)
                    except Exception as e:
                        logger.warning(f"Failed to move {key} to device: {e}")
                        device_inputs[key] = value
                else:
                    device_inputs[key] = value
            
            # 验证所有输入都在同一设备上
            input_devices = set()
            for key, value in device_inputs.items():
                if hasattr(value, 'device'):
                    input_devices.add(value.device)
            
            if len(input_devices) > 1:
                logger.warning(f"Input tensors are on multiple devices: {input_devices}")
                # 强制移动到主设备
                for key, value in device_inputs.items():
                    if hasattr(value, 'to'):
                        device_inputs[key] = value.to(self.main_device)
            
            # 选择正确的模型进行推理
            if self.use_multi_gpu:
                # 多GPU情况：使用DataParallel包装的模型，但确保输入在主设备上
                model_to_use = self.model
            else:
                # 单GPU情况：使用原始模型
                model_to_use = self.model_device
            
            # 提取特征
            with torch.no_grad():
                try:
                    # 根据检测到的方法选择特征提取策略
                    if self.feature_extraction_method == 'extract_features':
                        if self.use_multi_gpu:
                            # 对于DataParallel，使用原始模型避免设备问题
                            outputs = self.model_device.extract_features(**device_inputs)
                        else:
                            outputs = model_to_use.extract_features(**device_inputs)
                        
                        if isinstance(outputs, dict) and 'global_features' in outputs:
                            features = outputs['global_features']
                        else:
                            logger.warning("extract_features returned unexpected format")
                            return None, "failed"
                            
                    elif self.feature_extraction_method == 'encode_images':
                        if 'pixel_values' in device_inputs:
                            if self.use_multi_gpu:
                                features = self.model_device.encode_images(device_inputs['pixel_values'])
                            else:
                                features = model_to_use.encode_images(device_inputs['pixel_values'])
                        else:
                            logger.error("No pixel_values for encode_images")
                            return None, "failed"
                            
                    elif self.feature_extraction_method == 'get_image_features':
                        if 'pixel_values' in device_inputs:
                            if self.use_multi_gpu:
                                features = self.model_device.get_image_features(device_inputs['pixel_values'])
                            else:
                                features = model_to_use.get_image_features(device_inputs['pixel_values'])
                        else:
                            logger.error("No pixel_values for get_image_features")
                            return None, "failed"
                            
                    else:  # forward方法
                        if self.use_multi_gpu:
                            outputs = self.model_device(**device_inputs)
                        else:
                            outputs = model_to_use(**device_inputs)
                        
                        # 尝试从outputs中提取特征
                        if hasattr(outputs, 'last_hidden_state'):
                            features = outputs.last_hidden_state.mean(dim=1)
                        elif hasattr(outputs, 'pooler_output'):
                            features = outputs.pooler_output
                        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                            features = outputs.hidden_states[-1].mean(dim=1)
                        elif isinstance(outputs, tuple) and len(outputs) > 0:
                            features = outputs[0]
                            if features.dim() == 3:  # [batch, seq, hidden]
                                features = features.mean(dim=1)
                        else:
                            logger.error(f"Cannot extract features from outputs: {type(outputs)}")
                            return None, "failed"
                
                except Exception as e:
                    logger.error(f"Feature extraction failed with method {self.feature_extraction_method}: {e}")
                    return None, "failed"
            
            # 确保特征在主设备上
            features = self._ensure_tensor_device(features, self.main_device)
            
            # 处理特征维度
            if features.dim() == 1:
                features = features.unsqueeze(0)
            elif features.dim() > 2:
                # 如果是3D或更高维度，平均池化到2D
                while features.dim() > 2:
                    features = features.mean(dim=1)
            
            # 特征质量检查
            status = "success"
            feature_norm = torch.norm(features, p=2, dim=-1).item()
            
            # 检查并修复NaN特征
            if torch.isnan(features).any():
                nan_count = torch.isnan(features).sum().item()
                logger.warning(f"Image features contain {nan_count} NaN values, fixing...")
                features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
                status = "nan_fixed"
                feature_norm = torch.norm(features, p=2, dim=-1).item()
            
            # 检查并修复零范数特征
            if feature_norm < 1e-8:
                logger.warning(f"Image features have near-zero norm ({feature_norm:.2e}), using random fallback")
                features = torch.randn_like(features) * 0.01
                status = "zero_norm_fixed"
            
            return features, status
            
        except Exception as e:
            logger.error(f"Critical error in extract_image_features: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, "failed"

    def extract_text_features(self, text: str) -> Tuple[Optional[torch.Tensor], str]:
        """
        提取文本特征（设备安全版本）
        """
        try:
            # 处理疾病描述
            processed_text = text
            if "pneumonia" in text.lower() and "pneumonia" in self.disease_descriptions:
                processed_text = self.disease_descriptions["pneumonia"]
            elif "pneumothorax" in text.lower() and "pneumothorax" in self.disease_descriptions:
                processed_text = self.disease_descriptions["pneumothorax"]
            
            # 准备文本输入
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if inputs is None:
                return None, "failed"
            
            # 确保所有输入张量都在主设备上
            device_inputs = {}
            for key, value in inputs.items():
                if value is not None and hasattr(value, 'to'):
                    device_inputs[key] = self._ensure_tensor_device(value, self.main_device)
                else:
                    device_inputs[key] = value
            
            # 选择正确的模型进行推理
            if self.use_multi_gpu:
                model_to_use = self.model_device  # 文本特征提取使用原始模型
            else:
                model_to_use = self.model_device
            
            # 提取特征
            with torch.no_grad():
                try:
                    # 使用标准的forward方法
                    outputs = model_to_use(**device_inputs)
                    
                    # 从outputs中提取特征
                    if hasattr(outputs, 'last_hidden_state'):
                        features = outputs.last_hidden_state.mean(dim=1)
                    elif hasattr(outputs, 'pooler_output'):
                        features = outputs.pooler_output
                    elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        features = outputs.hidden_states[-1].mean(dim=1)
                    elif isinstance(outputs, tuple) and len(outputs) > 0:
                        features = outputs[0]
                        if features.dim() == 3:
                            features = features.mean(dim=1)
                    else:
                        logger.error(f"Cannot extract text features from outputs: {type(outputs)}")
                        return None, "failed"
                
                except Exception as e:
                    logger.error(f"Text feature extraction failed: {e}")
                    return None, "failed"
            
            # 确保特征在主设备上
            features = self._ensure_tensor_device(features, self.main_device)
            
            # 处理特征维度
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            # 特征质量检查
            status = "success"
            feature_norm = torch.norm(features, p=2, dim=-1).item()
            
            # 检查并修复NaN特征
            if torch.isnan(features).any():
                nan_count = torch.isnan(features).sum().item()
                logger.warning(f"Text features contain {nan_count} NaN values, fixing...")
                features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
                status = "nan_fixed"
                feature_norm = torch.norm(features, p=2, dim=-1).item()
            
            # 检查并修复零范数特征
            if feature_norm < 1e-8:
                logger.warning(f"Text features have near-zero norm ({feature_norm:.2e}), using random fallback")
                features = torch.randn_like(features) * 0.01
                status = "zero_norm_fixed"
            
            return features, status
            
        except Exception as e:
            logger.error(f"Critical error in extract_text_features: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, "failed"

    def _precompute_category_embeddings_llava_style(self):
        """预计算类别文本特征（LLaVA风格）"""
        # 创建类别描述（对齐LLaVA格式）
        categories = [f"This is a chest X-ray showing {category}" for category in self.dataset_classes]
        
        # 对类别进行编码（LLaVA风格批量处理）
        encoded_categories = [self.tokenizer(category, return_tensors="pt") for category in categories]
        category_ids = pad_sequence([item.input_ids.squeeze(0) for item in encoded_categories], batch_first=True).to(self.device)
        category_attention_mask = pad_sequence([item.attention_mask.squeeze(0) for item in encoded_categories], batch_first=True).to(self.device)
        
        # 预计算类别特征向量（LLaVA风格）
        global_category_embeddings_cache = []
        
        logger.info("Computing category embeddings using LLaVA style...")
        with torch.no_grad():
            for i in range(category_ids.size(0)):
                category_input_ids = category_ids[i].unsqueeze(0).to(self.device)
                category_attention = category_attention_mask[i].unsqueeze(0).to(self.device)

                try:
                    # 尝试使用LLaVA风格的特征提取
                    if self.use_multi_gpu:
                        category_output = self.model_device(
                            input_ids=category_input_ids, 
                            attention_mask=category_attention,
                            output_hidden_states=True,
                            return_dict=True
                        )
                    else:
                        category_output = self.model(
                            input_ids=category_input_ids, 
                            attention_mask=category_attention,
                            output_hidden_states=True,
                            return_dict=True
                        )

                    # 取最后指定层的隐藏状态，并取末尾 Txtcls_count 个 token
                    if hasattr(category_output, 'hidden_states') and category_output.hidden_states:
                        global_category_embedding = category_output.hidden_states[self.eval_args.feature_layer][:, -self.Txtcls_count:]
                    elif hasattr(category_output, 'last_hidden_state'):
                        global_category_embedding = category_output.last_hidden_state[:, -self.Txtcls_count:]
                    else:
                        # 备用方案：使用平均池化
                        global_category_embedding = category_output.last_hidden_state.mean(dim=1, keepdim=True)
                    
                    # 应用文本MLP（如果存在）
                    if hasattr(self.model_device, 'txt_mlp'):
                        global_category_embedding = self.model_device.txt_mlp(global_category_embedding)
                    
                    global_category_embedding = global_category_embedding.mean(dim=1)
                    global_category_embeddings_cache.append(global_category_embedding)
                    
                except Exception as e:
                    logger.warning(f"Failed to compute embedding for category {i} ({self.dataset_classes[i]}), using fallback: {e}")
                    # 使用随机特征作为后备
                    feature_dim = getattr(self.config, 'hidden_size', 768)
                    fallback_embedding = torch.randn(1, feature_dim, device=self.device) * 0.01
                    global_category_embeddings_cache.append(fallback_embedding)
        
        if global_category_embeddings_cache:
            global_category_embeddings_cache = torch.cat(global_category_embeddings_cache, dim=0).to(self.device)
            logger.info(f'Global Category embeddings computed: {global_category_embeddings_cache.shape}')
            return torch.nn.functional.normalize(global_category_embeddings_cache, dim=-1)
        else:
            # 如果所有都失败，创建随机特征
            feature_dim = getattr(self.config, 'hidden_size', 768)
            num_classes = len(self.dataset_classes)
            logger.warning("All category embedding computations failed, using random features")
            return torch.randn(num_classes, feature_dim, device=self.device) * 0.01

    def compute_similarity(self, image_features, text_features):
        """计算图像和文本特征之间的相似度"""
        # 确保特征在同一设备上
        image_features = self._ensure_tensor_device(image_features, self.main_device)
        text_features = self._ensure_tensor_device(text_features, self.main_device)
        
        # 归一化特征
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        
        # 计算余弦相似度
        similarities = torch.matmul(image_features, text_features.T)
        
        # 应用温度缩放
        similarities = similarities / self.temperature
        
        # 应用sigmoid获得概率
        probabilities = torch.sigmoid(similarities)
        
        return probabilities

    def _ensure_same_device(self, tensor1, tensor2):
        """确保两个张量在同一设备上"""
        tensor1 = self._ensure_tensor_device(tensor1, self.main_device)
        tensor2 = self._ensure_tensor_device(tensor2, self.main_device)
        return tensor1, tensor2

    def evaluate(self, questions):
        """执行评估主流程"""
        all_labels = []
        all_probs = []
        all_predictions = []
        
        batch_size = self.eval_args.batch_size
        num_batches = math.ceil(len(questions) / batch_size)
        
        successful_count = 0
        failed_count = 0
        
        for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {self.args.dataset}"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(questions))
            batch = questions[start_idx:end_idx]
            
            batch_images = []
            batch_labels = []
            
            # 收集批次数据
            for line in batch:
                # 构建图像路径
                if 'image' in line:
                    img_path = os.path.join(self.args.image_folder, line['image'])
                elif 'image_path' in line:
                    img_path = os.path.join(self.args.image_folder, line['image_path'])
                else:
                    logger.warning("No image path found in question, skipping")
                    continue
                
                # 检查图像文件是否存在
                if not os.path.exists(img_path):
                    logger.warning(f"Image file {img_path} not found, skipping")
                    continue
                
                # 加载图像
                try:
                    image = load_image_file(img_path, dataset_type=self.args.dataset)
                    batch_images.append(image)
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}, skipping")
                    continue
                
                # 处理标签
                label_vec = np.zeros(len(self.dataset_classes))
                label_dict = line.get('label', {})
                
                if isinstance(label_dict, dict):
                    for cls, value in label_dict.items():
                        if cls in self.dataset_classes and value == 1:
                            label_vec[self.dataset_classes.index(cls)] = 1
                elif isinstance(label_dict, list):
                    for cls in label_dict:
                        if cls in self.dataset_classes:
                            label_vec[self.dataset_classes.index(cls)] = 1
                
                batch_labels.append(label_vec)
            
            # 跳过空批次
            if not batch_images:
                continue
                    
            # 提取图像特征
            image_features = []
            successful_labels = []
            
            for i, img in enumerate(batch_images):
                img_feat, status = self.extract_image_features(img)
                if img_feat is None:
                    logger.warning(f"Skipping image {i} due to feature extraction failure")
                    failed_count += 1
                    continue
                
                image_features.append(img_feat)
                if i < len(batch_labels):
                    successful_labels.append(batch_labels[i])
                successful_count += 1
            
            if not image_features:
                logger.warning("No valid image features in this batch, skipping")
                continue
                    
            image_features = torch.cat(image_features, dim=0)
            
            # 确保特征设备一致性
            image_features, self.category_embeddings = self._ensure_same_device(
                image_features, self.category_embeddings
            )
            
            # 计算相似度/概率
            similarities = self.compute_similarity(image_features, self.category_embeddings)
            probs = similarities.cpu().numpy()
            
            # 生成二值预测（使用默认阈值，最优阈值在指标计算时确定）
            preds = (probs >= self.eval_args.threshold).astype(int)
            
            # 保存结果
            all_labels.extend(successful_labels)
            all_probs.extend(probs)
            all_predictions.extend(preds)
        
        # 转换为numpy数组
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_predictions = np.array(all_predictions)
        
        logger.info(f"Processing complete: {successful_count} successful, {failed_count} failed")
        logger.info(f"Successfully processed {len(all_labels)} samples")
        return all_labels, all_predictions, all_probs 
    
def calculate_metrics_with_optimal_threshold(labels, predictions, probs, class_names, args, use_optimal_threshold=True):
    """计算医疗图像分类任务的评估指标（支持最优F1阈值）"""
    result_metrics = {}
    
    # 计算每个类别的指标
    accuracies, auc_scores, auprc_scores = [], [], []
    f1_scores, precision_scores, recall_scores = [], [], []
    sensitivities, specificities = [], []
    optimal_thresholds = []  # 存储每个类别的最优阈值
    
    for i, class_name in enumerate(class_names):
        # 跳过没有正样本的类别
        if labels[:, i].sum() == 0:
            logger.info(f"Skipping {class_name} - no positive samples")
            continue
        
        if use_optimal_threshold:
            # 使用最优F1阈值（LLaVA风格）
            # 计算精确度、召回率和阈值
            precision_curve, recall_curve, thresholds = precision_recall_curve(labels[:, i], probs[:, i])

            # 计算 F1 分数并找到最大值
            f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-8)  # 避免分母为0
            max_f1_idx = np.argmax(f1_curve)  # 最大 F1 对应的索引
            
            # 选择最大 F1 对应的阈值
            if max_f1_idx < len(thresholds):
                best_threshold = thresholds[max_f1_idx]
            else:
                best_threshold = 0.5  # 默认阈值
            
            optimal_thresholds.append(best_threshold)
            
            # 使用最优阈值进行二值化预测
            optimal_predictions = (probs[:, i] >= best_threshold).astype(int)
            
            # 计算基于最优阈值的指标
            acc = accuracy_score(labels[:, i], optimal_predictions)
            precision = precision_score(labels[:, i], optimal_predictions, zero_division=0)
            recall = recall_score(labels[:, i], optimal_predictions, zero_division=0)
            f1 = f1_score(labels[:, i], optimal_predictions, zero_division=0)
            
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(labels[:, i], optimal_predictions).ravel()
            
            logger.info(f"\n{class_name} (Optimal Threshold: {best_threshold:.4f}):")
        else:
            # 使用固定阈值
            acc = accuracy_score(labels[:, i], predictions[:, i])
            precision = precision_score(labels[:, i], predictions[:, i], zero_division=0)
            recall = recall_score(labels[:, i], predictions[:, i], zero_division=0)
            f1 = f1_score(labels[:, i], predictions[:, i], zero_division=0)
            
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(labels[:, i], predictions[:, i]).ravel()
            optimal_thresholds.append(0.5)  # 使用默认阈值
            
            logger.info(f"\n{class_name} (Fixed Threshold: 0.5):")
        
        # 医学指标
        sensitivity = recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC-ROC
        try:
            if len(np.unique(labels[:, i])) > 1:
                auc_score = roc_auc_score(labels[:, i], probs[:, i])
            else:
                auc_score = np.nan
        except ValueError:
            auc_score = np.nan
        
        # AUC-PR
        precision_curve, recall_curve, _ = precision_recall_curve(labels[:, i], probs[:, i])
        auprc_score = auc(recall_curve, precision_curve)
        
        # 保存指标
        accuracies.append(acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        auc_scores.append(auc_score)
        auprc_scores.append(auprc_score)
        
        # 打印每个类别的结果
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall/Sensitivity: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Specificity: {specificity:.4f}")
        logger.info(f"  AUC-ROC: {auc_score:.4f}" if not np.isnan(auc_score) else "  AUC-ROC: N/A")
        logger.info(f"  AUC-PR: {auprc_score:.4f}")
        logger.info(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    # 计算宏平均指标
    if len(f1_scores) > 0:
        result_metrics = {
            "macro_accuracy": np.mean(accuracies),
            "macro_precision": np.mean(precision_scores),
            "macro_recall": np.mean(recall_scores),
            "macro_f1": np.mean(f1_scores),
            "macro_sensitivity": np.mean(sensitivities),
            "macro_specificity": np.mean(specificities),
            "mean_auc": np.nanmean(auc_scores),
            "mean_auprc": np.mean(auprc_scores),
            
            # 添加最优阈值信息
            "optimal_thresholds": optimal_thresholds,
            "use_optimal_threshold": use_optimal_threshold,
            
            "per_class_metrics": {
                class_names[i]: {
                    "accuracy": accuracies[i],
                    "precision": precision_scores[i],
                    "recall": recall_scores[i],
                    "f1": f1_scores[i],
                    "sensitivity": sensitivities[i],
                    "specificity": specificities[i],
                    "auc_roc": auc_scores[i],
                    "auc_pr": auprc_scores[i],
                    "optimal_threshold": optimal_thresholds[i]
                } for i in range(len(f1_scores))
            }
        }
        
        # 打印总体结果
        threshold_info = "with Optimal F1 Thresholds" if use_optimal_threshold else "with Fixed Threshold"
        logger.info(f"\n===== Overall Metrics ({threshold_info}) =====")
        logger.info(f"Macro Accuracy: {result_metrics['macro_accuracy']:.4f}")
        logger.info(f"Macro Precision: {result_metrics['macro_precision']:.4f}")
        logger.info(f"Macro Recall: {result_metrics['macro_recall']:.4f}")
        logger.info(f"Macro F1 Score: {result_metrics['macro_f1']:.4f}")
        logger.info(f"Mean Sensitivity: {result_metrics['macro_sensitivity']:.4f}")
        logger.info(f"Mean Specificity: {result_metrics['macro_specificity']:.4f}")
        logger.info(f"Mean AUC-ROC: {result_metrics['mean_auc']:.4f}" if not np.isnan(result_metrics['mean_auc']) else "Mean AUC-ROC: N/A")
        logger.info(f"Mean AUC-PR: {result_metrics['mean_auprc']:.4f}")
        
        if use_optimal_threshold:
            logger.info(f"Optimal Thresholds: {[f'{th:.3f}' for th in optimal_thresholds]}")
        
    else:
        logger.info("No valid classes to compute metrics")
        result_metrics = {}
    
    return result_metrics


def save_results(metrics, class_names, args):
    """保存评估结果到文件"""
    if not metrics:
        logger.info("No metrics to save")
        return
    
    # 确保结果目录存在
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    
    # 保存为JSON文件
    with open(args.result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": args.dataset,
            "model": args.model_path,
            "num_samples": len(next(iter(metrics['per_class_metrics'].values()))) if metrics.get('per_class_metrics') else 0,
            "metrics": metrics
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {args.result_file}")
    
    # 同时保存为CSV文件
    csv_path = os.path.splitext(args.result_file)[0] + ".csv"
    metrics_list = []
    
    if 'per_class_metrics' in metrics:
        for cls, cls_metrics in metrics['per_class_metrics'].items():
            metrics_list.append({
                "class": cls,
                "accuracy": cls_metrics['accuracy'],
                "precision": cls_metrics['precision'],
                "recall": cls_metrics['recall'],
                "f1": cls_metrics['f1'],
                "sensitivity": cls_metrics['sensitivity'],
                "specificity": cls_metrics['specificity'],
                "auc_roc": cls_metrics['auc_roc'],
                "auc_pr": cls_metrics['auc_pr'],
                "optimal_threshold": cls_metrics['optimal_threshold']
            })
        
        # 添加宏平均指标行
        metrics_list.append({
            "class": "macro_average",
            "accuracy": metrics['macro_accuracy'],
            "precision": metrics['macro_precision'],
            "recall": metrics['macro_recall'],
            "f1": metrics['macro_f1'],
            "sensitivity": metrics['macro_sensitivity'],
            "specificity": metrics['macro_specificity'],
            "auc_roc": metrics['mean_auc'],
            "auc_pr": metrics['mean_auprc'],
            "optimal_threshold": "N/A"
        })
        
        pd.DataFrame(metrics_list).to_csv(csv_path, index=False)
        logger.info(f"Class-level metrics saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen Radz Medical Evaluator (Optimized with LLaVA Style)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model_base", type=str, default=None, help="Path to the base model")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the image folder")
    parser.add_argument("--question_file", type=str, required=True, help="Path to the question JSONL file")
    parser.add_argument("--result_file", type=str, default="./results/qwen_radz_med_eval.json", help="Path to save results")
    parser.add_argument("--dataset", type=str, default="mimic", 
                        choices=["chestxray", "chexpert", "mimic", "rsna", "COVIDx_CXR", "SIIM_Pneumothorax"],
                        help="Dataset name")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks for parallel processing")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Current chunk index (0-based)")
    
    # 解析评估参数
    args, remaining_args = parser.parse_known_args()
    eval_parser = argparse.ArgumentParser()
    for field in EvaluationArguments.__dataclass_fields__:
        eval_parser.add_argument(f"--{field}", type=type(EvaluationArguments.__dataclass_fields__[field].default), 
                                default=EvaluationArguments.__dataclass_fields__[field].default)
    eval_args = eval_parser.parse_args(remaining_args)
    
    # 加载问题数据
    logger.info(f"Loading questions from {args.question_file}")
    with open(args.question_file, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
    
    # 分块处理
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    logger.info(f"Processing {len(questions)} questions (chunk {args.chunk_idx+1}/{args.num_chunks})")
    
    # 初始化评估器
    try:
        evaluator = QwenRadzMedEvaluator(args.model_path, args, eval_args)
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return
    
    # 执行评估
    labels, predictions, probs = evaluator.evaluate(questions)
    
    # 计算并保存指标（使用最优F1阈值）
    if len(labels) > 0:
        metrics = calculate_metrics_with_optimal_threshold(
            labels, predictions, probs, evaluator.dataset_classes, args, 
            use_optimal_threshold=eval_args.use_optimal_threshold
        )
        if metrics:
            save_results(metrics, evaluator.dataset_classes, args)
    else:
        logger.info("No valid data to evaluate")


if __name__ == "__main__":
    main()