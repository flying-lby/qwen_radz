"""
改进的CLIP风格Qwen2.5-VL训练脚本
实现基于LLM response的多模态特征提取和对比学习
"""

import os
import sys
import json
import math
import argparse
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    set_seed,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from torch.cuda import amp
from transformers.trainer_utils import get_last_checkpoint
from PIL import Image
import numpy as np

# 导入自定义模块
try:
    from .clip_modeling_improved import (
        ImprovedClipQwen2VLConfig as ClipQwen2VLConfig,
        ImprovedClipQwen2VLForConditionalGeneration as ClipQwen2VLForConditionalGeneration
    )
except ImportError:
    from clip_modeling_improved import (
        ImprovedClipQwen2VLConfig as ClipQwen2VLConfig,
        ImprovedClipQwen2VLForConditionalGeneration as ClipQwen2VLForConditionalGeneration
    )

try:
    from ..constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, LLAVA_IMAGE_TOKEN
except ImportError:
    # 兼容单文件运行
    try:
        from src.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, LLAVA_IMAGE_TOKEN
    except Exception:
        IGNORE_INDEX = -100
        DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
        LLAVA_IMAGE_TOKEN = "<image>"

# 数据预处理工具
try:
    from ..dataset.data_utils import get_image_info
except Exception:
    from src.dataset.data_utils import get_image_info

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rank_0_print(*args, **kwargs):
    """仅在rank 0进程上打印"""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def save_non_lora_module_weights(model: nn.Module, output_dir: str) -> str:
    """保存非LoRA小模块的权重（用于合并时覆盖）。返回保存路径。

    在 DeepSpeed/ZeRO-3 下，直接 state_dict 可能包含尺寸为 0 的占位张量。
    本函数尽可能在 rank0 聚合完整参数后再保存，避免 0 尺寸权重。
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import deepspeed  # type: ignore
        from deepspeed import zero as ds_zero  # type: ignore
        ds_available = True
    except Exception:
        ds_available = False

    model_sd = model.state_dict()
    target_keys = [
        k for k in model_sd.keys()
        if any(pat in k for pat in ['img_mlp', 'txt_mlp', 'knowledge_mlp', 'cross_attention', 'image_projector'])
    ]

    name_to_param: Dict[str, torch.nn.Parameter] = dict(model.named_parameters())
    name_to_buf: Dict[str, torch.Tensor] = dict(model.named_buffers())

    gathered: Dict[str, torch.Tensor] = {}
    for k in target_keys:
        if k in name_to_param:
            p = name_to_param[k]
            if ds_available:
                try:
                    with ds_zero.GatheredParameters([p], modifier_rank=0):
                        t = p.data.detach().cpu().clone()
                except Exception:
                    t = p.data.detach().cpu().clone()
            else:
                t = p.data.detach().cpu().clone()
            gathered[k] = t
        elif k in name_to_buf:
            b = name_to_buf[k]
            gathered[k] = b.detach().cpu().clone()
        else:
            v = model_sd[k]
            try:
                gathered[k] = v.detach().cpu().clone() if hasattr(v, 'detach') else torch.as_tensor(v).cpu().clone()
            except Exception:
                gathered[k] = v

    save_path = os.path.join(output_dir, 'non_lora_state_dict.bin')
    try:
        import torch.distributed as dist
        is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
    except Exception:
        is_rank0 = True

    if is_rank0:
        try:
            torch.save(gathered, save_path)
            rank_0_print(f"已保存非LoRA模块权重: {save_path} (tensors={len(gathered)})")
        except Exception as e:
            rank_0_print(f"保存非LoRA模块权重失败: {e}")
    else:
        rank_0_print("非rank0进程，跳过写入 non_lora_state_dict.bin")

    return save_path


@dataclass
class ImprovedClipModelArguments:
    """改进的CLIP模型参数"""
    model_name_or_path: str = field(
        metadata={"help": "预训练模型路径或Hugging Face模型标识符"}
    )
    model_max_length: int = field(
        default=8192,
        metadata={"help": "最大序列长度"}
    )
    # 特殊标记配置
    img_cls_token_count: int = field(
        default=4, 
        metadata={"help": "图像模态特殊标记数量"}
    )
    txt_cls_token_count: int = field(
        default=4, 
        metadata={"help": "文本模态特殊标记数量"}
    )
    # MLP配置
    hidden_dim: int = field(
        default=1024, 
        metadata={"help": "MLP隐藏层维度"}
    )
    output_dim: int = field(
        default=512, 
        metadata={"help": "输出特征维度"}
    )
    img_mlp_type: int = field(
        default=1, 
        metadata={"help": "图像MLP类型: 0=无, 1=GELU, 2=ReLU, 3=3层GELU, 4=线性, 5=LayerNorm+线性, 6=LayerNorm+Dropout+线性"}
    )
    txt_mlp_type: int = field(
        default=1, 
        metadata={"help": "文本MLP类型"}
    )
    # 损失函数配置
    temperature: float = field(
        default=0.05, 
        metadata={"help": "InfoNCE损失的温度参数"}
    )
    use_local_loss: bool = field(
        default=True, 
        metadata={"help": "是否使用局部特征损失"}
    )
    use_cross_attention_loss: bool = field(
        default=True, 
        metadata={"help": "是否使用交叉注意力损失"}
    )
    # 特征提取配置
    feature_extraction_layer: int = field(
        default=-1, 
        metadata={"help": "从哪一层提取特征，-1表示最后一层"}
    )
    pooling_strategy: str = field(
        default="mean", 
        metadata={"help": "池化策略: mean, max, cls"}
    )
    # 量化配置
    use_bnb: bool = field(
        default=False,
        metadata={"help": "是否使用BitsAndBytes量化"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "4bit量化计算数据类型"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "4bit量化类型"}
    )
    # LoRA配置
    use_lora: bool = field(
        default=True,
        metadata={"help": "是否使用LoRA微调"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA的rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA的alpha参数"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA的dropout"}
    )
    lora_target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA目标模块，用逗号分隔"}
    )
    # 是否启用疾病描述对齐分支
    use_disease_desc: bool = field(
        default=False,
        metadata={"help": "是否启用疾病描述第三支路进行对齐（I<->D, T<->D）"}
    )


@dataclass
class ImprovedClipDataArguments:
    """改进的CLIP数据参数"""
    data_path: str = field(
        metadata={"help": "训练数据路径"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "评估数据路径"}
    )
    image_folder: str = field(
        default="",
        metadata={"help": "图像文件夹路径"}
    )
    is_multimodal: bool = field(
        default=True,
        metadata={"help": "是否为多模态数据"}
    )
    clip_training_ratio: float = field(
        default=0.8,
        metadata={"help": "CLIP训练比例，0.0=无CLIP，1.0=仅CLIP"}
    )
    # 图像处理参数
    image_min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "图像最小像素数"}
    )
    image_max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "图像最大像素数"}
    )
    image_resized_width: Optional[int] = field(
        default=None,
        metadata={"help": "调整后的图像宽度"}
    )
    image_resized_height: Optional[int] = field(
        default=None,
        metadata={"help": "调整后的图像高度"}
    )
    # 数据增强
    use_data_augmentation: bool = field(
        default=True,
        metadata={"help": "是否使用数据增强"}
    )
    augmentation_prob: float = field(
        default=0.5,
        metadata={"help": "数据增强概率"}
    )
    # 疾病描述JSON路径（键值对或列表对象，作为图像/文本对齐的第三模态）
    disease_desc_path: Optional[str] = field(
        default=None,
        metadata={"help": "疾病描述JSON路径，键值对(疾病->描述)或列表[{name, desc}]"}
    )


@dataclass
class ImprovedClipTrainingArguments(TrainingArguments):
    """扩展的训练参数"""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_vision_tower: bool = field(default=False)
    freeze_language_model: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    # 添加新的训练参数
    warmup_ratio: float = field(default=0.03)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=50)
    save_total_limit: int = field(default=2)
    logging_steps: int = field(default=1)
    report_to: str = field(default="wandb")
    deepspeed: Optional[str] = field(default=None, metadata={"help": "DeepSpeed配置文件路径"})
    # CLIP子模块学习率
    clip_learning_rate: float = field(default=1e-4, metadata={"help": "CLIP头部模块的学习率（img_mlp/txt_mlp/cross_attention/image_projector）"})


class ImprovedClipDataset(Dataset):
    """改进的CLIP数据集，支持特殊标记注入"""
    
    def __init__(
        self,
        data_path: str,
        processor: transformers.ProcessorMixin,
        tokenizer: PreTrainedTokenizer,
        data_args: ImprovedClipDataArguments,
        model_args: ImprovedClipModelArguments,
        model_id: str,
    ):
        super().__init__()
        
        # 加载数据
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        
        self.processor = processor
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_args = model_args
        self.model_id = model_id
        # 加载疾病描述映射（可选）
        self.disease_desc_map = None
        if getattr(self.data_args, 'disease_desc_path', None):
            try:
                with open(self.data_args.disease_desc_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                mapping = {}
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        if isinstance(k, str) and isinstance(v, str):
                            mapping[k.strip().lower()] = v.strip()
                elif isinstance(raw, list):
                    for obj in raw:
                        name = (obj.get('name') or obj.get('disease') or '').strip().lower()
                        desc = (obj.get('desc') or obj.get('description') or '').strip()
                        if name and desc:
                            mapping[name] = desc
                self.disease_desc_map = mapping if mapping else None
            except Exception as e:
                rank_0_print(f"加载疾病描述JSON失败，将忽略: {e}")
        
        # 读取特殊标记ID（依赖于setup中已注入到tokenizer）
        self._load_special_token_ids()
        
    def _load_special_token_ids(self):
        """读取已注册到tokenizer的模态特殊标记ID，不再在此处新增token，避免与模型嵌入尺寸不一致。"""
        img_special_tokens = [f"<IMG_CLS_{i}>" for i in range(self.model_args.img_cls_token_count)]
        txt_special_tokens = [f"<TXT_CLS_{i}>" for i in range(self.model_args.txt_cls_token_count)]
        self.img_cls_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in img_special_tokens]
        self.txt_cls_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in txt_special_tokens]
        rank_0_print(
            f"使用特殊标记: 图像 {img_special_tokens} -> {self.img_cls_token_ids}, 文本 {txt_special_tokens} -> {self.txt_cls_token_ids}"
        )
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data_list[idx]
        
        # 处理图像（与SFT保持一致，控制像素数避免OOM）
        image_path = item.get("image", None)
        images = None
        if image_path:
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.data_args.image_folder, image_path)
            try:
                images = [
                    get_image_info(
                        image_path,
                        self.data_args.image_min_pixels,
                        self.data_args.image_max_pixels,
                        self.data_args.image_resized_width,
                        self.data_args.image_resized_height,
                    )
                ]
            except Exception as e:
                print(f"警告：无法预处理图像 {image_path}: {e}")
                images = None
        
        # 处理对话
        conversations = item["conversations"]
        # 可选的疾病描述文本（支持 key: 'disease_desc' 或 'diseases' -> 拼接）
        disease_desc_text = item.get("disease_desc")
        if disease_desc_text is None:
            diseases = item.get("diseases")
            if isinstance(diseases, list) and self.disease_desc_map:
                collected = []
                for d in diseases:
                    key = str(d).strip().lower()
                    if key in self.disease_desc_map:
                        collected.append(self.disease_desc_map[key])
                if collected:
                    # 用分号拼接多疾病描述
                    disease_desc_text = "; ".join(collected)
        
        # 构建带特殊标记的输入
        processed_data = self._process_conversations_with_special_tokens(
            conversations, images, disease_desc_text
        )
        
        return processed_data
    
    def _process_conversations_with_special_tokens(
        self, 
        conversations: List[Dict], 
        images: Optional[List],
        disease_desc_text: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """处理对话并添加特殊标记"""
        
        # 提取问题和答案
        human_input = None
        gpt_response = None
        
        for conv in conversations:
            if conv["from"] == "human":
                human_input = conv["value"]
            elif conv["from"] == "gpt":
                gpt_response = conv["value"]
        
        if human_input is None or gpt_response is None:
            raise ValueError(f"对话格式错误: {conversations}")
        
        # 为图像输入添加特殊标记
        img_special_tokens = " ".join([f"<IMG_CLS_{i}>" for i in range(self.model_args.img_cls_token_count)])
        
        # 处理图像输入（问题+图像） - 参考SFT数据集的处理方式
        has_images = images is not None and len(images) > 0
        contains_llava_token = (LLAVA_IMAGE_TOKEN in human_input) if isinstance(human_input, str) else False
        contains_qwen_token = (DEFAULT_IMAGE_TOKEN in human_input) if isinstance(human_input, str) else False

        if has_images:
            # 规范化：将 <image> 映射为 <|image_pad|>，并强制仅保留一个，占据首行
            # 为避免超长与mask错配，这里不再包含任何额外文本，仅保留图像占位符
            normalized_human = DEFAULT_IMAGE_TOKEN
            # 注意：不在传给processor的text里拼接模态特征标记，避免影响image mask对齐
            img_text = normalized_human
            
            # 使用processor处理图像和文本（与SFT一致）
            img_inputs = self.processor(
                text=[img_text],
                images=images,
                padding=False,
                do_resize=False,
                return_tensors="pt"
            )
            
            # 提取特征
            prompt_input_ids = img_inputs["input_ids"]  # shape (1, L)
            img_pixel_values = img_inputs["pixel_values"].squeeze(0)
            img_image_grid_thw = img_inputs["image_grid_thw"]  # 保持shape: (num_images, 3)
            # 将模态特殊标记追加到序列末尾（避免影响image mask对齐）
            img_tail = " ".join([f"<IMG_CLS_{i}>" for i in range(self.model_args.img_cls_token_count)])
            tail_ids = self.tokenizer(img_tail, add_special_tokens=False, return_tensors="pt")["input_ids"]
            img_input_ids = torch.cat([prompt_input_ids, tail_ids], dim=1).squeeze(0)
            
        else:
            # 纯文本输入（没有图像或图像加载失败）：移除任意图像占位符，避免“无图像token却有特征”
            normalized_human = (
                human_input.replace(LLAVA_IMAGE_TOKEN, "").replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            # 注意：不在传给processor的text里拼接模态特征标记，避免影响mask
            img_text = normalized_human
            # 使用tokenizer编码（与上游一致），并在末尾追加模态特殊标记
            prompt_input_ids = self.tokenizer(
                img_text,
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"]
            img_tail = " ".join([f"<IMG_CLS_{i}>" for i in range(self.model_args.img_cls_token_count)])
            tail_ids = self.tokenizer(img_tail, add_special_tokens=False, return_tensors="pt")["input_ids"]
            img_input_ids = torch.cat([prompt_input_ids, tail_ids], dim=1).squeeze(0)
            # 创建空的图像数据
            img_pixel_values = None
            img_image_grid_thw = None
        
        # 为文本响应添加特殊标记（文本支路采用 human 提示 + gpt 响应）
        txt_special_tokens = " ".join([f"<TXT_CLS_{i}>" for i in range(self.model_args.txt_cls_token_count)])
        # 清理 human 文本中的图像占位符
        txt_human = human_input.replace(LLAVA_IMAGE_TOKEN, "").replace(DEFAULT_IMAGE_TOKEN, "").strip()
        if len(txt_human) > 0:
            txt_text = txt_human + "\n" + gpt_response + " " + txt_special_tokens
        else:
            txt_text = gpt_response + " " + txt_special_tokens
        
        # 处理文本输入
        txt_input_ids = self.tokenizer(
            txt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.model_args.model_max_length
        )["input_ids"].squeeze(0)

        # 可选：疾病描述文本编码
        if self.model_args.use_disease_desc and isinstance(disease_desc_text, str) and len(disease_desc_text.strip()) > 0:
            desc_input_ids = self.tokenizer(
                disease_desc_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.model_args.model_max_length
            )["input_ids"].squeeze(0)
        else:
            desc_input_ids = None
        
        # 创建标签（用于语言建模损失，可选）
        labels = img_input_ids.clone()
        # 只计算响应部分的损失（粗略估计）
        labels[:len(img_input_ids)//2] = IGNORE_INDEX
        
        out = {
            "img_input_ids": img_input_ids,
            "img_pixel_values": img_pixel_values,
            "img_image_grid_thw": img_image_grid_thw,
            "txt_input_ids": txt_input_ids,
            "labels": labels,
            "img_cls_token_ids": torch.tensor(self.img_cls_token_ids),
            "txt_cls_token_ids": torch.tensor(self.txt_cls_token_ids),
        }
        if desc_input_ids is not None:
            out["desc_input_ids"] = desc_input_ids
        return out


class ImprovedClipDataCollator:
    """改进的数据整理器"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, model_max_length: int = 8192, spatial_merge_size: int = 1):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.spatial_merge_size = max(int(spatial_merge_size), 1)
        
    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """整理批次数据"""
        
        # 收集各种数据
        img_input_ids_list = []
        img_pixel_values_list = []
        img_image_grid_thw_list = []
        txt_input_ids_list = []
        desc_input_ids_list = []
        labels_list = []
        
        for instance in instances:
            img_input_ids_list.append(instance["img_input_ids"])
            img_pixel_values_list.append(instance["img_pixel_values"])
            img_image_grid_thw_list.append(instance["img_image_grid_thw"])
            txt_input_ids_list.append(instance["txt_input_ids"])
            desc_input_ids_list.append(instance.get("desc_input_ids"))
            if "labels" in instance:
                labels_list.append(instance["labels"])

        # 长度截断（防止超出模型max_length导致索引错误）。
        # 截断策略：保留序列开头（其中包含图像token展开），截断末尾。
        def truncate_list(tensor_list: List[torch.Tensor]) -> List[torch.Tensor]:
            truncated = []
            for t in tensor_list:
                if t is None:
                    truncated.append(t)
                    continue
                if t.dim() == 0:
                    truncated.append(t)
                    continue
                max_len = int(self.model_max_length)
                if t.size(0) > max_len:
                    truncated.append(t[:max_len])
                else:
                    truncated.append(t)
            return truncated

        img_input_ids_list = truncate_list(img_input_ids_list)
        txt_input_ids_list = truncate_list(txt_input_ids_list)
        # disease desc 可为空，只截断非空项
        desc_input_ids_list = [t if t is None else (t[:int(self.model_max_length)] if t.size(0) > int(self.model_max_length) else t) for t in desc_input_ids_list]
        if labels_list:
            labels_list = truncate_list(labels_list)
        
        # Padding
        img_input_ids = torch.nn.utils.rnn.pad_sequence(
            img_input_ids_list, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        txt_input_ids = torch.nn.utils.rnn.pad_sequence(
            txt_input_ids_list, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        # 可选疾病描述批
        if any(t is not None for t in desc_input_ids_list):
            non_null_desc = [t for t in desc_input_ids_list if t is not None]
            # 用 pad_token_id 对齐疾病描述序列，缺失样本用全 pad 占位
            max_len_desc = max(t.size(0) for t in non_null_desc)
            desc_tensors = []
            for t in desc_input_ids_list:
                if t is None:
                    desc_tensors.append(torch.full((max_len_desc,), self.tokenizer.pad_token_id, dtype=torch.long))
                else:
                    if t.size(0) < max_len_desc:
                        pad = torch.full((max_len_desc - t.size(0),), self.tokenizer.pad_token_id, dtype=torch.long)
                        desc_tensors.append(torch.cat([t, pad], dim=0))
                    else:
                        desc_tensors.append(t)
            desc_input_ids = torch.stack(desc_tensors, dim=0)
            desc_attention_mask = desc_input_ids.ne(self.tokenizer.pad_token_id).long()
        else:
            desc_input_ids = None
            desc_attention_mask = None
        
        # 注意力掩码（使用long的0/1，避免CUDA侧布尔索引行为差异）
        img_attention_mask = img_input_ids.ne(self.tokenizer.pad_token_id).long()
        txt_attention_mask = txt_input_ids.ne(self.tokenizer.pad_token_id).long()

        # 运行时一致性检查：有图像的样本必须包含图像token；无图像的样本不得包含图像token
        try:
            image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        except Exception:
            image_token_id = None

        if image_token_id is not None:
            # 逐样本统计（Qwen会将图像展开为大量占位token，因此只要求：有图像则 count>0，无图像则 count==0）
            per_sample_image_token_counts: List[int] = []
            for row in img_input_ids:
                per_sample_image_token_counts.append(int((row == image_token_id).sum().item()))

            has_image_flags: List[int] = []
            for pv in img_pixel_values_list:
                has_image_flags.append(1 if (pv is not None and pv.numel() > 0) else 0)

            for idx, (cnt, flag) in enumerate(zip(per_sample_image_token_counts, has_image_flags)):
                if flag == 1 and cnt <= 0:
                    raise ValueError(
                        f"样本{idx}图像标记缺失: image_token_count={cnt}, 但存在图像特征. "
                        f"请检查文本是否含有{DEFAULT_IMAGE_TOKEN}且图像是否可读。"
                    )
                if flag == 0 and cnt > 0:
                    raise ValueError(
                        f"样本{idx}存在图像标记但无图像特征: image_token_count={cnt}. "
                        f"请检查是否错误地插入了{DEFAULT_IMAGE_TOKEN}。"
                    )
        
        # 进一步一致性检查：基于每个样本的 grid_thw 预估视觉特征数量(考虑spatial_merge)，并与图像token计数比较
        try:
            image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        except Exception:
            image_token_id = None
        if image_token_id is not None:
            for idx, (ids, grid) in enumerate(zip(img_input_ids_list, img_image_grid_thw_list)):
                if ids is None:
                    continue
                token_cnt = int((ids == image_token_id).sum().item())
                if grid is None:
                    expected = 0
                else:
                    # grid_thw 为 (num_images, 3)，每个图像贡献 t*(h//merge)*(w//merge) 个特征
                    merge = self.spatial_merge_size
                    h_merged = (grid[:, 1] // merge)
                    w_merged = (grid[:, 2] // merge)
                    expected = int((grid[:, 0] * h_merged * w_merged).sum().item())
                if token_cnt != expected:
                    raise ValueError(
                        f"样本{idx}图像token与预估视觉特征不一致: tokens={token_cnt}, expected_features={expected}. "
                        f"请检查文本中 {DEFAULT_IMAGE_TOKEN} 的数量与 image_grid_thw 是否匹配。"
                    )

        # 处理图像数据（对齐为模型期望的batch扁平格式）
        # Qwen2.5-VL期望 pixel_values 为 (sum_i (t*h*w)_i, hidden_dim)
        # image_grid_thw 为 (num_images, 3)
        
        # 过滤掉None和空的pixel_values
        valid_pixel_values = [pv for pv in img_pixel_values_list if pv is not None and pv.numel() > 0]
        if valid_pixel_values:
            img_pixel_values = torch.cat(valid_pixel_values, dim=0)
        else:
            img_pixel_values = None
            
        # 过滤掉None和无效的image_grid_thw
        valid_grid_thw = [grid for grid in img_image_grid_thw_list if grid is not None]
        if valid_grid_thw:
            img_image_grid_thw = torch.cat(valid_grid_thw, dim=0)
        else:
            img_image_grid_thw = None
        
        # 处理标签
        if labels_list:
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, 
                batch_first=True, 
                padding_value=IGNORE_INDEX
            )
        else:
            labels = None
        
        # 获取特殊标记ID（假设批次中所有实例使用相同的特殊标记）
        img_cls_token_ids = instances[0]["img_cls_token_ids"]
        txt_cls_token_ids = instances[0]["txt_cls_token_ids"]
        
        return {
            "input_ids": img_input_ids,  # 用于图像模态
            "attention_mask": img_attention_mask,
            "pixel_values": img_pixel_values,
            "image_grid_thw": img_image_grid_thw,
            "txt_input_ids": txt_input_ids,  # 用于文本模态
            "txt_attention_mask": txt_attention_mask,
            "labels": labels,
            "img_cls_token_ids": img_cls_token_ids,
            "txt_cls_token_ids": txt_cls_token_ids,
            "return_clip_loss": True,  # 指示使用CLIP损失
        }


class ImprovedClipTrainer(Trainer):
    """改进的CLIP训练器"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算损失"""
        
        # 确保模型在训练模式
        model.train()
        
        # 前向传播
        outputs = model(**inputs)
        
        # 获取损失
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # 记录额外的损失信息
        if hasattr(outputs, 'clip_loss_dict'):
            for key, value in outputs.clip_loss_dict.items():
                if hasattr(value, 'item'):
                    self.log({f"train/{key}": value.item()})
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """预测步骤"""
        
        model.eval()
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        return (loss, None, None)


def setup_model_and_tokenizer(model_args: ImprovedClipModelArguments):
    """设置模型和分词器"""
    
    # 量化配置
    bnb_config = None
    if model_args.use_bnb:
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    
    # 创建配置
    config = ClipQwen2VLConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    # 基座一致性关键字段日志
    try:
        vh = getattr(getattr(config, 'vision_config', None), 'hidden_size', None)
        vm = getattr(getattr(config, 'vision_config', None), 'spatial_merge_size', None)
        rank_0_print(
            f"[Config] model_type={getattr(config, 'model_type', None)}, hidden_size={getattr(config, 'hidden_size', None)}, "
            f"vocab_size={getattr(config, 'vocab_size', None)}, vision_hidden_size={vh}, spatial_merge_size={vm}"
        )
    except Exception:
        pass
    
    # 先加载tokenizer与processor，并预先注册本任务的附加特殊token
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    # 设置pad_token，并确保processor与tokenizer使用同一词表
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        processor.tokenizer = tokenizer
    except Exception:
        pass
    # 预先添加模态特殊token，避免后续Embedding越界
    img_special_tokens = [f"<IMG_CLS_{i}>" for i in range(model_args.img_cls_token_count)]
    txt_special_tokens = [f"<TXT_CLS_{i}>" for i in range(model_args.txt_cls_token_count)]
    tokenizer.add_special_tokens({'additional_special_tokens': img_special_tokens + txt_special_tokens})

    # 更新CLIP相关配置
    config.clip_config = {
        "img_cls_token_count": model_args.img_cls_token_count,
        "txt_cls_token_count": model_args.txt_cls_token_count,
        "hidden_dim": model_args.hidden_dim,
        "output_dim": model_args.output_dim,
        "temperature": model_args.temperature,
        "use_local_features": model_args.use_local_loss,
        "use_global_features": True,
        "use_cross_attention": model_args.use_cross_attention_loss,
        "pooling_strategy": model_args.pooling_strategy,
        "feature_extraction_layer": model_args.feature_extraction_layer,
        "mlp_dropout": 0.1,
        "use_layer_norm": True,
    }
    
    
    # 加载模型
    # 默认严格加载，便于发现基座不匹配；如需放宽可设置环境变量 STRICT_LOAD=0
    strict_load = os.environ.get('STRICT_LOAD', '1') != '0'
    model = ClipQwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if model_args.use_bnb else torch.float16,
        trust_remote_code=True,
        ignore_mismatched_sizes=not strict_load
    )
    
    
    
    # 如果使用量化，准备模型进行量化训练
    if model_args.use_bnb:
        model = prepare_model_for_kbit_training(model)
        debug_model_parameters(model, "量化准备后")

    # 调整词表大小以适配新增的特殊token，防止Embedding越界
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        rank_0_print(f"警告：调整token嵌入大小失败，将继续训练。原因: {e}")
    
    # 配置LoRA
    if model_args.use_lora:
        rank_0_print("配置LoRA...")
        
        # 解析目标模块
        target_modules = model_args.lora_target_modules.split(",")
        target_modules = [module.strip() for module in target_modules]
        
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数
        model.print_trainable_parameters()
        
        # 确保CLIP相关层可训练
        for name, param in model.named_parameters():
            if any(clip_module in name for clip_module in ['img_mlp', 'txt_mlp', 'cross_attention', 'clip_loss']):
                param.requires_grad = True
                rank_0_print(f"设置CLIP层可训练: {name}")
    
    return model, tokenizer, processor


def debug_model_parameters(*args, **kwargs):
    # 已弃用：保留空实现以避免残留调用输出日志
    return


def main():
    # 解析参数
    parser = HfArgumentParser((
        ImprovedClipModelArguments,
        ImprovedClipDataArguments,
        ImprovedClipTrainingArguments
    ))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    # 加载模型和tokenizer
    rank_0_print("加载模型和tokenizer...")
    model, tokenizer, processor = setup_model_and_tokenizer(model_args)
    
    # 准备数据集
    rank_0_print("准备训练数据集...")
    train_dataset = ImprovedClipDataset(
        data_path=data_args.data_path,
        processor=processor,
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        model_id=model_args.model_name_or_path
    )
    
    # 准备评估数据集
    eval_dataset = None
    if data_args.eval_data_path:
        rank_0_print("准备评估数据集...")
        eval_dataset = ImprovedClipDataset(
            data_path=data_args.eval_data_path,
            processor=processor,
            tokenizer=tokenizer,
            data_args=data_args,
            model_args=model_args,
            model_id=model_args.model_name_or_path
        )
    
    
    
    # 更新模型的embedding层大小（如果添加了新的特殊标记）
    try:
        original_vocab_size = model.config.vocab_size
        new_vocab_size = len(tokenizer)
        
        if new_vocab_size != original_vocab_size:
            print(f"需要调整词表大小: {original_vocab_size} -> {new_vocab_size}")
            model.resize_token_embeddings(new_vocab_size)
            print("词表大小调整完成")
        else:
            print("词表大小无需调整")
    except Exception as e:
        print(f"词表调整失败: {e}")
        # 如果调整失败，继续训练
    
    # （移除调试输出）
    
    
    
    # 最终验证模型与DeepSpeed的兼容性
    
    
    # 数据整理器
    # 传入视觉spatial_merge_size（从模型config读取，不传则默认1）
    spatial_merge_size = getattr(getattr(model.config, 'vision_config', None), 'spatial_merge_size', 1)
    data_collator = ImprovedClipDataCollator(
        tokenizer=tokenizer,
        model_max_length=model_args.model_max_length,
        spatial_merge_size=spatial_merge_size,
    )
    
    # 构建参数组：CLIP子模块使用独立学习率
    clip_param_names = []
    for n, p in model.named_parameters():
        if any(key in n for key in [
            'img_mlp', 'txt_mlp', 'knowledge_mlp', 'cross_attention', 'image_projector'
        ]):
            clip_param_names.append(n)
    base_params = []
    clip_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in clip_param_names:
            clip_params.append(p)
        else:
            base_params.append(p)

    optimizer_grouped_parameters = [
        {
            'params': base_params,
            'lr': training_args.learning_rate,
        },
        {
            'params': clip_params,
            'lr': training_args.clip_learning_rate,
        },
    ]

    # 自定义优化器
    from torch.optim import AdamW
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

    # 创建训练器
    trainer = ImprovedClipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
    )
    
    # 仅在显式提供时才恢复，不再自动从 output_dir 搜索 last checkpoint
    if training_args.resume_from_checkpoint:
        checkpoint = training_args.resume_from_checkpoint
        rank_0_print(f"从显式提供的checkpoint恢复: {checkpoint}")
    else:
        checkpoint = None
        rank_0_print("未提供 resume_from_checkpoint，将从头开始训练（不自动扫描 output_dir）。")
    
    # 开始训练
    rank_0_print("开始训练...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # 保存模型
    rank_0_print("保存模型...")
    trainer.save_model()
    trainer.save_state()
    # 额外：保存非LoRA小模块权重，用于后续合并覆盖
    try:
        save_non_lora_module_weights(model, training_args.output_dir)
    except Exception as e:
        rank_0_print(f"保存非LoRA小模块权重失败: {e}")
    
    # 保存训练结果
    if training_args.local_rank in [-1, 0]:
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    
    rank_0_print("训练完成！")


if __name__ == "__main__":
    main()