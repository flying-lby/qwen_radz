"""
简化版CLIP风格Qwen2.5-VL训练脚本
基于LLaVA RadZ的简单高效设计理念，修复关键性能问题
"""

import os
import sys
import json
import logging
import random
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    set_seed,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# 导入简化的模型
try:
    from .new_clip_modeling_improved import (
        SimplifiedClipQwen2VLConfig as ClipQwen2VLConfig,
        SimplifiedClipQwen2VLForConditionalGeneration as ClipQwen2VLForConditionalGeneration
    )
except ImportError:
    from new_clip_modeling_improved import (
        SimplifiedClipQwen2VLConfig as ClipQwen2VLConfig,
        SimplifiedClipQwen2VLForConditionalGeneration as ClipQwen2VLForConditionalGeneration
    )

# 常量定义
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rank_0_print(*args, **kwargs):
    """仅在rank 0进程上打印"""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


@dataclass
class SimplifiedClipModelArguments:
    """简化的模型参数"""
    model_name_or_path: Optional[str] = field(default="Qwen2.5-VL-7B-Instruct")
    
    # CLIP特定参数 - 参考LLaVA RadZ的简单设计
    img_cls_token_count: int = field(default=4, metadata={"help": "图像分类token数量"})
    txt_cls_token_count: int = field(default=4, metadata={"help": "文本分类token数量（与图像平衡）"})
    hidden_dim: int = field(default=1024, metadata={"help": "MLP隐藏维度"})
    output_dim: int = field(default=512, metadata={"help": "输出特征维度"})
    temperature: float = field(default=0.05, metadata={"help": "InfoNCE温度参数"})
    
    # 关键修复：启用MLP
    img_mlp_type: int = field(default=1, metadata={"help": "图像MLP类型: 0=无, 1=GELU"})
    txt_mlp_type: int = field(default=1, metadata={"help": "文本MLP类型: 0=无, 1=GELU"})
    
    # 简化配置
    feature_extraction_layer: int = field(default=-2, metadata={"help": "特征提取层（与LLaVA RadZ一致）"})
    pooling_strategy: str = field(default="mean", metadata={"help": "池化策略"})
    
    # LoRA参数
    use_lora: bool = field(default=True)
    lora_r: int = field(default=128)
    lora_alpha: int = field(default=256)
    lora_dropout: float = field(default=0.05)


@dataclass
class SimplifiedClipDataArguments:
    """简化的数据参数"""
    data_path: str = field(default="", metadata={"help": "训练数据路径"})
    image_folder: str = field(default="", metadata={"help": "图像文件夹路径"})
    
    # 简化的CLIP训练控制
    clip_training_ratio: float = field(default=0.3, metadata={"help": "CLIP训练比例（降低以提高稳定性）"})
    
    # 图像处理参数
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=1048576)
    
    # 可选的疾病描述
    disease_desc_path: Optional[str] = field(default=None, metadata={"help": "疾病描述文件路径"})


class SimplifiedClipDataset(Dataset):
    """简化的CLIP数据集，参考LLaVA RadZ的简单设计"""
    
    def __init__(
        self,
        data_path: str,
        processor: transformers.ProcessorMixin,
        tokenizer: PreTrainedTokenizer,
        data_args: SimplifiedClipDataArguments,
        model_args: SimplifiedClipModelArguments,
    ):
        super().__init__()
        
        # 加载数据
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        
        self.processor = processor
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_args = model_args
        
        # 添加特殊标记
        self._add_special_tokens()
        
        # 简化的疾病描述处理
        self.disease_desc_map = None
        if data_args.disease_desc_path and os.path.exists(data_args.disease_desc_path):
            try:
                with open(data_args.disease_desc_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                    self.disease_desc_map = self._process_disease_descriptions(raw)
            except Exception as e:
                rank_0_print(f"加载疾病描述失败: {e}")
    
    def _add_special_tokens(self):
        """添加特殊分类标记"""
        # 图像分类标记
        img_tokens = [f"<IMG_CLS_{i}>" for i in range(self.model_args.img_cls_token_count)]
        txt_tokens = [f"<TXT_CLS_{i}>" for i in range(self.model_args.txt_cls_token_count)]
        
        # 添加到tokenizer
        special_tokens = img_tokens + txt_tokens
        num_new_tokens = self.tokenizer.add_tokens(special_tokens)
        rank_0_print(f"添加了 {num_new_tokens} 个特殊标记")
        
        # 获取token IDs
        self.img_cls_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in img_tokens]
        self.txt_cls_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in txt_tokens]
        
        rank_0_print(f"图像标记IDs: {self.img_cls_token_ids}")
        rank_0_print(f"文本标记IDs: {self.txt_cls_token_ids}")
    
    def _process_disease_descriptions(self, raw_data):
        """处理疾病描述数据"""
        mapping = {}
        if isinstance(raw_data, dict):
            for k, v in raw_data.items():
                if isinstance(k, str) and isinstance(v, str):
                    mapping[k.strip().lower()] = v.strip()
        elif isinstance(raw_data, list):
            for item in raw_data:
                name = (item.get('name') or item.get('disease') or '').strip().lower()
                desc = (item.get('desc') or item.get('description') or '').strip()
                if name and desc:
                    mapping[name] = desc
        return mapping if mapping else None
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data_list[idx]
        
        # 处理图像
        image_path = item.get("image", None)
        images = None
        if image_path:
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.data_args.image_folder, image_path)
            
            try:
                # 优先使用简化的数据处理工具
                try:
                    from src.dataset.new_data_utils import load_and_preprocess_image
                    image = load_and_preprocess_image(
                        image_path,
                        min_pixels=self.data_args.image_min_pixels,
                        max_pixels=self.data_args.image_max_pixels,
                        apply_augmentation=False  # 训练时可启用
                    )
                    images = [image]
                except ImportError:
                    # 回退到原始工具
                    from src.dataset.data_utils import get_image_info
                    images = [get_image_info(
                        image_path,
                        self.data_args.image_min_pixels,
                        self.data_args.image_max_pixels,
                    )]
            except Exception as e:
                rank_0_print(f"图像加载失败 {image_path}: {e}")
                images = None
        
        # 处理对话
        conversations = item["conversations"]
        
        # 决定是否进行CLIP训练（实现真正的混合训练）
        is_clip_training = random.random() < self.data_args.clip_training_ratio
        
        # 处理输入序列
        if is_clip_training:
            # CLIP训练模式：添加特殊标记
            img_conversation = self._prepare_clip_conversation(conversations, is_image=True)
            txt_conversation = self._prepare_clip_conversation(conversations, is_image=False)
            
            # 处理图像输入
            img_inputs = self._process_conversation(img_conversation, images)
            
            # 处理文本输入（不包含图像）
            txt_inputs = self._process_conversation(txt_conversation, images=None)
            
            return {
                'is_clip_training': True,
                'img_input_ids': img_inputs['input_ids'],
                'img_attention_mask': img_inputs['attention_mask'],
                'img_pixel_values': img_inputs.get('pixel_values'),
                'img_image_grid_thw': img_inputs.get('image_grid_thw'),
                'txt_input_ids': txt_inputs['input_ids'],
                'txt_attention_mask': txt_inputs['attention_mask'],
                'img_cls_token_ids': self.img_cls_token_ids,
                'txt_cls_token_ids': self.txt_cls_token_ids,
            }
        else:
            # 常规语言建模训练
            inputs = self._process_conversation(conversations, images)
            inputs['is_clip_training'] = False
            return inputs
    
    def _prepare_clip_conversation(self, conversations, is_image=True):
        """为CLIP训练准备对话，添加相应的特殊标记"""
        processed_conversations = []
        
        for conv in conversations:
            processed_conv = conv.copy()
            
            if conv['from'] == 'gpt':
                # 在回答末尾添加特殊标记
                if is_image:
                    tokens = " ".join([f"<IMG_CLS_{i}>" for i in range(self.model_args.img_cls_token_count)])
                else:
                    tokens = " ".join([f"<TXT_CLS_{i}>" for i in range(self.model_args.txt_cls_token_count)])
                processed_conv['value'] = conv['value'] + " " + tokens
            
            processed_conversations.append(processed_conv)
        
        return processed_conversations
    
    def _process_conversation(self, conversations, images=None):
        """处理单个对话，参考LLaVA的简单实现"""
        # 构建输入文本
        input_text = ""
        for i, conv in enumerate(conversations):
            if conv['from'] == 'human':
                if i == 0 and images is not None:
                    input_text += f"{DEFAULT_IMAGE_TOKEN}\n{conv['value']}\n"
                else:
                    input_text += f"{conv['value']}\n"
            elif conv['from'] == 'gpt':
                input_text += f"{conv['value']}\n"
        
        # 处理图像和文本
        if images is not None:
            inputs = self.processor(
                text=[input_text],
                images=images,
                return_tensors="pt",
                padding=False,
            )
        else:
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=False,
            )
        
        # 转换为单个样本
        result = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.squeeze(0) if value.dim() > 1 else value
            else:
                result[key] = value
        
        return result


class SimplifiedClipDataCollator:
    """简化的数据整理器，参考LLaVA RadZ设计"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, model_max_length: int = 4096):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
    
    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 分离CLIP训练和常规训练的样本
        clip_instances = [inst for inst in instances if inst.get('is_clip_training', False)]
        regular_instances = [inst for inst in instances if not inst.get('is_clip_training', False)]
        
        if clip_instances and regular_instances:
            # 混合批次：随机选择一种模式
            if random.random() < 0.5:
                instances = clip_instances
            else:
                instances = regular_instances
        
        if not instances:
            instances = regular_instances or clip_instances
        
        is_clip_batch = instances[0].get('is_clip_training', False)
        
        if is_clip_batch:
            return self._collate_clip_batch(instances)
        else:
            return self._collate_regular_batch(instances)
    
    def _collate_clip_batch(self, instances):
        """整理CLIP训练批次"""
        batch_size = len(instances)
        
        # 收集图像模态数据
        img_input_ids = [inst['img_input_ids'] for inst in instances]
        img_attention_mask = [inst['img_attention_mask'] for inst in instances]
        
        # 收集文本模态数据  
        txt_input_ids = [inst['txt_input_ids'] for inst in instances]
        txt_attention_mask = [inst['txt_attention_mask'] for inst in instances]
        
        # Padding
        img_input_ids = torch.nn.utils.rnn.pad_sequence(
            img_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        img_attention_mask = torch.nn.utils.rnn.pad_sequence(
            img_attention_mask, batch_first=True, padding_value=0
        )
        
        txt_input_ids = torch.nn.utils.rnn.pad_sequence(
            txt_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        txt_attention_mask = torch.nn.utils.rnn.pad_sequence(
            txt_attention_mask, batch_first=True, padding_value=0
        )
        
        # 处理图像数据
        pixel_values_list = [inst.get('img_pixel_values') for inst in instances if inst.get('img_pixel_values') is not None]
        image_grid_thw_list = [inst.get('img_image_grid_thw') for inst in instances if inst.get('img_image_grid_thw') is not None]
        
        batch = {
            'input_ids': img_input_ids,
            'attention_mask': img_attention_mask,
            'txt_input_ids': txt_input_ids,
            'txt_attention_mask': txt_attention_mask,
            'is_clip_training': True,
            'img_cls_token_ids': instances[0]['img_cls_token_ids'],
            'txt_cls_token_ids': instances[0]['txt_cls_token_ids'],
        }
        
        if pixel_values_list:
            batch['pixel_values'] = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            batch['image_grid_thw'] = torch.cat(image_grid_thw_list, dim=0)
        
        return batch
    
    def _collate_regular_batch(self, instances):
        """整理常规训练批次"""
        input_ids = [inst['input_ids'] for inst in instances]
        attention_mask = [inst.get('attention_mask', torch.ones_like(inst['input_ids'])) for inst in instances]
        
        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'is_clip_training': False,
        }
        
        # 处理图像数据
        pixel_values_list = [inst.get('pixel_values') for inst in instances if inst.get('pixel_values') is not None]
        image_grid_thw_list = [inst.get('image_grid_thw') for inst in instances if inst.get('image_grid_thw') is not None]
        
        if pixel_values_list:
            batch['pixel_values'] = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            batch['image_grid_thw'] = torch.cat(image_grid_thw_list, dim=0)
        
        return batch


class SimplifiedClipTrainer(Trainer):
    """简化的CLIP训练器，参考LLaVA RadZ设计"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """简化的损失计算"""
        
        model.train()
        
        # 根据批次类型选择前向传播方式
        is_clip_training = inputs.pop('is_clip_training', False)
        
        if is_clip_training:
            # CLIP训练模式
            outputs = model(is_clip_training=True, **inputs)
            loss = outputs.loss
            
            # 记录CLIP损失细节
            if hasattr(outputs, 'clip_loss_dict') and outputs.clip_loss_dict:
                for key, value in outputs.clip_loss_dict.items():
                    if hasattr(value, 'item'):
                        self.log({f"train/{key}": value.item()})
        else:
            # 常规语言建模训练
            outputs = model(is_clip_training=False, **inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        return (loss, outputs) if return_outputs else loss


def setup_model_and_tokenizer(model_args: SimplifiedClipModelArguments):
    """设置模型和tokenizer"""
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载processor
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    
    # 创建配置
    config = ClipQwen2VLConfig.from_pretrained(model_args.model_name_or_path)
    config.clip_config.update({
        "img_cls_token_count": model_args.img_cls_token_count,
        "txt_cls_token_count": model_args.txt_cls_token_count,
        "hidden_dim": model_args.hidden_dim,
        "output_dim": model_args.output_dim,
        "temperature": model_args.temperature,
        "img_mlp_type": model_args.img_mlp_type,
        "txt_mlp_type": model_args.txt_mlp_type,
        "feature_extraction_layer": model_args.feature_extraction_layer,
        "pooling_strategy": model_args.pooling_strategy,
    })
    
    # 加载模型
    model = ClipQwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    
    # 应用LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 调整embedding大小（如果添加了新tokens）
    if len(tokenizer) > model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))
        rank_0_print(f"调整embedding大小到: {len(tokenizer)}")
    
    return model, tokenizer, processor


def main():
    # 解析参数
    parser = HfArgumentParser((SimplifiedClipModelArguments, SimplifiedClipDataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 设置模型和tokenizer
    model, tokenizer, processor = setup_model_and_tokenizer(model_args)
    
    # 创建数据集
    train_dataset = SimplifiedClipDataset(
        data_path=data_args.data_path,
        processor=processor,
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
    )
    
    # 创建数据整理器
    data_collator = SimplifiedClipDataCollator(
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )
    
    # 创建训练器
    trainer = SimplifiedClipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    if training_args.do_train:
        rank_0_print("开始训练...")
        trainer.train()
        
        # 保存最终模型
        trainer.save_model()
        rank_0_print(f"训练完成，模型保存至: {training_args.output_dir}")


if __name__ == "__main__":
    main()
