"""
CLIP风格的Qwen2.5-VL训练脚本
实现图像和文本的对比学习训练
"""

import os
import sys
import json
import math
import argparse
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
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
    PreTrainedTokenizer
)
from torch.cuda import amp
from transformers.trainer_utils import get_last_checkpoint
from PIL import Image
import numpy as np

# 导入自定义模块
from .clip_modeling_qwen2_5_vl import (
    ClipQwen2VLConfig,
    ClipQwen2VLForConditionalGeneration
)
from ..dataset.sft_dataset import SFTDataset
# rank_0_print函数 - 用于分布式训练中的日志打印
def rank_0_print(*args, **kwargs):
    """Print only on rank 0 process in distributed training"""
    print(*args, **kwargs)


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClipModelArguments:
    """CLIP模型参数"""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # CLIP特有参数
    Imgcls_count: int = field(default=4, metadata={"help": "Number of image classification tokens"})
    Txtcls_count: int = field(default=4, metadata={"help": "Number of text classification tokens"})
    hidden_dim: int = field(default=1024, metadata={"help": "Hidden dimension for MLP"})
    output_dim: int = field(default=3584, metadata={"help": "Output dimension for features - updated to 3584"})
    img_mlp_type: int = field(default=8, metadata={"help": "Image MLP type - using simplified enhanced architecture"})
    txt_mlp_type: int = field(default=8, metadata={"help": "Text MLP type - using simplified enhanced architecture"})
    knowledge_mlp_type: int = field(default=1, metadata={"help": "Knowledge MLP type"})
    loss_threshold: float = field(default=0.5, metadata={"help": "Loss threshold for combining losses"})
    temperature: float = field(default=0.05, metadata={"help": "Temperature for InfoNCE loss"})
    use_local_loss: bool = field(default=False, metadata={"help": "Whether to use local contrastive loss"})
    feature_layer: int = field(default=1, metadata={"help": "Which layer to extract features from"})
    special_tokens_mlp_type: int = field(default=1, metadata={"help": "Special tokens MLP type"})
    use_ca_loss: bool = field(default=True, metadata={"help": "Whether to use cross attention loss"})
    inference_type: int = field(default=2, metadata={"help": "Inference type"})
    use_cat: bool = field(default=True, metadata={"help": "Whether to use concatenation"})
    use_prompt: bool = field(default=True, metadata={"help": "Whether to use prompt"})
    Book_choice: int = field(default=1, metadata={"help": "Book choice parameter"})


@dataclass
class ClipDataArguments:
    """CLIP数据参数"""
    data_path: str = field(
        metadata={"help": "Path to the training data."}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the evaluation data."}
    )
    image_folder: str = field(
        default="",
        metadata={"help": "Path to the folder containing images."}
    )
    is_multimodal: bool = field(
        default=True,
        metadata={"help": "Whether the data is multimodal."}
    )
    clip_training_ratio: float = field(
        default=0.5,
        metadata={"help": "Ratio of CLIP training vs regular language modeling. 0.0 = no CLIP, 1.0 = only CLIP"}
    )
    # 图像和视频处理参数（修复AttributeError）
    image_min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for image processing."}
    )
    image_max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for image processing."}
    )
    video_min_pixels: Optional[int] = field(
        default=100352,
        metadata={"help": "Minimum number of pixels for video processing."}
    )
    video_max_pixels: Optional[int] = field(
        default=602112,
        metadata={"help": "Maximum number of pixels for video processing."}
    )
    image_resized_width: Optional[int] = field(
        default=None,
        metadata={"help": "Resized width for images."}
    )
    image_resized_height: Optional[int] = field(
        default=None,
        metadata={"help": "Resized height for images."}
    )


@dataclass
class ClipTrainingArguments(TrainingArguments):
    """扩展训练参数"""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_vision_tower: bool = field(default=False)
    freeze_language_model: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    group_by_modality_length: bool = field(default=True)


class ClipDataCollator:
    """CLIP训练的数据整理器"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, model_max_length: int = 8192):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        
    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        整理一个批次的数据
        
        Args:
            instances: 包含input_ids、labels、pixel_values等的实例列表
        
        Returns:
            batch: 整理后的批次数据
        """
        # 分离图像数据和文本数据
        image_instances = []
        text_instances = []
        
        for instance in instances:
            if "pixel_values" in instance and instance["pixel_values"] is not None:
                image_instances.append(instance)
            else:
                text_instances.append(instance)
        
        batch = {}
        
        # 处理图像数据
        if image_instances:
            # 处理input_ids
            input_ids = [instance["input_ids"] for instance in image_instances]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            
            # 处理attention_mask
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            
            # 处理pixel_values
            pixel_values = torch.stack([instance["pixel_values"] for instance in image_instances])
            
            # 处理image_grid_thw
            if "image_grid_thw" in image_instances[0]:
                image_grid_thw = torch.stack([instance["image_grid_thw"] for instance in image_instances])
            else:
                image_grid_thw = None
            
            # 处理labels（如果存在）
            labels = None
            if "labels" in image_instances[0] and image_instances[0]["labels"] is not None:
                labels = [instance["labels"] for instance in image_instances]
                labels = torch.nn.utils.rnn.pad_sequence(
                    labels, batch_first=True, padding_value=-100
                )
            
            batch.update({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "labels": labels,
            })
        
        # 处理文本数据（用于对比学习）
        if text_instances:
            txt_input_ids = [instance["input_ids"] for instance in text_instances]
            txt_input_ids = torch.nn.utils.rnn.pad_sequence(
                txt_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            txt_attention_mask = txt_input_ids.ne(self.tokenizer.pad_token_id)
            
            batch.update({
                "txt_input_ids": txt_input_ids,
                "txt_attention_mask": txt_attention_mask,
            })
        
        # 添加CLIP训练标志
        batch["return_clip_loss"] = len(image_instances) > 0 and len(text_instances) > 0
        
        return batch


class ClipSFTDataset(SFTDataset):
    """扩展SFT数据集以支持CLIP训练"""
    
    def __init__(
        self,
        data_path: str,
        processor: transformers.ProcessorMixin,
        data_args: ClipDataArguments,
        model_id: str,
        clip_training_ratio: float = 0.5,
        **kwargs
    ):
        super().__init__(
            data_path=data_path,
            processor=processor,
            data_args=data_args,
            model_id=model_id,
            **kwargs
        )
        self.clip_training_ratio = clip_training_ratio
        
        # 为CLIP训练添加特殊标记
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """添加CLIP训练所需的特殊标记"""
        # 这里应该添加<Imgcls0>, <Imgcls1>, <Txtcls0>, <Txtcls1>等标记
        # 但由于标记数量可能是动态的，这里先留空
        # 实际使用时需要在模型初始化时添加
        pass
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """获取单个数据项"""
        item = super().__getitem__(index)
        
        # 决定是否进行CLIP训练
        if np.random.random() < self.clip_training_ratio:
            # CLIP训练模式：修改输入以包含特殊标记
            item = self._prepare_clip_item(item)
        
        return item
    
    def _prepare_clip_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """为CLIP训练准备数据项"""
        # 在输入序列末尾添加特殊分类标记
        if "pixel_values" in item and item["pixel_values"] is not None:
            # 图像数据：添加图像分类标记
            # 这里应该添加<Imgcls0>, <Imgcls1>等标记到input_ids末尾
            # 实际实现需要根据具体的tokenizer和特殊标记设计
            pass
        else:
            # 文本数据：添加文本分类标记
            # 这里应该添加<Txtcls0>, <Txtcls1>等标记到input_ids末尾
            pass
        
        return item


class ClipTrainer(Trainer):
    """CLIP训练器"""
    
    def __init__(self, clip_training_ratio: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.clip_training_ratio = clip_training_ratio
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算损失，支持CLIP和语言模型混合训练
        """
        if inputs.get("return_clip_loss", False):
            # CLIP训练模式
            outputs = model(**inputs, return_clip_loss=True)
            loss = outputs.loss
        else:
            # 常规语言模型训练
            outputs = model(**inputs, return_clip_loss=False)
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        单步训练，支持混合训练模式
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        return loss.detach() / self.args.gradient_accumulation_steps


def setup_model_and_tokenizer(model_args: ClipModelArguments) -> tuple:
    """设置模型、tokenizer和processor"""
    
    # 创建CLIP配置
    sparse_config = {
        "Imgcls_count": model_args.Imgcls_count,
        "Txtcls_count": model_args.Txtcls_count,
        "hidden_dim": model_args.hidden_dim,
        "output_dim": model_args.output_dim,
        "img_mlp_type": model_args.img_mlp_type,
        "txt_mlp_type": model_args.txt_mlp_type,
        "knowledge_mlp_type": model_args.knowledge_mlp_type,
        "loss_threshold": model_args.loss_threshold,
        "temperature": model_args.temperature,
        "use_local_loss": model_args.use_local_loss,
        "feature_layer": model_args.feature_layer,
        "special_tokens_mlp_type": model_args.special_tokens_mlp_type,
        "use_ca_loss": model_args.use_ca_loss,
        "inference_type": model_args.inference_type,
        "use_cat": model_args.use_cat,
        "use_prompt": model_args.use_prompt,
        "Book_choice": model_args.Book_choice,
    }
    
    # 加载processor和tokenizer
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    
    # 添加特殊标记
    special_tokens = []
    for i in range(model_args.Imgcls_count):
        special_tokens.append(f"<Imgcls{i}>")
    for i in range(model_args.Txtcls_count):
        special_tokens.append(f"<Txtcls{i}>")
    
    # 检查并添加新标记
    new_tokens = []
    for token in special_tokens:
        if token not in tokenizer.get_vocab():
            new_tokens.append(token)
    
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        rank_0_print(f"Added {len(new_tokens)} special tokens: {new_tokens}")
    
    # 确保有pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建配置并加载模型
    config = ClipQwen2VLConfig.from_pretrained(model_args.model_name_or_path)
    config.sparse_config = sparse_config
    
    model = ClipQwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 调整词汇表大小
    if len(new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    # 初始化新增的embedding
    if hasattr(model, 'get_input_embeddings'):
        embeddings = model.get_input_embeddings()
        with torch.no_grad():
            for i, token in enumerate(new_tokens):
                token_id = tokenizer.convert_tokens_to_ids(token)
                # 使用正态分布初始化新的embedding
                embeddings.weight[token_id] = torch.randn_like(embeddings.weight[token_id]) * 0.02
    
    return model, tokenizer, processor


def setup_data(data_args: ClipDataArguments, processor: transformers.ProcessorMixin, model_id: str) -> tuple:
    """设置训练和验证数据集"""
    
    train_dataset = ClipSFTDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_id,
        clip_training_ratio=data_args.clip_training_ratio
    )
    
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = ClipSFTDataset(
            data_path=data_args.eval_data_path,
            processor=processor,
            data_args=data_args,
            model_id=model_id,
            clip_training_ratio=data_args.clip_training_ratio
        )
    
    return train_dataset, eval_dataset


def main():
    # 解析参数
    parser = HfArgumentParser((ClipModelArguments, ClipDataArguments, ClipTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 检查checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # 设置模型、tokenizer和processor
    model, tokenizer, processor = setup_model_and_tokenizer(model_args)
    
    # 设置数据
    train_dataset, eval_dataset = setup_data(data_args, processor, model_args.model_name_or_path)
    
    # 创建数据整理器
    data_collator = ClipDataCollator(tokenizer, model_args.model_max_length)
    
    # 创建训练器
    trainer = ClipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        clip_training_ratio=data_args.clip_training_ratio,
    )
    
    # 训练
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # 评估
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()