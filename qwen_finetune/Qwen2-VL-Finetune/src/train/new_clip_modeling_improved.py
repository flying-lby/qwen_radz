"""
简化版CLIP-Style Qwen2.5-VL模型实现
参考LLaVA RadZ的简单高效设计，去除过度复杂的多分支架构
"""

import math
from typing import List, Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass

# 使用官方实现
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration,
)


@dataclass
class SimplifiedClipModelOutput(CausalLMOutputWithPast):
    """简化的CLIP模型输出"""
    loss: Optional[torch.FloatTensor] = None
    clip_loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    clip_loss_dict: Optional[Dict[str, torch.FloatTensor]] = None


class SimplifiedClipQwen2VLConfig(Qwen2_5_VLConfig):
    """简化的Qwen2VL配置，去除复杂特性"""
    model_type = "qwen2_5_vl"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 简化的CLIP配置 - 参考LLaVA RadZ
        self.clip_config = kwargs.get("clip_config", {
            "img_cls_token_count": 4,      # 图像特殊标记数量
            "txt_cls_token_count": 4,      # 文本特殊标记数量（与图像平衡）
            "hidden_dim": 1024,            # MLP隐藏维度
            "output_dim": 512,             # 输出特征维度
            "temperature": 0.05,           # 对比学习温度
            "use_local_features": False,   # 简化：禁用局部特征
            "use_cross_attention": False,  # 简化：禁用交叉注意力
            "pooling_strategy": "mean",    # 简单的均值池化
            "feature_extraction_layer": -2, # 使用倒数第二层，与LLaVA RadZ一致
            "mlp_dropout": 0.1,
            "img_mlp_type": 1,             # 启用GELU MLP（修复关键问题）
            "txt_mlp_type": 1,             # 启用GELU MLP
        })


class SimplifiedMLP(nn.Module):
    """简化的MLP层，参考LLaVA RadZ设计"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 简单的2层MLP，类似LLaVA RadZ的mm_projector
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Xavier初始化
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SimplifiedCLIPLoss(nn.Module):
    """简化的CLIP损失，参考LLaVA RadZ的简单设计"""
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        简化的InfoNCE损失计算
        
        Args:
            image_features: (B, D) 图像特征
            text_features: (B, D) 文本特征
        """
        # L2归一化
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # 相似度矩阵
        batch_size = image_features.size(0)
        sim_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 标签（对角线为正样本）
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # 双向InfoNCE损失
        loss_i2t = F.cross_entropy(sim_matrix, labels)
        loss_t2i = F.cross_entropy(sim_matrix.T, labels)
        
        clip_loss = (loss_i2t + loss_t2i) / 2
        
        loss_dict = {
            'clip_loss': clip_loss,
            'loss_i2t': loss_i2t,
            'loss_t2i': loss_t2i
        }
        
        return clip_loss, loss_dict


class SimplifiedClipQwen2VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """简化的CLIP风格Qwen2VL模型，参考LLaVA RadZ的设计理念"""
    
    config_class = SimplifiedClipQwen2VLConfig
    
    def __init__(self, config: SimplifiedClipQwen2VLConfig):
        super().__init__(config)
        
        # 保存配置
        self.clip_config = config.clip_config
        
        # 简化的MLP层 - 修复关键性能问题
        if self.clip_config["img_mlp_type"] > 0:
            self.img_mlp = SimplifiedMLP(
                input_dim=config.hidden_size,
                hidden_dim=self.clip_config["hidden_dim"],
                output_dim=self.clip_config["output_dim"],
                dropout=self.clip_config["mlp_dropout"]
            )
        else:
            self.img_mlp = None
            
        if self.clip_config["txt_mlp_type"] > 0:
            self.txt_mlp = SimplifiedMLP(
                input_dim=config.hidden_size,
                hidden_dim=self.clip_config["hidden_dim"],
                output_dim=self.clip_config["output_dim"],
                dropout=self.clip_config["mlp_dropout"]
            )
        else:
            self.txt_mlp = None
        
        # 简化的损失函数
        self.clip_loss_fn = SimplifiedCLIPLoss(
            temperature=self.clip_config["temperature"]
        )
        
        # 确保正确初始化
        self.post_init()
    
    def extract_image_features(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, 
                             special_token_ids: List[int]) -> torch.Tensor:
        """简化的图像特征提取，参考LLaVA RadZ的inference_pipeline"""
        
        if not special_token_ids:
            # 如果没有特殊标记，使用简单的均值池化（类似LLaVA RadZ）
            return hidden_states.mean(dim=1)
        
        batch_size = hidden_states.size(0)
        features_list = []
        
        for b in range(batch_size):
            # 查找特殊标记位置
            special_mask = torch.zeros_like(input_ids[b], dtype=torch.bool)
            for token_id in special_token_ids:
                special_mask |= (input_ids[b] == token_id)
            
            special_positions = special_mask.nonzero(as_tuple=True)[0]
            
            if len(special_positions) > 0:
                # 提取特殊标记的特征并平均
                special_features = hidden_states[b, special_positions, :]
                batch_feature = special_features.mean(dim=0)
            else:
                # 回退到均值池化
                batch_feature = hidden_states[b].mean(dim=0)
            
            features_list.append(batch_feature)
        
        return torch.stack(features_list, dim=0)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        # 简化的CLIP训练参数
        is_clip_training: bool = False,
        txt_input_ids: Optional[torch.LongTensor] = None,
        txt_attention_mask: Optional[torch.Tensor] = None,
        img_cls_token_ids: Optional[List[int]] = None,
        txt_cls_token_ids: Optional[List[int]] = None,
        **kwargs
    ) -> Union[Tuple, SimplifiedClipModelOutput]:
        
        # 强制输出隐藏状态
        output_hidden_states = True
        return_dict = True
        
        if not is_clip_training:
            # 常规语言建模模式
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs
            )
        
        # CLIP训练模式：简化的单次前向传播
        if txt_input_ids is None:
            raise ValueError("txt_input_ids is required for CLIP training")
        
        # 1. 图像模态前向传播
        img_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=None,  # CLIP训练时不使用language modeling loss
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # 2. 文本模态前向传播（简化处理）
        txt_outputs = super().forward(
            input_ids=txt_input_ids,
            attention_mask=txt_attention_mask,
            pixel_values=None,  # 文本模态不需要图像
            image_grid_thw=None,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # 3. 特征提取（使用指定层，参考LLaVA RadZ）
        feature_layer = self.clip_config["feature_extraction_layer"]
        img_hidden = img_outputs.hidden_states[feature_layer]
        txt_hidden = txt_outputs.hidden_states[feature_layer]
        
        # 4. 简化的特征提取
        img_features = self.extract_image_features(
            img_hidden, input_ids, img_cls_token_ids or []
        )
        txt_features = self.extract_image_features(  # 复用相同逻辑
            txt_hidden, txt_input_ids, txt_cls_token_ids or []
        )
        
        # 5. MLP映射（修复关键问题）
        if self.img_mlp is not None:
            img_features = self.img_mlp(img_features)
        if self.txt_mlp is not None:
            txt_features = self.txt_mlp(txt_features)
        
        # 6. 计算CLIP损失
        clip_loss, loss_dict = self.clip_loss_fn(img_features, txt_features)
        
        # 7. 组合损失（可选的语言建模损失）
        total_loss = clip_loss
        if labels is not None and img_outputs.loss is not None:
            lm_loss = img_outputs.loss
            total_loss = 0.7 * clip_loss + 0.3 * lm_loss
            loss_dict['lm_loss'] = lm_loss
        
        return SimplifiedClipModelOutput(
            loss=total_loss,
            clip_loss=clip_loss,
            lm_loss=img_outputs.loss if labels is not None else None,
            clip_loss_dict=loss_dict,
            logits=img_outputs.logits,
            past_key_values=img_outputs.past_key_values,
            hidden_states=img_outputs.hidden_states,
            attentions=img_outputs.attentions,
        )
    
    def inference_pipeline(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        img_cls_token_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        """简化的推理管道，参考LLaVA RadZ设计"""
        
        # 前向传播获取特征
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            is_clip_training=False
        )
        
        # 提取图像特征（使用指定层）
        feature_layer = self.clip_config["feature_extraction_layer"]
        hidden_states = outputs.hidden_states[feature_layer]
        
        # 简化的特征提取
        image_features = self.extract_image_features(
            hidden_states, input_ids, img_cls_token_ids or []
        )
        
        # MLP映射
        if self.img_mlp is not None:
            image_features = self.img_mlp(image_features)
        
        return image_features
