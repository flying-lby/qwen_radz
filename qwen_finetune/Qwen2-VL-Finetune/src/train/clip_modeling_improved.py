"""改进的CLIP-Style Qwen2.5-VL模型实现"""

import math
from typing import List, Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from dataclasses import dataclass

# 强制使用官方实现，避免本地自定义文件覆盖导致与checkpoint不匹配
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration,
)


@dataclass
class ClipModelOutput(CausalLMOutputWithPast):
    """CLIP模型输出，包含对比学习损失"""
    loss: Optional[torch.FloatTensor] = None
    clip_loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    clip_loss_dict: Optional[Dict[str, torch.FloatTensor]] = None
    img_features: Optional[torch.FloatTensor] = None
    txt_features: Optional[torch.FloatTensor] = None


class ImprovedClipQwen2VLConfig(Qwen2_5_VLConfig):
    """改进的Qwen2VL配置，支持基于LLM response的特征提取"""
    model_type = "qwen2_5_vl"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 改进的CLIP配置
        self.clip_config = kwargs.get("clip_config", {
            "img_cls_token_count": 4,      # 图像特殊标记数量
            "txt_cls_token_count": 4,      # 文本特殊标记数量
            "hidden_dim": 1024,            # MLP隐藏维度
            "output_dim": 512,             # 输出特征维度
            "temperature": 0.05,           # 对比学习温度
            "use_local_features": True,    # 使用局部特征
            "use_global_features": True,   # 使用全局特征
            "use_cross_attention": True,   # 使用交叉注意力
            "pooling_strategy": "mean",    # 池化策略: mean, max, cls
            "feature_extraction_layer": -1, # 特征提取层
            "mlp_dropout": 0.1,            # MLP dropout
            "use_layer_norm": True,        # 是否使用LayerNorm
        })


class ModalityMLP(nn.Module):
    """通用的模态特征映射MLP"""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        
        # Layer normalization
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        
        # First linear layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Second linear layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化MLP权重"""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FeaturePooling(nn.Module):
    """特征池化层"""
    
    def __init__(self, strategy: str = "mean"):
        super().__init__()
        self.strategy = strategy
    
    def forward(
        self, 
        features: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) 可选的注意力掩码
        
        Returns:
            pooled_features: (batch_size, hidden_dim)
        """
        if mask is not None:
            # 扩展mask维度以匹配features
            mask = mask.unsqueeze(-1).expand_as(features)
            features = features * mask
        
        if self.strategy == "mean":
            if mask is not None:
                # 计算有效token的平均值
                pooled = features.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = features.mean(dim=1)
        elif self.strategy == "max":
            if mask is not None:
                features = features.masked_fill(~mask.bool(), -1e9)
            pooled = features.max(dim=1)[0]
        elif self.strategy == "cls":
            # 使用第一个token作为CLS token
            pooled = features[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.strategy}")
        
        return pooled


class CrossModalAttention(nn.Module):
    """跨模态注意力模块"""
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len_q, hidden_dim)
            key: (batch_size, seq_len_k, hidden_dim)
            value: (batch_size, seq_len_v, hidden_dim) 可选，默认使用key
        
        Returns:
            output: (batch_size, seq_len_q, hidden_dim)
        """
        if value is None:
            value = key
        
        # 多头注意力
        attn_output, _ = self.multihead_attn(query, key, value)
        
        # 残差连接和层归一化
        output = self.layer_norm(query + self.dropout(attn_output))
        
        return output


class ImprovedCLIPLoss(nn.Module):
    """改进的CLIP对比学习损失"""
    
    def __init__(
        self, 
        temperature: float = 0.05,
        use_local_features: bool = True,
        use_cross_modal_loss: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.use_local_features = use_local_features
        self.use_cross_modal_loss = use_cross_modal_loss
    
    def forward(
        self,
        global_img_features: torch.Tensor,
        global_txt_features: torch.Tensor,
        local_img_features: Optional[torch.Tensor] = None,
        local_txt_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算多层次的对比学习损失
        
        Args:
            global_img_features: (B, D) 全局图像特征
            global_txt_features: (B, D) 全局文本特征
            local_img_features: (B, D) 局部图像特征
            local_txt_features: (B, D) 局部文本特征
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}
        
        # L2归一化
        global_img_features = F.normalize(global_img_features, p=2, dim=-1)
        global_txt_features = F.normalize(global_txt_features, p=2, dim=-1)
        
        # 全局特征的InfoNCE损失
        batch_size = global_img_features.size(0)
        
        # 计算相似度矩阵
        global_sim = torch.matmul(global_img_features, global_txt_features.T) / self.temperature
        
        # 创建标签（对角线为正样本）
        labels = torch.arange(batch_size, device=global_sim.device)
        
        # 双向损失
        loss_i2t = F.cross_entropy(global_sim, labels)
        loss_t2i = F.cross_entropy(global_sim.T, labels)
        global_loss = (loss_i2t + loss_t2i) / 2
        
        loss_dict['global_loss'] = global_loss
        loss_dict['loss_i2t'] = loss_i2t
        loss_dict['loss_t2i'] = loss_t2i
        
        total_loss = global_loss
        
        # 局部特征损失
        if self.use_local_features and local_img_features is not None and local_txt_features is not None:
            # L2归一化
            local_img_features = F.normalize(local_img_features, p=2, dim=-1)
            local_txt_features = F.normalize(local_txt_features, p=2, dim=-1)
            
            # 局部特征相似度
            local_sim = torch.matmul(local_img_features, local_txt_features.T) / self.temperature
            
            # 局部特征InfoNCE损失
            local_loss_i2t = F.cross_entropy(local_sim, labels)
            local_loss_t2i = F.cross_entropy(local_sim.T, labels)
            local_loss = (local_loss_i2t + local_loss_t2i) / 2
            
            loss_dict['local_loss'] = local_loss
            
            # 交叉模态损失（全局-局部）
            if self.use_cross_modal_loss:
                # 全局图像 - 局部文本
                cross_sim_gi2lt = torch.matmul(global_img_features, local_txt_features.T) / self.temperature
                cross_loss_gi2lt = F.cross_entropy(cross_sim_gi2lt, labels)
                
                # 局部图像 - 全局文本
                cross_sim_li2gt = torch.matmul(local_img_features, global_txt_features.T) / self.temperature
                cross_loss_li2gt = F.cross_entropy(cross_sim_li2gt, labels)
                
                cross_modal_loss = (cross_loss_gi2lt + cross_loss_li2gt) / 2
                loss_dict['cross_modal_loss'] = cross_modal_loss
                
                # 组合损失
                total_loss = 0.4 * global_loss + 0.3 * local_loss + 0.3 * cross_modal_loss
            else:
                total_loss = 0.6 * global_loss + 0.4 * local_loss
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict


class ImprovedClipQwen2VLModel(Qwen2_5_VLModel):
    """改进的Qwen2VL模型，支持基于LLM response的特征提取"""
    
    def __init__(self, config: ImprovedClipQwen2VLConfig):
        super().__init__(config)
        self.config = config
        clip_config = config.clip_config
        
        # 特征提取配置
        self.img_cls_token_count = clip_config["img_cls_token_count"]
        self.txt_cls_token_count = clip_config["txt_cls_token_count"]
        self.hidden_dim = clip_config["hidden_dim"]
        self.output_dim = clip_config["output_dim"]
        self.feature_extraction_layer = clip_config["feature_extraction_layer"]
        
        # MLP层
        self.img_mlp = ModalityMLP(
            input_dim=config.hidden_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=clip_config["mlp_dropout"],
            use_layer_norm=clip_config["use_layer_norm"]
        )
        
        self.txt_mlp = ModalityMLP(
            input_dim=config.hidden_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=clip_config["mlp_dropout"],
            use_layer_norm=clip_config["use_layer_norm"]
        )
        
        # 池化层
        self.global_pooling = FeaturePooling(strategy=clip_config["pooling_strategy"])
        self.local_pooling = FeaturePooling(strategy="mean")
        
        # 交叉注意力（可选）
        if clip_config["use_cross_attention"]:
            self.cross_attention = CrossModalAttention(
                hidden_dim=config.hidden_size,
                num_heads=config.num_attention_heads // 4,  # 使用较少的头
                dropout=clip_config["mlp_dropout"]
            )
        else:
            self.cross_attention = None
        
        # 损失函数
        self.clip_loss_fn = ImprovedCLIPLoss(
            temperature=clip_config["temperature"],
            use_local_features=clip_config["use_local_features"],
            use_cross_modal_loss=clip_config.get("use_cross_modal_loss", True)
        )


class ImprovedClipQwen2VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """改进的CLIP风格Qwen2VL条件生成模型"""
    
    config_class = ImprovedClipQwen2VLConfig
    
    def __init__(self, config: ImprovedClipQwen2VLConfig):
        # 首先调用父类初始化，这会创建标准的Qwen2VL模型结构
        super().__init__(config)
        
        
        # 保存CLIP配置
        self.clip_config = config.clip_config
        
        # 在原有模型基础上添加对比学习组件（不替换self.model）
        # 添加MLP层用于特征映射
        self.img_mlp = ModalityMLP(
            input_dim=config.hidden_size,
            hidden_dim=self.clip_config["hidden_dim"],
            output_dim=self.clip_config["output_dim"],
            dropout=self.clip_config["mlp_dropout"],
            use_layer_norm=self.clip_config["use_layer_norm"]
        )
        
        self.txt_mlp = ModalityMLP(
            input_dim=config.hidden_size,
            hidden_dim=self.clip_config["hidden_dim"],
            output_dim=self.clip_config["output_dim"],
            dropout=self.clip_config["mlp_dropout"],
            use_layer_norm=self.clip_config["use_layer_norm"]
        )
        
        # 池化层
        self.global_pooling = FeaturePooling(strategy=self.clip_config["pooling_strategy"])
        self.local_pooling = FeaturePooling(strategy="mean")
        
        # 交叉注意力（可选）
        if self.clip_config["use_cross_attention"]:
            self.cross_attention = CrossModalAttention(
                hidden_dim=config.hidden_size,
                num_heads=config.num_attention_heads // 4,
                dropout=self.clip_config["mlp_dropout"]
            )
        else:
            self.cross_attention = None
        
        # 对比学习损失函数
        self.clip_loss_fn = ImprovedCLIPLoss(
            temperature=self.clip_config["temperature"],
            use_local_features=self.clip_config["use_local_features"],
            use_cross_modal_loss=self.clip_config.get("use_cross_modal_loss", True)
        )
        
        # 在初始化阶段根据配置确定视觉->LLM 的投影器，避免在DeepSpeed包裹后再动态创建参数
        visual_dim = None
        try:
            vc = getattr(config, 'vision_config', None)
            if vc is not None:
                # 兼容不同字段名
                visual_dim = getattr(vc, 'hidden_size', None) or getattr(vc, 'embed_dim', None) or getattr(vc, 'vision_hidden_size', None)
        except Exception:
            visual_dim = None

        if visual_dim is not None and int(visual_dim) != int(config.hidden_size):
            self.image_projector = nn.Linear(int(visual_dim), int(config.hidden_size), bias=False)
            nn.init.xavier_uniform_(self.image_projector.weight)
        else:
            self.image_projector = None
        
        # 确保模型正确初始化
        self.post_init()
    
    
    def extract_features_from_response(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        special_token_ids: List[int],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从LLM response中提取特征
        
        Args:
            hidden_states: (B, seq_len, hidden_dim) LLM的隐藏状态
            input_ids: (B, seq_len) 输入token IDs
            special_token_ids: 特殊标记的token IDs列表
            attention_mask: (B, seq_len) 注意力掩码
        
        Returns:
            global_features: (B, hidden_dim) 全局特征（特殊标记）
            local_features: (B, hidden_dim) 局部特征（序列平均）
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 创建特殊标记的掩码
        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in special_token_ids:
            special_token_mask |= (input_ids == token_id)
        
        # 提取特殊标记的特征（全局特征）
        special_token_features = []
        for b in range(batch_size):
            # 获取当前样本的特殊标记位置
            special_positions = special_token_mask[b].nonzero(as_tuple=True)[0]
            
            if len(special_positions) > 0:
                # 提取特殊标记对应的隐藏状态
                special_hidden = hidden_states[b, special_positions, :]
                # 平均池化
                special_feature = special_hidden.mean(dim=0)
            else:
                # 如果没有找到特殊标记，使用序列末尾的特征
                special_feature = hidden_states[b, -1, :]
            
            special_token_features.append(special_feature)
        
        global_features = torch.stack(special_token_features, dim=0)
        
        # 提取局部特征（排除特殊标记的序列平均）
        if attention_mask is not None:
            # 创建排除特殊标记的掩码
            local_mask = attention_mask & ~special_token_mask
            local_mask = local_mask.unsqueeze(-1).expand_as(hidden_states)
            
            # 应用掩码并计算平均
            masked_hidden = hidden_states * local_mask
            local_features = masked_hidden.sum(dim=1) / local_mask.sum(dim=1).clamp(min=1e-9)
        else:
            # 简单地排除特殊标记位置
            local_mask = ~special_token_mask
            local_mask = local_mask.unsqueeze(-1).expand_as(hidden_states)
            masked_hidden = hidden_states * local_mask
            local_features = masked_hidden.sum(dim=1) / local_mask.sum(dim=1).clamp(min=1e-9)
        
        return global_features, local_features
    
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
        txt_input_ids: Optional[torch.LongTensor] = None,
        txt_attention_mask: Optional[torch.Tensor] = None,
        img_cls_token_ids: Optional[torch.Tensor] = None,
        txt_cls_token_ids: Optional[torch.Tensor] = None,
        return_clip_loss: Optional[bool] = False,
        # 可选：疾病描述对齐支路
        desc_input_ids: Optional[torch.LongTensor] = None,
        desc_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple, ClipModelOutput]:
        """
        前向传播
        
        Args:
            return_clip_loss: 是否计算CLIP损失
            txt_input_ids: 文本模态的输入IDs
            txt_attention_mask: 文本模态的注意力掩码
            img_cls_token_ids: 图像特殊标记IDs
            txt_cls_token_ids: 文本特殊标记IDs
        """
        
        # 强制输出隐藏状态以提取特征
        output_hidden_states = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 在进入上游 forward 前进行强一致性检查，避免在 masked_scatter 处触发 CUDA 断言
        # 若存在图像输入，优先在此构造输入嵌入，避免上游在维度不匹配时于 masked_scatter 处触发 CUDA 断言
        custom_inputs_embeds = None
        if inputs_embeds is None and input_ids is not None:
            image_token_id = getattr(self.config, "image_token_id", None)
            if image_token_id is not None:
                n_image_tokens = int((input_ids == image_token_id).sum().item())
                if pixel_values is None:
                    if n_image_tokens > 0:
                        raise ValueError(f"发现 {n_image_tokens} 个图像token，但未提供 pixel_values。")
                else:
                    # 1) 计算视觉特征
                    pv = pixel_values.type(self.visual.dtype)
                    image_embeds = self.visual(pv, grid_thw=image_grid_thw)
                    n_image_features = int(image_embeds.shape[0])
                    hidden_from_visual = int(image_embeds.shape[-1])
                    hidden_expected = int(getattr(self.config, "hidden_size", hidden_from_visual))
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            "图像特征与图像token数量不匹配: "
                            f"tokens={n_image_tokens}, features={n_image_features}"
                        )

                    # 2) 若隐藏维度不匹配，使用初始化阶段创建的投影器
                    if hidden_from_visual != hidden_expected:
                        if self.image_projector is None:
                            raise RuntimeError("需要 image_projector 但未初始化。请检查 vision_config.hidden_size 与 hidden_size。")
                        # 将输入对齐到与 token_embeds 相同的 dtype/device，保持与当前rank设备一致
                        # 先生成 token_embeds 以确定目标 dtype/device
                        token_embeds_for_device = self.model.embed_tokens(input_ids)
                        image_embeds = image_embeds.to(dtype=token_embeds_for_device.dtype, device=token_embeds_for_device.device)
                        image_embeds = self.image_projector(image_embeds)

                    # 3) 构造 inputs_embeds 并在图像 token 处替换
                    token_embeds = self.model.embed_tokens(input_ids)
                    mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(token_embeds)
                    image_embeds = image_embeds.to(token_embeds.device, token_embeds.dtype)
                    custom_inputs_embeds = token_embeds.masked_scatter(mask, image_embeds)

                    # 4) 将 inputs_embeds 转换为与模型 LayerNorm 参数一致的 dtype，避免 "expected Float but found BFloat16"
                    try:
                        first_ln_dtype = None
                        for module in self.model.modules():
                            if isinstance(module, nn.LayerNorm):
                                first_ln_dtype = module.weight.dtype
                                break
                        if first_ln_dtype is not None and custom_inputs_embeds.dtype != first_ln_dtype:
                            custom_inputs_embeds = custom_inputs_embeds.to(dtype=first_ln_dtype)
                    except Exception:
                        # 安全降级：维持现有 dtype
                        pass

        # 处理图像模态（如已构造 custom_inputs_embeds，则不再传递 pixel_values，避免上游重复替换）
        img_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=custom_inputs_embeds if custom_inputs_embeds is not None else inputs_embeds,
            labels=labels if not return_clip_loss else None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            pixel_values=None if custom_inputs_embeds is not None else pixel_values,
            image_grid_thw=None if custom_inputs_embeds is not None else image_grid_thw,
            **kwargs
        )
        
        if not return_clip_loss:
            # 常规语言建模模式
            return img_outputs
        
        # CLIP对比学习模式
        if txt_input_ids is None:
            raise ValueError("txt_input_ids is required for CLIP training")
        
        # 处理文本模态
        txt_outputs = super().forward(
            input_ids=txt_input_ids,
            attention_mask=txt_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            pixel_values=None,
            image_grid_thw=None,
            **kwargs
        )
        
        # 获取隐藏状态
        img_hidden_states = img_outputs.hidden_states[self.clip_config["feature_extraction_layer"]]
        txt_hidden_states = txt_outputs.hidden_states[self.clip_config["feature_extraction_layer"]]
        
        # 不在forward中迁移模块dtype/device，避免与DeepSpeed钩子冲突

        # 从response中提取特征
        img_global_features, img_local_features = self.extract_features_from_response(
            img_hidden_states,
            input_ids,
            img_cls_token_ids.tolist() if img_cls_token_ids is not None else [],
            attention_mask
        )
        
        txt_global_features, txt_local_features = self.extract_features_from_response(
            txt_hidden_states,
            txt_input_ids,
            txt_cls_token_ids.tolist() if txt_cls_token_ids is not None else [],
            txt_attention_mask
        )

        # 可选：疾病描述支路（仅在提供且启用时）
        use_desc_branch = (desc_input_ids is not None) and (desc_attention_mask is not None)
        if use_desc_branch:
            desc_outputs = super().forward(
                input_ids=desc_input_ids,
                attention_mask=desc_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                pixel_values=None,
                image_grid_thw=None,
                **kwargs
            )
            desc_hidden_states = desc_outputs.hidden_states[self.clip_config["feature_extraction_layer"]]
            # 复用 TXT 的特征抽取逻辑
            desc_global, desc_local = self.extract_features_from_response(
                desc_hidden_states,
                desc_input_ids,
                [],
                desc_attention_mask
            )
        
        # 应用交叉注意力（可选）
        if self.cross_attention is not None:
            # 图像特征作为query，文本特征作为key/value
            img_global_features_ca = self.cross_attention(
                img_global_features.unsqueeze(1),
                txt_global_features.unsqueeze(1)
            ).squeeze(1)
            
            # 文本特征作为query，图像特征作为key/value
            txt_global_features_ca = self.cross_attention(
                txt_global_features.unsqueeze(1),
                img_global_features.unsqueeze(1)
            ).squeeze(1)
            
            # 残差连接
            img_global_features = img_global_features + 0.5 * img_global_features_ca
            txt_global_features = txt_global_features + 0.5 * txt_global_features_ca
        
        # 通过MLP映射到共享空间（确保输入输出dtype一致）
        target_dtype = img_hidden_states.dtype
        img_global_features = self.img_mlp(img_global_features.to(dtype=target_dtype))
        txt_global_features = self.txt_mlp(txt_global_features.to(dtype=target_dtype))
        img_local_features = self.img_mlp(img_local_features.to(dtype=target_dtype))
        txt_local_features = self.txt_mlp(txt_local_features.to(dtype=target_dtype))
        if use_desc_branch:
            desc_global = self.txt_mlp(desc_global.to(dtype=target_dtype))
            desc_local = self.txt_mlp(desc_local.to(dtype=target_dtype))
        
        # 计算CLIP损失
        clip_loss, loss_dict = self.clip_loss_fn(
            img_global_features,
            txt_global_features,
            img_local_features if self.clip_config["use_local_features"] else None,
            txt_local_features if self.clip_config["use_local_features"] else None
        )
        if use_desc_branch:
            # I<->D, T<->D 对齐（全局）
            i_norm = torch.nn.functional.normalize(img_global_features, p=2, dim=-1)
            t_norm = torch.nn.functional.normalize(txt_global_features, p=2, dim=-1)
            d_norm = torch.nn.functional.normalize(desc_global, p=2, dim=-1)
            temp = self.model.clip_loss_fn.temperature
            sim_i_d = (i_norm @ d_norm.t()) / temp
            sim_t_d = (t_norm @ d_norm.t()) / temp
            labels_sim = torch.arange(sim_i_d.size(0), device=sim_i_d.device)
            loss_i2d = torch.nn.functional.cross_entropy(sim_i_d, labels_sim)
            loss_t2d = torch.nn.functional.cross_entropy(sim_t_d, labels_sim)
            loss_desc = 0.5 * (loss_i2d + loss_t2d)
            clip_loss = clip_loss + loss_desc
            loss_dict['loss_i2d'] = loss_i2d
            loss_dict['loss_t2d'] = loss_t2d
        
        # 组合损失（可选地包含语言建模损失）
        total_loss = clip_loss
        if labels is not None and img_outputs.loss is not None:
            lm_loss = img_outputs.loss
            total_loss = 0.7 * clip_loss + 0.3 * lm_loss
            loss_dict['lm_loss'] = lm_loss
        
        if return_dict:
            return ClipModelOutput(
                loss=total_loss,
                clip_loss=clip_loss,
                lm_loss=img_outputs.loss if labels is not None else None,
                clip_loss_dict=loss_dict,
                img_features=img_global_features,
                txt_features=txt_global_features,
                logits=img_outputs.logits,
                past_key_values=img_outputs.past_key_values,
                hidden_states=img_outputs.hidden_states,
                attentions=img_outputs.attentions,
            )
        
        return (total_loss,) + img_outputs[1:]