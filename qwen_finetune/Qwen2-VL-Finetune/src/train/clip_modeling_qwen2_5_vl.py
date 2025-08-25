"""
CLIP-Style Qwen2.5-VL Model Implementation
基于LLaVA-Med的CLIP架构，实现图像和文本的对比学习
"""

import math
from typing import List, Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .modeling_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration
)


class ClipQwen2VLConfig(Qwen2_5_VLConfig):
    """扩展Qwen2VL配置，支持CLIP风格的对比学习参数"""
    model_type = "clip_qwen2_vl"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # CLIP风格配置参数
        self.sparse_config = kwargs.get("sparse_config", {
            "Imgcls_count": 4,  # 图像分类标记数量
            "Txtcls_count": 4,  # 文本分类标记数量
            "hidden_dim": 1024,  # MLP隐藏层维度
            "output_dim": 512,   # 输出特征维度
            "img_mlp_type": 1,   # 图像MLP类型
            "txt_mlp_type": 1,   # 文本MLP类型
            "knowledge_mlp_type": 1,  # 知识MLP类型
            "loss_threshold": 0.5,    # 损失函数阈值
            "temperature": 0.05,      # InfoNCE温度参数
            "use_local_loss": False,  # 是否使用局部损失
            "feature_layer": 1,       # 特征提取层数
            "special_tokens_mlp_type": 1,  # 特殊标记MLP类型
            "use_ca_loss": True,      # 是否使用交叉注意力损失
            "inference_type": 2,      # 推理类型
            "use_cat": True,          # 是否使用连接
            "use_prompt": True,       # 是否使用提示
            "Book_choice": 1          # 书籍选择
        })


class ImageMLP(nn.Module):
    """图像特征映射MLP - 数值稳定优化版本"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, mlp_type: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mlp_type = mlp_type
        
        if mlp_type == 1:
            # 数值稳定：LayerNorm移到中间，添加输入预处理
            self.input_stabilizer = nn.Dropout(0.05)
            self.out_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
        elif mlp_type == 2:
            # 完全无LayerNorm版本 - 最稳定
            self.input_stabilizer = None
            self.out_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
        elif mlp_type == 3:
            # 深层架构，LayerNorm在中间
            self.input_stabilizer = None
            self.out_mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
        elif mlp_type == 4:
            # 最简单版本 - 用于调试
            self.input_stabilizer = None
            self.out_mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim),
            )
        elif mlp_type == 5:
            # 轻量级改进版本
            self.input_stabilizer = None
            self.out_mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim // 2),
                nn.LayerNorm(output_dim // 2, eps=1e-6),
                nn.GELU(),
                nn.Linear(output_dim // 2, output_dim),
            )
        elif mlp_type == 6:
            # 极简版本
            self.input_stabilizer = None
            self.out_mlp = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(input_dim, output_dim),
            )
        else:
            # mlp_type == 0, 直接返回输入
            self.input_stabilizer = None
            self.out_mlp = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Invalid input to ImageMLP, applying stabilization")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x = torch.clamp(x, -1e6, 1e6)
        
        # 零范数检查
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        if (norms < 1e-8).any():
            # 添加小的随机扰动避免零输入
            x = x + torch.randn_like(x) * 1e-6
        
        # 应用输入稳定器
        if self.input_stabilizer is not None:
            x = self.input_stabilizer(x)
            
        if self.out_mlp is None:
            return x
        return self.out_mlp(x)


class TextMLP(nn.Module):
    """文本特征映射MLP - 数值稳定优化版本"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, mlp_type: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mlp_type = mlp_type
        
        if mlp_type == 1:
            # 数值稳定：LayerNorm移到中间，添加输入预处理
            self.input_stabilizer = nn.Dropout(0.05)
            self.out_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
        elif mlp_type == 2:
            # 完全无LayerNorm版本 - 最稳定
            self.input_stabilizer = None
            self.out_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
        elif mlp_type == 3:
            # 深层架构，LayerNorm在中间
            self.input_stabilizer = None
            self.out_mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
        elif mlp_type == 4:
            # 最简单版本 - 用于调试
            self.input_stabilizer = None
            self.out_mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim),
            )
        elif mlp_type == 5:
            # 轻量级改进版本
            self.input_stabilizer = None
            self.out_mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim // 2),
                nn.LayerNorm(output_dim // 2, eps=1e-6),
                nn.GELU(),
                nn.Linear(output_dim // 2, output_dim),
            )
        elif mlp_type == 6:
            # 极简版本
            self.input_stabilizer = None
            self.out_mlp = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(input_dim, output_dim),
            )
        else:
            # mlp_type == 0, 直接返回输入
            self.input_stabilizer = None
            self.out_mlp = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入数值稳定性检查 - 与ImageMLP保持一致
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Invalid input to TextMLP, applying stabilization")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x = torch.clamp(x, -1e6, 1e6)
        
        # 零范数检查
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        if (norms < 1e-8).any():
            # 添加小的随机扰动避免零输入
            x = x + torch.randn_like(x) * 1e-6
        
        # 应用输入稳定器
        if self.input_stabilizer is not None:
            x = self.input_stabilizer(x)
            
        if self.out_mlp is None:
            return x
        return self.out_mlp(x)


class KnowledgeMLP(nn.Module):
    """知识特征映射MLP"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, mlp_type: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mlp_type = mlp_type
        
        if mlp_type == 1:
            self.out_mlp = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim)
            )
        elif mlp_type == 2:
            self.out_mlp = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
            )
        elif mlp_type == 3:
            self.out_mlp = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Dropout(0.3),
                nn.Linear(input_dim, output_dim),
            )
        else:
            # mlp_type == 0, 直接返回输入
            self.out_mlp = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_mlp is None:
            return x
        return self.out_mlp(x)


class CrossAttentionModule(nn.Module):
    """交叉注意力模块"""
    
    def __init__(self, hidden_size: int, num_heads: int = 32, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, query_features: torch.Tensor, key_value_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_features: (B, hidden_size) 查询特征
            key_value_features: (B, hidden_size) 键值特征
        
        Returns:
            attn_output: (B, hidden_size) 注意力输出
            attn_weights: (B, 1, 1) 注意力权重
        """
        # 扩展维度以适应多头注意力
        query = query_features.unsqueeze(1)  # (B, 1, hidden_size)
        key_value = key_value_features.unsqueeze(1)  # (B, 1, hidden_size)
        
        attn_output, attn_weights = self.attention(query, key_value, key_value)
        
        # 返回压缩后的输出
        attn_output = attn_output.squeeze(1)  # (B, hidden_size)
        
        return attn_output, attn_weights


class CLIPLoss(nn.Module):
    """改进的CLIP风格InfoNCE对比学习损失 - 自适应权重和数值稳定"""
    
    def __init__(self, temperature: float = 0.05, use_local_loss: bool = False):
        super().__init__()
        self.temperature = temperature
        self.use_local_loss = use_local_loss
        
        # 自适应权重参数（可学习）
        self.global_weight = nn.Parameter(torch.tensor(1.0))
        self.local_weight = nn.Parameter(torch.tensor(0.5))
        self.cross_weight = nn.Parameter(torch.tensor(0.3))
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        local_image_features: Optional[torch.Tensor] = None,
        local_text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算InfoNCE对比学习损失
        
        Args:
            image_features: (B, D) 图像全局特征
            text_features: (B, D) 文本全局特征
            local_image_features: (B, D) 图像局部特征
            local_text_features: (B, D) 文本局部特征
        
        Returns:
            loss: InfoNCE损失
        """
        # L2归一化
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 创建标签（正样本在对角线上）
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # 计算双向InfoNCE损失
        loss_i2t = F.cross_entropy(similarity_matrix, labels)  # 图像到文本
        loss_t2i = F.cross_entropy(similarity_matrix.T, labels)  # 文本到图像
        
        # 全局损失
        global_loss = (loss_i2t + loss_t2i) / 2
        
        if self.use_local_loss and local_image_features is not None and local_text_features is not None:
            # 数值稳定性检查
            if torch.isnan(local_image_features).any() or torch.isnan(local_text_features).any():
                print("Warning: NaN detected in local features, using global loss only")
                return global_loss
                
            # 归一化局部特征
            local_image_features = F.normalize(local_image_features, p=2, dim=-1)
            local_text_features = F.normalize(local_text_features, p=2, dim=-1)
            
            # 计算全局-局部交叉损失
            global_img_to_local_txt_sim = torch.matmul(image_features, local_text_features.T) / self.temperature
            local_img_to_global_txt_sim = torch.matmul(local_image_features, text_features.T) / self.temperature
            
            # 全局图像到局部文本
            loss_gi2lt = F.cross_entropy(global_img_to_local_txt_sim, labels)
            loss_lt2gi = F.cross_entropy(global_img_to_local_txt_sim.T, labels)
            cross_loss_1 = (loss_gi2lt + loss_lt2gi) / 2
            
            # 局部图像到全局文本
            loss_li2gt = F.cross_entropy(local_img_to_global_txt_sim, labels)
            loss_gt2li = F.cross_entropy(local_img_to_global_txt_sim.T, labels)
            cross_loss_2 = (loss_li2gt + loss_gt2li) / 2
            
            # 局部-局部损失
            local_similarity_matrix = torch.matmul(local_image_features, local_text_features.T) / self.temperature
            loss_li2lt = F.cross_entropy(local_similarity_matrix, labels)
            loss_lt2li = F.cross_entropy(local_similarity_matrix.T, labels)
            local_loss = (loss_li2lt + loss_lt2li) / 2
            
            # 交叉损失
            cross_loss = (cross_loss_1 + cross_loss_2) / 2
            
            # 自适应权重组合（使用可学习权重并归一化）
            weights = torch.softmax(torch.stack([self.global_weight, self.local_weight, self.cross_weight]), dim=0)
            total_loss = weights[0] * global_loss + weights[1] * local_loss + weights[2] * cross_loss
            
            return total_loss
        
        return global_loss


class ClipQwen2VLModel(Qwen2_5_VLModel):
    """扩展Qwen2VL模型，支持CLIP风格的对比学习"""
    
    def __init__(self, config: ClipQwen2VLConfig):
        super().__init__(config)
        self.config = config
        
        # 获取CLIP配置
        sparse_config = config.sparse_config
        self.Imgcls_count = sparse_config["Imgcls_count"]
        self.Txtcls_count = sparse_config["Txtcls_count"]
        self.hidden_dim = sparse_config["hidden_dim"]
        self.output_dim = sparse_config["output_dim"]
        self.img_mlp_type = sparse_config["img_mlp_type"]
        self.txt_mlp_type = sparse_config["txt_mlp_type"]
        self.knowledge_mlp_type = sparse_config["knowledge_mlp_type"]
        self.temperature = sparse_config["temperature"]
        self.use_local_loss = sparse_config["use_local_loss"]
        self.feature_layer = sparse_config["feature_layer"]
        self.special_tokens_mlp_type = sparse_config["special_tokens_mlp_type"]
        self.use_ca_loss = sparse_config["use_ca_loss"]
        
        # 初始化MLP层 - 统一使用config.hidden_size作为输入维度
        self.img_mlp = ImageMLP(
            input_dim=config.hidden_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            mlp_type=self.img_mlp_type
        )
        
        self.txt_mlp = TextMLP(
            input_dim=config.hidden_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            mlp_type=self.txt_mlp_type
        )
        
        self.knowledge_mlp = KnowledgeMLP(
            input_dim=config.hidden_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            mlp_type=self.knowledge_mlp_type
        )
        
        # 特殊标记MLP - 保持维度不变
        if self.special_tokens_mlp_type == 1:
            self.special_token_mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.GELU(),
                nn.Linear(config.hidden_size // 4, config.hidden_size)  # 保持hidden_size维度
            )
        elif self.special_tokens_mlp_type == 2:
            self.special_token_mlp = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Dropout(0.3),
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.GELU(),
                nn.Linear(config.hidden_size // 4, config.hidden_size)  # 保持hidden_size维度
            )
        else:
            self.special_token_mlp = None
        
        # 交叉注意力模块
        if self.use_ca_loss:
            self.cross_attention_module = CrossAttentionModule(
                hidden_size=config.hidden_size
            )
        
        # CLIP损失函数
        self.clip_loss = CLIPLoss(
            temperature=self.temperature,
            use_local_loss=self.use_local_loss
        )


class ClipQwen2VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """CLIP风格的Qwen2VL条件生成模型"""
    
    config_class = ClipQwen2VLConfig
    
    def __init__(self, config: ClipQwen2VLConfig):
        super().__init__(config)
        self.model = ClipQwen2VLModel(config)
        self.config = config
        
        # 获取CLIP配置
        sparse_config = config.sparse_config
        self.Imgcls_count = sparse_config["Imgcls_count"]
        self.Txtcls_count = sparse_config["Txtcls_count"]
        self.output_dim = sparse_config["output_dim"]  # 添加缺失的output_dim属性
        self.loss_threshold = sparse_config["loss_threshold"]
        self.use_ca_loss = sparse_config["use_ca_loss"]
        self.feature_layer = sparse_config["feature_layer"]
        
        # 确保模型权重初始化
        self.post_init()
    
    def get_special_token_features(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        token_type: str = "img"
    ) -> torch.Tensor:
        """
        提取特殊标记的特征 - 增强版本，包含鲁棒性检查
        
        Args:
            hidden_states: (B, seq_len, hidden_size) 隐藏状态
            input_ids: (B, seq_len) 输入ID
            token_type: 标记类型，"img" 或 "txt"
        
        Returns:
            features: (B, num_tokens, hidden_size) 特殊标记特征
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if token_type == "img":
            # 查找图像分类标记
            token_count = self.Imgcls_count
        else:
            # 查找文本分类标记
            token_count = self.Txtcls_count
        
        # 安全检查：确保序列长度足够
        if seq_len < token_count:
            print(f"Warning: Sequence length {seq_len} < token_count {token_count}, using available tokens")
            # 如果序列太短，使用所有可用的隐藏状态
            features = hidden_states  # (B, seq_len, hidden_size)
            # 如果还是不够，用零填充
            if seq_len < token_count:
                padding_needed = token_count - seq_len
                padding = torch.zeros(batch_size, padding_needed, hidden_size, 
                                    device=hidden_states.device, dtype=hidden_states.dtype)
                features = torch.cat([features, padding], dim=1)
        else:
            # 正常情况：从序列末尾提取特殊标记
            features = hidden_states[:, -token_count:, :]  # (B, token_count, hidden_size)
        
        # 验证提取的特征
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"Warning: Invalid values in extracted {token_type} features, applying cleanup")
            # 清理NaN和Inf值
            features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
            features = torch.where(torch.isinf(features), torch.clamp(features, -1e6, 1e6), features)
        
        # 检查特征是否全零
        feature_norms = torch.norm(features, p=2, dim=-1)  # (B, token_count)
        if (feature_norms < 1e-8).all():
            print(f"Warning: All {token_type} features have near-zero norm, applying random initialization")
            features = torch.randn_like(features) * 0.01
        
        return features
    
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
        return_clip_loss: Optional[bool] = False,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        前向传播，支持CLIP风格的对比学习
        
        Args:
            return_clip_loss: 是否返回CLIP损失而不是语言模型损失
            txt_input_ids: 文本输入ID（用于对比学习）
            txt_attention_mask: 文本注意力掩码
            其他参数与父类相同
        """
        output_hidden_states = True  # 强制输出隐藏状态以提取特征
        
        # 处理图像输入的前向传播
        img_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels if not return_clip_loss else None,  # CLIP训练时不计算语言模型损失
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs
        )
        
        if not return_clip_loss:
            # 常规训练，返回语言模型损失
            return img_outputs
        
        # CLIP训练模式
        if txt_input_ids is None:
            raise ValueError("txt_input_ids is required for CLIP training")
        
        # 处理文本输入的前向传播
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
        
        # 提取特征
        img_hidden_states = img_outputs.hidden_states[-self.feature_layer]
        txt_hidden_states = txt_outputs.hidden_states[-self.feature_layer]
        
        # 获取全局特征（特殊标记）
        global_img_features = self.get_special_token_features(
            img_hidden_states, input_ids, "img"
        )  # (B, Imgcls_count, hidden_size)
        
        global_txt_features = self.get_special_token_features(
            txt_hidden_states, txt_input_ids, "txt"
        )  # (B, Txtcls_count, hidden_size)
        
        # 池化全局特征
        global_img_features = global_img_features.mean(dim=1)  # (B, hidden_size)
        global_txt_features = global_txt_features.mean(dim=1)  # (B, hidden_size)
        
        # 获取局部特征（序列的平均）
        local_img_features = img_hidden_states[:, :-self.Imgcls_count, :].mean(dim=1)  # (B, hidden_size)
        local_txt_features = txt_hidden_states[:, :-self.Txtcls_count, :].mean(dim=1)  # (B, hidden_size)
        
        # 应用特殊标记MLP
        if self.model.special_token_mlp is not None:
            global_img_features = self.model.special_token_mlp(global_img_features)
            global_txt_features = self.model.special_token_mlp(global_txt_features)
        
        # 应用MLP映射
        global_img_features = self.model.img_mlp(global_img_features)  # (B, output_dim)
        global_txt_features = self.model.txt_mlp(global_txt_features)  # (B, output_dim)
        local_img_features = self.model.img_mlp(local_img_features)   # (B, output_dim)
        local_txt_features = self.model.txt_mlp(local_txt_features)   # (B, output_dim)
        
        # 计算CLIP损失
        clip_loss = self.model.clip_loss(
            image_features=global_img_features,
            text_features=global_txt_features,
            local_image_features=local_img_features if self.model.use_local_loss else None,
            local_text_features=local_txt_features if self.model.use_local_loss else None
        )
        
        # 如果使用交叉注意力损失
        if self.use_ca_loss and hasattr(self.model, 'cross_attention_module'):
            ca_output, ca_weights = self.model.cross_attention_module(
                global_img_features, global_txt_features
            )
            # 可以在这里添加额外的交叉注意力损失
        
        return CausalLMOutputWithPast(
            loss=clip_loss,
            logits=None,  # CLIP训练时不需要logits
            past_key_values=img_outputs.past_key_values,
            hidden_states=img_outputs.hidden_states,
            attentions=img_outputs.attentions,
        )
    
    @torch.no_grad()
    def extract_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        提取图像和文本特征，用于推理
        
        Returns:
            features: 包含global_features和local_features的字典
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        hidden_states = outputs.hidden_states[-self.feature_layer]
        
        # 提取全局特征
        if pixel_values is not None:
            # 图像输入 - 增强的特征提取和NaN处理
            
            # 1. 首先检查隐藏状态的有效性
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                print(f"Warning: Invalid hidden_states detected, using fallback features")
                batch_size = hidden_states.shape[0]
                # 通过MLP生成正确维度的fallback特征
                fallback_hidden = torch.randn(batch_size, self.config.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                try:
                    fallback_global = self.model.img_mlp(fallback_hidden)
                    fallback_local = self.model.img_mlp(fallback_hidden)
                except Exception as e:
                    print(f"Warning: MLP fallback failed: {e}, using direct random features")
                    fallback_global = torch.randn(batch_size, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                    fallback_local = torch.randn(batch_size, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                return {"global_features": fallback_global, "local_features": fallback_local}
            
            # 2. 提取特殊标记特征 (全局)
            special_tokens = self.get_special_token_features(
                hidden_states, input_ids, "img"
            )  # (B, token_count, hidden_size)
            raw_global_features = special_tokens.mean(dim=1)  # (B, hidden_size)
            
            # 3. 改进的局部特征提取 - 多尺度池化
            non_special_states = hidden_states[:, :-self.Imgcls_count, :]  # (B, seq_len-token_count, hidden_size)
            seq_len_local = non_special_states.shape[1]
            
            if seq_len_local > 0:
                # 多尺度特征聚合：平均池化 + 最大池化 + 注意力加权池化
                avg_local = non_special_states.mean(dim=1)  # (B, hidden_size)
                max_local, _ = non_special_states.max(dim=1)  # (B, hidden_size)
                
                # 简单注意力加权池化
                attention_weights = torch.softmax(
                    torch.matmul(non_special_states, raw_global_features.unsqueeze(-1)).squeeze(-1), 
                    dim=1
                )  # (B, seq_len_local)
                att_local = torch.sum(
                    non_special_states * attention_weights.unsqueeze(-1), 
                    dim=1
                )  # (B, hidden_size)
                
                # 组合局部特征：给注意力池化更高权重
                local_features = 0.5 * att_local + 0.3 * avg_local + 0.2 * max_local
            else:
                # 如果没有非特殊标记，使用全局特征作为局部特征
                local_features = raw_global_features.clone()
            
            # 3. 验证提取的特征
            if torch.isnan(raw_global_features).any() or torch.isinf(raw_global_features).any():
                print(f"Warning: NaN/Inf in raw global_features after get_special_token_features (img)")
                # 使用局部特征的平均值作为回退
                raw_global_features = local_features.clone()
            
            # 检查特征范数
            global_norm = torch.norm(raw_global_features, p=2, dim=-1)
            if (global_norm < 1e-8).any():
                print(f"Warning: Global features have near-zero norm, applying regularization")
                zero_mask = (global_norm < 1e-8).unsqueeze(-1)
                random_fix = torch.randn_like(raw_global_features) * 0.01
                raw_global_features = torch.where(zero_mask, random_fix, raw_global_features)
            
            # 4. 应用特殊标记MLP（如果存在）
            if self.model.special_token_mlp is not None:
                global_features = self.model.special_token_mlp(raw_global_features)
                # 验证MLP输出
                if torch.isnan(global_features).any() or torch.isinf(global_features).any():
                    print(f"Warning: NaN/Inf in global_features after special_token_mlp, using raw features")
                    global_features = raw_global_features
            else:
                global_features = raw_global_features
            
            # 5. 应用图像MLP
            try:
                global_features = self.model.img_mlp(global_features)
                local_features = self.model.img_mlp(local_features)
            except Exception as e:
                print(f"Warning: Error in img_mlp: {e}, using fallback")
                batch_size = hidden_states.shape[0]
                # 创建正确维度的fallback特征
                fallback_hidden = torch.randn(batch_size, self.config.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                try:
                    global_features = self.model.img_mlp(fallback_hidden)
                    local_features = self.model.img_mlp(fallback_hidden)
                except:
                    # 如果MLP也失败，直接创建output_dim的特征
                    print(f"Warning: Complete MLP failure, using direct random features")
                    global_features = torch.randn(batch_size, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                    local_features = torch.randn(batch_size, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
            
            # 6. 最终验证
            if torch.isnan(global_features).any() or torch.isnan(local_features).any():
                print(f"Warning: Final NaN detected after img_mlp, applying fallback")
                batch_size = hidden_states.shape[0]
                if torch.isnan(global_features).any():
                    fallback_hidden = torch.randn(batch_size, self.config.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                    try:
                        global_features = self.model.img_mlp(fallback_hidden)
                    except:
                        global_features = torch.randn(batch_size, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                if torch.isnan(local_features).any():
                    fallback_hidden = torch.randn(batch_size, self.config.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                    try:
                        local_features = self.model.img_mlp(fallback_hidden)
                    except:
                        local_features = torch.randn(batch_size, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
        else:
            # 文本输入 - 增强的特征提取和NaN处理
            
            # 1. 首先检查隐藏状态的有效性
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                print(f"Warning: Invalid text hidden_states detected, using fallback features")
                batch_size = hidden_states.shape[0]
                # 通过MLP生成正确维度的fallback特征
                fallback_hidden = torch.randn(batch_size, self.config.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                try:
                    fallback_global = self.model.txt_mlp(fallback_hidden)
                    fallback_local = self.model.txt_mlp(fallback_hidden)
                except Exception as e:
                    print(f"Warning: Text MLP fallback failed: {e}, using direct random features")
                    fallback_global = torch.randn(batch_size, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                    fallback_local = torch.randn(batch_size, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                return {"global_features": fallback_global, "local_features": fallback_local}
            
            # 2. 提取特殊标记特征 (全局)
            special_tokens = self.get_special_token_features(
                hidden_states, input_ids, "txt"
            )  # (B, token_count, hidden_size)
            raw_global_features = special_tokens.mean(dim=1)  # (B, hidden_size)
            
            # 3. 改进的局部特征提取 - 多尺度池化
            non_special_states = hidden_states[:, :-self.Txtcls_count, :]  # (B, seq_len-token_count, hidden_size)
            seq_len_local = non_special_states.shape[1]
            
            if seq_len_local > 0:
                # 多尺度特征聚合：平均池化 + 最大池化 + 注意力加权池化
                avg_local = non_special_states.mean(dim=1)  # (B, hidden_size)
                max_local, _ = non_special_states.max(dim=1)  # (B, hidden_size)
                
                # 简单注意力加权池化
                attention_weights = torch.softmax(
                    torch.matmul(non_special_states, raw_global_features.unsqueeze(-1)).squeeze(-1), 
                    dim=1
                )  # (B, seq_len_local)
                att_local = torch.sum(
                    non_special_states * attention_weights.unsqueeze(-1), 
                    dim=1
                )  # (B, hidden_size)
                
                # 组合局部特征：给注意力池化更高权重
                local_features = 0.5 * att_local + 0.3 * avg_local + 0.2 * max_local
            else:
                # 如果没有非特殊标记，使用全局特征作为局部特征
                local_features = raw_global_features.clone()
            
            # 3. 验证提取的特征
            if torch.isnan(raw_global_features).any() or torch.isinf(raw_global_features).any():
                print(f"Warning: NaN/Inf in raw global_features after get_special_token_features (txt)")
                # 使用局部特征的平均值作为回退
                raw_global_features = local_features.clone()
            
            # 检查特征范数
            global_norm = torch.norm(raw_global_features, p=2, dim=-1)
            if (global_norm < 1e-8).any():
                print(f"Warning: Text global features have near-zero norm, applying regularization")
                zero_mask = (global_norm < 1e-8).unsqueeze(-1)
                random_fix = torch.randn_like(raw_global_features) * 0.01
                raw_global_features = torch.where(zero_mask, random_fix, raw_global_features)
            
            # 4. 应用特殊标记MLP（如果存在）
            if self.model.special_token_mlp is not None:
                global_features = self.model.special_token_mlp(raw_global_features)
                # 验证MLP输出
                if torch.isnan(global_features).any() or torch.isinf(global_features).any():
                    print(f"Warning: NaN/Inf in global_features after special_token_mlp (txt), using raw features")
                    global_features = raw_global_features
            else:
                global_features = raw_global_features
            
            # 5. 应用文本MLP
            try:
                global_features = self.model.txt_mlp(global_features)
                local_features = self.model.txt_mlp(local_features)
            except Exception as e:
                print(f"Warning: Error in txt_mlp: {e}, using fallback")
                batch_size = hidden_states.shape[0]
                # 创建正确维度的fallback特征
                fallback_hidden = torch.randn(batch_size, self.config.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                try:
                    global_features = self.model.txt_mlp(fallback_hidden)
                    local_features = self.model.txt_mlp(fallback_hidden)
                except:
                    # 如果MLP也失败，直接创建output_dim的特征
                    print(f"Warning: Complete Text MLP failure, using direct random features")
                    global_features = torch.randn(batch_size, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                    local_features = torch.randn(batch_size, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
            
            # 6. 最终验证
            if torch.isnan(global_features).any() or torch.isnan(local_features).any():
                print(f"Warning: Final NaN detected after txt_mlp, applying fallback")
                batch_size = hidden_states.shape[0]
                if torch.isnan(global_features).any():
                    fallback_hidden = torch.randn(batch_size, self.config.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                    try:
                        global_features = self.model.txt_mlp(fallback_hidden)
                    except:
                        global_features = torch.randn(batch_size, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                if torch.isnan(local_features).any():
                    fallback_hidden = torch.randn(batch_size, self.config.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                    try:
                        local_features = self.model.txt_mlp(fallback_hidden)
                    except:
                        local_features = torch.randn(batch_size, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
        
        # 检查特征是否包含NaN或无穷大
        if torch.isnan(global_features).any() or torch.isinf(global_features).any():
            print(f"Warning: global_features contains NaN/Inf before normalization")
            print(f"Global features stats: min={global_features.min():.6f}, max={global_features.max():.6f}, mean={global_features.mean():.6f}")
        
        if torch.isnan(local_features).any() or torch.isinf(local_features).any():
            print(f"Warning: local_features contains NaN/Inf before normalization")
        
        # 安全的归一化，添加eps参数避免除零
        return {
            "global_features": F.normalize(global_features, p=2, dim=-1, eps=1e-8),
            "local_features": F.normalize(local_features, p=2, dim=-1, eps=1e-8)
        }
    
    @torch.no_grad()
    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        计算图像和文本特征的相似度
        
        Args:
            image_features: (B, D) 图像特征
            text_features: (C, D) 文本特征
            temperature: 温度参数
        
        Returns:
            similarity: (B, C) 相似度矩阵
        """
        if temperature is None:
            temperature = self.model.temperature
        
        # 检查温度参数
        if temperature == 0 or torch.isnan(torch.tensor(temperature)):
            print(f"Warning: Invalid temperature value: {temperature}")
            temperature = 1.0  # 使用默认值
        
        # 检查并修复特征中的NaN值
        if torch.isnan(image_features).any():
            print(f"Warning: image_features contains NaN values, fixing...")
            image_features = torch.where(torch.isnan(image_features), torch.zeros_like(image_features), image_features)
        if torch.isnan(text_features).any():
            print(f"Warning: text_features contains NaN values, fixing...")
            text_features = torch.where(torch.isnan(text_features), torch.zeros_like(text_features), text_features)
        
        # 检查并修复特征范数为零的情况
        img_norm = torch.norm(image_features, p=2, dim=-1)
        txt_norm = torch.norm(text_features, p=2, dim=-1)
        
        if (img_norm == 0).any():
            print(f"Warning: image_features has zero norm, adding small random values")
            zero_mask = (img_norm == 0).unsqueeze(-1)
            random_fix = torch.randn_like(image_features) * 0.01
            image_features = torch.where(zero_mask, random_fix, image_features)
            
        if (txt_norm == 0).any():
            print(f"Warning: text_features has zero norm, adding small random values")
            zero_mask = (txt_norm == 0).unsqueeze(-1)
            random_fix = torch.randn_like(text_features) * 0.01
            text_features = torch.where(zero_mask, random_fix, text_features)
        
        # 归一化（添加小值避免除零）
        image_features = F.normalize(image_features, p=2, dim=-1, eps=1e-8)
        text_features = F.normalize(text_features, p=2, dim=-1, eps=1e-8)
        
        # 计算相似度
        similarity = torch.matmul(image_features, text_features.T) / temperature
        
        # 最终NaN检查和修复
        if torch.isnan(similarity).any():
            print(f"Warning: similarity contains NaN after computation, fixing...")
            similarity = torch.where(torch.isnan(similarity), torch.zeros_like(similarity), similarity)
        
        # 确保返回的tensor与numpy兼容（BFloat16转换）
        if similarity.dtype == torch.bfloat16:
            similarity = similarity.float()
        
        return similarity


# 注册模型配置和类
AutoConfig.register("clip_qwen2_vl", ClipQwen2VLConfig)
AutoModelForCausalLM.register(ClipQwen2VLConfig, ClipQwen2VLForConditionalGeneration)