# 改进的CLIP风格Qwen2.5-VL训练代码

## 项目概述

本项目实现了基于大模型(LLM)响应的多模态特征提取和对比学习训练，参考了LLaVA-Med的架构，通过在输入序列末尾添加特殊标记，从LLM的response中提取对应模态特征，并通过MLP和池化构建全局和局部特征，实现交叉损失优化。

## 主要特性

### 1. 基于LLM Response的特征提取
- 在图像和文本输入序列末尾添加特殊标记 (`<IMG_CLS_0>`, `<IMG_CLS_1>`, ..., `<TXT_CLS_0>`, `<TXT_CLS_1>`, ...)
- 从LLM的隐藏状态中提取特殊标记对应的特征作为全局特征
- 计算序列中非特殊标记部分的平均作为局部特征

### 2. 改进的模型架构
- **ModalityMLP**: 通用的模态特征映射层，支持不同的激活函数和dropout
- **FeaturePooling**: 灵活的特征池化策略 (mean, max, cls)
- **CrossModalAttention**: 跨模态注意力机制，增强特征表示
- **ImprovedCLIPLoss**: 多层次对比学习损失，包含全局损失、局部损失和交叉模态损失

### 3. 数据处理改进
- **特殊标记注入**: 自动为tokenizer添加模态特殊标记
- **智能数据整理**: 处理变长序列和批次数据
- **多模态数据加载**: 支持图像和文本的联合处理

## 文件结构

```
src/train/
├── clip_train_improved.py       # 改进的训练主脚本
├── clip_modeling_improved.py    # 改进的模型实现
└── clip_train_sft.py           # 原始训练脚本 (参考)

scripts/
├── train_clip_improved.sh       # 改进的训练启动脚本
└── clip_train_script.sh        # 原始训练脚本 (参考)

test_offline_clip.py             # 离线测试脚本
```

## 训练配置

### 模型参数
- `img_cls_token_count`: 图像模态特殊标记数量 (默认: 4)
- `txt_cls_token_count`: 文本模态特殊标记数量 (默认: 4)
- `hidden_dim`: MLP隐藏层维度 (默认: 1024)
- `output_dim`: 输出特征维度 (默认: 512)
- `temperature`: InfoNCE损失温度参数 (默认: 0.05)

### 损失函数配置
- `use_local_loss`: 是否使用局部特征损失 (默认: True)
- `use_cross_attention_loss`: 是否使用交叉注意力损失 (默认: True)
- `pooling_strategy`: 池化策略 (默认: "mean")

### 训练参数
- `clip_training_ratio`: CLIP训练比例 (默认: 0.8)
- `feature_extraction_layer`: 特征提取层 (默认: -1, 最后一层)
- `learning_rate`: 学习率 (默认: 2e-5)
- `batch_size`: 批次大小 (默认: 4)

## 使用方法

### 1. 环境准备

确保已安装必要的依赖：
```bash
# 激活conda环境
conda activate qwen_vl

# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 数据准备

训练数据格式 (JSON):
```json
[
  {
    "id": 0,
    "image": "path/to/image.jpg",
    "conversations": [
      {"from": "human", "value": "What is in this image? <image>"},
      {"from": "gpt", "value": "This is a medical X-ray showing..."}
    ]
  }
]
```

### 3. 配置修改

编辑 `scripts/train_clip_improved.sh`:
```bash
# 模型和数据路径
MODEL_NAME_OR_PATH="/path/to/your/model"  # 本地模型路径
DATA_PATH="/path/to/your/train_data.json"  # 训练数据
IMAGE_FOLDER="/path/to/your/images"  # 图像文件夹
```

### 4. 运行训练

```bash
# 给脚本执行权限
chmod +x scripts/train_clip_improved.sh

# 启动训练
./scripts/train_clip_improved.sh
```

### 5. 离线测试

在没有实际模型的情况下测试代码逻辑：
```bash
python test_offline_clip.py
```

## 核心算法流程

### 1. 特殊标记注入
```
原始输入: "What is in this image? <image>"
注入后:   "What is in this image? <image> <IMG_CLS_0> <IMG_CLS_1> <IMG_CLS_2> <IMG_CLS_3>"

原始回答: "This is a chest X-ray."
注入后:   "This is a chest X-ray. <TXT_CLS_0> <TXT_CLS_1> <TXT_CLS_2> <TXT_CLS_3>"
```

### 2. 特征提取
```python
# 从LLM隐藏状态中提取特征
img_global_features = extract_special_token_features(img_hidden_states, img_cls_token_ids)
txt_global_features = extract_special_token_features(txt_hidden_states, txt_cls_token_ids)

# 计算局部特征 (排除特殊标记的序列平均)
img_local_features = compute_local_features(img_hidden_states, exclude_special_tokens=True)
txt_local_features = compute_local_features(txt_hidden_states, exclude_special_tokens=True)
```

### 3. 对比学习损失
```python
# 多层次损失计算
global_loss = infonce_loss(img_global_features, txt_global_features)
local_loss = infonce_loss(img_local_features, txt_local_features)
cross_loss = cross_modal_loss(img_global_features, txt_local_features) + 
             cross_modal_loss(img_local_features, txt_global_features)

total_loss = 0.4 * global_loss + 0.3 * local_loss + 0.3 * cross_loss
```

## 测试结果

离线测试全部通过：
- ✓ 模块导入: 通过
- ✓ 配置创建: 通过  
- ✓ 模型组件: 通过
- ✓ 数据处理: 通过
- ✓ 特殊标记: 通过

## 主要改进点

1. **基于LLM Response的特征提取**: 不再依赖固定的视觉编码器，而是利用LLM的表示能力
2. **特殊标记机制**: 通过添加模态特殊标记来标识和提取对应的特征
3. **多层次对比学习**: 结合全局特征、局部特征和交叉模态特征
4. **灵活的模型架构**: 模块化设计，易于扩展和修改
5. **完善的数据处理**: 支持变长序列和复杂的多模态数据

## 注意事项

1. **模型路径**: 确保 `MODEL_NAME_OR_PATH` 指向有效的本地模型
2. **数据格式**: 训练数据必须包含 `image` 字段和 `conversations` 字段
3. **GPU内存**: 建议使用至少24GB显存的GPU，或调整批次大小
4. **特殊标记数量**: 可根据任务需求调整特殊标记数量
5. **损失权重**: 可根据验证效果调整不同损失的权重

## 故障排除

1. **导入错误**: 检查PYTHONPATH设置和模块路径
2. **CUDA错误**: 检查GPU可用性和CUDA版本兼容性
3. **内存不足**: 减少批次大小或使用梯度累积
4. **数据加载错误**: 检查图像路径和数据格式

## 下一步计划

1. 添加评估脚本
2. 支持多GPU训练
3. 添加模型量化支持
4. 优化内存使用
5. 添加更多的数据增强策略