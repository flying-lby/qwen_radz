# CLIP风格Qwen2.5-VL实现

基于LLaVA-Med项目的CLIP架构，在Qwen2.5-VL上实现图像和文本的对比学习训练。

## 项目特点

本实现参考了LLaVA-Med项目中的CLIP风格架构，主要特点包括：

1. **特殊标记机制**：使用`<Imgcls0>`、`<Imgcls1>`等图像分类标记和`<Txtcls0>`、`<Txtcls1>`等文本分类标记
2. **多层MLP映射**：将图像和文本特征映射到相同的嵌入空间，支持多种MLP架构
3. **InfoNCE对比损失**：实现双向对比学习损失，支持全局和局部特征对比
4. **交叉注意力机制**：增强图像和文本特征的交互
5. **混合训练模式**：支持常规语言模型训练和CLIP对比学习的混合训练

## 文件结构

```
src/
├── train/
│   ├── clip_modeling_qwen2_5_vl.py  # CLIP风格的Qwen2.5-VL模型实现
│   └── clip_train_sft.py            # CLIP训练脚本
├── eval/
│   └── clip_eval.py                 # CLIP评估脚本
└── scripts/
    └── clip_train_script.sh         # 训练启动脚本
```

## 核心组件

### 1. ClipQwen2VLForConditionalGeneration

扩展的Qwen2.5-VL模型，支持：
- 特殊分类标记的嵌入和特征提取
- 多种类型的MLP映射层
- InfoNCE对比学习损失
- 图像-文本特征检索接口

### 2. 特殊标记系统

- **图像标记**：`<Imgcls0>`, `<Imgcls1>`, ..., `<Imgcls{N-1}>`
- **文本标记**：`<Txtcls0>`, `<Txtcls1>`, ..., `<Txtcls{M-1}>`

这些标记会被添加到输入序列的末尾，用于提取全局分类特征。

### 3. MLP映射层

支持6种不同的MLP架构类型：
1. **Type 1**: LayerNorm + Linear + GELU + Linear
2. **Type 2**: LayerNorm + Linear + ReLU + Linear  
3. **Type 3**: LayerNorm + 3层深度网络
4. **Type 4**: 单层Linear
5. **Type 5**: LayerNorm + Linear
6. **Type 6**: LayerNorm + Dropout + Linear

### 4. InfoNCE损失函数

实现标准的InfoNCE对比学习损失：
- 双向损失：图像到文本 + 文本到图像
- 支持全局和局部特征对比
- 可配置的温度参数

## 使用方法

### 1. 训练

#### 基本训练命令：

```bash
# 编辑scripts/clip_train_script.sh中的路径配置
vim scripts/clip_train_script.sh

# 运行训练
bash scripts/clip_train_script.sh
```

#### 主要训练参数：

```bash
# CLIP特定参数
--Imgcls_count 4                    # 图像分类标记数量
--Txtcls_count 4                    # 文本分类标记数量  
--hidden_dim 1024                   # MLP隐藏层维度
--output_dim 512                    # 输出特征维度
--temperature 0.05                  # InfoNCE温度参数
--clip_training_ratio 0.5           # CLIP训练比例 (0.0-1.0)
--img_mlp_type 1                    # 图像MLP类型
--txt_mlp_type 1                    # 文本MLP类型
--use_local_loss False              # 是否使用局部对比损失
--use_ca_loss True                  # 是否使用交叉注意力损失
```

### 2. 评估

#### 检索任务评估：

```bash
python src/eval/clip_eval.py \
    --model_path /path/to/trained/model \
    --data_path /path/to/eval/data.json \
    --image_folder /path/to/images \
    --task_type "retrieval" \
    --batch_size 8 \
    --output_path "eval_results.json"
```

#### 分类任务评估：

```bash
python src/eval/clip_eval.py \
    --model_path /path/to/trained/model \
    --data_path /path/to/eval/data.json \
    --image_folder /path/to/images \
    --task_type "classification" \
    --class_texts_path /path/to/class_texts.json \
    --batch_size 8 \
    --output_path "eval_results.json"
```

### 3. 特征提取

```python
from src.train.clip_modeling_qwen2_5_vl import ClipQwen2VLForConditionalGeneration
from transformers import AutoTokenizer

# 加载模型
model = ClipQwen2VLForConditionalGeneration.from_pretrained("/path/to/model")
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")

# 提取图像特征
image_features = model.extract_features(
    input_ids=image_input_ids,
    attention_mask=image_attention_mask,
    pixel_values=pixel_values
)

# 提取文本特征  
text_features = model.extract_features(
    input_ids=text_input_ids,
    attention_mask=text_attention_mask
)

# 计算相似度
similarity = model.compute_similarity(
    image_features["global_features"],
    text_features["global_features"]
)
```

## 数据格式

### 训练数据格式

训练数据应为JSON格式，包含图像和文本对：

```json
[
    {
        "conversations": [
            {
                "from": "human", 
                "value": "<image>\nDescribe this image."
            },
            {
                "from": "gpt",
                "value": "This is a medical X-ray showing..."
            }
        ],
        "image": "image_001.jpg"
    },
    {
        "conversations": [
            {
                "from": "human",
                "value": "What is pneumonia?"
            },
            {
                "from": "gpt", 
                "value": "Pneumonia is an infection..."
            }
        ]
    }
]
```

### 评估数据格式

#### 检索任务：
```json
[
    {"image": "img1.jpg", "caption": "A chest X-ray showing pneumonia"},
    {"image": "img2.jpg", "caption": "Normal chest X-ray"},
    {"text": "Chest X-ray with pneumonia findings"},
    {"text": "Normal lung appearance"}
]
```

#### 分类任务：
```json
[
    {"image": "img1.jpg", "text": "Medical image", "label": 0},
    {"image": "img2.jpg", "text": "Chest X-ray", "label": 1}
]
```

## 配置参数详解

### 模型架构参数

- `Imgcls_count`: 图像分类标记数量，用于提取图像全局特征
- `Txtcls_count`: 文本分类标记数量，用于提取文本全局特征
- `hidden_dim`: MLP隐藏层维度
- `output_dim`: 最终特征输出维度
- `feature_layer`: 从第几层提取特征 (1=最后一层)

### 训练参数

- `temperature`: InfoNCE损失的温度参数，控制对比学习的硬度
- `clip_training_ratio`: CLIP训练占总训练的比例
- `loss_threshold`: 不同损失项之间的权重阈值
- `use_local_loss`: 是否使用局部特征的对比损失
- `use_ca_loss`: 是否使用交叉注意力损失

### MLP类型选择

建议的MLP类型组合：
- **轻量级**: img_mlp_type=4, txt_mlp_type=4 (单层Linear)
- **标准**: img_mlp_type=1, txt_mlp_type=1 (LayerNorm + GELU)
- **鲁棒**: img_mlp_type=6, txt_mlp_type=6 (包含Dropout)

## 实验建议

### 1. 超参数调优

关键超参数及建议范围：
- `temperature`: 0.01-0.1 (较小值使对比更加尖锐)
- `learning_rate`: 1e-5 to 5e-5
- `clip_training_ratio`: 0.3-0.7 (平衡对比学习和语言建模)
- `output_dim`: 256-1024 (根据下游任务调整)

### 2. 训练策略

1. **预热阶段**: 先用较低的clip_training_ratio (0.2-0.3)训练几个epoch
2. **主训练**: 逐步提高clip_training_ratio到目标值
3. **微调阶段**: 降低学习率，专注优化对比学习部分

### 3. 评估指标

- **检索任务**: Recall@1, Recall@5, Recall@10
- **分类任务**: Top-1 Accuracy, Top-5 Accuracy
- **对比学习**: 特征空间的聚类质量

## 注意事项

1. **内存使用**: CLIP训练需要同时处理图像和文本，内存占用较大
2. **批次大小**: 建议根据GPU内存调整batch_size和gradient_accumulation_steps
3. **特殊标记**: 确保训练数据包含足够的<Imgcls>和<Txtcls>标记
4. **数据平衡**: 保持图像-文本对的数量平衡，避免模态偏差

## 引用

本实现基于以下工作：
- LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day
- Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
- Learning Transferable Visual Representations with Contrastive Learning (InfoNCE)

## 问题排查

### 常见问题

1. **特殊标记未找到**: 确保模型加载时正确添加了特殊标记
2. **显存不足**: 减少batch_size或启用gradient_checkpointing
3. **收敛慢**: 检查temperature参数是否合适，尝试调整学习率
4. **特征提取失败**: 确认输入格式正确，特别是pixel_values的维度

### 调试建议

1. 使用小数据集验证pipeline的正确性
2. 监控训练过程中的损失变化
3. 定期检查提取的特征向量是否合理
4. 使用TensorBoard可视化训练过程

---

更多技术细节请参考源代码中的注释和docstring。