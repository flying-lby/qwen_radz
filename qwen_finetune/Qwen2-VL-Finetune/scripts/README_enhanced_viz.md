# 增强Grounding可视化使用指南

本指南介绍如何使用新增的GT vs Prediction对比可视化功能。

## 📁 可用脚本

### 1. `enhanced_grounding_viz.sh` - 专用增强可视化脚本
专门用于生成高质量的GT vs Prediction对比可视化。

### 2. `grounding_eval_script.sh` - 集成的评估脚本  
在原有评估功能基础上，可选启用增强可视化。

## 🚀 快速开始

### 方法1：使用专用可视化脚本（推荐）

```bash
# 进入脚本目录
cd /home/lby/iclr2026/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/scripts

# 基础使用（默认balanced策略，15个样本）
./enhanced_grounding_viz.sh

# 选择高质量样本（20个样本）
./enhanced_grounding_viz.sh quality 20

# 选择多样化样本（12个样本）  
./enhanced_grounding_viz.sh diverse 12

# 选择挑战性样本（10个样本）
./enhanced_grounding_viz.sh challenging 10
```

### 方法2：使用集成评估脚本

```bash
# 正常评估（不生成增强可视化）
./grounding_eval_script.sh

# 启用增强可视化（默认配置）
ENABLE_ENHANCED_VIZ=true ./grounding_eval_script.sh

# 自定义增强可视化配置
VIZ_STRATEGY=quality NUM_VIZ_SAMPLES=20 ENABLE_ENHANCED_VIZ=true ./grounding_eval_script.sh
```

## 🎨 可视化策略说明

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| `balanced` | 平衡选择正负样本，优先质量高的 | 全面展示模型性能 |
| `quality` | 选择Dice分数最高的样本 | 展示模型最佳表现 |
| `diverse` | 在不同质量区间选择样本 | 展示模型在各种情况下的表现 |
| `challenging` | 选择模型表现困难的样本 | 分析模型弱点和改进空间 |

## 📂 输出文件结构

### 专用可视化脚本输出
```
enhanced_visualizations/
└── gt_vs_prediction_[策略]_[样本数]samples_[时间戳]/
    ├── rsna/
    │   ├── evaluation_results.json
    │   └── gt_vs_prediction/
    │       ├── sample_XXX.png
    │       └── samples_grid.png
    ├── siim/
    │   ├── evaluation_results.json  
    │   └── gt_vs_prediction/
    │       ├── sample_YYY.png
    │       └── samples_grid.png
    └── view_results.sh  # 快速查看脚本
```

### 集成评估脚本输出
```
results/
├── rsna_grounding/
│   ├── rsna_grounding_results_[时间戳].json
│   └── visualizations/
│       └── gt_vs_prediction/  # 仅在启用增强可视化时存在
├── siim_grounding/
│   ├── siim_grounding_results_[时间戳].json
│   └── visualizations/
│       └── gt_vs_prediction/  # 仅在启用增强可视化时存在
```

## 🖼️ 可视化特性

### GT vs Prediction对比图
- **左侧**: Ground Truth（真实标注）+ 红色高亮病理区域
- **右侧**: Prediction（模型预测）+ 红色高亮病理区域  
- **标题**: 显示样本ID、Dice分数、IoU分数
- **分辨率**: 300 DPI高质量输出

### 网格展示图
- 将所有选定样本排列成整齐的网格
- 3列布局，自动计算行数
- 便于快速对比多个样本

## 🔧 配置参数

### 环境变量配置
```bash
# 启用/禁用增强可视化
export ENABLE_ENHANCED_VIZ=true

# 设置可视化策略
export VIZ_STRATEGY=quality

# 设置样本数量
export NUM_VIZ_SAMPLES=20
```

### 脚本参数
```bash
# enhanced_grounding_viz.sh
./enhanced_grounding_viz.sh [策略] [样本数]

# 示例
./enhanced_grounding_viz.sh quality 20
./enhanced_grounding_viz.sh diverse 15
```

## 💡 使用建议

### 快速预览
1. 首先使用 `balanced` 策略生成15个样本，获得全面概览
2. 查看网格图，快速了解模型整体表现

### 深入分析  
1. 使用 `quality` 策略查看模型最佳表现
2. 使用 `challenging` 策略分析模型弱点
3. 使用 `diverse` 策略了解模型在不同质量区间的表现

### 论文展示
1. 使用 `quality` 策略生成高质量样本用于论文
2. 调整样本数量以符合论文版面要求
3. 使用高分辨率输出保证打印质量

## 🚨 注意事项

1. **依赖检查**: 确保安装了必要的Python库（matplotlib, opencv-python, PIL）
2. **GPU内存**: 大批量样本可能需要较多GPU内存
3. **存储空间**: 高分辨率图像会占用较多磁盘空间
4. **数据路径**: 确保图像数据路径正确配置

## 🔍 快速查看结果

### 自动生成的查看脚本
专用可视化脚本会自动生成 `view_results.sh`：
```bash
# 运行自动查看脚本
./enhanced_visualizations/gt_vs_prediction_*/view_results.sh
```

### 手动查看
```bash
# 查看网格图
eog enhanced_visualizations/*/rsna/gt_vs_prediction/samples_grid.png
eog enhanced_visualizations/*/siim/gt_vs_prediction/samples_grid.png

# 查看单个样本
ls enhanced_visualizations/*/rsna/gt_vs_prediction/sample_*.png
```

## 📞 问题反馈

如果遇到问题，请检查：
1. Python环境和依赖库是否正确安装
2. 模型路径和数据路径是否正确
3. 磁盘空间是否充足
4. GPU内存是否足够

---

**享受高质量的医学图像可视化体验！** 🎉

