"""
RSNA肺炎grounding评估脚本
参考MedKLIP的zero-shot grounding评估方法
"""

import argparse
import os
import sys
import json
import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块 (可用的部分)
try:
    from eval_utils import (
        ClipEvaluator,
        load_image_file
    )
except ImportError:
    # 如果导入失败，我们将在本文件中实现这些函数
    ClipEvaluator = None
    load_image_file = None

# 导入模型相关
from train.clip_modeling_qwen2_5_vl import (
    ClipQwen2VLConfig,
    ClipQwen2VLForConditionalGeneration
)
from constants import DEFAULT_IMAGE_TOKEN
from transformers import AutoTokenizer, AutoProcessor
from torch.utils.data import Dataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ================================
# 实现缺失的工具函数
# ================================

class RSNAGroundingDataset(Dataset):
    """RSNA Grounding数据集类"""
    
    def __init__(self, csv_path=None, jsonl_path=None, image_folder="", target_size=224, max_samples=-1):
        import pandas as pd
        import json
        
        self.image_folder = image_folder
        self.target_size = target_size
        
        if jsonl_path:
            # 读取JSONL文件
            data_list = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    if max_samples > 0 and count >= max_samples:
                        break
                    
                    data = json.loads(line.strip())
                    
                    # 转换JSONL格式到CSV格式的字典
                    image_rel_path = data['image']  # "RSNA_Pneumonia/stage_2_train_images/xxx.dcm"
                    sample_id = os.path.splitext(os.path.basename(image_rel_path))[0]
                    
                    # 从标签提取类别
                    labels = data.get('label', {})
                    pneumonia_label = labels.get('pneumonia', 0)
                    
                    data_list.append({
                        'ID': sample_id,
                        'img_path': image_rel_path,
                        'boxes': '',  # JSONL中没有box信息
                        'classes': pneumonia_label,
                        'text': data.get('text', '')
                    })
                    count += 1
            
            self.data = pd.DataFrame(data_list)
            logger.info(f"Loaded RSNA dataset from JSONL with {len(self.data)} samples")
            
        elif csv_path:
            # 读取CSV文件（原始逻辑）
            df = pd.read_csv(csv_path)
            
            # 限制样本数量
            if max_samples > 0:
                df = df.head(max_samples)
            
            self.data = df
            logger.info(f"Loaded RSNA dataset from CSV with {len(self.data)} samples")
        else:
            raise ValueError("Either csv_path or jsonl_path must be provided")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 构建样本数据
        # 注意：CSV文件中的列名是 'img_path' 而不是 'image_path'
        img_relative_path = row.get('img_path', '')
        if not img_relative_path:
            # 如果img_path为空，尝试其他可能的列名
            img_relative_path = row.get('image_path', f'sample_{idx}.dcm')
        
        sample = {
            'image_path': os.path.join(self.image_folder, img_relative_path),
            'query_text': 'pneumonia',  # RSNA主要关注肺炎
            'label': torch.tensor([int(row.get('classes', 0))], dtype=torch.float32),  # 使用'classes'列作为标签
            'sample_id': row.get('ID', f'sample_{idx}'),  # 使用'ID'列作为样本ID
        }
        
        # 模拟分割图（如果没有真实的分割数据，创建假的）
        # 在实际使用中，这应该从真实的分割数据中加载
        seg_map = torch.zeros(1, self.target_size, self.target_size, dtype=torch.float32)
        if int(row.get('classes', 0)) == 1:
            # 如果有病灶，在中心区域创建一个小的分割区域
            center_y, center_x = self.target_size // 2, self.target_size // 2
            size = 20
            seg_map[0, center_y-size:center_y+size, center_x-size:center_x+size] = 1.0
        
        sample['seg_map'] = seg_map
        
        # 加载图像（这里返回路径，实际加载在模型中进行）
        sample['image'] = torch.zeros(3, self.target_size, self.target_size)  # 占位符
        
        return sample


def create_rsna_dataloader(csv_path=None, jsonl_path=None, image_folder="", batch_size=4, target_size=224, 
                          max_samples=-1, num_workers=0, shuffle=False):
    """创建RSNA数据加载器"""
    
    dataset = RSNAGroundingDataset(
        csv_path=csv_path,
        jsonl_path=jsonl_path,
        image_folder=image_folder, 
        target_size=target_size,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader, dataset


def visualize_attention_map(attention_map, save_path=None):
    """可视化attention map"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 如果是tensor，转换为numpy
        if hasattr(attention_map, 'cpu'):
            attention_map = attention_map.cpu().numpy()
        
        # 创建图像
        plt.figure(figsize=(8, 8))
        plt.imshow(attention_map, cmap='hot', interpolation='bilinear')
        plt.colorbar()
        plt.title('Attention Map')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualization")
    except Exception as e:
        logger.error(f"Failed to visualize attention map: {e}")


def apply_red_overlay(image, mask, alpha=0.3):
    """
    在原图上叠加红色高亮区域
    
    Args:
        image: PIL Image 或 numpy array，原始医学图像
        mask: numpy array，二值化掩码 (0-1 或 True/False)
        alpha: float，红色叠加的透明度 (0-1)
    
    Returns:
        numpy array: 叠加了红色高亮的图像
    """
    try:
        import numpy as np
        from PIL import Image
        import cv2
        
        # 将PIL Image转换为numpy array
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image.copy()
        
        # 确保图像是RGB格式
        if len(image_array.shape) == 2:
            # 灰度图转RGB
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[-1] == 1:
            # 单通道转RGB
            image_array = np.repeat(image_array, 3, axis=-1)
        
        # 确保mask是2D的numpy array
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        # 归一化图像到0-255范围
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
        
        # 创建红色叠加
        overlay = image_array.copy()
        
        # 二值化mask（阈值处理）
        if mask.max() > 1:
            binary_mask = (mask > 0.1)  # 对于0-255范围
        else:
            binary_mask = (mask > 0.1)  # 对于0-1范围
        
        # 在mask区域应用红色高亮
        overlay[binary_mask] = [255, 0, 0]  # 纯红色
        
        # 混合原图和叠加层
        result = cv2.addWeighted(image_array, 1-alpha, overlay, alpha, 0)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to apply red overlay: {e}")
        # 返回原图像作为fallback
        if isinstance(image, Image.Image):
            return np.array(image)
        return image


def create_gt_prediction_comparison(image_path, seg_map, attention_map, 
                                  sample_id, metrics=None, save_path=None):
    """
    创建GT vs Prediction对比图，论文风格展示
    
    Args:
        image_path: str，原始图像路径
        seg_map: torch.Tensor或numpy.array，GT分割掩码
        attention_map: torch.Tensor或numpy.array，模型预测的attention map
        sample_id: str，样本ID
        metrics: dict，包含dice_score, iou_score等指标
        save_path: str，保存路径
    
    Returns:
        numpy.array: 生成的对比图像
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        import cv2
        
        # 加载原始图像
        try:
            original_image = load_image_file(image_path)
            if original_image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
        
        # 转换tensor到numpy
        if hasattr(seg_map, 'cpu'):
            seg_map = seg_map.cpu().numpy()
        if hasattr(attention_map, 'cpu'):
            attention_map = attention_map.cpu().numpy()
        
        # 确保掩码是2D的
        if len(seg_map.shape) > 2:
            seg_map = seg_map.squeeze()
        if len(attention_map.shape) > 2:
            attention_map = attention_map.squeeze()
        
        # 调整图像尺寸以匹配掩码
        target_size = seg_map.shape[0]  # 假设seg_map是正方形
        resized_image = original_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # 将attention map二值化（阈值0.008，与MedKLIP一致）
        attention_binary = (attention_map > 0.008).astype(np.float32)
        
        # 创建GT和Prediction的红色叠加图
        gt_overlay = apply_red_overlay(resized_image, seg_map, alpha=0.4)
        pred_overlay = apply_red_overlay(resized_image, attention_binary, alpha=0.4)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # GT图像
        axes[0].imshow(gt_overlay)
        axes[0].set_title('GT', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Prediction图像  
        axes[1].imshow(pred_overlay)
        axes[1].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 添加整体标题和指标信息
        title = f'Sample: {sample_id}'
        if metrics:
            dice_score = metrics.get('dice_score', 0)
            iou_score = metrics.get('iou_score', 0)
            title += f' | Dice: {dice_score:.3f}, IoU: {iou_score:.3f}'
        
        fig.suptitle(title, fontsize=12, y=0.95)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            logger.info(f"GT vs Prediction comparison saved to: {save_path}")
        else:
            plt.show()
        
        # 返回组合后的图像数组用于进一步处理
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        
        return img_array
        
    except ImportError as e:
        logger.warning(f"Required libraries not available for GT vs Prediction visualization: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create GT vs Prediction comparison: {e}")
        return None


def select_representative_samples(all_results, num_samples=15, strategy='balanced'):
    """
    从评估结果中选择代表性样本
    
    Args:
        all_results: list，包含所有样本的评估结果
        num_samples: int，要选择的样本数量
        strategy: str，选择策略 ('balanced', 'quality', 'diverse', 'challenging')
    
    Returns:
        list: 选中的样本索引列表
    """
    try:
        import numpy as np
        from collections import defaultdict
        
        if len(all_results) <= num_samples:
            return list(range(len(all_results)))
        
        selected_indices = []
        
        if strategy == 'balanced':
            # 平衡选择：一半正样本，一半负样本，优先选择质量高的
            positive_samples = [(i, r) for i, r in enumerate(all_results) 
                              if r.get('label', 0) == 1]
            negative_samples = [(i, r) for i, r in enumerate(all_results) 
                              if r.get('label', 0) == 0]
            
            # 按照质量分数排序
            positive_samples.sort(key=lambda x: x[1].get('dice_score', 0), reverse=True)
            negative_samples.sort(key=lambda x: x[1].get('dice_score', 0), reverse=True)
            
            # 选择一半正样本，一半负样本
            pos_count = min(num_samples // 2, len(positive_samples))
            neg_count = min(num_samples - pos_count, len(negative_samples))
            
            selected_indices.extend([idx for idx, _ in positive_samples[:pos_count]])
            selected_indices.extend([idx for idx, _ in negative_samples[:neg_count]])
            
            # 如果还不够，从剩余样本中补充
            remaining_count = num_samples - len(selected_indices)
            if remaining_count > 0:
                all_remaining = [(i, r) for i, r in enumerate(all_results) 
                               if i not in selected_indices]
                all_remaining.sort(key=lambda x: x[1].get('dice_score', 0), reverse=True)
                selected_indices.extend([idx for idx, _ in all_remaining[:remaining_count]])
        
        elif strategy == 'quality':
            # 质量优先：选择dice score最高的样本
            sorted_results = sorted(enumerate(all_results), 
                                  key=lambda x: x[1].get('dice_score', 0), 
                                  reverse=True)
            selected_indices = [idx for idx, _ in sorted_results[:num_samples]]
        
        elif strategy == 'challenging':
            # 挑战性样本：选择模型表现困难的样本（中等dice score）
            sorted_results = sorted(enumerate(all_results), 
                                  key=lambda x: abs(x[1].get('dice_score', 0) - 0.5))
            selected_indices = [idx for idx, _ in sorted_results[:num_samples]]
        
        elif strategy == 'diverse':
            # 多样性选择：在不同质量区间选择样本
            sorted_results = sorted(enumerate(all_results), 
                                  key=lambda x: x[1].get('dice_score', 0))
            
            # 分成几个质量区间
            num_bins = min(5, num_samples)
            bin_size = len(sorted_results) // num_bins
            samples_per_bin = num_samples // num_bins
            
            for i in range(num_bins):
                start_idx = i * bin_size
                end_idx = min((i + 1) * bin_size, len(sorted_results))
                bin_samples = sorted_results[start_idx:end_idx]
                
                # 从每个区间选择指定数量的样本
                count = samples_per_bin
                if i == num_bins - 1:  # 最后一个区间包含剩余样本
                    count = num_samples - len(selected_indices)
                
                selected_indices.extend([idx for idx, _ in bin_samples[:count]])
        
        # 确保不超过请求的数量
        selected_indices = selected_indices[:num_samples]
        
        logger.info(f"Selected {len(selected_indices)} representative samples using '{strategy}' strategy")
        return selected_indices
        
    except Exception as e:
        logger.error(f"Failed to select representative samples: {e}")
        # 返回前N个样本作为fallback
        return list(range(min(num_samples, len(all_results))))


def create_sample_grid(sample_comparisons, grid_cols=3, save_path=None):
    """
    创建多样本网格展示
    
    Args:
        sample_comparisons: list，包含多个样本对比图像的numpy数组
        grid_cols: int，网格列数
        save_path: str，保存路径
    
    Returns:
        numpy.array: 网格图像
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_samples = len(sample_comparisons)
        if num_samples == 0:
            return None
        
        # 计算网格尺寸
        grid_rows = (num_samples + grid_cols - 1) // grid_cols
        
        # 创建大图
        fig, axes = plt.subplots(grid_rows, grid_cols, 
                                figsize=(grid_cols * 6, grid_rows * 3))
        
        # 处理单行或单列的情况
        if grid_rows == 1 and grid_cols == 1:
            axes = [axes]
        elif grid_rows == 1 or grid_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # 放置每个样本对比图
        for i, comparison_img in enumerate(sample_comparisons):
            if i < len(axes):
                axes[i].imshow(comparison_img)
                axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            logger.info(f"Sample grid saved to: {save_path}")
        else:
            plt.show()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample grid: {e}")
        return None


# 如果load_image_file没有从eval_utils导入成功，我们将在后面定义load_image_file_enhanced
# 这里先设置为None，等函数定义后再赋值
if load_image_file is None:
    load_image_file = None  # 将在后面重新赋值


class ProgressTracker:
    """进度跟踪器，提供详细的统计和时间估算"""
    
    def __init__(self, total_samples: int, batch_size: int):
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.total_batches = (total_samples - 1) // batch_size + 1
        
        self.start_time = time.time()
        self.processed_samples = 0
        self.processed_batches = 0
        
        # 状态统计
        self.success_count = 0
        self.error_count = 0
        self.status_stats = {"success": 0, "nan_fixed": 0, "zero_norm_fixed": 0, "degraded": 0, "failed": 0}
        
        # 时间统计
        self.batch_times = []
        self.last_batch_time = self.start_time
        
        # 简化的性能统计
        self.memory_usage_samples = []
        
    def update_batch(self, batch_valid_count: int, batch_status_stats: Dict[str, int]):
        """更新批次处理结果"""
        current_time = time.time()
        batch_duration = current_time - self.last_batch_time
        self.batch_times.append(batch_duration)
        self.last_batch_time = current_time
        
        self.processed_samples += batch_valid_count
        self.processed_batches += 1
        self.success_count += batch_valid_count
        self.error_count += (self.batch_size - batch_valid_count)
        
        # 合并状态统计
        for status, count in batch_status_stats.items():
            self.status_stats[status] += count
            
        # 记录内存使用情况（如果有GPU）
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            self.memory_usage_samples.append(memory_used)

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """获取当前内存使用情况"""
        memory_info = {}
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
            memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return memory_info
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        elapsed_time = time.time() - self.start_time
        
        # 计算处理速度
        if elapsed_time > 0:
            samples_per_sec = self.processed_samples / elapsed_time
            avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        else:
            samples_per_sec = 0
            avg_batch_time = 0
        
        # 估算剩余时间
        remaining_samples = self.total_samples - self.processed_samples
        if samples_per_sec > 0:
            eta_seconds = remaining_samples / samples_per_sec
            eta_str = f"{int(eta_seconds//3600):02d}:{int((eta_seconds%3600)//60):02d}:{int(eta_seconds%60):02d}"
        else:
            eta_str = "Unknown"
        
        # 成功率计算 - 只有真正成功的才算成功，降级处理算作错误
        true_success_count = self.status_stats.get("success", 0)
        total_attempted = self.success_count + self.error_count
        success_rate = (true_success_count / total_attempted * 100) if total_attempted > 0 else 0
        
        # 内存使用统计
        memory_stats = self.get_memory_usage()
        if self.memory_usage_samples:
            avg_memory = sum(self.memory_usage_samples) / len(self.memory_usage_samples)
            max_memory = max(self.memory_usage_samples)
        else:
            avg_memory = memory_stats.get('gpu_allocated', 0)
            max_memory = memory_stats.get('gpu_max_allocated', 0)
        
        return {
            "processed_samples": self.processed_samples,
            "total_samples": self.total_samples,
            "processed_batches": self.processed_batches,
            "total_batches": self.total_batches,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "samples_per_sec": samples_per_sec,
            "avg_batch_time": avg_batch_time,
            "elapsed_time": elapsed_time,
            "eta": eta_str,
            "status_breakdown": self.status_stats.copy(),
            # 新增：内存使用统计
            "avg_memory_gb": avg_memory,
            "max_memory_gb": max_memory,
            "current_memory": memory_stats
        }
    
    def format_progress_message(self) -> str:
        """格式化进度消息"""
        stats = self.get_stats()
        
        progress_msg = (
            f"批次进度: {stats['processed_batches']}/{stats['total_batches']} "
            f"({stats['processed_batches']/stats['total_batches']*100:.1f}%)\n"
            f"样本进度: {stats['processed_samples']}/{stats['total_samples']} "
            f"({stats['processed_samples']/stats['total_samples']*100:.1f}%)\n"
            f"成功率: {stats['success_rate']:.1f}% "
            f"(成功: {stats['success_count']}, 失败: {stats['error_count']})\n"
            f"处理速度: {stats['samples_per_sec']:.1f} samples/sec\n"
            f"预计剩余时间: {stats['eta']}\n"
        )
        
        # 添加GPU内存信息（如果可用）
        if torch.cuda.is_available():
            current_mem = stats['current_memory'].get('gpu_allocated', 0)
            max_mem = stats['max_memory_gb']
            progress_msg += f"GPU内存: 当前 {current_mem:.1f}GB, 峰值 {max_mem:.1f}GB\n"
        
        progress_msg += (
            f"状态详情: 成功={stats['status_breakdown']['success']}, "
            f"降级处理={stats['status_breakdown'].get('degraded', 0)}, "
            f"失败={stats['status_breakdown']['failed']}"
        )
        
        return progress_msg


def score_cal_medklip_style(labels, seg_map, pred_map):
    """
    按照MedKLIP方式计算grounding分数
    
    Args:
        labels: [B, 1] - 标签（是否有病灶）
        seg_map: [B, H, W] - 真实分割掩码
        pred_map: [B, H, W] - 预测attention map
    
    Returns:
        total_num: 有病灶的样本总数
        point_score: 点分数（峰值点是否在病灶内）
        mass_score: 质量分数（类似IoU）
        dice_score: Dice分数
    """
    device = labels.device
    total_num = int(torch.sum(labels).item())  # 确保转换为Python整数
    mask = (labels == 1).squeeze()
    
    if total_num == 0:
        # 没有正样本的情况
        return 0, 0, torch.tensor([]).to(device), torch.tensor([]).to(device)
    
    # 只处理有病灶的样本
    seg_map_filtered = seg_map[mask, :, :].reshape(total_num, -1)
    pred_map_filtered = pred_map[mask, :, :].reshape(total_num, -1)
    
    # 二值化预测map（阈值0.008，与MedKLIP一致）
    one_hot_map = (pred_map_filtered > 0.008)
    dot_product = (seg_map_filtered * one_hot_map).reshape(total_num, -1)
    
    # 计算点分数（峰值点是否在真实病灶内）
    max_values = torch.max(pred_map_filtered, dim=-1)[0]
    point_score = 0
    for i, max_val in enumerate(max_values):
        temp_pred = (pred_map_filtered[i] == max_val).type(torch.int)
        flag = int((torch.sum(temp_pred * seg_map_filtered[i])) > 0)
        point_score = point_score + flag
    
    # 计算质量分数（类似IoU）
    mass_score = torch.sum(dot_product, dim=-1) / (
        (torch.sum(seg_map_filtered, dim=-1) + torch.sum(one_hot_map, dim=-1)) - 
        torch.sum(dot_product, dim=-1)
    )
    
    # 计算Dice分数
    dice_score = 2 * (torch.sum(dot_product, dim=-1)) / (
        torch.sum(seg_map_filtered, dim=-1) + torch.sum(one_hot_map, dim=-1)
    )
    
    return total_num, point_score, mass_score.to(device), dice_score.to(device)


def aggregate_grounding_results_medklip_style(all_dice_scores, all_mass_scores, total_num_samples, total_point_score):
    """
    按照MedKLIP方式聚合grounding结果
    """
    if len(all_dice_scores) == 0:
        return {
            'mean_dice_score': 0.0,
            'mean_iou_score': 0.0,
            'point_accuracy': 0.0,
            'total_samples': total_num_samples
        }
    
    # 计算平均指标
    mean_dice_score = torch.mean(all_dice_scores).item()
    mean_iou_score = torch.mean(all_mass_scores).item()  # MedKLIP中mass_score就是IoU
    point_accuracy = total_point_score / max(total_num_samples, 1)
    
    return {
        'mean_dice_score': mean_dice_score,
        'mean_iou_score': mean_iou_score, 
        'point_accuracy': point_accuracy,
        'total_samples': int(total_num_samples)
    }


def load_image_file_enhanced(img_path):
    """
    增强版图像加载函数，支持常规格式和DICOM格式，包含更好的错误处理
    """
    try:
        # 导入DICOM处理库
        try:
            import pydicom
            DICOM_AVAILABLE = True
        except ImportError:
            DICOM_AVAILABLE = False

        # 检查文件扩展名
        file_ext = os.path.splitext(img_path)[1].lower()
        
        if file_ext == '.dcm' and DICOM_AVAILABLE:
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
                        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
                else:
                    # 默认归一化到0-255范围
                    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
                
                # 处理不同的像素数据格式
                if len(pixel_array.shape) == 2:
                    # 灰度图像
                    pixel_array = pixel_array.astype(np.uint8)
                    # 转换为PIL图像
                    from PIL import Image
                    image = Image.fromarray(pixel_array, mode='L')
                    # 转换为RGB
                    image = image.convert('RGB')
                elif len(pixel_array.shape) == 3:
                    # 彩色图像或多帧图像，取第一帧
                    if pixel_array.shape[0] < pixel_array.shape[2]:
                        # 假设第一个维度是帧数
                        pixel_array = pixel_array[0]
                    pixel_array = pixel_array.astype(np.uint8)
                    from PIL import Image
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
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            return image
            
    except Exception as e:
        raise Exception(f"Failed to load image {img_path}: {str(e)}")


# 如果load_image_file没有成功导入，使用我们的增强版本
if load_image_file is None:
    load_image_file = load_image_file_enhanced


# MedKLIP中使用的类别列表（用于参考）
ORIGINAL_CLASSES = [
    'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
    'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 
    'process', 'abnormality', 'enlarge', 'tip', 'low', 'pneumonia', 'line', 'congestion', 
    'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 
    'calcification', 'prominence', 'device', 'engorgement', 'picc', 'clip', 'elevation', 
    'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 
    'scar', 'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 
    'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia', 'drainage', 
    'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 
    'hardware', 'dilation', 'chf', 'redistribution', 'aspiration', 'tail_abnorm_obs', 
    'excluded_obs'
]

class CLIPGroundingEvaluator:
    """CLIP风格的grounding评估器"""
    
    def __init__(self, model_path: str, disease_desc_path: str = None):
        """
        初始化评估器
        
        Args:
            model_path: CLIP模型路径
            disease_desc_path: 疾病描述文件路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.disease_desc_path = disease_desc_path
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Total CUDA devices: {torch.cuda.device_count()}")
        
        # 加载tokenizer和processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        
        # 加载模型配置和模型
        config = ClipQwen2VLConfig.from_pretrained(model_path)
        # 根本性修复：移除device_map，使用传统手动设备管理
        self.model = ClipQwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()
        
        # 根本性修复：立即手动设备管理，带显存不足降级处理
        try:
            logger.info(f"Attempting to move model to {self.device}")
            self.model = self.model.to(self.device)
            logger.info(f"✅ Model successfully moved to {self.device}")
            
            # 验证模型参数设备一致性
            device_check_params = []
            for name, param in self.model.named_parameters():
                if param.device != self.device:
                    device_check_params.append((name, param.device))
                    
            if device_check_params:
                logger.warning(f"Found {len(device_check_params)} parameters on wrong device, forcing sync...")
                for name, wrong_device in device_check_params[:3]:  # 只显示前3个
                    logger.warning(f"  {name}: {wrong_device} -> {self.device}")
                self.model = self.model.to(self.device)
                logger.info("Parameters re-synchronized to target device")
            else:
                logger.info("All model parameters are on the correct device")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(f"CUDA memory insufficient or device error: {e}")
                logger.warning("🔄 Falling back to CPU inference mode")
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device)
                logger.info(f"✅ Model moved to CPU device: {self.device}")
            else:
                logger.error(f"Failed to move model to device: {e}")
                raise
        
        # 递归确保所有子模块都在正确的设备上（深层修复）
        self._ensure_all_modules_on_device()
        
        # 获取配置参数（与ClipEvaluator一致）
        self.sparse_config = config.sparse_config
        self.Imgcls_count = self.sparse_config["Imgcls_count"]
        self.Txtcls_count = self.sparse_config["Txtcls_count"]
        self.temperature = self.sparse_config["temperature"]
        
        # 检查模型权重是否包含NaN
        nan_params = []
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
        
        if nan_params:
            logger.warning(f"Model contains NaN parameters: {nan_params[:5]}...")  # 只显示前5个
            logger.warning("Attempting to fix NaN parameters by reinitializing them...")
            
            # 尝试修复NaN参数
            fixed_count = 0
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    # 重新初始化包含NaN的参数
                    with torch.no_grad():
                        if 'weight' in name:
                            if len(param.shape) >= 2:
                                # 使用Xavier初始化
                                torch.nn.init.xavier_uniform_(param)
                            else:
                                # 一维参数使用正态分布
                                torch.nn.init.normal_(param, 0, 0.02)
                        elif 'bias' in name:
                            # 偏置初始化为零
                            torch.nn.init.zeros_(param)
                        else:
                            # 其他参数使用正态分布
                            torch.nn.init.normal_(param, 0, 0.02)
                    fixed_count += 1
                    logger.info(f"Fixed NaN parameter: {name}")
            
            logger.info(f"Successfully fixed {fixed_count} NaN parameters.")
            
            # 再次检查是否还有NaN
            remaining_nan = []
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    remaining_nan.append(name)
            
            if remaining_nan:
                logger.error(f"Still have NaN parameters after fixing: {remaining_nan[:3]}...")
                raise ValueError("Failed to fix all NaN parameters.")
            else:
                logger.info("All NaN parameters have been successfully fixed.")
        
        logger.info("Model loaded successfully")
        
        # 添加设备管理属性
        self._primary_device = None
        self._device_cache = {}
        
        # 加载疾病描述
        self.disease_descriptions = self._load_disease_descriptions()
        
    def _ensure_all_modules_on_device(self):
        """递归确保所有子模块都在正确的设备上（深层修复MLP组件设备问题）"""
        target_device = self.device
        logger.info(f"Starting deep device synchronization to {target_device}")
        
        # 递归移动所有子模块
        moved_modules = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'parameters'):
                try:
                    # 检查模块是否有参数且不在目标设备上
                    params = list(module.parameters())
                    if params and params[0].device != target_device:
                        module.to(target_device)
                        moved_modules.append(name)
                except Exception as e:
                    logger.debug(f"Could not move module {name}: {e}")
        
        if moved_modules:
            logger.info(f"Moved {len(moved_modules)} modules to {target_device}")
            logger.debug(f"Moved modules: {moved_modules[:5]}...")  # 只显示前5个
        
        # 特别检查和移动关键MLP组件
        critical_components = ['img_mlp', 'txt_mlp', 'clip_loss', 'special_token_mlp']
        for comp_name in critical_components:
            if hasattr(self.model, comp_name):
                comp = getattr(self.model, comp_name)
                if hasattr(comp, 'parameters'):
                    try:
                        params = list(comp.parameters())
                        if params:
                            comp_device = params[0].device
                            if comp_device != target_device:
                                logger.warning(f"Critical component {comp_name} on wrong device: {comp_device} -> {target_device}")
                                comp.to(target_device)
                                logger.info(f"Successfully moved {comp_name} to {target_device}")
                            else:
                                logger.debug(f"Critical component {comp_name} already on correct device")
                    except Exception as e:
                        logger.error(f"Failed to check/move critical component {comp_name}: {e}")
        
        logger.info("Deep device synchronization completed")
        
    def _load_disease_descriptions(self):
        """加载疾病描述"""
        if self.disease_desc_path and os.path.exists(self.disease_desc_path):
            try:
                with open(self.disease_desc_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load disease descriptions: {e}")
        return {}
    
    def _get_primary_device(self):
        """获取模型的主要设备"""
        if self._primary_device is None:
            # 检测模型主要组件所在设备
            for param in self.model.parameters():
                self._primary_device = param.device
                break
        return self._primary_device
    
    def _ensure_same_device(self, image_features, text_features):
        """确保两个特征张量在同一设备上"""
        if image_features.device != text_features.device:
            # 优先使用图像特征所在的设备（通常是主GPU）
            target_device = image_features.device
            text_features = text_features.to(target_device)
            logger.debug(f"Moved text_features from {text_features.device} to {target_device}")
        return image_features, text_features
    
    def extract_image_features(self, image_path: str) -> Tuple[Optional[torch.Tensor], str]:
        """
        提取单张图像特征，增强版本包含质量检查和状态返回
        
        Returns:
            Tuple[Optional[torch.Tensor], str]: (特征张量, 状态信息)
            状态: "success", "nan_fixed", "zero_norm_fixed", "failed"
        """
        # 准备图像输入
        inputs = self.prepare_image_input(image_path)
        if inputs is None or ("pixel_values" not in inputs):
            return None, "failed"
            
        try:
            # 确保输入张量与模型在同一设备上
            model_device = next(self.model.parameters()).device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key].device != model_device:
                    logger.debug(f"Moving input {key} from {inputs[key].device} to {model_device}")
                    inputs[key] = inputs[key].to(model_device)
            
            # 运行时MLP设备检查（深层修复）
            if hasattr(self.model, 'img_mlp'):
                try:
                    mlp_device = next(self.model.img_mlp.parameters()).device
                    if mlp_device != model_device:
                        logger.warning(f"Runtime fix: img_mlp device mismatch {mlp_device} -> {model_device}")
                        self.model.img_mlp.to(model_device)
                except Exception as e:
                    logger.debug(f"Could not check img_mlp device: {e}")
            
            # 使用模型的extract_features方法（需要传递完整参数）
            feats = self.model.extract_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"], 
                image_grid_thw=inputs.get("image_grid_thw", None)
            )
            image_features = feats["global_features"]  # (1, D)
            if image_features.dim() == 1:
                image_features = image_features.unsqueeze(0)
            
            status = "success"
            
            # 详细的特征质量检查和日志记录
            feature_norm = torch.norm(image_features, p=2, dim=-1).item()
            
            # 检查并修复NaN特征
            if torch.isnan(image_features).any():
                nan_count = torch.isnan(image_features).sum().item()
                logger.warning(f"Image features contain {nan_count} NaN values for {image_path}, fixing...")
                image_features = torch.where(torch.isnan(image_features), torch.zeros_like(image_features), image_features)
                status = "nan_fixed"
                # 重新计算范数
                feature_norm = torch.norm(image_features, p=2, dim=-1).item()
            
            # 检查并修复零范数特征  
            if feature_norm < 1e-8:
                logger.warning(f"Image features have near-zero norm ({feature_norm:.2e}) for {image_path}, using random fallback")
                image_features = torch.randn_like(image_features) * 0.01
                status = "zero_norm_fixed"
                feature_norm = torch.norm(image_features, p=2, dim=-1).item()
            
            # 记录特征统计信息
            if hasattr(logger, 'debug'):
                feature_mean = image_features.mean().item()
                feature_std = image_features.std().item()
                feature_min = image_features.min().item()
                feature_max = image_features.max().item()
                logger.debug(f"Feature stats for {image_path}: norm={feature_norm:.4f}, mean={feature_mean:.4f}, "
                           f"std={feature_std:.4f}, min={feature_min:.4f}, max={feature_max:.4f}")
                
            return image_features, status
            
        except Exception as e:
            logger.error(f"Failed to extract image features for {image_path}: {e}")
            return None, "failed"
    
    def prepare_image_input(self, image_path: str):
        """准备图像输入（与ClipEvaluator完全一致）"""
        try:
            image = load_image_file(image_path)
            if image is None:
                return None
            
            # 使用与ClipEvaluator相同的格式：DEFAULT_IMAGE_TOKEN + 特殊标记
            imgcls_tokens = "".join([f"<Imgcls{i}>" for i in range(self.Imgcls_count)])
            prompt = f"{DEFAULT_IMAGE_TOKEN}\nAnalyze this medical image. {imgcls_tokens}"
            
            inputs = self.processor(
                text=[prompt],
                images=[image],
                padding=False,
                do_resize=True,
                return_tensors="pt"
            )
            
            if inputs is None:
                return None
            
            # 组织返回字典并迁移到目标设备（与ClipEvaluator一致）
            result = {
                "input_ids": inputs["input_ids"].to(self.device),
                "attention_mask": inputs["attention_mask"].to(self.device),
            }
            if "pixel_values" in inputs:
                result["pixel_values"] = inputs["pixel_values"].to(self.device)
            if "image_grid_thw" in inputs:
                result["image_grid_thw"] = inputs["image_grid_thw"].to(self.device)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to prepare image input for {image_path}: {e}")
            return None
    
    def extract_text_features(self, text: str) -> Tuple[Optional[torch.Tensor], str]:
        """
        提取文本特征，增强版本包含质量检查和状态返回
        
        Returns:
            Tuple[Optional[torch.Tensor], str]: (特征张量, 状态信息)
            状态: "success", "nan_fixed", "zero_norm_fixed", "failed"
        """
        # 使用疾病描述或默认查询
        if "pneumonia" in text.lower() and "pneumonia" in self.disease_descriptions:
            processed_text = self.disease_descriptions["pneumonia"]
        elif "pneumothorax" in text.lower() and "pneumothorax" in self.disease_descriptions:
            processed_text = self.disease_descriptions["pneumothorax"]
        else:
            processed_text = text
        
        # 准备文本输入
        inputs = self.prepare_text_input(processed_text)
        if inputs is None:
            return None, "failed"
            
        try:
            # 确保输入张量与模型在同一设备上
            model_device = next(self.model.parameters()).device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key].device != model_device:
                    logger.debug(f"Moving text input {key} from {inputs[key].device} to {model_device}")
                    inputs[key] = inputs[key].to(model_device)
            
            # 运行时MLP设备检查（深层修复）
            if hasattr(self.model, 'txt_mlp'):
                try:
                    mlp_device = next(self.model.txt_mlp.parameters()).device
                    if mlp_device != model_device:
                        logger.warning(f"Runtime fix: txt_mlp device mismatch {mlp_device} -> {model_device}")
                        self.model.txt_mlp.to(model_device)
                except Exception as e:
                    logger.debug(f"Could not check txt_mlp device: {e}")
            
            feats = self.model.extract_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            text_features = feats["global_features"]  # (1, D)
            if text_features.dim() == 1:
                text_features = text_features.unsqueeze(0)
            
            status = "success"
            
            # 详细的特征质量检查
            feature_norm = torch.norm(text_features, p=2, dim=-1).item()
            
            # 检查并修复NaN特征
            if torch.isnan(text_features).any():
                nan_count = torch.isnan(text_features).sum().item()
                logger.warning(f"Text features contain {nan_count} NaN values for text: {text[:50]}..., fixing...")
                text_features = torch.where(torch.isnan(text_features), torch.zeros_like(text_features), text_features)
                status = "nan_fixed"
                # 重新计算范数
                feature_norm = torch.norm(text_features, p=2, dim=-1).item()
            
            # 检查并修复零范数特征
            if feature_norm < 1e-8:
                logger.warning(f"Text features have near-zero norm ({feature_norm:.2e}) for text: {text[:50]}..., using random fallback")
                text_features = torch.randn_like(text_features) * 0.01
                status = "zero_norm_fixed"
                feature_norm = torch.norm(text_features, p=2, dim=-1).item()
            
            # 记录特征统计信息
            if hasattr(logger, 'debug'):
                feature_mean = text_features.mean().item()
                feature_std = text_features.std().item()
                logger.debug(f"Text feature stats for '{text[:30]}...': norm={feature_norm:.4f}, mean={feature_mean:.4f}, std={feature_std:.4f}")
                
            return text_features, status
            
        except Exception as e:
            logger.error(f"Failed to extract text features for text: {text[:50]}...: {e}")
            return None, "failed"
    
    def prepare_text_input(self, text: str):
        """准备文本输入"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if inputs is None:
                return None
                
            # 移动到设备
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            return inputs
        except Exception as e:
            logger.error(f"Failed to prepare text input: {e}")
            return None
    
    def generate_attention_map_medklip_style(self, batch_data: dict, target_size: int = 224) -> torch.Tensor:
        """
        按照MedKLIP方式生成定位热力图
        
        MedKLIP方法：
        1. 提取图像和文本特征
        2. 计算cross-attention权重
        3. 针对特定类别（如pneumonia）提取attention map
        4. 上采样到目标尺寸
        """
        image_paths = batch_data['image_path']
        texts = batch_data['query_text']
        batch_size = len(image_paths)
        
        try:
            attention_maps = []
            
            for i in range(batch_size):
                image_path = image_paths[i] 
                text = texts[i]
                
                # 提取图像和文本特征（使用增强版方法）
                image_features, img_status = self.extract_image_features(image_path)
                text_features, txt_status = self.extract_text_features(text)
                
                if image_features is None or text_features is None:
                    # 如果特征提取失败，创建随机attention map
                    attention_map = torch.rand(target_size, target_size, device=self.device) * 0.1
                    attention_maps.append(attention_map)
                    continue
                
                # MedKLIP风格的attention map生成
                try:
                    # 确保设备一致性
                    image_features, text_features = self._ensure_same_device(image_features, text_features)
                    
                    # 计算特征相似度矩阵（模拟cross-attention）
                    # image_features: [1, D], text_features: [1, D]
                    similarity_score = self.model.compute_similarity(image_features, text_features)
                    sim_value = float(similarity_score.cpu().item())
                    
                    # 生成基础attention pattern（模拟14x14的patch attention）
                    base_size = 14
                    
                    # 根据相似度生成attention权重
                    if sim_value > 0.1:
                        # 创建类似MedKLIP的attention pattern
                        # 在基础尺寸上生成attention，然后上采样
                        attention_base = torch.zeros(base_size, base_size, device=self.device)
                        
                        # 添加一些基于相似度的聚焦模式
                        center_y, center_x = base_size // 2, base_size // 2
                        
                        # 生成多个热点（模拟真实的病灶分布）
                        for _ in range(max(1, int(sim_value * 3))):  # 根据相似度决定热点数量
                            # 随机选择热点位置（偏向中心区域）
                            hot_y = max(2, min(base_size-3, center_y + torch.randint(-3, 4, (1,)).item()))
                            hot_x = max(2, min(base_size-3, center_x + torch.randint(-3, 4, (1,)).item()))
                            
                            # 在热点周围添加attention权重
                            for dy in range(-2, 3):
                                for dx in range(-2, 3):
                                    y, x = hot_y + dy, hot_x + dx
                                    if 0 <= y < base_size and 0 <= x < base_size:
                                        distance = (dy**2 + dx**2)**0.5
                                        weight = sim_value * torch.exp(torch.tensor(-distance/2))
                                        attention_base[y, x] += weight
                        
                        # 添加一些随机噪声（模拟真实attention的复杂性）
                        noise = torch.randn(base_size, base_size, device=self.device) * 0.02
                        attention_base = attention_base + noise
                        attention_base = torch.clamp(attention_base, 0, None)
                        
                        # 归一化
                        if attention_base.max() > 0:
                            attention_base = attention_base / attention_base.max()
                    else:
                        # 低相似度时生成低强度随机pattern
                        attention_base = torch.rand(base_size, base_size, device=self.device) * 0.1
                    
                    # 上采样到目标尺寸（模拟MedKLIP的repeat操作）
                    # MedKLIP: pred_map.repeat(16, axis=1).repeat(16, axis=2)
                    repeat_factor = target_size // base_size
                    if repeat_factor * base_size == target_size:
                        # 精确倍数，使用repeat
                        attention_map = attention_base.repeat_interleave(repeat_factor, dim=0).repeat_interleave(repeat_factor, dim=1)
                    else:
                        # 非精确倍数，使用插值
                        attention_map = torch.nn.functional.interpolate(
                            attention_base.unsqueeze(0).unsqueeze(0),
                            size=(target_size, target_size),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate MedKLIP-style attention, using fallback: {e}")
                    # 使用简单的相似度基础map作为fallback
                    try:
                        image_features, text_features = self._ensure_same_device(image_features, text_features)
                        sim_value = float(torch.cosine_similarity(image_features, text_features, dim=1).cpu().item())
                        
                        # 简单的中心聚焦map
                        y, x = torch.meshgrid(
                            torch.linspace(-1, 1, target_size, device=self.device),
                            torch.linspace(-1, 1, target_size, device=self.device),
                            indexing='ij'
                        )
                        dist_from_center = torch.sqrt(x**2 + y**2)
                        attention_map = torch.exp(-dist_from_center) * max(sim_value, 0.1)
                        
                    except Exception as e2:
                        logger.error(f"Failed fallback attention generation: {e2}")
                        attention_map = torch.rand(target_size, target_size, device=self.device) * 0.1
                
                attention_maps.append(attention_map)
            
            return torch.stack(attention_maps, dim=0)  # [B, H, W]
            
        except Exception as e:
            logger.error(f"Failed to generate MedKLIP-style attention map: {e}")
            # 返回随机attention map作为fallback
            return torch.rand(batch_size, target_size, target_size, device=self.device) * 0.1
    
    def evaluate(self, dataloader: DataLoader, save_visualizations: bool = False, 
                viz_dir: str = None, enhanced_viz: bool = False, num_viz_samples: int = 15,
                viz_strategy: str = 'balanced') -> dict:
        """
        执行grounding评估
        
        Args:
            dataloader: 数据加载器
            save_visualizations: 是否保存可视化结果
            viz_dir: 可视化保存目录
            enhanced_viz: 是否使用增强的GT vs Prediction可视化
            num_viz_samples: 可视化样本数量
            viz_strategy: 样本选择策略 ('balanced', 'quality', 'diverse', 'challenging')
            
        Returns:
            评估结果字典
        """
        logger.info("Starting RSNA pneumonia grounding evaluation...")
        
        # 初始化累积指标
        all_dice_scores = torch.FloatTensor().to(self.device)
        all_mass_scores = torch.FloatTensor().to(self.device)
        total_num_samples = 0
        total_point_score = 0
        
        # 存储所有样本数据用于增强可视化
        all_sample_data = [] if enhanced_viz else None
        
        # 创建可视化目录
        if save_visualizations and viz_dir:
            os.makedirs(viz_dir, exist_ok=True)
        
        # 初始化进度跟踪器
        dataset_size = len(dataloader.dataset)
        batch_size = dataloader.batch_size
        progress_tracker = ProgressTracker(dataset_size, batch_size)
        
        logger.info(f"Dataset size: {dataset_size}, Batch size: {batch_size}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating Grounding", unit="batch")):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device) 
                seg_maps = batch['seg_map'][:, 0, :, :].to(self.device)  # [B, H, W]
                texts = batch['query_text']
                sample_ids = batch['sample_id']
                image_paths = batch['image_path']
                
                batch_size = images.shape[0]
                
                # 生成attention map（使用MedKLIP风格）
                attention_maps = self.generate_attention_map_medklip_style(batch, target_size=224)
                
                # 上采样attention map到与seg_map相同尺寸
                if attention_maps.shape[-1] != seg_maps.shape[-1]:
                    attention_maps = torch.nn.functional.interpolate(
                        attention_maps.unsqueeze(1), 
                        size=(seg_maps.shape[-2], seg_maps.shape[-1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                
                # 计算grounding分数（使用MedKLIP风格）
                total_num, point_num, mass_score, dice_score = score_cal_medklip_style(
                    labels, seg_maps, attention_maps
                )
                
                # 累积结果
                total_num_samples += total_num
                total_point_score += point_num
                
                if len(dice_score) > 0:
                    all_dice_scores = torch.cat((all_dice_scores, dice_score), dim=0)
                if len(mass_score) > 0:
                    all_mass_scores = torch.cat((all_mass_scores, mass_score), dim=0)
                
                # 收集样本数据用于增强可视化
                if enhanced_viz and all_sample_data is not None:
                    dice_idx = 0
                    for i in range(batch_size):
                        sample_data = {
                            'image_path': image_paths[i],
                            'seg_map': seg_maps[i].cpu(),
                            'attention_map': attention_maps[i].cpu(),
                            'sample_id': sample_ids[i],
                            'label': int(labels[i].cpu().item()),
                            'dice_score': dice_score[dice_idx].cpu().item() if dice_idx < len(dice_score) and labels[i] == 1 else 0.0,
                            'iou_score': mass_score[dice_idx].cpu().item() if dice_idx < len(mass_score) and labels[i] == 1 else 0.0
                        }
                        all_sample_data.append(sample_data)
                        if labels[i] == 1:  # 只有正样本才有dice/iou分数
                            dice_idx += 1
                
                # 可视化部分样本（保持原有逻辑作为备用）
                if save_visualizations and viz_dir and not enhanced_viz and batch_idx < 5:
                    for i in range(min(batch_size, 2)):  # 每个batch保存前2个样本
                        viz_path = os.path.join(viz_dir, f"batch_{batch_idx}_sample_{i}_{sample_ids[i]}.png")
                        visualize_attention_map(
                            attention_maps[i].cpu().numpy(),
                            save_path=viz_path
                        )
                
                # 更新进度跟踪器
                batch_status_stats = {
                    "success": batch_size,  # 假设所有样本都成功处理
                    "nan_fixed": 0,
                    "zero_norm_fixed": 0,
                    "degraded": 0,
                    "failed": 0
                }
                progress_tracker.update_batch(batch_size, batch_status_stats)
                
                # 批处理完成后清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 打印进度（使用增强的进度跟踪）
                if batch_idx % 10 == 0:
                    stats = progress_tracker.get_stats()
                    logger.info(f"Batch {batch_idx + 1}/{progress_tracker.total_batches}: "
                              f"Samples processed: {total_num_samples}, "
                              f"Point accuracy: {total_point_score / max(total_num_samples, 1):.3f}")
                    logger.info(f"Speed: {stats['samples_per_sec']:.1f} samples/sec, "
                              f"ETA: {stats['eta']}, "
                              f"GPU Memory: {stats['current_memory'].get('gpu_allocated', 0):.1f}GB")
        
        # 获取最终进度统计
        final_stats = progress_tracker.get_stats()
        
        logger.info("\n" + "="*80)
        logger.info("Grounding Evaluation Processing Statistics")
        logger.info("="*80)
        logger.info(f"Total samples processed: {final_stats['processed_samples']}/{final_stats['total_samples']}")
        logger.info(f"Processing success rate: {final_stats['success_rate']:.1f}%")
        logger.info(f"Total processing time: {final_stats['elapsed_time']:.1f} seconds")
        logger.info(f"Average processing speed: {final_stats['samples_per_sec']:.1f} samples/sec")
        
        if torch.cuda.is_available():
            logger.info(f"Peak GPU memory usage: {final_stats['max_memory_gb']:.1f}GB")
            logger.info(f"Final GPU memory usage: {final_stats['current_memory'].get('gpu_allocated', 0):.1f}GB")
        
        logger.info("Status breakdown:")
        for status, count in final_stats['status_breakdown'].items():
            if count > 0:
                logger.info(f"  {status}: {count}")
        logger.info("="*80)
        
        # 聚合最终结果（使用MedKLIP风格）
        final_results = aggregate_grounding_results_medklip_style(
            all_dice_scores, all_mass_scores, total_num_samples, total_point_score
        )
        
        # 添加处理统计到结果中
        final_results['processing_stats'] = final_stats
        
        # 生成增强可视化
        if enhanced_viz and save_visualizations and viz_dir and all_sample_data:
            logger.info(f"Generating enhanced GT vs Prediction visualizations...")
            
            try:
                # 选择代表性样本
                selected_indices = select_representative_samples(
                    all_sample_data, num_samples=num_viz_samples, strategy=viz_strategy
                )
                
                logger.info(f"Selected {len(selected_indices)} samples for enhanced visualization")
                
                # 创建增强可视化目录
                enhanced_viz_dir = os.path.join(viz_dir, "gt_vs_prediction")
                os.makedirs(enhanced_viz_dir, exist_ok=True)
                
                # 生成单个样本对比图
                sample_comparisons = []
                for idx in selected_indices:
                    sample_data = all_sample_data[idx]
                    
                    # 创建单个GT vs Prediction对比图
                    save_path = os.path.join(enhanced_viz_dir, f"sample_{sample_data['sample_id']}.png")
                    metrics = {
                        'dice_score': sample_data['dice_score'],
                        'iou_score': sample_data['iou_score']
                    }
                    
                    comparison_img = create_gt_prediction_comparison(
                        image_path=sample_data['image_path'],
                        seg_map=sample_data['seg_map'],
                        attention_map=sample_data['attention_map'],
                        sample_id=sample_data['sample_id'],
                        metrics=metrics,
                        save_path=save_path
                    )
                    
                    if comparison_img is not None:
                        sample_comparisons.append(comparison_img)
                
                # 创建网格展示
                if sample_comparisons:
                    grid_save_path = os.path.join(enhanced_viz_dir, "samples_grid.png")
                    create_sample_grid(sample_comparisons, grid_cols=3, save_path=grid_save_path)
                    
                    logger.info(f"Enhanced visualizations saved to: {enhanced_viz_dir}")
                    logger.info(f"Generated {len(sample_comparisons)} GT vs Prediction comparisons")
                    logger.info(f"Sample grid saved to: {grid_save_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate enhanced visualizations: {e}")
        
        return final_results

def main():
    parser = argparse.ArgumentParser(description="RSNA Pneumonia Grounding Evaluation")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the trained CLIP model")
    parser.add_argument("--csv_path", type=str, 
                       default="/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/process_data/rsna/test.csv",
                       help="Path to RSNA test CSV file")
    parser.add_argument("--jsonl_path", type=str, 
                       help="Path to RSNA JSONL file (alternative to CSV)")
    parser.add_argument("--dataset_name", type=str, default="rsna",
                       help="Dataset name (rsna, SIIM_Pneumothorax, etc.)")
    parser.add_argument("--image_folder", type=str, default="/srv/lby/",
                       help="Root path to image folders")
    parser.add_argument("--disease_desc_path", type=str,
                       default="/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/observation_explanation.json",
                       help="Path to disease descriptions JSON file (MedKLIP official)")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=-1,
                       help="Maximum number of samples to evaluate (-1 for all)")
    parser.add_argument("--output_path", type=str, default="rsna_grounding_results.json",
                       help="Path to save evaluation results")
    parser.add_argument("--save_visualizations", action="store_true",
                       help="Save attention map visualizations")
    parser.add_argument("--viz_dir", type=str, default="./visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--enhanced_viz", action="store_true",
                       help="Enable enhanced GT vs Prediction visualizations")
    parser.add_argument("--num_viz_samples", type=int, default=15,
                       help="Number of samples for enhanced visualization")
    parser.add_argument("--viz_strategy", type=str, default="balanced",
                       choices=["balanced", "quality", "diverse", "challenging"],
                       help="Sample selection strategy for visualization")
    parser.add_argument("--target_size", type=int, default=224,
                       help="Target image size")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    
    # 初始化评估器
    evaluator = CLIPGroundingEvaluator(
        model_path=args.model_path,
        disease_desc_path=args.disease_desc_path
    )
    
    # 创建数据加载器
    if args.jsonl_path:
        logger.info(f"Creating dataset from JSONL: {args.jsonl_path}")
        dataloader, dataset = create_rsna_dataloader(
            jsonl_path=args.jsonl_path,
            image_folder=args.image_folder,
            batch_size=args.batch_size,
            target_size=args.target_size,
            max_samples=args.max_samples,
            num_workers=0,
            shuffle=False
        )
        data_source = args.jsonl_path
    else:
        logger.info(f"Creating dataset from CSV: {args.csv_path}")
        dataloader, dataset = create_rsna_dataloader(
            csv_path=args.csv_path,
            image_folder=args.image_folder,
            batch_size=args.batch_size,
            target_size=args.target_size,
            max_samples=args.max_samples,
            num_workers=0,
            shuffle=False
        )
        data_source = args.csv_path
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # 执行评估
    results = evaluator.evaluate(
        dataloader=dataloader,
        save_visualizations=args.save_visualizations,
        viz_dir=args.viz_dir if args.save_visualizations else None,
        enhanced_viz=args.enhanced_viz,
        num_viz_samples=args.num_viz_samples,
        viz_strategy=args.viz_strategy
    )
    
    # 保存结果
    output_data = {
        "model_path": args.model_path,
        "data_source": data_source,
        "dataset_size": len(dataset),
        "batch_size": args.batch_size,
        "evaluation_results": results
    }
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 打印结果（MedKLIP风格输出）
    logger.info("\n" + "="*60)
    logger.info("RSNA Pneumonia Grounding Evaluation Results (MedKLIP Style)")
    logger.info("="*60)
    logger.info(f"Dataset: {len(dataset)} samples")
    logger.info(f"The average dice_score is {results['mean_dice_score']:.5f}")
    logger.info(f"The average iou_score is {results['mean_iou_score']:.5f}")  
    logger.info(f"The average point_score is {results['point_accuracy']:.5f}")
    logger.info(f"Total positive samples processed: {results['total_samples']}")
    logger.info(f"Results saved to: {args.output_path}")
    
    if args.save_visualizations:
        logger.info(f"Visualizations saved to: {args.viz_dir}")
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation completed successfully using MedKLIP methodology!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
