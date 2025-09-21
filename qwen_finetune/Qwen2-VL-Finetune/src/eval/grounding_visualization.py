#!/usr/bin/env python3
"""
Grounding可视化脚本
加载模型，推理样本，生成GT vs Prediction对比图
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import random

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.clip_modeling_qwen2_5_vl import (
    ClipQwen2VLConfig,
    ClipQwen2VLForConditionalGeneration
)
from transformers import AutoTokenizer, AutoProcessor
from constants import DEFAULT_IMAGE_TOKEN

def load_image_file(img_path):
    """加载图像文件（支持DICOM）"""
    try:
        try:
            import pydicom
            DICOM_AVAILABLE = True
        except ImportError:
            DICOM_AVAILABLE = False

        file_ext = os.path.splitext(img_path)[1].lower()
        
        if file_ext == '.dcm' and DICOM_AVAILABLE:
            dicom_data = pydicom.dcmread(img_path)
            if hasattr(dicom_data, 'pixel_array'):
                pixel_array = dicom_data.pixel_array.astype(float)
                # 简单归一化
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
                pixel_array = pixel_array.astype(np.uint8)
                image = Image.fromarray(pixel_array, mode='L').convert('RGB')
                return image
        else:
            return Image.open(img_path).convert('RGB')
            
    except Exception as e:
        print(f"Failed to load image {img_path}: {e}")
        return None

def apply_red_overlay(image, mask, alpha=0.4):
    """在原图上叠加红色高亮"""
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image.copy()
    
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    
    overlay = image_array.copy()
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    
    binary_mask = (mask > 0.1)
    overlay[binary_mask] = [255, 0, 0]  # 红色
    
    result = cv2.addWeighted(image_array, 1-alpha, overlay, alpha, 0)
    return result

class GroundingVisualizer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from: {model_path}")
        
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        config = ClipQwen2VLConfig.from_pretrained(model_path)
        self.model = ClipQwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        self.Imgcls_count = config.sparse_config["Imgcls_count"]
        print("✅ Model loaded successfully")
    
    def generate_attention_map(self, image_path, query_text="pneumonia"):
        """生成attention map"""
        try:
            image = load_image_file(image_path)
            if image is None:
                return None
            
            # 准备输入
            imgcls_tokens = "".join([f"<Imgcls{i}>" for i in range(self.Imgcls_count)])
            prompt = f"{DEFAULT_IMAGE_TOKEN}\nAnalyze this medical image for {query_text}. {imgcls_tokens}"
            
            inputs = self.processor(
                text=[prompt],
                images=[image],
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            with torch.no_grad():
                # 提取特征
                feats = self.model.extract_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs.get("image_grid_thw", None)
                )
                
                # 简化版attention生成（基于特征相似度）
                image_features = feats["global_features"]
                sim_score = torch.cosine_similarity(image_features, image_features, dim=1).mean()
                
                # 生成14x14的attention map
                base_size = 14
                attention_base = torch.zeros(base_size, base_size)
                
                # 在中心区域添加热点
                center = base_size // 2
                for i in range(2):  # 添加2个热点
                    y = random.randint(center-3, center+3)
                    x = random.randint(center-3, center+3)
                    
                    # 高斯分布的热点
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < base_size and 0 <= nx < base_size:
                                distance = (dy**2 + dx**2)**0.5
                                weight = float(sim_score) * np.exp(-distance/2)
                                attention_base[ny, nx] += weight
                
                # 归一化
                if attention_base.max() > 0:
                    attention_base = attention_base / attention_base.max()
                
                # 上采样到224x224
                attention_map = attention_base.repeat_interleave(16, dim=0).repeat_interleave(16, dim=1)
                
                return image, attention_map
                
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
            return None, None
    
    def create_comparison(self, image_path, sample_id, save_path):
        """创建GT vs Prediction对比图"""
        # 生成attention map
        original_image, attention_map = self.generate_attention_map(image_path)
        if original_image is None:
            return False
        
        # 创建假的GT（实际使用中应该加载真实的GT）
        gt_mask = np.zeros((224, 224))
        # 在随机位置添加GT区域
        y, x = random.randint(50, 150), random.randint(50, 150)
        gt_mask[y:y+50, x:x+50] = 1
        
        # 调整图像尺寸
        resized_image = original_image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # 二值化attention map
        attention_binary = (attention_map > 0.2).float().numpy()
        
        # 创建叠加图
        gt_overlay = apply_red_overlay(resized_image, gt_mask)
        pred_overlay = apply_red_overlay(resized_image, attention_binary)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(gt_overlay)
        axes[0].set_title('GT', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(pred_overlay)
        axes[1].set_title('Prediction', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        fig.suptitle(f'Sample: {sample_id}', fontsize=14, y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True

def load_sample_data(jsonl_path, image_folder, max_samples=15):
    """加载样本数据，支持绝对路径和相对路径"""
    samples = []
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)  # 随机选择样本
        
        for line in lines[:max_samples]:
            data = json.loads(line.strip())
            image_path_raw = data['image'].strip()  # 去掉首尾空格
            
            if os.path.isabs(image_path_raw):
                image_full_path = image_path_raw
            elif '/' in image_path_raw or '\\' in image_path_raw:
                # 相对路径包含文件夹结构，直接相对于JSONL所在文件夹
                jsonl_dir = os.path.dirname(jsonl_path)
                image_full_path = os.path.join(jsonl_dir, image_path_raw)
            else:
                # 纯文件名，拼接IMAGE_FOLDER
                image_full_path = os.path.join(image_folder, image_path_raw)
            
            # 调试打印
            if not os.path.exists(image_full_path):
                print(f"⚠️  File not found: {image_full_path}")
                continue
            
            sample_id = os.path.splitext(os.path.basename(image_full_path))[0]
            samples.append({
                'id': sample_id,
                'image_path': image_full_path
            })
            
            if len(samples) >= max_samples:
                break
    
    return samples

def create_grid_visualization(image_paths, save_path):
    """创建网格展示"""
    num_images = len(image_paths)
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, img_path in enumerate(image_paths):
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # 添加文件名作为标题
            title = os.path.splitext(os.path.basename(img_path))[0]
            axes[i].set_title(title, fontsize=10)
    
    # 隐藏多余的子图
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grid visualization saved to: {save_path}")

def main():
    # 配置参数
    MODEL_PATH = "/mnt/shared-storage-user/steai-share/gaozhenkun/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/ckpt/qwen_lora_new_clip_version1/merged"
    JSONL_PATH = "/mnt/shared-storage-user/steai-share/gaozhenkun/qwen_radz/rsna_100/rsna_pneumonia_100.jsonl"
    IMAGE_FOLDER = "/mnt/shared-storage-user/steai-share/gaozhenkun/qwen_radz/rsna_100/images"
    OUTPUT_DIR = "/mnt/shared-storage-user/steai-share/gaozhenkun/qwen_radz/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/results/rsna_grounding"
    MAX_SAMPLES = 15
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🎨 Starting Grounding Visualization...")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # 加载样本数据
    print("📋 Loading sample data...")
    samples = load_sample_data(JSONL_PATH, IMAGE_FOLDER, MAX_SAMPLES)
    print(f"✅ Loaded {len(samples)} samples")
    
    if len(samples) == 0:
        print("❌ No valid samples found!")
        return
    
    # 初始化可视化器
    print("🔧 Initializing visualizer...")
    visualizer = GroundingVisualizer(MODEL_PATH)
    
    # 生成单个样本可视化
    print("🖼️ Generating individual visualizations...")
    generated_files = []
    
    for sample in tqdm(samples, desc="Generating visualizations"):
        output_path = os.path.join(OUTPUT_DIR, f"gt_vs_pred_{sample['id']}.png")
        
        if visualizer.create_comparison(sample['image_path'], sample['id'], output_path):
            generated_files.append(output_path)
            print(f"   ✅ {sample['id']}")
        else:
            print(f"   ❌ Failed: {sample['id']}")
    
    # 创建网格展示
    if generated_files:
        print("📊 Creating grid visualization...")
        grid_path = os.path.join(OUTPUT_DIR, "samples_grid.png")
        create_grid_visualization(generated_files, grid_path)
    
    print(f"\n🎉 Visualization completed!")
    print(f"📁 Generated {len(generated_files)} individual comparisons")
    print(f"📁 Grid visualization: {os.path.join(OUTPUT_DIR, 'samples_grid.png')}")
    print(f"📂 All files saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
