#!/usr/bin/env python3
"""
复制MIMIC训练数据中的图像文件到指定目录
从train_data_100.json中提取图像路径，添加前缀，并复制到目标目录
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    # 文件路径配置
    json_file = "/home/lby/iclr2026/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/train_data_100.json"
    image_prefix = "/srv/lby/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
    target_dir = "/home/lby/iclr2026/qwen_radz/qwen_finetune/Qwen2-VL-Finetune/mimic_train_data_100"
    
    print(f"开始处理图像复制任务...")
    print(f"JSON文件: {json_file}")
    print(f"图像前缀: {image_prefix}")
    print(f"目标目录: {target_dir}")
    
    # 读取JSON文件
    print(f"\n正在读取JSON文件...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取JSON文件，包含 {len(data)} 个条目")
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return
    
    # 提取图像路径
    print(f"\n正在提取图像路径...")
    image_paths = []
    for item in data:
        if 'image' in item:
            image_paths.append(item['image'])
    
    print(f"提取到 {len(image_paths)} 个图像路径")
    
    # 创建目标目录
    print(f"\n正在创建目标目录...")
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"目标目录已创建: {target_dir}")
    
    # 复制图像文件
    print(f"\n开始复制图像文件...")
    success_count = 0
    error_count = 0
    
    for i, rel_path in enumerate(tqdm(image_paths, desc="复制图像")):
        try:
            # 构建完整的源路径
            source_path = os.path.join(image_prefix, rel_path)
            
            # 构建目标路径，保持相对路径结构
            target_file_path = os.path.join(target_dir, rel_path)
            
            # 创建目标文件的父目录
            target_file_dir = os.path.dirname(target_file_path)
            os.makedirs(target_file_dir, exist_ok=True)
            
            # 检查源文件是否存在
            if os.path.exists(source_path):
                # 复制文件
                shutil.copy2(source_path, target_file_path)
                success_count += 1
            else:
                print(f"警告: 源文件不存在: {source_path}")
                error_count += 1
                
        except Exception as e:
            print(f"复制文件失败 {rel_path}: {e}")
            error_count += 1
    
    # 输出统计结果
    print(f"\n" + "="*60)
    print(f"图像复制任务完成!")
    print(f"总文件数: {len(image_paths)}")
    print(f"成功复制: {success_count}")
    print(f"失败/不存在: {error_count}")
    print(f"成功率: {success_count/len(image_paths)*100:.1f}%")
    print(f"目标目录: {target_dir}")
    
    # 验证目标目录结构
    print(f"\n正在验证目标目录结构...")
    if os.path.exists(target_dir):
        # 统计复制的文件数量
        copied_files = []
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    copied_files.append(os.path.join(root, file))
        
        print(f"目标目录中的图像文件数量: {len(copied_files)}")
        
        # 显示目录结构示例
        print(f"\n目录结构示例:")
        count = 0
        for root, dirs, files in os.walk(target_dir):
            level = root.replace(target_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # 只显示前3个文件
                print(f"{subindent}{file}")
            if len(files) > 3:
                print(f"{subindent}... 还有 {len(files)-3} 个文件")
            count += 1
            if count >= 5:  # 只显示前5个目录
                break
    
    print(f"\n任务完成!")

if __name__ == "__main__":
    main()
