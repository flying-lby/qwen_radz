"""
简化的数据处理工具
参考LLaVA RadZ的简单设计
"""

import os
from PIL import Image
import torch
from typing import Optional, Union, Tuple


def get_image_info(
    image_path: str,
    min_pixels: int = 3136,
    max_pixels: int = 1048576,
    resized_width: Optional[int] = None,
    resized_height: Optional[int] = None,
) -> Image.Image:
    """
    加载和预处理图像，简化版实现
    
    Args:
        image_path: 图像文件路径
        min_pixels: 最小像素数
        max_pixels: 最大像素数
        resized_width: 调整后的宽度（可选）
        resized_height: 调整后的高度（可选）
    
    Returns:
        PIL.Image: 处理后的图像
    """
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    try:
        # 加载图像
        image = Image.open(image_path)
        
        # 转换为RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 获取原始尺寸
        width, height = image.size
        total_pixels = width * height
        
        # 如果指定了具体尺寸，直接调整
        if resized_width and resized_height:
            image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
            return image
        
        # 根据像素数调整尺寸
        if total_pixels < min_pixels:
            # 图像太小，放大
            scale_factor = (min_pixels / total_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        elif total_pixels > max_pixels:
            # 图像太大，缩小
            scale_factor = (max_pixels / total_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
        
    except Exception as e:
        raise RuntimeError(f"图像处理失败 {image_path}: {str(e)}")


def process_dicom_image(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """
    处理DICOM图像（如果需要）
    
    Args:
        image_path: DICOM文件路径
        target_size: 目标尺寸 (width, height)
    
    Returns:
        PIL.Image: 处理后的图像
    """
    
    try:
        import pydicom
        from PIL import ImageOps
        
        # 读取DICOM文件
        dicom_data = pydicom.dcmread(image_path)
        
        # 获取像素数据
        pixel_array = dicom_data.pixel_array
        
        # 归一化到0-255
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
        pixel_array = pixel_array.astype('uint8')
        
        # 转换为PIL图像
        image = Image.fromarray(pixel_array)
        
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 调整尺寸（如果指定）
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image
        
    except ImportError:
        # 如果没有pydicom，尝试作为普通图像处理
        return get_image_info(image_path)
    except Exception as e:
        # 如果DICOM处理失败，尝试作为普通图像处理
        return get_image_info(image_path)


def smart_resize_image(
    image: Image.Image,
    target_pixels: int = 262144,  # 512x512
    maintain_aspect_ratio: bool = True
) -> Image.Image:
    """
    智能调整图像尺寸
    
    Args:
        image: PIL图像
        target_pixels: 目标像素数
        maintain_aspect_ratio: 是否保持长宽比
    
    Returns:
        PIL.Image: 调整后的图像
    """
    
    width, height = image.size
    current_pixels = width * height
    
    if current_pixels <= target_pixels:
        return image
    
    if maintain_aspect_ratio:
        # 保持长宽比
        scale_factor = (target_pixels / current_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
    else:
        # 固定为正方形
        side_length = int(target_pixels ** 0.5)
        new_width = side_length
        new_height = side_length
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def apply_simple_augmentation(image: Image.Image, augmentation_prob: float = 0.5) -> Image.Image:
    """
    应用简单的数据增强
    
    Args:
        image: PIL图像
        augmentation_prob: 增强概率
    
    Returns:
        PIL.Image: 增强后的图像
    """
    
    import random
    
    if random.random() > augmentation_prob:
        return image
    
    # 简单的增强操作
    augmentations = []
    
    # 随机水平翻转
    if random.random() < 0.5:
        augmentations.append(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT))
    
    # 随机旋转（小角度）
    if random.random() < 0.3:
        angle = random.uniform(-5, 5)
        augmentations.append(lambda img: img.rotate(angle, expand=False, fillcolor=(128, 128, 128)))
    
    # 应用增强
    for aug in augmentations:
        try:
            image = aug(image)
        except:
            pass  # 如果增强失败，跳过
    
    return image


def load_and_preprocess_image(
    image_path: str,
    min_pixels: int = 3136,
    max_pixels: int = 262144,
    apply_augmentation: bool = False,
    augmentation_prob: float = 0.5
) -> Image.Image:
    """
    完整的图像加载和预处理pipeline
    
    Args:
        image_path: 图像路径
        min_pixels: 最小像素数
        max_pixels: 最大像素数
        apply_augmentation: 是否应用数据增强
        augmentation_prob: 增强概率
    
    Returns:
        PIL.Image: 处理后的图像
    """
    
    # 判断文件类型
    if image_path.lower().endswith('.dcm') or 'dicom' in image_path.lower():
        image = process_dicom_image(image_path)
    else:
        image = get_image_info(image_path, min_pixels, max_pixels)
    
    # 智能调整尺寸
    image = smart_resize_image(image, target_pixels=max_pixels)
    
    # 应用数据增强
    if apply_augmentation:
        image = apply_simple_augmentation(image, augmentation_prob)
    
    return image
