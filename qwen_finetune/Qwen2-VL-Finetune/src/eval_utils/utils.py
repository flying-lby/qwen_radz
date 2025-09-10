"""
工具函数模块
包含图像加载、列表分割等通用功能
"""

import os
import math
import numpy as np
from PIL import Image
from typing import List, Any

# 导入DICOM处理库
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("Warning: pydicom not available. DCM files will not be supported.")


def load_image_file(img_path):
    """
    加载图像文件，支持常规格式和DICOM格式
    """
    try:
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
                    image = Image.fromarray(pixel_array, mode='L')
                    # 转换为RGB
                    image = image.convert('RGB')
                elif len(pixel_array.shape) == 3:
                    # 彩色图像或多帧图像，取第一帧
                    if pixel_array.shape[0] < pixel_array.shape[2]:
                        # 假设第一个维度是帧数
                        pixel_array = pixel_array[0]
                    pixel_array = pixel_array.astype(np.uint8)
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
            image = Image.open(img_path).convert('RGB')
            return image
            
    except Exception as e:
        raise Exception(f"Failed to load image {img_path}: {str(e)}")


def split_list(lst: List[Any], n: int) -> List[List[Any]]:
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst: List[Any], n: int, k: int) -> List[Any]:
    """Get the k-th chunk from a list split into n chunks"""
    chunks = split_list(lst, n)
    return chunks[k]
