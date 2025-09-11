# 核心CLIP评估模块 - 清理后只保留实际存在的模块
from .evaluator import ClipEvaluator
from .dataset import ClipClassificationDataset
from .progress_tracker import ProgressTracker
from .utils import load_image_file, split_list, get_chunk

__all__ = [
    # 实际使用的核心模块
    'ClipEvaluator',
    'ClipClassificationDataset', 
    'ProgressTracker',
    'load_image_file',
    'split_list',
    'get_chunk'
]