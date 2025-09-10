# 核心CLIP评估模块 - 实际使用的模块
from .evaluator import ClipEvaluator
from .dataset import ClipClassificationDataset
from .progress_tracker import ProgressTracker
from .utils import load_image_file, split_list, get_chunk

# Grounding评估模块
from .dataset_rsna import RSNAGroundingDataset, create_rsna_dataloader
from .grounding_metrics import (
    calculate_grounding_scores, 
    aggregate_grounding_results,
    visualize_attention_map,
    extract_attention_from_model_output
)

# 未使用的模块 (可考虑移除或保留作为扩展)
# from .metrics import MetricCalculator, calculate_aucs, calculate_f1_scores, calculate_accuracy

__all__ = [
    # 实际使用的核心模块
    'ClipEvaluator',
    'ClipClassificationDataset', 
    'ProgressTracker',
    'load_image_file',
    'split_list',
    'get_chunk',
    # Grounding模块
    'RSNAGroundingDataset',
    'create_rsna_dataloader',
    'calculate_grounding_scores',
    'aggregate_grounding_results',
    'visualize_attention_map', 
    'extract_attention_from_model_output'
]