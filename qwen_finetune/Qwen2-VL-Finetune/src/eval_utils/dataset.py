"""
CLIP分类数据集模块
包含ClipClassificationDataset类的实现
"""

import os
import json
import numpy as np
from typing import Dict, List
from torch.utils.data import Dataset


class ClipClassificationDataset(Dataset):
    """CLIP分类评估数据集"""
    
    def __init__(
        self,
        data_path: str,
        image_folder: str = "",
        dataset_name: str = "mimic"
    ):
        self.image_folder = image_folder
        self.dataset_name = dataset_name
        
        # 加载数据
        self.questions = []
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.questions.append(json.loads(line))
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.questions = json.load(f)
        
        # 数据集类别定义
        self.dataset_classes = {
            'chestxray': ["fibrosis", "edema", "pneumothorax", "cardiomegaly", "atelectasis", 
                         "nodule", "emphysema", "no finding", "mass", "pleural_thickening", 
                         "effusion", "infiltration", "pneumonia", "hernia", "consolidation"],
            'chexpert': ['no finding', 'enlarged cardiomediastinum', 'cardiomegaly', 
                        'lung opacity', 'lung lesion', 'edema', 'consolidation', 
                        'pneumonia', 'atelectasis', 'pneumothorax', 'pleural effusion', 
                        'pleural other', 'fracture', 'support devices'],
            'mimic': ["atelectasis", "cardiomegaly", "consolidation", "edema", "enlarged cardiomediastinum",
                     "fracture", "lung lesion", "lung opacity", "no finding", "pleural effusion", 
                     "pleural other", "pneumonia", "pneumothorax", "support devices"],
            'rsna': ["pneumonia", "normal"],
            'COVIDx_CXR': ["covid-19", "normal"],
            'SIIM_Pneumothorax': ["pneumothorax", "non-pneumothorax"]
        }
        
        # 根据数据集选择类别
        if dataset_name in self.dataset_classes:
            self.target_classes = self.dataset_classes[dataset_name]
        else:
            self.target_classes = self.dataset_classes['mimic']  # 默认使用MIMIC类别
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        question = self.questions[index]
        
        # 构建图像路径
        if 'image' in question:
            img_path = os.path.join(self.image_folder, question['image'])
        elif 'image_path' in question:
            img_path = os.path.join(self.image_folder, question['image_path'])
        else:
            raise ValueError("No image path found in question")
        
        # 获取真实标签
        true_vector = np.zeros(len(self.target_classes))
        if 'label' in question:
            labels = question['label']
            if isinstance(labels, dict):
                for cls, value in labels.items():
                    if cls in self.target_classes and value == 1:
                        true_vector[self.target_classes.index(cls)] = 1
            elif isinstance(labels, list):
                for cls in labels:
                    if cls in self.target_classes:
                        true_vector[self.target_classes.index(cls)] = 1
        
        return {
            "image_path": img_path,
            "true_labels": true_vector,
            "question_data": question
        }
