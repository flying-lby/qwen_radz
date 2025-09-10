"""         
CLIP风格Qwen2.5-VL模型的分类评估脚本
采用CLIP风格的图像-文本相似度计算进行分类任务评估
"""

import os
import sys
import json
import argparse
import logging

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入封装的评估工具
from eval_utils import (
    ProgressTracker,
    ClipClassificationDataset, 
    ClipEvaluator,
    load_image_file,
    split_list,
    get_chunk
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP Qwen2.5-VL model for classification")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained CLIP model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to evaluation data (jsonl format)")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to images folder")
    parser.add_argument("--dataset", type=str, default="mimic", 
                       choices=["chestxray", "chexpert", "mimic", "rsna", "COVIDx_CXR", "SIIM_Pneumothorax"],
                       help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--output_path", type=str, default="clip_eval_results.json", 
                       help="Path to save results")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks for parallel processing")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Current chunk index")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of samples to evaluate (-1 for all)")
    
    # 疾病描述相关参数
    parser.add_argument("--use_disease_descriptions", action="store_true", 
                       help="Use detailed disease descriptions instead of simple templates")
    parser.add_argument("--disease_desc_path", type=str, 
                       default="/home/lby/iclr2026/llava_med/LLaVA-Med/llava/run/data/new_full_disease.json",
                       help="Path to disease descriptions JSON file")
    parser.add_argument("--description_source", type=str, default="template", 
                       choices=["template", "file"],
                       help="Source of class descriptions: 'template' for simple templates, 'file' for detailed descriptions")
    
    # MedKLIP对齐参数
    parser.add_argument("--medklip_style_output", action="store_true",
                       help="Use MedKLIP-style output format and evaluation metrics")
    
    args = parser.parse_args()
    
    print(f"Starting CLIP-style evaluation...")
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Image folder: {args.image_folder}")
    print(f"Dataset: {args.dataset}")
    print(f"Use disease descriptions: {args.use_disease_descriptions}")
    if args.use_disease_descriptions:
        print(f"Disease description path: {args.disease_desc_path}")
        print(f"Description source: {args.description_source}")
    print(f"MedKLIP style output: {args.medklip_style_output}")
    
    # 创建评估器
    try:
        evaluator = ClipEvaluator(
            model_path=args.model_path,
            batch_size=args.batch_size,
            use_disease_descriptions=args.use_disease_descriptions,
            disease_desc_path=args.disease_desc_path,
            description_source=args.description_source
        )
        print(f"Model loaded successfully on {evaluator.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 创建数据集
    try:
        dataset = ClipClassificationDataset(
            data_path=args.data_path,
            image_folder=args.image_folder,
            dataset_name=args.dataset
        )
        
        # 分块处理（支持多进程评估）
        questions = dataset.questions
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        
        # 限制样本数量（用于快速测试）
        if args.max_samples > 0:
            questions = questions[:args.max_samples]
        
        # 更新数据集
        dataset.questions = questions
        
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Target classes: {dataset.target_classes}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # 执行CLIP风格分类评估
    try:
        # 使用MedKLIP对齐的评估策略
        print(f"Using MedKLIP-aligned evaluation strategy with optimal F1 thresholds")
            
        results = evaluator.evaluate_clip_classification(dataset)
        
        if not results:
            print("Evaluation failed - no results generated")
            return
        
        # 保存结果
        result_data = {
            "model_path": args.model_path,
            "dataset": args.dataset,
            "num_samples": len(dataset),
            "chunk_info": f"{args.chunk_idx+1}/{args.num_chunks}",
            "evaluation_method": "CLIP-style similarity",
            "metrics": results
        }
        
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {args.output_path}")
        
        # 打印关键指标摘要
        print(f"\n===== Key Performance Metrics =====")
        print(f"Macro F1 Score: {results.get('macro_f1', 0):.3f}")
        print(f"Macro Balanced Accuracy: {results.get('macro_balanced_accuracy', 0):.3f}")
        print(f"Mean AUC-ROC: {results.get('mean_auc', 0):.3f}")
        print(f"Overall Accuracy: {results.get('overall_accuracy', 0):.3f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()