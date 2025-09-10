"""
进度跟踪器模块
提供详细的统计和时间估算功能
"""

import time
from typing import Dict, Any
import torch


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
