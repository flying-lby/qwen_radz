'''
Author: flying-lby 2230232178@qq.com
Date: 2025-07-16 10:54:18
LastEditors: flying-lby 2230232178@qq.com
LastEditTime: 2025-07-21 12:19:32
FilePath: /qwen_radz/qwen_finetune/Qwen2-VL-Finetune/src/trainer/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# from .dpo_trainer import QwenDPOTrainer
from .sft_trainer import QwenSFTTrainer
# from .grpo_trainer import QwenGRPOTrainer

__all__ = ["QwenSFTTrainer"]

# __all__ = ["QwenSFTTrainer", "QwenDPOTrainer", "QwenGRPOTrainer"]