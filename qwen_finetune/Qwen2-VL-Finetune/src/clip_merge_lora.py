#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge LoRA adapter into base model and overlay non-LoRA module weights.

Usage:
  python src/clip_merge_lora.py \
    --model-base /path/to/base \
    --model-path /path/to/lora_output \
    --save-model-path /path/to/save_merged \
    --safe-serialization

This script will:
- Load base Qwen2.5-VL model (official transformers implementation)
- Load LoRA adapter from --model-path and merge into base
- Try to locate non-LoRA modules (img_mlp/txt_mlp/cross_attention/image_projector) weights from --model-path
  and overlay them onto the merged model
- Save merged model (and tokenizer/processor from base)
"""

import argparse
import os
import sys
from typing import Dict, List
import json

import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

try:
    from peft import PeftModel
except Exception as e:  # pragma: no cover
    print("[WARN] peft is not installed; cannot merge LoRA.", e)
    PeftModel = None


def _find_adapter_dir(path: str) -> str:
    """Find directory that contains adapter_model.safetensors under path."""
    target = os.path.join(path, "adapter_model.safetensors")
    if os.path.exists(target):
        return path
    for root, _, files in os.walk(path):
        if "adapter_model.safetensors" in files:
            return root
    return ""


def _peek_adapter_vocab_size(adapter_dir: str) -> int:
    """Read adapter_model.safetensors and try to infer expected vocab size
    from keys like base_model.model.model.embed_tokens.weight or lm_head.weight.
    Returns -1 if not found.
    """
    try:
        from safetensors.torch import load_file as safe_load  # type: ignore
        st = safe_load(os.path.join(adapter_dir, "adapter_model.safetensors"))
        for k in (
            "base_model.model.model.embed_tokens.weight",
            "base_model.model.lm_head.weight",
            "model.model.embed_tokens.weight",
            "model.lm_head.weight",
        ):
            if k in st and st[k].dim() == 2:
                return int(st[k].shape[0])
    except Exception as e:
        print(f"[WARN] Failed to peek adapter vocab size: {e}")
    return -1


def _load_state_dict_any(path: str) -> Dict[str, torch.Tensor]:
    """Load a state dict from possible files under path.
    Tries model.safetensors, pytorch_model.bin, non_lora_state_dict.bin.
    Returns empty dict if none found.
    """
    try:
        from safetensors.torch import load_file as safe_load  # type: ignore
    except Exception:
        safe_load = None

    cand_files = [
        "model.safetensors",
        "model.safetensors.index.json",  # multi-shard safetensors
        "pytorch_model.bin",
        "non_lora_state_dict.bin",
    ]

    # 优先在更深层目录中查找高优先级文件（尤其是 deepspeed 聚合后位于子目录的 pytorch_model.bin 或其同名子目录）
    for root, _, files in os.walk(path):
        for fname in ("model.safetensors.index.json", "model.safetensors", "pytorch_model.bin"):
            # 支持两种形式：文件 或 目录/pytorch_model.bin
            fpath = os.path.join(root, fname)
            alt_dir_file = os.path.join(fpath, fname)  # e.g. /.../pytorch_model.bin/pytorch_model.bin
            if os.path.isfile(fpath) or os.path.isfile(alt_dir_file):
                real_path = alt_dir_file if os.path.isfile(alt_dir_file) else fpath
                try:
                    if real_path.endswith(".safetensors") and safe_load is not None:
                        return safe_load(real_path)
                    elif real_path.endswith(".index.json") and safe_load is not None:
                        with open(real_path, "r", encoding="utf-8") as f:
                            index = json.load(f)
                        weight_map = index.get("weight_map", {})
                        shard_to_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
                        for tensor_name, shard_fname in weight_map.items():
                            shard_path = os.path.join(os.path.dirname(real_path), shard_fname)
                            if shard_fname not in shard_to_tensors:
                                shard_to_tensors[shard_fname] = safe_load(shard_path)
                        assembled: Dict[str, torch.Tensor] = {}
                        for shard_fname, tensors in shard_to_tensors.items():
                            for k, v in tensors.items():
                                assembled[k] = v
                        return assembled
                    else:
                        return torch.load(real_path, map_location="cpu")
                except Exception as e:
                    print(f"[WARN] Failed to load {real_path}: {e}")
                    continue

    # 然后尝试在根目录直接加载候选文件（fallback，包括 non_lora_state_dict.bin）
    for fname in cand_files:
        fpath = os.path.join(path, fname)
        if os.path.exists(fpath):
            try:
                if fname.endswith(".safetensors") and safe_load is not None:
                    return safe_load(fpath)
                elif fname.endswith(".index.json") and safe_load is not None:
                    # Load multi-shard safetensors via index
                    with open(fpath, "r", encoding="utf-8") as f:
                        index = json.load(f)
                    weight_map = index.get("weight_map", {})
                    shard_to_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
                    for tensor_name, shard_fname in weight_map.items():
                        shard_path = os.path.join(path, shard_fname)
                        if shard_fname not in shard_to_tensors:
                            shard_to_tensors[shard_fname] = safe_load(shard_path)
                    # assemble
                    assembled: Dict[str, torch.Tensor] = {}
                    for shard_fname, tensors in shard_to_tensors.items():
                        for k, v in tensors.items():
                            assembled[k] = v
                    return assembled
                else:
                    return torch.load(fpath, map_location="cpu")
            except Exception as e:
                print(f"[WARN] Failed to load {fpath}: {e}")
                continue
    # 若仍未找到，最后再深度扫描所有候选（与上方顺序一致）
    for root, _, files in os.walk(path):
        for fname in cand_files:
            if fname in files:
                fpath = os.path.join(root, fname)
                try:
                    if fname.endswith(".safetensors") and safe_load is not None:
                        return safe_load(fpath)
                    elif fname.endswith(".index.json") and safe_load is not None:
                        with open(fpath, "r", encoding="utf-8") as f:
                            index = json.load(f)
                        weight_map = index.get("weight_map", {})
                        shard_to_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
                        for tensor_name, shard_fname in weight_map.items():
                            shard_path = os.path.join(root, shard_fname)
                            if shard_fname not in shard_to_tensors:
                                shard_to_tensors[shard_fname] = safe_load(shard_path)
                        assembled: Dict[str, torch.Tensor] = {}
                        for shard_fname, tensors in shard_to_tensors.items():
                            for k, v in tensors.items():
                                assembled[k] = v
                        return assembled
                    else:
                        return torch.load(fpath, map_location="cpu")
                except Exception as e:
                    print(f"[WARN] Failed to load {fpath}: {e}")
                    continue
    return {}


def _overlay_non_lora_modules(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], patterns: List[str]) -> int:
    """Overlay weights for specified module name patterns onto model.
    Returns number of parameters updated.
    """
    if not state_dict:
        return 0

    normalized = {}
    for k, v in state_dict.items():
        nk = k
        # 训练产物中来自 PeftModel 的 state_dict 往往包含如下前缀层级：
        #   - "base_model."（PEFT 包装器）
        #   - "model."（底座模型中的子模块命名空间）
        # 改进导出模型中的小模块（如 img_mlp/txt_mlp/cross_attention/image_projector）
        # 位于顶层命名空间，因此这里需要去除这些前缀以对齐键名。
        if nk.startswith("base_model."):
            nk = nk[len("base_model."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        normalized[nk] = v

    model_sd = model.state_dict()
    to_update = {}
    skipped_mismatch: Dict[str, str] = {}
    for name, tensor in normalized.items():
        if name not in model_sd:
            continue
        if not any(pat in name for pat in patterns):
            continue
        tgt_tensor = model_sd[name]
        # 跳过大小为0的占位参数（常见于 ZeRO-3 未聚合的状态字典）
        if tensor.numel() == 0:
            skipped_mismatch[name] = f"src has 0 elements, target shape={tuple(tgt_tensor.shape)}"
            continue
        # 仅当形状完全一致时才进行覆盖，避免 load_state_dict 抛出 size mismatch 异常
        if tuple(tensor.shape) != tuple(tgt_tensor.shape):
            skipped_mismatch[name] = f"shape mismatch: src={tuple(tensor.shape)}, tgt={tuple(tgt_tensor.shape)}"
            continue
        
        # 检查tensor是否包含NaN值
        if torch.isnan(tensor).any():
            print(f"[WARN] Found NaN values in {name}, skipping this parameter.")
            skipped_mismatch[name] = f"contains NaN values"
            continue
            
        to_update[name] = tensor

    if not to_update:
        return 0

    missing, unexpected = model.load_state_dict(to_update, strict=False)
    if missing:
        print(f"[INFO] Missing keys while overlaying (ignored count={len(missing)}): first few: {missing[:5]}")
    if unexpected:
        print(f"[INFO] Unexpected keys while overlaying (ignored count={len(unexpected)}): first few: {unexpected[:5]}")
    if skipped_mismatch:
        first_few = list(skipped_mismatch.items())[:5]
        print(f"[INFO] Skipped {len(skipped_mismatch)} non-LoRA tensors due to mismatch/zero-size. Examples: {first_few}")
    return len(to_update)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", required=True, help="Base model path (e.g., Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--model-path", required=True, help="LoRA output directory containing adapter_model.safetensors and (optionally) full state dict")
    parser.add_argument("--save-model-path", required=True, help="Path to save merged model")
    parser.add_argument("--safe-serialization", action="store_true", help="Use safetensors to save model if supported")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "auto"], help="Dtype for loading base model")
    parser.add_argument(
        "--non-lora-patterns",
        nargs="*",
        default=["img_mlp", "txt_mlp", "knowledge_mlp", "cross_attention", "image_projector"],
        help="Module name patterns to overlay from training outputs",
    )
    parser.add_argument(
        "--export-with-clip-head",
        action="store_true",
        help="Export improved model with CLIP head (img_mlp/txt_mlp/etc). If set, base weights are loaded into improved model before overlaying",
    )
    args = parser.parse_args()

    os.makedirs(args.save_model_path, exist_ok=True)

    if PeftModel is None:
        print("[ERROR] peft is required to merge LoRA. Please install peft.")
        sys.exit(1)

    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = None

    print("[INFO] Loading base model:", args.model_base)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_base,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    print("[INFO] Locating LoRA adapter under:", args.model_path)
    adapter_dir = _find_adapter_dir(args.model_path)
    if not adapter_dir:
        print("[ERROR] adapter_model.safetensors not found under --model-path")
        sys.exit(1)
    print("[INFO] Found adapter at:", adapter_dir)

    # Align base vocab size with adapter if needed (common when special tokens were added during finetuning)
    try:
        expected_vocab = _peek_adapter_vocab_size(adapter_dir)
        if expected_vocab > 0:
            current_vocab = base_model.get_output_embeddings().weight.shape[0]
            if current_vocab != expected_vocab:
                print(f"[INFO] Resizing token embeddings: {current_vocab} -> {expected_vocab} (per adapter)")
                base_model.resize_token_embeddings(expected_vocab)
        else:
            # Fallback: try tokenizer saved in model-path
            try:
                tok_tmp = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
                tok_vocab = len(tok_tmp)
                current_vocab = base_model.get_output_embeddings().weight.shape[0]
                if tok_vocab != current_vocab:
                    print(f"[INFO] Resizing token embeddings via tokenizer: {current_vocab} -> {tok_vocab}")
                    base_model.resize_token_embeddings(tok_vocab)
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] Vocab alignment skipped: {e}")

    print("[INFO] Merging LoRA adapter into base model...")
    merged_base = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_base = merged_base.merge_and_unload()

    # Optionally build improved model and load base weights into it
    if args.export_with_clip_head:
        print("[INFO] Building improved model with CLIP head and loading base weights...")
        try:
            import sys as _sys, os as _os
            proj_root = _os.path.dirname(_os.path.dirname(__file__))
            if proj_root not in _sys.path:
                _sys.path.append(proj_root)
            from train.clip_modeling_improved import (
                ImprovedClipQwen2VLConfig as _Cfg,
                ImprovedClipQwen2VLForConditionalGeneration as _Imp,
            )
            cfg = _Cfg.from_pretrained(args.model_base, trust_remote_code=True)
            improved = _Imp(cfg)
            # 先对齐 improved 的词表大小到 merged_base（避免 embed/lm_head 形状不匹配）
            try:
                target_vocab = int(merged_base.get_output_embeddings().weight.shape[0])
                cur_vocab = int(improved.get_output_embeddings().weight.shape[0])
                if target_vocab != cur_vocab:
                    print(f"[INFO] Resizing improved token embeddings: {cur_vocab} -> {target_vocab}")
                    improved.resize_token_embeddings(target_vocab)
            except Exception as _e:
                print(f"[WARN] Failed to resize improved embeddings: {_e}")

            # 将合并后的基座权重加载到 improved 中
            mb_sd = merged_base.state_dict()
            missing, unexpected = improved.load_state_dict(mb_sd, strict=False)
            if missing:
                print(f"[INFO] While loading base into improved, missing={len(missing)} (ok)")
            if unexpected:
                print(f"[INFO] While loading base into improved, unexpected={len(unexpected)} (ok)")
            target_model = improved
        except Exception as e:
            print(f"[WARN] Failed to build improved model; fallback to base-only export. Reason: {e}")
            target_model = merged_base
    else:
        target_model = merged_base

    print("[INFO] Trying to overlay non-LoRA module weights...")
    sd = _load_state_dict_any(args.model_path)
    updated = _overlay_non_lora_modules(target_model, sd, args.non_lora_patterns)
    print(f"[INFO] Overlay updated {updated} tensors for non-LoRA modules")

    print("[INFO] Saving merged model to:", args.save_model_path)
    target_model.save_pretrained(args.save_model_path, safe_serialization=args.safe_serialization)
    tok = AutoTokenizer.from_pretrained(args.model_base, use_fast=False, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(args.model_base, trust_remote_code=True)
    tok.save_pretrained(args.save_model_path)
    proc.save_pretrained(args.save_model_path)
    print("[INFO] Done.")


if __name__ == "__main__":  # pragma: no cover
    main()

