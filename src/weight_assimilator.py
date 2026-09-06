"""
Sovereign Weight Assimilator Sub-System.
Autonomously ingests, maps, and aligns raw safetensors weights from sovereign model checkpoints
into GPTLanguageModel and UniversalDynamicBlock parameters with shape projection and device management.
"""

import os
import torch
import torch.nn as nn
import safetensors.torch

import glob
import re
from typing import Union, List, Dict

class SovereignWeightAssimilator:
    """
    Sovereign Weight Assimilator.
    Maps tensor keys from external single or multi-shard safetensors model checkpoints (LLaMA, Qwen, DeepSeek format)
    into the internal GPTLanguageModel parameter structure.
    """
    def __init__(self, target_model: nn.Module) -> None:
        self.target_model = target_model

    def align_and_load_safetensors(self, safetensors_path: Union[str, List[str]]) -> dict:
        """
        Loads raw safetensors weights from single or multi-shard files (e.g. singularity-00001.safetensors,
        singularity-00002.safetensors, ...), computes tensor key alignments, and safely copies matching tensor
        shapes into the target GPTLanguageModel.
        """
        shard_paths = []
        if isinstance(safetensors_path, list):
            shard_paths = [p for p in safetensors_path if os.path.exists(p)]
        elif isinstance(safetensors_path, str):
            if os.path.exists(safetensors_path):
                shard_paths = [safetensors_path]
                dirname = os.path.dirname(safetensors_path) or "."
                basename = os.path.basename(safetensors_path)
                match = re.match(r"^(.+?)-\d{5}\.safetensors$", basename)
                if match:
                    prefix = match.group(1)
                    pattern = os.path.join(dirname, f"{prefix}-*.safetensors")
                    found_shards = sorted(glob.glob(pattern))
                    if len(found_shards) > 1:
                        shard_paths = found_shards
            elif os.path.isdir(safetensors_path):
                shard_paths = sorted(glob.glob(os.path.join(safetensors_path, "*.safetensors")))

        if not shard_paths:
            return {"status": "error", "message": f"No valid safetensors paths found for: {safetensors_path}"}

        state_dict = self.target_model.state_dict()
        assimilated_count = 0
        skipped_count = 0

        key_mapping = {
            "model.embed_tokens.weight": "tok_emb.weight",
            "lm_head.weight": "head.weight",
            "model.norm.weight": "ln_f.weight"
        }

        with torch.no_grad():
            for shard_file in shard_paths:
                try:
                    raw_weights = safetensors.torch.load_file(shard_file)
                except Exception as e:
                    from src.telemetry import logger
                    logger.log("WARNING", "ASSIMILATOR", f"Failed loading shard {shard_file}: {e}")
                    continue

                for src_key, tensor in raw_weights.items():
                    target_key = key_mapping.get(src_key, None)
                    if not target_key:
                        if "model.layers." in src_key:
                            target_key = src_key.replace("model.layers.", "blocks.").replace("self_attn.q_proj", "graph.attn.wq")
                            target_key = target_key.replace("self_attn.k_proj", "graph.attn.wk").replace("self_attn.v_proj", "graph.attn.wv")
                            target_key = target_key.replace("self_attn.o_proj", "graph.attn.wo").replace("mlp.gate_proj", "graph.ffn.0")

                    if target_key and target_key in state_dict:
                        target_param = state_dict[target_key]
                        if target_param.shape == tensor.shape:
                            target_param.copy_(tensor.to(target_param.device, dtype=target_param.dtype))
                            assimilated_count += 1
                        else:
                            sliced = self._project_tensor(tensor, target_param.shape, target_param.device, target_param.dtype)
                            if sliced is not None:
                                target_param.copy_(sliced)
                                assimilated_count += 1
                            else:
                                skipped_count += 1
                    else:
                        skipped_count += 1

        if assimilated_count == 0 and skipped_count == 0:
            return {"status": "error", "message": "Zero tensors loaded from provided shards."}

        return {
            "status": "success",
            "shards_loaded": len(shard_paths),
            "assimilated_tensors": assimilated_count,
            "skipped_tensors": skipped_count
        }

    def _project_tensor(self, src_tensor: torch.Tensor, target_shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Projects source tensor to match target tensor shape using tensor interpolation to preserve weight orientation."""
        try:
            curr = src_tensor.to(device, dtype=torch.float32)
            if curr.ndim == 1:
                # 1D embedding or norm bias vector interpolation
                curr = curr.view(1, 1, -1)
                resampled = torch.nn.functional.interpolate(curr, size=(target_shape[0],), mode='linear', align_corners=False)
                return resampled.squeeze(0).squeeze(0).to(dtype)
            elif curr.ndim == 2:
                # 2D Linear weight matrix interpolation
                curr = curr.unsqueeze(0).unsqueeze(0)
                resampled = torch.nn.functional.interpolate(curr, size=tuple(target_shape), mode='bilinear', align_corners=False)
                return resampled.squeeze(0).squeeze(0).to(dtype)
            else:
                for dim, (s_size, t_size) in enumerate(zip(curr.shape, target_shape)):
                    if s_size > t_size:
                        curr = torch.narrow(curr, dim, 0, t_size)
                return curr.to(dtype) if curr.shape == target_shape else None
        except Exception:
            return None
