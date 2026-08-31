"""
Sovereign Weight Assimilator Sub-System.
Autonomously ingests, maps, and aligns raw safetensors weights from sovereign model checkpoints
into GPTLanguageModel and UniversalDynamicBlock parameters with shape projection and device management.
"""

import os
import torch
import torch.nn as nn
import safetensors.torch

class SovereignWeightAssimilator:
    """
    Sovereign Weight Assimilator.
    Maps tensor keys from external safetensors model checkpoints (LLaMA, Qwen, DeepSeek format)
    into the internal GPTLanguageModel parameter structure.
    """
    def __init__(self, target_model: nn.Module):
        self.target_model = target_model

    def align_and_load_safetensors(self, safetensors_path: str) -> dict:
        """
        Loads raw safetensors weights from disk, computes tensor key alignments,
        and safely copies matching tensor shapes into the target GPTLanguageModel.
        """
        if not os.path.exists(safetensors_path):
            return {"status": "error", "message": f"Path not found: {safetensors_path}"}

        state_dict = self.target_model.state_dict()
        try:
            raw_weights = safetensors.torch.load_file(safetensors_path)
        except Exception as e:
            return {"status": "error", "message": f"Failed to load safetensors: {e}"}

        assimilated_count = 0
        skipped_count = 0

        key_mapping = {
            "model.embed_tokens.weight": "tok_emb.weight",
            "lm_head.weight": "head.weight",
            "model.norm.weight": "ln_f.weight"
        }

        with torch.no_grad():
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

        return {
            "status": "success",
            "assimilated_tensors": assimilated_count,
            "skipped_tensors": skipped_count
        }

    def _project_tensor(self, src_tensor: torch.Tensor, target_shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Projects or crops source tensor to match target tensor shape for weight assimilation."""
        try:
            curr = src_tensor.to(device, dtype=dtype)
            for dim, (s_size, t_size) in enumerate(zip(curr.shape, target_shape)):
                if s_size > t_size:
                    curr = torch.narrow(curr, dim, 0, t_size)
                elif s_size < t_size:
                    pad_sizes = [0] * (2 * curr.ndim)
                    pad_sizes[2 * (curr.ndim - 1 - dim) + 1] = t_size - s_size
                    curr = torch.nn.functional.pad(curr, pad_sizes)
            return curr if curr.shape == target_shape else None
        except Exception:
            return None
