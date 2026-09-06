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
                
                if "of-" in basename and "model-" in basename:
                    pattern = os.path.join(dirname, "model-*-of-*.safetensors")
                    found_shards = sorted(glob.glob(pattern))
                    if found_shards:
                        shard_paths = found_shards
                else:
                    match = re.match(r"^(.+?)-\d{5}\.safetensors$", basename)
                    if match:
                        prefix = match.group(1)
                        pattern = os.path.join(dirname, f"{prefix}-*.safetensors")
                        found_shards = sorted(glob.glob(pattern))
                        if len(found_shards) > 1:
                            shard_paths = found_shards
            elif os.path.isdir(safetensors_path):
                shard_paths = sorted(glob.glob(os.path.join(safetensors_path, "**", "*.safetensors"), recursive=True))

        if not shard_paths:
            return {"status": "error", "message": f"No valid safetensors paths found for: {safetensors_path}"}

        state_dict = self.target_model.state_dict()
        assimilated_count = 0
        skipped_count = 0

        key_mapping = {
            "model.embed_tokens.weight": "graph.tok_emb.weight",
            "model.language_model.embed_tokens.weight": "graph.tok_emb.weight",
            "lm_head.weight": "graph.lm_head.weight",
            "model.language_model.lm_head.weight": "graph.lm_head.weight",
            "model.norm.weight": "graph.ln_f.weight",
            "model.language_model.norm.weight": "graph.ln_f.weight"
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
                    target_key = None
                    if src_key in state_dict:
                        target_key = src_key
                    elif ("graph." + src_key) in state_dict:
                        target_key = "graph." + src_key
                    elif src_key.startswith("graph.") and src_key[6:] in state_dict:
                        target_key = src_key[6:]
                    else:
                        target_key = key_mapping.get(src_key, None)
                        if not target_key:
                            clean_k = src_key
                            if "model.language_model.layers." in clean_k:
                                clean_k = clean_k.replace("model.language_model.layers.", "graph.blocks.")
                            elif "model.layers." in clean_k:
                                clean_k = clean_k.replace("model.layers.", "graph.blocks.")
                            
                            clean_k = clean_k.replace("self_attn.q_proj", "graph.attn.wq")
                            clean_k = clean_k.replace("self_attn.k_proj", "graph.attn.wk")
                            clean_k = clean_k.replace("self_attn.v_proj", "graph.attn.wv")
                            clean_k = clean_k.replace("self_attn.o_proj", "graph.attn.wo")
                            clean_k = clean_k.replace("mlp.gate_proj", "graph.ffn.0")
                            clean_k = clean_k.replace("mlp.down_proj", "graph.ffn.2")
                            clean_k = clean_k.replace("input_layernorm.weight", "graph.norm1.weight")
                            clean_k = clean_k.replace("post_attention_layernorm.weight", "graph.norm2.weight")

                            if clean_k in state_dict:
                                target_key = clean_k
                            elif ("graph." + clean_k) in state_dict:
                                target_key = "graph." + clean_k

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

    def fuse_all_sovereign_checkpoints_to_single_brain(self, models_dir: str = "models", output_save_path: str = "models/singularity-00001.safetensors") -> dict:
        """
        Single-Minded Knowledge Fusion.
        Sequentially ingests and merges layer weights from all sovereign models (DeepSeek, Qwen, Llama, etc.)
        present in models_dir into a single unified checkpoint file.
        """
        if not os.path.exists(models_dir):
            return {"status": "error", "message": f"Models directory not found: {models_dir}"}
        
        all_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".safetensors") and not f.startswith("singularity-")]
        if not all_files:
            return {"status": "info", "message": "No external model files to fuse."}
            
        from src.telemetry import logger
        logger.log("INFO", "ASSIMILATOR", f"Initiating Single-Minded Knowledge Fusion for {len(all_files)} external model files...")
        
        total_assimilated = 0
        for model_file in all_files:
            res = self.align_and_load_safetensors(model_file)
            if res.get("status") == "success":
                total_assimilated += res.get("assimilated_tensors", 0)
                logger.log("INFO", "ASSIMILATOR", f"Fused weights from [ {os.path.basename(model_file)} ]: {res}")
                
        # Save unified single brain checkpoint
        state_dict = {k: v.cpu().contiguous() for k, v in self.target_model.state_dict().items()}
        tmp_path = output_save_path + ".tmp"
        safetensors.torch.save_file(state_dict, tmp_path)
        os.replace(tmp_path, output_save_path)
        
        logger.log("INFO", "ASSIMILATOR", f"Single-Minded Knowledge Fusion complete -> Saved to {output_save_path}")
        return {"status": "success", "fused_tensors": total_assimilated, "unified_brain": output_save_path}

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

WeightAssimilator = SovereignWeightAssimilator
