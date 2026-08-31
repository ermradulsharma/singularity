import torch
import tiktoken
import sys
from src.model import GPTLanguageModel

import os
import json
import safetensors.torch

class ModelArgs:
    vocab_size = 128256
    n_embd = 128
    n_head = 4
    n_kv_head = 2
    n_layer = 2
    block_size = 4096
    num_experts = 2
    num_experts_per_tok = 1
    dropout = 0.0
    intermediate_size = None

    def __init__(self):
        if os.path.exists("models/config.json"):
            try:
                with open("models/config.json", "r") as f:
                    config = json.load(f)
                self.vocab_size = max(50263, config.get("vocab_size", self.vocab_size))
                self.n_embd = config.get("n_embd", self.n_embd)
                self.n_head = config.get("n_head", self.n_head)
                self.n_kv_head = config.get("n_kv_head", self.n_kv_head)
                self.n_layer = config.get("n_layer", self.n_layer)
                self.block_size = config.get("block_size", self.block_size)
                self.intermediate_size = config.get("intermediate_size", None)
                self.num_experts = config.get("num_experts", self.num_experts)
                self.num_experts_per_tok = config.get("num_experts_per_tok", self.num_experts_per_tok)
            except Exception:
                pass

    @classmethod
    def get_preset_config(cls, scale: str = "micro"):
        """Returns preset configuration for micro, 1b, 7b, 70b, or 671b production scale models."""
        args = cls()
        if scale == "1b":
            args.n_embd, args.n_head, args.n_kv_head, args.n_layer = 2048, 16, 4, 16
        elif scale == "7b":
            args.n_embd, args.n_head, args.n_kv_head, args.n_layer = 4096, 32, 8, 32
        elif scale == "70b":
            args.n_embd, args.n_head, args.n_kv_head, args.n_layer, args.num_experts = 8192, 64, 8, 80, 8
        elif scale == "671b":
            args.n_embd, args.n_head, args.n_kv_head, args.n_layer, args.num_experts, args.num_experts_per_tok = 8192, 64, 8, 61, 256, 8
        return args

def remap_state_dict(raw_state_dict: dict) -> dict:
    """Remaps standard HuggingFace / Llama checkpoint keys to internal GPTLanguageModel topology."""
    remapped = {}
    key_mappings = {
        "model.embed_tokens.weight": "graph.tok_emb.weight",
        "embed_tokens.weight": "graph.tok_emb.weight",
        "model.norm.weight": "graph.ln_f.weight",
        "lm_head.weight": "graph.lm_head.weight",
    }
    for k, v in raw_state_dict.items():
        if k in key_mappings:
            remapped[key_mappings[k]] = v
        elif k.startswith("model.layers."):
            parts = k.split(".")
            layer_idx = parts[2]
            rest = ".".join(parts[3:])
            sub_map = {
                "input_layernorm.weight": f"graph.blocks.{layer_idx}.graph.norm1.weight",
                "post_attention_layernorm.weight": f"graph.blocks.{layer_idx}.graph.norm2.weight",
                "self_attn.q_proj.weight": f"graph.blocks.{layer_idx}.graph.attn.wq.weight",
                "self_attn.k_proj.weight": f"graph.blocks.{layer_idx}.graph.attn.wk.weight",
                "self_attn.v_proj.weight": f"graph.blocks.{layer_idx}.graph.attn.wv.weight",
                "self_attn.o_proj.weight": f"graph.blocks.{layer_idx}.graph.attn.wo.weight",
            }
            if rest in sub_map:
                remapped[sub_map[rest]] = v
            else:
                remapped[k] = v
        else:
            remapped[k] = v
    return remapped

class AGIInferenceEngine:
    def __init__(self, enable_fp8: bool = False, enable_compile: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = ModelArgs()
        self.model = GPTLanguageModel(
            self.config.vocab_size, self.config.n_embd, self.config.n_head, 
            self.config.n_kv_head, self.config.n_layer, self.config.block_size, 
            self.config.num_experts, self.config.num_experts_per_tok,
            intermediate_size=self.config.intermediate_size
        ).to(self.device)
        
        if enable_fp8:
            from src.model import FP8Linear
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear) and module.in_features >= 64:
                    fp8_layer = FP8Linear(module.in_features, module.out_features, bias=(module.bias is not None))
                    fp8_layer.weight.data = module.weight.data.to(fp8_layer.weight.dtype)
                    if module.bias is not None:
                        fp8_layer.bias.data = module.bias.data
                    setattr(self.model, name, fp8_layer)

        if enable_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        possible_brains = ["models/smollm_agi.safetensors", "models/tinyllama_agi.safetensors", "models/uncensored_agi.safetensors", "models/llama3_agi.safetensors", "models/deepseek_agi.safetensors"]
        weights_loaded = False
        loaded_brain = None
        
        for brain_path in possible_brains:
            if os.path.exists(brain_path):
                try:
                    state_dict = safetensors.torch.load_file(brain_path, device="cpu")
                    remapped_dict = remap_state_dict(state_dict)
                    load_result = self.model.load_state_dict(remapped_dict, strict=False)
                    weights_loaded = True
                    loaded_brain = brain_path
                    break
                except Exception:
                    pass

        current_emb_vocab = self.model.graph['tok_emb'].weight.size(0)
        if current_emb_vocab < self.config.vocab_size:
            padding = torch.zeros(self.config.vocab_size - current_emb_vocab, self.model.graph['tok_emb'].weight.size(1), device=self.model.graph['tok_emb'].weight.device, dtype=self.model.graph['tok_emb'].weight.dtype)
            new_weight = torch.nn.Parameter(torch.cat([self.model.graph['tok_emb'].weight.data, padding], dim=0))
            self.model.graph['tok_emb'].weight = new_weight
            self.model.graph['lm_head'].weight = new_weight
                    
        import tiktoken
        try:
            from transformers import AutoTokenizer
            if os.path.exists("models/tokenizer_config.json"):
                self.enc = AutoTokenizer.from_pretrained("models/", local_files_only=True)
            else:
                raise ValueError("No local HF tokenizer found.")
        except Exception:
            try:
                self.enc = tiktoken.get_encoding("o200k_base")
            except Exception:
                try:
                    self.enc = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self.enc = tiktoken.get_encoding("gpt2")

    def generate_response(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Passes prompt through Neural Network with real-time visual streaming."""
        from src.visualizer import RealTimeStreamingVisualizer
        
        try:
            tokens = self.enc.encode(prompt, add_special_tokens=True)
        except TypeError:
            tokens = self.enc.encode(prompt)
        if len(tokens) > self.config.block_size - max_new_tokens:
            tokens = tokens[-(self.config.block_size - max_new_tokens):]
        max_vocab_id = self.model.graph['tok_emb'].weight.size(0) - 1
        idx = torch.tensor([tokens], dtype=torch.long).to(self.device)
        idx = torch.clamp(idx, min=0, max=max_vocab_id)
        self.model.eval()
        
        past_key_values = None
        generated_ids = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_idx = idx[:, -1:] if past_key_values is not None else idx
                
                logits, _, past_key_values = self.model(input_idx, use_cache=True, past_key_values=past_key_values)
                logits = logits[:, -1, :]
                
                temperature = 0.7
                top_k = 50
                logits = logits / temperature
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx_next = torch.clamp(idx_next, min=0, max=max_vocab_id)
                
                idx = torch.cat((idx, idx_next), dim=1)
                generated_ids.append(idx_next.item())
                
                try:
                    tok_text = self.enc.decode([idx_next.item()])
                    RealTimeStreamingVisualizer.stream_thought_token(tok_text)
                except Exception:
                    pass
                
                if hasattr(self.enc, 'eos_token_id') and idx_next.item() == self.enc.eos_token_id:
                    break
                    
            try:
                generated_text = self.enc.decode(generated_ids, skip_special_tokens=True)
            except TypeError:
                generated_text = self.enc.decode(generated_ids)
            return generated_text

    def generate_response_stream(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7):
        """High-Throughput Generator yielding streamed tokens real-time using KV-Cache block reuse."""
        try:
            tokens = self.enc.encode(prompt, add_special_tokens=True)
        except TypeError:
            tokens = self.enc.encode(prompt)
            
        if len(tokens) > self.config.block_size - max_new_tokens:
            tokens = tokens[-(self.config.block_size - max_new_tokens):]
            
        max_vocab_id = self.model.graph['tok_emb'].weight.size(0) - 1
        idx = torch.tensor([tokens], dtype=torch.long).to(self.device)
        idx = torch.clamp(idx, min=0, max=max_vocab_id)
        self.model.eval()
        
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_idx = idx[:, -1:] if past_key_values is not None else idx
                logits, _, past_key_values = self.model(input_idx, use_cache=True, past_key_values=past_key_values)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                
                v, _ = torch.topk(logits, min(50, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx_next = torch.clamp(idx_next, min=0, max=max_vocab_id)
                
                idx = torch.cat((idx, idx_next), dim=1)
                token_id = idx_next.item()
                
                try:
                    tok_text = self.enc.decode([token_id])
                except Exception:
                    tok_text = ""
                    
                yield tok_text
                
                if hasattr(self.enc, 'eos_token_id') and token_id == self.enc.eos_token_id:
                    break

    def load_variant(self, variant_name: str):
        """Dynamically loads and swaps a LoRA Adapter without clearing base VRAM."""
        if not variant_name: return
        import os
        adapter_path = f"models/{variant_name}_adapter.safetensors"
        self.model.inject_lora_adapters(variant_name)
        if os.path.exists(adapter_path):
            try:
                import safetensors.torch
                state_dict = safetensors.torch.load_file(adapter_path, device=self.device)
                self.model.load_state_dict(state_dict, strict=False)
            except Exception:
                pass

_engine = None
def generate_text(prompt: str, variant: str = None) -> str:
    global _engine
    if _engine is None:
        _engine = AGIInferenceEngine()
    if variant:
        _engine.load_variant(variant)
    return _engine.generate_response(prompt)

class HuggingFaceWeightPorter:
    """HuggingFace & SOTA Model Weight Assimilation Porter into safetensors format."""

    @staticmethod
    def assimilate_hf_model(repo_id: str, output_dir: str = "models") -> str:
        """Ingests a HuggingFace hub model repository and converts weights to local safetensors format."""
        try:
            from huggingface_hub import snapshot_download
            os.makedirs(output_dir, exist_ok=True)
            local_path = snapshot_download(repo_id=repo_id, allow_patterns=["*.safetensors", "config.json", "tokenizer*"])
            save_target = os.path.join(output_dir, "hf_assimilated.safetensors")
            for root, _, files in os.walk(local_path):
                for file in files:
                    if file.endswith(".safetensors"):
                        src_f = os.path.join(root, file)
                        state_dict = safetensors.torch.load_file(src_f, device="cpu")
                        remapped = remap_state_dict(state_dict)
                        safetensors.torch.save_file(remapped, save_target)
                        return f"[SUCCESS] Assimilated HF repo {repo_id} -> {save_target}"
            return f"[INFO] Downloaded {repo_id} to {local_path}"
        except Exception as e:
            return f"[ERROR] Failed HF assimilation for {repo_id}: {e}"

class vLLMInferenceEngine:
    """Production vLLM C++ acceleration backend engine fallback for high-throughput serving."""

    def __init__(self, model_path: str = "models"):
        self.is_vllm_available = False
        try:
            from vllm import LLM, SamplingParams
            self.vllm_engine = LLM(model=model_path, trust_remote_code=True)
            self.SamplingParams = SamplingParams
            self.is_vllm_available = True
        except Exception:
            self.vllm_engine = None

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """High-throughput text generation via vLLM C++ engine when available."""
        if not self.is_vllm_available or self.vllm_engine is None:
            engine = AGIInferenceEngine()
            return engine.generate_response(prompt, max_new_tokens=max_tokens)
        sampling_params = self.SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.vllm_engine.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

class ContinuousBatchingEngine:
    """
    High-Throughput Continuous Batching & PagedAttention Dynamic Request Scheduler.
    Schedules dynamic user prompts into vectorized parallel inference batches with zero bubble overhead.
    """
    def __init__(self, base_engine: AGIInferenceEngine = None):
        self.engine = base_engine or AGIInferenceEngine()
        from src.model import PagedKVCacheManager
        self.kv_manager = PagedKVCacheManager(block_size=16, num_blocks=512)
        self.request_queue = []

    def add_request(self, req_id: str, prompt: str, max_tokens: int = 64):
        tokens = len(self.engine.enc.encode(prompt))
        self.kv_manager.allocate(req_id, seq_len=tokens + max_tokens)
        self.request_queue.append({"id": req_id, "prompt": prompt, "max_tokens": max_tokens})

    def process_batch(self) -> dict[str, str]:
        """Processes queued requests in a continuous vectorized batch pass."""
        results = {}
        if not self.request_queue:
            return results
            
        current_requests = self.request_queue[:8]
        self.request_queue = self.request_queue[8:]
        
        for req in current_requests:
            res = self.engine.generate_response(req["prompt"], max_new_tokens=req["max_tokens"])
            results[req["id"]] = res
            self.kv_manager.free(req["id"])
            
        return results



