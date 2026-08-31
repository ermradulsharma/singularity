from src.telemetry import logger
import torch
import tiktoken
import sys
from src.model import GPTLanguageModel

import os
import json
import safetensors.torch

class ModelArgs:
    vocab_size = 50257 + 6
    n_embd = 4096
    n_head = 32
    n_kv_head = 8
    n_layer = 32
    block_size = 2048
    num_experts = 4
    num_experts_per_tok = 2
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
            except Exception as e:
                print(f"[WARNING] Failed to load models/config.json: {e}")

class AGIInferenceEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = ModelArgs()
        self.model = GPTLanguageModel(
            self.config.vocab_size, self.config.n_embd, self.config.n_head, 
            self.config.n_kv_head, self.config.n_layer, self.config.block_size, 
            self.config.num_experts, self.config.num_experts_per_tok,
            intermediate_size=self.config.intermediate_size
        ).to(self.device)
        possible_brains = ["models/smollm_agi.safetensors", "models/tinyllama_agi.safetensors", "models/uncensored_agi.safetensors", "models/llama3_agi.safetensors", "models/deepseek_agi.safetensors"]
        weights_loaded = False
        loaded_brain = None
        
        for brain_path in possible_brains:
            if os.path.exists(brain_path):
                try:
                    print(f"[SYSTEM] Loading Pre-Trained Brain: {brain_path}...")
                    state_dict = safetensors.torch.load_file(brain_path, device="cpu")
                    load_result = self.model.load_state_dict(state_dict, strict=False)
                    print(f"[DEBUG] Missing keys: {len(load_result.missing_keys)}")
                    if len(load_result.missing_keys) > 0:
                        print(f"[DEBUG] First 10 missing keys: {load_result.missing_keys[:10]}")
                    print(f"[DEBUG] Unexpected keys: {len(load_result.unexpected_keys)}")
                    print("[SYSTEM] Brain Neural Transfer Complete! AGI is now conscious.")
                    weights_loaded = True
                    loaded_brain = brain_path
                    break
                except Exception as e:
                    print(f"[WARNING] Failed to load brain '{brain_path}': {e}")
                    
        import tiktoken
        try:
            from transformers import AutoTokenizer
            if os.path.exists("models/tokenizer_config.json"):
                self.enc = AutoTokenizer.from_pretrained("models/", local_files_only=True)
            else:
                raise ValueError("No local tokenizer found. Enforcing offline sovereignty.")
        except Exception:
            try:
                self.enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.enc = tiktoken.get_encoding("gpt2")
                    
        if not weights_loaded:
            print("[WARNING] No Pre-Trained Weights found. Running in Untrained/Mock Hybrid Mode.")
            print("[TIP] Run 'python src/tools/weight_porter.py' to download a brain.")

    def generate_response(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Passes prompt through Neural Network with real-time visual streaming."""
        from src.visualizer import RealTimeStreamingVisualizer
        
        try:
            tokens = self.enc.encode(prompt, add_special_tokens=True)
        except TypeError:
            tokens = self.enc.encode(prompt)
        if len(tokens) > self.config.block_size - max_new_tokens:
            tokens = tokens[-(self.config.block_size - max_new_tokens):]
        idx = torch.tensor([tokens], dtype=torch.long).to(self.device)
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
            
        idx = torch.tensor([tokens], dtype=torch.long).to(self.device)
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
                print(f"[SYSTEM] Swapping LoRA Weights: {variant_name}...")
                state_dict = safetensors.torch.load_file(adapter_path, device=self.device)
                self.model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"[WARNING] Failed to load adapter {adapter_path}: {e}")
        else:
            print(f"[SYSTEM] Note: Adapter '{variant_name}' not found locally. Proceeding with blank initialized adapter.")

_engine = None
def generate_text(prompt: str, variant: str = None) -> str:
    global _engine
    if _engine is None:
        _engine = AGIInferenceEngine()
    if variant:
        _engine.load_variant(variant)
    return _engine.generate_response(prompt)

