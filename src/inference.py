import torch
import tiktoken
import sys
from src.model import GPTLanguageModel

import os
import json

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
        # Dynamically scale architecture based on downloaded config
        if os.path.exists("models/config.json"):
            try:
                with open("models/config.json", "r") as f:
                    config = json.load(f)
                self.vocab_size = config.get("vocab_size", self.vocab_size)
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
        # Attempt to load ported weights from the Neural Weight Porter
        possible_brains = ["models/smollm_agi.pt", "models/tinyllama_agi.pt", "models/uncensored_agi.pt", "models/llama3_agi.pt", "models/deepseek_agi.pt"]
        weights_loaded = False
        loaded_brain = None
        
        for brain_path in possible_brains:
            if os.path.exists(brain_path):
                try:
                    print(f"[SYSTEM] Loading Pre-Trained Brain: {brain_path}...")
                    # Load on CPU first to prevent VRAM spikes, then the model handles device placement
                    state_dict = torch.load(brain_path, map_location=self.device)
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
                    
        # Load correct tokenizer from LOCAL storage (Offline Mode)
        try:
            from transformers import AutoTokenizer
            if os.path.exists("models/tokenizer_config.json"):
                self.enc = AutoTokenizer.from_pretrained("models/")
            else:
                # Fallback to loading dynamically if local files are missing
                if loaded_brain and "smollm" in loaded_brain:
                    self.enc = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
                elif loaded_brain and "tinyllama" in loaded_brain:
                    self.enc = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                elif loaded_brain and "deepseek" in loaded_brain:
                    self.enc = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base")
                else:
                    self.enc = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        except Exception as e:
            print(f"[WARNING] Tokenizer load failed, falling back to GPT-2: {e}")
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
                    
        if not weights_loaded:
            print("[WARNING] No Pre-Trained Weights found. Running in Untrained/Mock Hybrid Mode.")
            print("[TIP] Run 'python src/tools/weight_porter.py' to download a brain.")

    def generate_response(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Passes the prompt through the Neural Network using KV-Cache.
        """
        tokens = self.enc.encode(prompt, add_special_tokens=True)
        if len(tokens) > self.config.block_size - max_new_tokens:
            tokens = tokens[-(self.config.block_size - max_new_tokens):]
        idx = torch.tensor([tokens], dtype=torch.long).to(self.device)
        self.model.eval()
        
        past_key_values = None
        generated_ids = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Pass only the last token if cache exists
                input_idx = idx[:, -1:] if past_key_values is not None else idx
                
                logits, _, past_key_values = self.model(input_idx, use_cache=True, past_key_values=past_key_values)
                logits = logits[:, -1, :]
                
                # Temperature & Top-K Sampling
                temperature = 0.7
                top_k = 50
                logits = logits / temperature
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                idx = torch.cat((idx, idx_next), dim=1)
                generated_ids.append(idx_next.item())
                
                # Basic EOT break
                if hasattr(self.enc, 'eos_token_id') and idx_next.item() == self.enc.eos_token_id:
                    break
                    
            generated_text = self.enc.decode(generated_ids, skip_special_tokens=True)
            return generated_text

# Global singleton to prevent reloading model into GPU multiple times
_engine = None
def generate_text(prompt: str) -> str:
    global _engine
    if _engine is None:
        _engine = AGIInferenceEngine()
    return _engine.generate_response(prompt)
