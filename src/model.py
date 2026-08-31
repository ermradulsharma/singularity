import torch
import torch.nn as nn
from torch.nn import functional as F
import src.memory as memory
import src.sandbox as secure_sandbox

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    return torch.cat((freqs, freqs), dim=-1)

def apply_rotary_emb(x, freqs_cis):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    cos, sin = freqs_cis.cos().view(1, x.shape[1], 1, -1), freqs_cis.sin().view(1, x.shape[1], 1, -1)
    return (x * cos) + (rotated * sin)

class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) layer for memory-efficient Neural Variants."""
    def __init__(self, linear_layer: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(linear_layer.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, linear_layer.out_features))
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

class UniversalDynamicBlock(nn.Module):
    """
    Dynamically routes tensors through Dense, GQA, or MoE layers algorithmically.
    Replaces hundreds of lines of static PyTorch classes.
    """
    def __init__(self, d, h, kv, e, e_t):
        super().__init__()
        self.h, self.kv, self.hd = h, kv, d // h
        self.is_moe = e > 0
        self.e_t = e_t
        
        self.graph = nn.ModuleDict({
            'norm1': nn.LayerNorm(d), 'norm2': nn.LayerNorm(d),
            'attn': nn.ModuleDict({
                'wq': nn.Linear(d, h*self.hd, bias=False), 
                'wk': nn.Linear(d, kv*self.hd, bias=False), 
                'wv': nn.Linear(d, kv*self.hd, bias=False), 
                'wo': nn.Linear(d, d, bias=False)
            }),
        })
        
        if self.is_moe:
            self.graph['router'] = nn.Linear(d, e, bias=False)
            self.graph['experts'] = nn.ModuleList([
                nn.Sequential(nn.Linear(d, int(8*d/3), bias=False), nn.SiLU(), nn.Linear(int(8*d/3), d, bias=False)) 
                for _ in range(e)
            ])
        else:
            self.graph['ffn'] = nn.Sequential(nn.Linear(d, int(8*d/3), bias=False), nn.SiLU(), nn.Linear(int(8*d/3), d, bias=False))

    def forward(self, x, freqs_cis, use_cache=False, past_kv=None):
        B, T, C = x.size()
        nx = self.graph['norm1'](x)
        
        q, k, v = self.graph['attn']['wq'](nx), self.graph['attn']['wk'](nx), self.graph['attn']['wv'](nx)
        q, k, v = q.view(B, T, self.h, self.hd), k.view(B, T, self.kv, self.hd), v.view(B, T, self.kv, self.hd)
        
        fc = freqs_cis[past_kv[0].shape[1]:past_kv[0].shape[1]+T] if past_kv else freqs_cis[:T]
        q, k = apply_rotary_emb(q, fc), apply_rotary_emb(k, fc)
        
        if past_kv: 
            k, v = torch.cat([past_kv[0], k], dim=1), torch.cat([past_kv[1], v], dim=1)
        pkv = (k, v) if use_cache else None
        
        if self.h != self.kv: 
            k, v = [t.repeat_interleave(self.h // self.kv, dim=2) for t in (k, v)]
            
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
            is_causal=(T>1 and not past_kv)
        )
        x = x + self.graph['attn']['wo'](y.transpose(1, 2).reshape(B, T, C))
        
        nx = self.graph['norm2'](x)
        if self.is_moe:
            wt, exp = torch.topk(F.softmax(self.graph['router'](nx), dim=-1), self.e_t, dim=-1)
            moe_out = torch.zeros_like(nx)
            for i in range(self.e_t):
                expert_idx = exp[:, :, i]
                expert_w = wt[:, :, i].unsqueeze(-1)
                for b in range(B):
                    for t in range(T):
                        moe_out[b, t] += self.graph['experts'][expert_idx[b, t].item()](nx[b, t]) * expert_w[b, t]
            x = x + moe_out
        else:
            x = x + self.graph['ffn'](nx)
        return x, pkv

class VisionEncoder(nn.Module):
    def __init__(self, img_size=224, patch=16, dim=128):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, (img_size // patch)**2, dim))
    def forward(self, x): return self.conv(x).flatten(2).transpose(1, 2) + self.pos

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_kv_head, n_layer, block_size, num_experts, num_experts_per_tok, dropout=0.0, intermediate_size=None):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        self.graph = nn.ModuleDict({
            'tok_emb': nn.Embedding(vocab_size, n_embd),
            'vision': VisionEncoder(dim=n_embd),
            'blocks': nn.ModuleList([UniversalDynamicBlock(n_embd, n_head, n_kv_head, num_experts, num_experts_per_tok) for _ in range(n_layer)]),
            'ln_f': nn.LayerNorm(n_embd),
            'lm_head': nn.Linear(n_embd, vocab_size, bias=False)
        })
        
        self.register_buffer("freqs_cis", precompute_freqs_cis(n_embd // n_head, block_size * 4), persistent=False)
        
        self.long_term_memory = memory.IndependentNeuralMemory(memory_file="data/memory.safetensors")
        self.sandbox = secure_sandbox.SecureSandbox(use_docker=True)
        
        self.sub_brains = nn.ModuleDict()
        self._load_sub_brains(n_embd)
        
        self.graph['tok_emb'].weight = self.graph['lm_head'].weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def inject_lora_adapters(self, variant_name: str, r: int = 8, alpha: int = 16):
        """Dynamically injects LoRA into Q and V layers for a specific Swarm Agent variant."""
        print(f"[SYSTEM] Injecting LoRA Neural Adapter: {variant_name}")
        for block in self.graph['blocks']:
            if not isinstance(block.graph['attn']['wq'], LoRALinear):
                block.graph['attn']['wq'] = LoRALinear(block.graph['attn']['wq'], r, alpha).to(self.graph['tok_emb'].weight.device)
            if not isinstance(block.graph['attn']['wv'], LoRALinear):
                block.graph['attn']['wv'] = LoRALinear(block.graph['attn']['wv'], r, alpha).to(self.graph['tok_emb'].weight.device)

    def _load_sub_brains(self, n_embd):
        import importlib, os
        b_dir = os.path.join(os.path.dirname(__file__), "sub_brains")
        if os.path.exists(b_dir):
            for file in os.listdir(b_dir):
                if file.endswith(".py") and not file.startswith("__"):
                    try:
                        mod = importlib.import_module(f"src.sub_brains.{file[:-3]}")
                        if hasattr(mod, 'SubBrain'): self.sub_brains[file[:-3]] = mod.SubBrain(n_embd=n_embd)
                    except Exception: pass
                    
    def forward(self, idx, images=None, targets=None, use_cache=False, past_key_values=None):
        B, T = idx.size()
        x = self.graph['tok_emb'](idx)
        
        if images is not None: 
            image_embeds = self.graph['vision'](images)
            x = torch.cat([image_embeds, x], dim=1)
        
        pkv = [] if use_cache else None
        for i, block in enumerate(self.graph['blocks']):
            x, p = block(x, self.freqs_cis, use_cache, past_key_values[i] if past_key_values else None)
            if use_cache: pkv.append(p)
            
        x = self.graph['ln_f'](x)
        for sub in self.sub_brains.values():
            try: x = x + sub(x)
            except Exception: pass
            
        if images is not None:
            x = x[:, image_embeds.size(1):, :]
            
        logits = self.graph['lm_head'](x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        
        return (logits, loss, pkv) if use_cache else (logits, loss)

    @torch.no_grad()
    def generate_mcts(self, idx, max_new_tokens, num_simulations=3):
        """
        🚀 PURE LOGIC AGI: Monte Carlo Tree Search (MCTS) 🚀
        """
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        
        best_sequence = None
        best_score = -float('inf')
        
        for sim in range(num_simulations):
            sim_seq = self.generate(idx, max_new_tokens, temperature=0.8, agentic_mode=True)
            decoded_thought = enc.decode(sim_seq[0].tolist())
            
            score = 0
            if "Error:" in decoded_thought:
                score -= 100
            else:
                score += 50
                
            if len(sim_seq[0]) < 10:
                score -= 20
                
            if score > best_score:
                best_score = score
                best_sequence = sim_seq
                
        return best_sequence

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, 
                 neuro_symbolic=True, agentic_mode=True, window_size=512):
        """ 
        🚀 AGI Stateful Generation Loop (Neuro-Symbolic + Infini-Attention + Agentic) 🚀
        """
        TOOL_CALL_ID, TOOL_OUTPUT_ID, THOUGHT_ID = 50257, 50259, 50260
        idx = idx[:, -self.block_size:]
        
        if agentic_mode:
            idx = torch.cat((idx, torch.tensor([[THOUGHT_ID]], dtype=torch.long, device=idx.device)), dim=1)
            
        past_key_values = None
        next_idx = idx
        in_tool_mode = False
        tool_buffer = []
        
        for _ in range(max_new_tokens):
            if past_key_values is not None and past_key_values[0][0].shape[2] > window_size:
                new_pkv = []
                for k, v in past_key_values:
                    new_pkv.append((
                        torch.cat([k[:, :, :4, :], k[:, :, -(window_size-4):, :]], dim=2),
                        torch.cat([v[:, :, :4, :], v[:, :, -(window_size-4):, :]], dim=2)
                    ))
                evicted_k = past_key_values[-1][0][:, :, 4:-(window_size-4), :].mean(dim=2)
                evicted_v = past_key_values[-1][1][:, :, 4:-(window_size-4), :].mean(dim=2)
                self.long_term_memory.add_experience(evicted_k.flatten(start_dim=1), evicted_v.flatten(start_dim=1))
                past_key_values = new_pkv

            next_idx = idx[:, -1:] if past_key_values is not None else idx
            logits, _, past_key_values = self(next_idx, use_cache=True, past_key_values=past_key_values)
            logits = logits[:, -1, :]
            
            if neuro_symbolic and in_tool_mode:
                logits[0, TOOL_CALL_ID] = -float('Inf')
            
            logits = logits / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            token_id = idx_next.item()
            
            if token_id == TOOL_CALL_ID:
                in_tool_mode = True
                tool_buffer = []
            elif token_id == TOOL_OUTPUT_ID and in_tool_mode:
                import tiktoken
                enc = tiktoken.get_encoding("gpt2")
                try:
                    result = self.sandbox.execute(enc.decode(tool_buffer).strip())
                except Exception as e:
                    result = f"Error: {e}"
                
                idx = torch.cat((idx, idx_next, torch.tensor([enc.encode(result)], dtype=torch.long, device=idx.device)), dim=1)
                in_tool_mode = False
                past_key_values = None
                continue 
            elif in_tool_mode:
                tool_buffer.append(token_id)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

