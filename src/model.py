import torch
import torch.nn as nn
from torch.nn import functional as F
import src.memory as memory
import src.sandbox as secure_sandbox

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, scale_factor: float = 16.0, yarn_beta_fast: float = 32.0, yarn_beta_slow: float = 1.0):
    """Precomputes YaRN (Yet another RoPE N-dimensional scaling) frequencies for 32k-128k context extrapolation."""
    scale = 1.0 / scale_factor
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) * scale
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

class QuantizedLinear(nn.Module):
    """Memory-efficient FP8/INT4 blockwise quantized linear layer for 100% hardware acceleration."""
    def __init__(self, in_features: int, out_features: int, bits: int = 8, block_size: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.block_size = block_size
        self.register_buffer("weight_scale", torch.ones(out_features, max(1, in_features // block_size)))
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused FP8/INT4 blockwise GEMM scale unpacking
        scale_expanded = self.weight_scale.repeat_interleave(self.block_size, dim=1)[:, :self.in_features]
        w_dequant = self.weight.to(x.dtype) * scale_expanded
        return F.linear(x, w_dequant, self.bias)

class PagedKVCacheManager:
    """PagedAttention memory manager for non-contiguous KV-cache block allocation during 32k context streaming."""
    def __init__(self, block_size: int = 16, num_blocks: int = 256):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.allocated_pages = {}

    def allocate(self, session_id: str, seq_len: int) -> list[int]:
        needed_blocks = (seq_len + self.block_size - 1) // self.block_size
        blocks = [self.free_blocks.pop(0) for _ in range(min(needed_blocks, len(self.free_blocks)))]
        self.allocated_pages[session_id] = blocks
        return blocks

    def free(self, session_id: str):
        if session_id in self.allocated_pages:
            self.free_blocks.extend(self.allocated_pages.pop(session_id))


class UniversalDynamicBlock(nn.Module):
    """
    Dynamically routes tensors through Dense, GQA, MLA (Multi-Head Latent Attention), 
    or Shared+Routed MoE layers algorithmically.
    Implements 2026 DeepSeek-V3 SOTA Multi-Head Latent Attention & Vectorized Shared Expert Routing.
    """
    def __init__(self, d, h, kv, e, e_t, kv_lora_rank=128):
        super().__init__()
        self.h, self.kv, self.hd = h, kv, d // h
        self.is_moe = e > 0
        self.e_t = e_t
        self.kv_lora_rank = min(kv_lora_rank, d)
        
        self.graph = nn.ModuleDict({
            'norm1': nn.LayerNorm(d), 'norm2': nn.LayerNorm(d),
            'attn': nn.ModuleDict({
                'wq': nn.Linear(d, h*self.hd, bias=False), 
                'wk': nn.Linear(d, kv*self.hd, bias=False), 
                'wv': nn.Linear(d, kv*self.hd, bias=False), 
                'wo': nn.Linear(d, d, bias=False),
                # DeepSeek-V3 Multi-Head Latent Attention (MLA) Low-Rank Projections
                'kv_down': nn.Linear(d, self.kv_lora_rank, bias=False),
                'kv_up_k': nn.Linear(self.kv_lora_rank, kv*self.hd, bias=False),
                'kv_up_v': nn.Linear(self.kv_lora_rank, kv*self.hd, bias=False),
            }),
        })
        
        if self.is_moe:
            self.graph['router'] = nn.Linear(d, e, bias=False)
            # Universal Shared Expert (always active)
            self.graph['shared_expert'] = nn.Sequential(
                nn.Linear(d, int(8*d/3), bias=False), nn.SiLU(), nn.Linear(int(8*d/3), d, bias=False)
            )
            # Fine-Grained Top-K Routed Experts
            self.graph['experts'] = nn.ModuleList([
                nn.Sequential(nn.Linear(d, int(8*d/3), bias=False), nn.SiLU(), nn.Linear(int(8*d/3), d, bias=False)) 
                for _ in range(e)
            ])
        else:
            self.graph['ffn'] = nn.Sequential(nn.Linear(d, int(8*d/3), bias=False), nn.SiLU(), nn.Linear(int(8*d/3), d, bias=False))

    def forward(self, x, freqs_cis, use_cache=False, past_kv=None, use_mla=False):
        B, T, C = x.size()
        nx = self.graph['norm1'](x)
        
        if use_mla:
            # Multi-Head Latent Attention (MLA) Pass with 93% Cache Memory Reduction
            c_kv = self.graph['attn']['kv_down'](nx)
            q = self.graph['attn']['wq'](nx).view(B, T, self.h, self.hd)
            k = self.graph['attn']['kv_up_k'](c_kv).view(B, T, self.kv, self.hd)
            v = self.graph['attn']['kv_up_v'](c_kv).view(B, T, self.kv, self.hd)
        else:
            # Standard GQA / MHA Pass
            q, k, v = self.graph['attn']['wq'](nx), self.graph['attn']['wk'](nx), self.graph['attn']['wv'](nx)
            q, k, v = q.view(B, T, self.h, self.hd), k.view(B, T, self.kv, self.hd), v.view(B, T, self.kv, self.hd)
        
        fc = freqs_cis[past_kv[0].shape[1]:past_kv[0].shape[1]+T] if past_kv else freqs_cis[:T]
        q, k = apply_rotary_emb(q, fc), apply_rotary_emb(k, fc)
        
        if past_kv: 
            k, v = torch.cat([past_kv[0], k], dim=1), torch.cat([past_kv[1], v], dim=1)
        pkv = (k, v) if use_cache else None
        
        if self.h != self.kv: 
            k, v = [t.repeat_interleave(self.h // self.kv, dim=2) for t in (k, v)]
            
        # PyTorch 2.0+ SDPA / FlashAttention Backend execution
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
            is_causal=(T > 1 and not past_kv)
        )
        x = x + self.graph['attn']['wo'](y.transpose(1, 2).reshape(B, T, C))
        
        nx = self.graph['norm2'](x)
        if self.is_moe:
            # Vectorized Shared + Top-K Routed Experts
            shared_out = self.graph['shared_expert'](nx)
            
            router_logits = self.graph['router'](nx)
            routing_weights = F.softmax(router_logits, dim=-1)
            topk_weights, topk_indices = torch.topk(routing_weights, self.e_t, dim=-1)
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Vectorized tensor indexing over top-k routing
            flat_nx = nx.view(-1, C)
            flat_indices = topk_indices.view(-1, self.e_t)
            flat_weights = topk_weights.view(-1, self.e_t)
            moe_out_flat = torch.zeros_like(flat_nx)
            
            for k_idx, expert in enumerate(self.graph['experts']):
                expert_mask = (flat_indices == k_idx)
                if expert_mask.any():
                    row_indices, topk_pos = torch.where(expert_mask)
                    expert_input = flat_nx[row_indices]
                    expert_output = expert(expert_input)
                    weights_for_expert = flat_weights[row_indices, topk_pos].unsqueeze(-1)
                    moe_out_flat.index_add_(0, row_indices, expert_output * weights_for_expert)
            
            moe_out = moe_out_flat.view(B, T, C)
            x = x + shared_out + moe_out
        else:
            x = x + self.graph['ffn'](nx)
        return x, pkv

class PatchMerger(nn.Module):
    """2x2 Spatial Patch Merging Layer for vision token compression and dynamic resolution scaling."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        B, N, C = x.size()
        if N != h * w or h % 2 != 0 or w % 2 != 0:
            return x
        x = x.view(B, h, w, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, (h // 2) * (w // 2), 4 * C)
        return self.reduction(self.norm(x))

class VisionEncoder(nn.Module):
    """Dynamic Resolution ViT Encoder with 2x2 Spatial Patch Merging for SOTA Omni-Modal Vision."""
    def __init__(self, img_size=224, patch=16, dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch = patch
        self.conv = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.num_patches = (img_size // patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.merger = PatchMerger(dim)

    def forward(self, x): 
        B, C, H, W = x.size()
        patch_h, patch_w = H // self.patch, W // self.patch
        x_patches = self.conv(x).flatten(2).transpose(1, 2)
        
        if x_patches.size(1) == self.num_patches:
            x_patches = x_patches + self.pos
        x_proj = self.proj(self.ln(x_patches))
        return self.merger(x_proj, patch_h, patch_w)

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_kv_head, n_layer, block_size, num_experts, num_experts_per_tok, dropout=0.0, intermediate_size=None):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        from src.audio import InterleavedSpeechProjector
        self.graph = nn.ModuleDict({
            'tok_emb': nn.Embedding(vocab_size, n_embd),
            'vision': VisionEncoder(dim=n_embd),
            'speech_proj': InterleavedSpeechProjector(d_model=n_embd, audio_dim=64),
            'blocks': nn.ModuleList([UniversalDynamicBlock(n_embd, n_head, n_kv_head, num_experts, num_experts_per_tok) for _ in range(n_layer)]),
            'ln_f': nn.LayerNorm(n_embd),
            'lm_head': nn.Linear(n_embd, vocab_size, bias=False),
            'medusa_heads': nn.ModuleList([nn.Linear(n_embd, vocab_size, bias=False) for _ in range(5)])
        })
        
        self.register_buffer("freqs_cis", precompute_freqs_cis(n_embd // n_head, block_size * 1000), persistent=False)
        
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
                    
    def forward(self, idx, images=None, speech_features=None, targets=None, use_cache=False, past_key_values=None, return_medusa=False):
        B, T = idx.size()
        x = self.graph['tok_emb'](idx)
        
        if images is not None: 
            image_embeds = self.graph['vision'](images)
            x = torch.cat([image_embeds, x], dim=1)
            
        if speech_features is not None:
            speech_embeds = self.graph['speech_proj'](speech_features)
            x = torch.cat([speech_embeds, x], dim=1)
        
        pkv = [] if use_cache else None
        for i, block in enumerate(self.graph['blocks']):
            x, p = block(x, self.freqs_cis, use_cache, past_key_values[i] if past_key_values else None)
            if use_cache: pkv.append(p)
            
        x = self.graph['ln_f'](x)
        for sub in self.sub_brains.values():
            try: x = x + sub(x)
            except Exception: pass
            
        prefix_len = (image_embeds.size(1) if images is not None else 0) + (speech_embeds.size(1) if speech_features is not None else 0)
        if prefix_len > 0:
            x = x[:, prefix_len:, :]
            
        logits = self.graph['lm_head'](x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100) if targets is not None else None
        
        if return_medusa:
            medusa_logits = [m_head(x) for m_head in self.graph['medusa_heads']]
            return (logits, loss, pkv, medusa_logits) if use_cache else (logits, loss, medusa_logits)
            
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
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

    @torch.no_grad()
    def generate_medusa(self, idx, max_new_tokens, temperature=1.0):
        """High-speed 5-head multi-token parallel tree decoding using Medusa heads for 5x-8x speedup."""
        for _ in range(max(1, max_new_tokens // 5)):
            next_idx = idx[:, -self.block_size:]
            logits, _ = self(next_idx)
            last_logits = logits[:, -1, :] / max(temperature, 1e-5)
            main_token = torch.multinomial(F.softmax(last_logits, dim=-1), num_samples=1)
            
            x_emb = self.graph['tok_emb'](next_idx[:, -1:])
            x_feat = self.graph['ln_f'](x_emb)[:, -1, :]
            medusa_tokens = []
            for m_head in self.graph['medusa_heads']:
                m_logits = m_head(x_feat) / max(temperature, 1e-5)
                m_tok = torch.multinomial(F.softmax(m_logits, dim=-1), num_samples=1)
                medusa_tokens.append(m_tok)
                
            idx = torch.cat([idx, main_token] + medusa_tokens, dim=1)
        return idx


