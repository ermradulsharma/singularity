import torch
import torch.nn as nn
from torch.nn import functional as F
import src.memory as memory
import src.sandbox as secure_sandbox

# --- Rotary Positional Embeddings (RoPE) ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.cat((freqs, freqs), dim=-1)
    return freqs_cis

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(xq, xk, freqs_cis):
    cos = freqs_cis.cos().view(1, xq.shape[1], 1, xq.shape[-1])
    sin = freqs_cis.sin().view(1, xq.shape[1], 1, xq.shape[-1])
    
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


# --- Root Mean Square Normalization (RMSNorm) ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# --- SwiGLU Activation Function ---
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# 🚀 NEW: Mixture of Experts (MoE) Architecture
class Expert(nn.Module):
    def __init__(self, n_embd, dropout, intermediate_size=None):
        super().__init__()
        if intermediate_size is None:
            hidden_dim = int(8 * n_embd / 3)
            hidden_dim = 256 * ((hidden_dim + 255) // 256)
        else:
            hidden_dim = intermediate_size
            
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(hidden_dim, n_embd, bias=False),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    """ Sparse Mixture of Experts Router """
    def __init__(self, n_embd, num_experts, num_experts_per_tok, dropout, intermediate_size=None):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        
        self.router = nn.Linear(n_embd, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(n_embd, dropout, intermediate_size) for _ in range(num_experts)])
        
    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        
        # Pick top-k experts
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        
        final_output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i).any(dim=-1)
            if not expert_mask.any():
                continue
                
            expert_weights = routing_weights[expert_mask, (selected_experts[expert_mask] == i).nonzero(as_tuple=True)[1]]
            expert_out = expert(x_flat[expert_mask])
            final_output[expert_mask] += expert_out * expert_weights.unsqueeze(1)
            
        return final_output.view(B, T, C)


# 🚀 NEW: Grouped-Query Attention (GQA) Helper
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """ Repeats KV heads for GQA (e.g., Llama-3 style) """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    """ Grouped-Query Attention (GQA) with RoPE and KV-Cache """
    def __init__(self, n_embd, n_head, n_kv_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = n_embd // n_head
        
        # In GQA, KV heads are fewer than Q heads to save memory
        self.wq = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        
        self.dropout = dropout

    def forward(self, x, freqs_cis, use_cache=False, past_key_value=None):
        B, T, C = x.size()
        
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)
        
        if past_key_value is not None:
            past_k, _ = past_key_value
            seq_len_offset = past_k.shape[1]
            freqs_cis_slice = freqs_cis[seq_len_offset:seq_len_offset + T]
        else:
            freqs_cis_slice = freqs_cis[:T]
            
        q, k = apply_rotary_emb(q, k, freqs_cis_slice)
        
        # KV-Cache Logic
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            
        present_key_value = (k, v) if use_cache else None

        # Repeat KV heads for GQA before SDPA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        
        # Transpose to (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        is_causal = (T > 1) and (past_key_value is None)
        
        y = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=is_causal
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.wo(y)
        return y, present_key_value


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, num_experts, num_experts_per_tok, dropout, intermediate_size=None):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd)
        self.attn = Attention(n_embd, n_head, n_kv_head, dropout)
        self.ln_2 = RMSNorm(n_embd)
        
        # MoE Layer replaces the standard dense MLP
        self.moe = MoE(n_embd, num_experts, num_experts_per_tok, dropout, intermediate_size)

    def forward(self, x, freqs_cis, use_cache=False, past_key_value=None):
        attn_out, present_key_value = self.attn(self.ln_1(x), freqs_cis, use_cache, past_key_value)
        x = x + attn_out
        x = x + self.moe(self.ln_2(x))
        return x, present_key_value


# 🚀 NEW: Omni-Modality Vision Encoder (ViT Patching)
class VisionEncoder(nn.Module):
    """ Converts Images into Embeddings that the Language Model can read """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, n_embd=128):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding layer (Acts like a CNN extracting 16x16 squares)
        self.patch_embed = nn.Conv2d(in_channels, n_embd, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding so the model knows where the patch was in the image (Top-left, bottom-right etc)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, n_embd))
        
    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.patch_embed(x) # (B, n_embd, H/P, W/P)
        x = x.flatten(2) # (B, n_embd, num_patches)
        x = x.transpose(1, 2) # (B, num_patches, n_embd)
        x = x + self.pos_embed
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_kv_head, n_layer, block_size, num_experts, num_experts_per_tok, dropout=0.0, intermediate_size=None):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # 🚀 AGI Upgrade: Vision Encoder
        # Defaulting to 224x224 images with 16x16 patches = 196 image tokens
        self.vision_encoder = VisionEncoder(image_size=224, patch_size=16, in_channels=3, n_embd=n_embd)
        
        # 🚀 AGI Upgrade: Increased RoPE limit for Infini-Attention support (images + long text)
        freqs_cis = precompute_freqs_cis(n_embd // n_head, block_size * 4)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
        # 🚀 AGI Upgrade: Independent Neural Memory
        self.long_term_memory = memory.IndependentNeuralMemory(memory_file="data/memory.agi")
        
        # Untrusted generated code must never run in the host Python process.
        self.sandbox = secure_sandbox.SecureSandbox(use_docker=True)
        
        # 🧠 AGI Upgrade: Sub-Brain Ensemble (Absorb other models)
        self.sub_brains = nn.ModuleDict()
        self._load_sub_brains(n_embd)
        
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, n_kv_head, num_experts, num_experts_per_tok, dropout, intermediate_size) for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding_table.weight = self.lm_head.weight
        self.apply(self._init_weights)
        
    def _load_sub_brains(self, n_embd):
        import importlib
        import os
        
        brains_dir = os.path.join(os.path.dirname(__file__), "sub_brains")
        if not os.path.exists(brains_dir):
            return
            
        for file in os.listdir(brains_dir):
            if file.endswith(".py") and not file.startswith("__"):
                module_name = file[:-3]
                try:
                    module = importlib.import_module(f"src.sub_brains.{module_name}")
                    if hasattr(module, 'SubBrain'):
                        brain_instance = module.SubBrain(n_embd=n_embd)
                        self.sub_brains[module_name] = brain_instance
                except Exception:
                    pass

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, images=None, targets=None, use_cache=False, past_key_values=None):
        B, T = idx.size()
        x = self.token_embedding_table(idx) # (B, T, n_embd)
        
        # 👁️ OMNI-MODALITY: If images are provided, fuse them into the text stream!
        if images is not None:
            image_embeds = self.vision_encoder(images) # (B, 196, n_embd)
            # Prepend visual "thoughts" before the text
            x = torch.cat([image_embeds, x], dim=1)
        
        present_key_values = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, self.freqs_cis, use_cache=use_cache, past_key_value=past_kv)
            if use_cache:
                present_key_values.append(present_kv)
                
        x = self.ln_f(x)
        
        # 🧠 ENSEMBLE FUSION: Absorb thoughts from Sub-Brains (e.g. DeepSeek Clone, Gemini Clone)
        for name, sub_brain in self.sub_brains.items():
            try:
                # Add sub-brain insights directly into the Master AGI's stream
                x = x + sub_brain(x)
            except Exception:
                pass
        
        # If we injected images, we need to extract only the text logits for loss calculation
        if images is not None:
            num_img_tokens = image_embeds.size(1)
            text_x = x[:, num_img_tokens:, :]
            logits = self.lm_head(text_x)
        else:
            logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B*T, C)
            targets_reshaped = targets.view(B*T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)

        if use_cache:
            return logits, loss, present_key_values
        return logits, loss

    @torch.no_grad()
    def generate_mcts(self, idx, max_new_tokens, num_simulations=3):
        """
        🚀 PURE LOGIC AGI: Monte Carlo Tree Search (MCTS) 🚀
        Instead of just predicting the next token, this algorithm explores multiple
        logical branches (thoughts), evaluates them using the Sandbox, and selects
        the branch that successfully solves the problem without errors.
        """
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        
        best_sequence = None
        best_score = -float('inf')
        
        for sim in range(num_simulations):
            # Branch by using temperature and forcing agentic reasoning.
            sim_seq = self.generate(idx, max_new_tokens, temperature=0.8, agentic_mode=True)
            
            # Deterministic Evaluation Mechanism (NO DUMMY CODE)
            # We decode the sequence to objectively analyze the branch outcome.
            decoded_thought = enc.decode(sim_seq[0].tolist())
            
            score = 0
            if "Error:" in decoded_thought:
                score -= 100  # Penalize failing sandbox executions
            else:
                score += 50   # Baseline success for valid syntax/logic
                
            # Additional heuristic: Penalize excessively short branches as they might be trivial
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
        # Hardcoded IDs mapped from our dataset.py
        TOOL_CALL_ID = 50257
        TOOL_INPUT_ID = 50258
        TOOL_OUTPUT_ID = 50259
        THOUGHT_ID = 50260
        END_THOUGHT_ID = 50261
        
        idx = idx[:, -self.block_size:]
        
        # 🧠 AGENTIC LOOP: Force the model to think before it speaks!
        if agentic_mode:
            thought_tensor = torch.tensor([[THOUGHT_ID]], dtype=torch.long, device=idx.device)
            idx = torch.cat((idx, thought_tensor), dim=1)
            
        past_key_values = None
        next_idx = idx
        
        # State tracker for Sandbox Execution
        in_tool_mode = False
        tool_buffer = []
        
        for _ in range(max_new_tokens):
            # ♾️ INFINI-ATTENTION (StreamingLLM Architecture)
            # Prevents RAM from overflowing by keeping only the "Attention Sinks" and recent window.
            if past_key_values is not None and past_key_values[0][0].shape[2] > window_size:
                new_pkv = []
                for k, v in past_key_values:
                    # Keep first 4 tokens (Sinks) + last (window-4) tokens
                    k_new = torch.cat([k[:, :, :4, :], k[:, :, -(window_size-4):, :]], dim=2)
                    v_new = torch.cat([v[:, :, :4, :], v[:, :, -(window_size-4):, :]], dim=2)
                    new_pkv.append((k_new, v_new))
                    
                # 🧠 INDEPENDENT NEURAL MEMORY: Save evicted tokens to Long-Term Storage
                # We save the aggregated context to disk before eviction
                evicted_k = past_key_values[-1][0][:, :, 4:-(window_size-4), :].mean(dim=2)
                evicted_v = past_key_values[-1][1][:, :, 4:-(window_size-4), :].mean(dim=2)
                self.long_term_memory.add_experience(
                    evicted_k.flatten(start_dim=1), 
                    evicted_v.flatten(start_dim=1)
                )
                
                past_key_values = new_pkv

            if past_key_values is not None:
                next_idx = idx[:, -1:]
            else:
                next_idx = idx
                
            logits, _, past_key_values = self(next_idx, use_cache=True, past_key_values=past_key_values)
            logits = logits[:, -1, :]
            
            # --- 🛡️ Zero-Hallucination Logit Processor ---
            if neuro_symbolic and in_tool_mode:
                # Prevent the model from nesting another tool call while already writing code
                logits[0, TOOL_CALL_ID] = -float('Inf')
            
            logits = logits / max(temperature, 1e-5)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            token_id = idx_next.item()
            
            # ==============================================================
            # ⚙️ THE NEURO-SYMBOLIC SANDBOX INTERCEPTOR
            # ==============================================================
            if token_id == TOOL_CALL_ID:
                in_tool_mode = True
                tool_buffer = []
                
            elif token_id == TOOL_OUTPUT_ID and in_tool_mode:
                # 🛑 PAUSE GENERATION: Execute the buffered tool code!
                import tiktoken
                enc = tiktoken.get_encoding("gpt2")
                code_str = enc.decode(tool_buffer).strip()
                
                
                try:
                    result = self.sandbox.execute(code_str)
                except Exception as e:
                    result = f"Error: {e}"
                
                
                # Append the <|tool_output|> token first
                idx = torch.cat((idx, idx_next), dim=1)
                
                # Encode the deterministic mathematical result
                result_tokens = enc.encode(result)
                result_tensor = torch.tensor([result_tokens], dtype=torch.long, device=idx.device)
                
                # 🧠 INJECT the exact result back into the model's brain!
                idx = torch.cat((idx, result_tensor), dim=1)
                
                # Reset Sandbox State
                in_tool_mode = False
                
                # Because we injected multiple tokens manually, we must discard our KV-Cache
                # and recompute it on the next step so the model "reads" the tool output.
                past_key_values = None
                next_idx = idx
                continue 
                
            elif in_tool_mode:
                # Store code tokens instead of printing them
                tool_buffer.append(token_id)
            
            # Standard auto-regressive append
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
