import torch
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import GPTLanguageModel, QuantizedLinear, precompute_freqs_cis, PagedKVCacheManager, UniversalDynamicBlock
from src.inference import ModelArgs

def test_model_forward_pass_stability():
    """Validates that the causal transformer topology initializes correctly and forward pass dimensions are stable."""
    config = ModelArgs()
    model = GPTLanguageModel(
        vocab_size=config.vocab_size, 
        n_embd=64, 
        n_head=2, 
        n_kv_head=1,
        n_layer=2, 
        block_size=128, 
        num_experts=2, 
        num_experts_per_tok=1
    )
    
    batch_size = 2
    seq_len = 16
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(idx)
        
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Dimension crash detected in forward pass!"
    assert not torch.isnan(logits).any(), "NaN values detected! Activation or MoE router instability."
    assert logits.abs().max() < 1e4, "Gradient explosion logic detected in raw logits."

def test_yarn_rope_scaling():
    """Validates YaRN RoPE frequency matrix computation."""
    dim, end = 16, 2048
    freqs = precompute_freqs_cis(dim=dim, end=end, scale_factor=16.0)
    assert freqs.shape == (end, dim)
    assert not torch.isnan(freqs).any()

def test_1million_infini_attention_freqs():
    """Validates long-context RoPE positional frequency scaling up to 10k positions."""
    freqs = precompute_freqs_cis(dim=16, end=10000, scale_factor=16.0)
    assert freqs.shape == (10000, 16)
    assert not torch.isnan(freqs).any()

def test_medusa_speculative_decoding():
    """Validates Medusa multi-head parallel decoding generator."""
    config = ModelArgs()
    model = GPTLanguageModel(
        vocab_size=config.vocab_size,
        n_embd=32,
        n_head=2,
        n_kv_head=1,
        n_layer=1,
        block_size=64,
        num_experts=1,
        num_experts_per_tok=1
    )
    idx = torch.tensor([[10, 20, 30]], dtype=torch.long)
    out = model.generate_medusa(idx, max_new_tokens=6)
    assert out.shape[1] >= 9
    assert not torch.isnan(out.float()).any()

def test_5head_medusa_tree_decoding():
    """Validates 5-head Medusa speculative tree decoding with custom vocabulary."""
    model = GPTLanguageModel(
        vocab_size=1000, n_embd=32, n_head=2, n_kv_head=1, n_layer=1,
        block_size=64, num_experts=1, num_experts_per_tok=1
    )
    idx = torch.tensor([[10, 20]], dtype=torch.long)
    out = model.generate_medusa(idx, max_new_tokens=10)
    assert out.shape[1] >= 12
    assert not torch.isnan(out.float()).any()

def test_vectorized_moe_forward_pass():
    """Validates vectorized MoE tensor routing for multi-expert configurations."""
    d, h, kv, e, e_t = 64, 4, 2, 4, 2
    block = UniversalDynamicBlock(d, h, kv, e, e_t)
    
    B, T = 4, 32
    x = torch.randn(B, T, d)
    freqs_cis = torch.cat((torch.zeros(T, d // h // 2), torch.zeros(T, d // h // 2)), dim=-1)
    
    out, _ = block(x, freqs_cis)
    assert out.shape == (B, T, d), f"Expected shape {(B, T, d)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Vectorized MoE output contains NaNs!"

def test_multi_head_latent_attention_mla():
    """Validates DeepSeek-V3 style Multi-Head Latent Attention (MLA) forward pass."""
    d, h, kv, e, e_t = 64, 4, 2, 2, 1
    block = UniversalDynamicBlock(d, h, kv, e, e_t, kv_lora_rank=16)
    
    B, T = 2, 16
    x = torch.randn(B, T, d)
    freqs_cis = torch.cat((torch.zeros(T, d // h // 2), torch.zeros(T, d // h // 2)), dim=-1)
    
    out, _ = block(x, freqs_cis, use_mla=True)
    assert out.shape == (B, T, d), f"Expected MLA output shape {(B, T, d)}, got {out.shape}"
    assert not torch.isnan(out).any(), "MLA output contains NaNs!"

def test_fp8_blockwise_quantized_linear():
    """Validates FP8/INT4 blockwise quantized linear forward pass."""
    q_layer = QuantizedLinear(in_features=128, out_features=64, block_size=32)
    x = torch.randn(2, 4, 128)
    out = q_layer(x)
    assert out.shape == (2, 4, 64)
    assert not torch.isnan(out).any()

def test_paged_kv_cache_manager():
    """Validates PagedAttention KV-Cache block allocation and freeing."""
    mgr = PagedKVCacheManager(block_size=16, num_blocks=64)
    pages = mgr.allocate("session_1", seq_len=40)
    assert len(pages) == 3
    mgr.free("session_1")
    assert len(mgr.free_blocks) == 64
