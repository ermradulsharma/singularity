import torch
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import GPTLanguageModel
from src.inference import ModelArgs

def test_model_forward_pass_stability():
    """
    Architectural CI Test: 
    Validates that the causal transformer topology initializes correctly
    and the forward pass dimensions (MoE / GQA) are mathematically stable.
    """
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

def test_vectorized_moe_forward_pass():
    """
    Validates that the vectorized MoE tensor routing works for multi-expert configurations.
    """
    from src.model import UniversalDynamicBlock
    d, h, kv, e, e_t = 64, 4, 2, 4, 2
    block = UniversalDynamicBlock(d, h, kv, e, e_t)
    
    B, T = 4, 32
    x = torch.randn(B, T, d)
    freqs_cis = torch.cat((torch.zeros(T, d // h // 2), torch.zeros(T, d // h // 2)), dim=-1)
    
    out, _ = block(x, freqs_cis)
    assert out.shape == (B, T, d), f"Expected shape {(B, T, d)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Vectorized MoE output contains NaNs!"

def test_multi_head_latent_attention_mla():
    """
    Validates DeepSeek-V3 style Multi-Head Latent Attention (MLA) forward pass.
    """
    from src.model import UniversalDynamicBlock
    d, h, kv, e, e_t = 64, 4, 2, 2, 1
    block = UniversalDynamicBlock(d, h, kv, e, e_t, kv_lora_rank=16)
    
    B, T = 2, 16
    x = torch.randn(B, T, d)
    freqs_cis = torch.cat((torch.zeros(T, d // h // 2), torch.zeros(T, d // h // 2)), dim=-1)
    
    out, _ = block(x, freqs_cis, use_mla=True)
    assert out.shape == (B, T, d), f"Expected MLA output shape {(B, T, d)}, got {out.shape}"
    assert not torch.isnan(out).any(), "MLA output contains NaNs!"

