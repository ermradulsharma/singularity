import torch
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import GPTLanguageModel, precompute_freqs_cis
from src.inference import ModelArgs

def test_yarn_rope_scaling():
    dim, end = 16, 2048
    freqs = precompute_freqs_cis(dim=dim, end=end, scale_factor=16.0)
    assert freqs.shape == (end, dim)
    assert not torch.isnan(freqs).any()

def test_medusa_speculative_decoding():
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
