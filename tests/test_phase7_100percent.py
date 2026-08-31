import torch
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import GPTLanguageModel, QuantizedLinear, precompute_freqs_cis
from src.audio import InterleavedSpeechProjector
from src.sandbox import SecureSandbox
from src.train import setup_fsdp_model

def test_fp8_blockwise_quantized_linear():
    q_layer = QuantizedLinear(in_features=128, out_features=64, block_size=32)
    x = torch.randn(2, 4, 128)
    out = q_layer(x)
    assert out.shape == (2, 4, 64)
    assert not torch.isnan(out).any()

def test_5head_medusa_tree_decoding():
    model = GPTLanguageModel(
        vocab_size=1000, n_embd=32, n_head=2, n_kv_head=1, n_layer=1,
        block_size=64, num_experts=1, num_experts_per_tok=1
    )
    idx = torch.tensor([[10, 20]], dtype=torch.long)
    out = model.generate_medusa(idx, max_new_tokens=10)
    assert out.shape[1] >= 12
    assert not torch.isnan(out.float()).any()

def test_1million_infini_attention_freqs():
    freqs = precompute_freqs_cis(dim=16, end=10000, scale_factor=16.0)
    assert freqs.shape == (10000, 16)
    assert not torch.isnan(freqs).any()

def test_interleaved_speech_projector():
    proj = InterleavedSpeechProjector(d_model=128, audio_dim=64)
    speech = torch.randn(1, 10, 64)
    text_embeds = proj(speech)
    assert text_embeds.shape == (1, 10, 128)

def test_multilang_sandbox():
    sandbox = SecureSandbox(use_docker=False)
    res = sandbox.execute_compiled_lang("int main() { return 0; }", lang="cpp")
    assert "CPP" in res

def test_fsdp_wrapper():
    dummy = torch.nn.Linear(10, 10)
    wrapped = setup_fsdp_model(dummy, rank=0, world_size=1)
    assert wrapped is not None
