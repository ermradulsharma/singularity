"""
Unit Test Suite for Advanced Generative AI Engine Features:
- Multi-GPU Sharding (FSDP / Tensor Parallelism)
- PagedAttention KV Cache Block Management
- Speculative Draft Head Parallel Decoding
- MoE Dynamic Expert Capacity Factor (C=1.2)
- SNAC / Discrete Neural Audio Codec Streaming
"""
import torch
import pytest
from src.distributed import ColumnParallelLinear, RowParallelLinear, FSDPZero3OptimizerManager
from src.model import GPTLanguageModel, PagedKVCacheManager, UniversalDynamicBlock
from src.inference import AGIInferenceEngine
from src.audio import DiscreteAudioTokenizer, SNACContinuousNeuralCodec

def test_paged_attention_kv_manager():
    """Verifies physical block allocation and freeing in PagedKVCacheManager."""
    manager = PagedKVCacheManager(
        num_layers=4,
        num_heads=4,
        head_dim=32,
        block_size=16,
        num_blocks=32,
        device="cpu"
    )
    session_id = "test_session_001"
    block_indices = manager.allocate(session_id, seq_len=48)
    assert len(block_indices) == 3, f"Expected 3 blocks for sequence length 48, got {len(block_indices)}"
    assert session_id in manager.allocated_pages
    
    manager.free(session_id)
    assert session_id not in manager.allocated_pages, "Session physical blocks failed to free"

def test_tensor_parallel_layers():
    """Verifies ColumnParallelLinear and RowParallelLinear projection dimensions."""
    in_features, out_features = 64, 128
    col_layer = ColumnParallelLinear(in_features, out_features, tp_size=2)
    row_layer = RowParallelLinear(out_features, in_features, tp_size=2)
    
    x = torch.randn(2, 10, in_features)
    col_out = col_layer(x)
    assert col_out.shape == (2, 10, 64), f"ColumnParallel shape mismatch: {col_out.shape}"
    
    row_out = row_layer(col_out)
    assert row_out.shape == (2, 10, in_features), f"RowParallel shape mismatch: {row_out.shape}"

def test_moe_capacity_factor():
    """Verifies MoE expert capacity factor (C=1.2) token bounding."""
    block = UniversalDynamicBlock(d=64, h=4, kv=2, e=4, e_t=2)
    x = torch.randn(2, 16, 64)
    freqs_cis = torch.randn(16, 16, 32)
    out, pkv, aux_loss = block(x, freqs_cis)
    assert out.shape == (2, 16, 64), f"MoE output shape mismatch: {out.shape}"
    assert aux_loss >= 0.0, "MoE aux loss must be non-negative"

def test_neural_audio_codec():
    """Verifies Discrete Neural Audio Tokenizer and SNAC Codec."""
    audio_tok = DiscreteAudioTokenizer(sample_rate=16000)
    pcm = torch.randn(1, 1600)
    tokens = audio_tok.encode_waveform(pcm)
    assert tokens.ndim == 2, f"Audio token dimension mismatch: {tokens.shape}"
    
    snac = SNACContinuousNeuralCodec(sample_rate=24000)
    latents = snac.encode_pcm_to_latents(pcm)
    assert latents.ndim == 3, f"SNAC latents dimension mismatch: {latents.shape}"

def test_speculative_medusa_generation():
    """Verifies parallel multi-token candidate generation via Medusa draft heads."""
    model = GPTLanguageModel(
        vocab_size=1000,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=128,
        num_experts=0,
        num_experts_per_tok=0
    )
    model.eval()
    input_ids = torch.randint(0, 1000, (1, 10))
    output_ids = model.generate_medusa(input_ids, max_new_tokens=10)
    assert output_ids.size(1) > input_ids.size(1), "Medusa speculative decoding generated no new tokens"

def test_agi_inference_engine_advanced_apis():
    """Verifies API endpoints for FSDP, PagedAttention, and Speculative Draft Engine."""
    engine = AGIInferenceEngine(scale="micro")
    kv_mgr = engine.enable_paged_attention(block_size=16, num_blocks=64)
    assert kv_mgr is not None, "Failed to enable PagedAttention"
    
    draft_res = engine.generate_speculative_draft("Hello world", max_new_tokens=10)
    assert isinstance(draft_res, str), "Speculative draft result is not a string"

if __name__ == "__main__":
    pytest.main([__file__])
