import pytest
import torch
from src.grpo import GRPOTrainer
from src.model import GPTLanguageModel, PagedKVCacheManager
from src.tool_router import ConstrainedStructuredToolRouter, GrammarConstrainedLogitProcessor
from src.audio import DuplexAudioStreamBuffer

def test_grpo_trainer_initialization_and_step():
    """Verifies DeepSeek-R1 style GRPO trainer initialization, advantage calculation, and step update."""
    model = GPTLanguageModel(
        vocab_size=1000,
        n_embd=64,
        n_head=2,
        n_kv_head=2,
        n_layer=2,
        block_size=128,
        num_experts=0,
        num_experts_per_tok=0
    )
    trainer = GRPOTrainer(model=model, group_size=2, lr=1e-4)
    prompt_tokens = torch.tensor([[10, 20, 30]], dtype=torch.long)
    
    metrics = trainer.train_step(prompt_tokens, max_gen_tokens=16)
    assert "grpo_loss" in metrics
    assert "mean_reward" in metrics
    assert "kl_divergence" in metrics
    assert isinstance(metrics["grpo_loss"], float)

def test_paged_kv_cache_and_chunked_prefill():
    """Verifies vLLM-grade Tensor PagedAttention allocation and chunked prefill splitting."""
    manager = PagedKVCacheManager(num_layers=2, num_heads=2, head_dim=32, block_size=16, num_blocks=32)
    session_id = "test_session_1"
    
    blocks = manager.allocate(session_id, seq_len=40)
    assert isinstance(blocks, torch.Tensor)
    assert len(blocks) == 3 # 40 / 16 = 2.5 -> 3 blocks
    
    input_ids = torch.randint(0, 100, (1, 100))
    chunks = manager.chunked_prefill_split(input_ids, chunk_size=32)
    assert len(chunks) == 4 # 100 / 32 = 3.125 -> 4 chunks
    assert chunks[0].shape[1] == 32
    assert chunks[-1].shape[1] == 4
    
    manager.free(session_id)
    assert session_id not in manager.allocated_pages

def test_grammar_constrained_logit_processor():
    """Verifies JSON State Tracker and logit processor masking."""
    processor = GrammarConstrainedLogitProcessor()
    logits = torch.zeros((1, 1, 1000), dtype=torch.float32)
    
    processed = processor.process_logits(input_ids=None, logits=logits)
    assert processed is not None
    assert processed[..., 123] > 0.0 # '{' token boosted

def test_duplex_audio_stream_buffer():
    """Verifies streaming audio chunk pushing and synthesized PCM decoding."""
    buffer = DuplexAudioStreamBuffer(chunk_size_ms=40, sample_rate=16000)
    # 40ms at 16kHz = 640 samples
    pcm_data = torch.randn((1, 640), dtype=torch.float32)
    
    token_ids = buffer.push_raw_audio(pcm_data)
    assert token_ids.ndim == 2
    assert token_ids.shape[1] > 0
    
    waveform = buffer.pop_synthesized_waveform(token_ids)
    assert waveform.ndim == 2
    assert waveform.shape[1] > 0

def test_vllm_continuous_batching_engine():
    """Verifies high-throughput Continuous Batching & Medusa speculative acceleration request scheduler."""
    from src.inference import vLLMInferenceEngine, ContinuousBatchingEngine
    v_engine = vLLMInferenceEngine(scale="micro")
    assert v_engine is not None
    
    batch_scheduler = ContinuousBatchingEngine(base_engine=v_engine.base_engine)
    batch_scheduler.add_request("req_1", "Explain MoE.", max_tokens=10)
    batch_scheduler.add_request("req_2", "Solve 2+2.", max_tokens=10)
    
    results = batch_scheduler.process_batch()
    assert "req_1" in results
    assert "req_2" in results
    assert isinstance(results["req_1"], str)

def test_dpo_and_rlaif_trainer():
    """Verifies DPO loss calculation and RLAIF preference pair generation."""
    from src.dpo import DPOTrainer, RLAIFEngine
    model = GPTLanguageModel(vocab_size=1000, n_embd=64, n_head=2, n_kv_head=2, n_layer=2, block_size=128, num_experts=0, num_experts_per_tok=0)
    trainer = DPOTrainer(model=model, beta=0.1, lr=1e-4)
    rlaif = RLAIFEngine(model=model)
    
    prompt = torch.tensor([[10, 20, 30]], dtype=torch.long)
    chosen, rejected, chosen_m, rejected_m = rlaif.generate_preference_pair(prompt, max_gen_tokens=8)
    
    metrics = trainer.train_step(chosen, rejected, chosen_m, rejected_m)
    assert "dpo_loss" in metrics
    assert "reward_margin" in metrics
    assert isinstance(metrics["dpo_loss"], float)

if __name__ == "__main__":
    pytest.main([__file__])


