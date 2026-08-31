import torch
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio import DiscreteAudioTokenizer
from src.model import PagedKVCacheManager
from src.tools.dashboard import get_telemetry_summary

def test_discrete_audio_tokenizer():
    tokenizer = DiscreteAudioTokenizer(num_tokens=128)
    dummy_waveform = torch.randn(1, 1600)
    token_ids = tokenizer.encode_waveform(dummy_waveform)
    assert token_ids.shape[1] == 10
    recon = tokenizer.decode_tokens(token_ids)
    assert recon.shape[1] == 1600
    assert not torch.isnan(recon).any()

def test_paged_kv_cache_manager():
    mgr = PagedKVCacheManager(block_size=16, num_blocks=64)
    pages = mgr.allocate("session_1", seq_len=40)
    assert len(pages) == 3
    mgr.free("session_1")
    assert len(mgr.free_blocks) == 64

def test_dashboard_summary():
    res = get_telemetry_summary()
    assert isinstance(res, str)
