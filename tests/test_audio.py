import os
import sys
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio import DiscreteAudioTokenizer, InterleavedSpeechProjector

def test_discrete_audio_tokenizer():
    """Validates DiscreteAudioTokenizer waveform encoding and decoding."""
    tokenizer = DiscreteAudioTokenizer(num_tokens=128)
    dummy_waveform = torch.randn(1, 1600)
    token_ids = tokenizer.encode_waveform(dummy_waveform)
    assert token_ids.shape[1] == 10
    recon = tokenizer.decode_tokens(token_ids)
    assert recon.shape[1] == 1600
    assert not torch.isnan(recon).any()

def test_interleaved_speech_projector():
    """Validates InterleavedSpeechProjector text embedding projection."""
    proj = InterleavedSpeechProjector(d_model=128, audio_dim=64)
    speech = torch.randn(1, 10, 64)
    text_embeds = proj(speech)
    assert text_embeds.shape == (1, 10, 128)
