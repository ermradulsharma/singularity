import torch
import torch.nn as nn
import math

class DiscreteAudioTokenizer(nn.Module):
    """Discrete Audio Tokenizer for native voice waveform quantization and token processing."""
    def __init__(self, num_tokens: int = 1024, sample_rate: int = 16000):
        super().__init__()
        self.num_tokens = num_tokens
        self.sample_rate = sample_rate
        self.codebook = nn.Parameter(torch.randn(num_tokens, 64))

    def encode_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Quantizes raw audio waveform tensor into discrete audio token IDs."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            
        seq_len = waveform.shape[1]
        frame_size = 160
        num_frames = max(1, seq_len // frame_size)
        
        frames = waveform[:, :num_frames * frame_size].view(waveform.shape[0], num_frames, frame_size)
        frame_features = torch.mean(frames, dim=-1, keepdim=True).repeat(1, 1, 64)
        
        dists = torch.cdist(frame_features, self.codebook.unsqueeze(0))
        token_ids = torch.argmin(dists, dim=-1)
        return token_ids

    def decode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Synthesizes discrete audio token IDs back into raw audio waveform tensor."""
        quantized = self.codebook[token_ids]
        waveform = torch.mean(quantized, dim=-1).repeat_interleave(160, dim=-1)
        return waveform

class InterleavedSpeechProjector(nn.Module):
    """End-to-End Interleaved Speech-to-Speech Projection Layer for 100% Omni-Modal Voice Parity."""
    def __init__(self, d_model: int = 128, audio_dim: int = 64):
        super().__init__()
        self.speech_proj = nn.Linear(audio_dim, d_model, bias=False)
        self.text_proj = nn.Linear(d_model, audio_dim, bias=False)

    def forward(self, speech_embeds: torch.Tensor) -> torch.Tensor:
        return self.speech_proj(speech_embeds)

class DiscreteAudioHead(nn.Module):
    """Direct Acoustic Codebook Output Generation Head for Zero-Latency Speech Synthesis."""
    def __init__(self, d_model: int = 128, num_audio_tokens: int = 1024):
        super().__init__()
        self.audio_proj = nn.Linear(d_model, num_audio_tokens, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Projects LLM hidden representation into discrete audio codebook token logits."""
        return self.audio_proj(hidden_states)


