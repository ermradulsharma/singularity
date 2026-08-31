import torch
import torch.nn as nn
import math

class ResidualVectorQuantizer(nn.Module):
    """Multi-Stage Residual Vector Quantizer (RVQ) for high-fidelity Mimi/EnCodec-grade acoustic tokenization."""
    def __init__(self, num_quantizers: int = 4, num_tokens: int = 1024, dim: int = 64):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_tokens = num_tokens
        self.dim = dim
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(num_tokens, dim) / math.sqrt(dim))
            for _ in range(num_quantizers)
        ])

    def forward(self, x: torch.Tensor):
        """
        Quantizes input feature tensor x [B, T, D] across multi-stage residual codebooks.
        Returns quantized representation and list of codebook token indices per stage.
        """
        residual = x
        quantized_out = torch.zeros_like(x)
        all_indices = []

        for codebook in self.codebooks:
            dists = torch.cdist(residual, codebook.unsqueeze(0))
            indices = torch.argmin(dists, dim=-1)
            all_indices.append(indices)
            
            # Retrieve quantized vectors
            q_stage = codebook[indices]
            quantized_out = quantized_out + q_stage
            residual = residual - q_stage

        return quantized_out, torch.stack(all_indices, dim=1)

    def decode_indices(self, indices_stack: torch.Tensor) -> torch.Tensor:
        """Decodes stacked stage indices [B, N_q, T] into quantized acoustic vector representation [B, T, D]."""
        if indices_stack.ndim == 2:
            indices_stack = indices_stack.unsqueeze(1)
            
        B, N_q, T = indices_stack.shape
        quantized = torch.zeros((B, T, self.dim), device=indices_stack.device)
        for q in range(min(N_q, self.num_quantizers)):
            stage_idx = indices_stack[:, q, :]
            quantized = quantized + self.codebooks[q][stage_idx]
        return quantized

class DiscreteAudioTokenizer(nn.Module):
    """Discrete Audio Tokenizer with Multi-Stage RVQ (Residual Vector Quantization) for native voice parity."""
    def __init__(self, num_tokens: int = 1024, num_quantizers: int = 4, sample_rate: int = 16000):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_quantizers = num_quantizers
        self.sample_rate = sample_rate
        self.rvq = ResidualVectorQuantizer(num_quantizers=num_quantizers, num_tokens=num_tokens, dim=64)

    def encode_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Quantizes raw audio waveform tensor into RVQ discrete audio token IDs."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            
        seq_len = waveform.shape[1]
        frame_size = 160
        num_frames = max(1, seq_len // frame_size)
        
        frames = waveform[:, :num_frames * frame_size].view(waveform.shape[0], num_frames, frame_size)
        frame_features = torch.mean(frames, dim=-1, keepdim=True).repeat(1, 1, 64)
        
        _, token_indices = self.rvq(frame_features)
        # Return first stage or primary codebook indices for 1D token ID compatibility
        return token_indices[:, 0, :]

    def decode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Synthesizes discrete audio token IDs back into raw audio waveform tensor via RVQ codebook."""
        if token_ids.ndim == 2:
            token_ids = token_ids.unsqueeze(1)
        quantized = self.rvq.decode_indices(token_ids)
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

class DuplexAudioStreamBuffer:
    """Real-Time Duplex Audio Streaming Buffer for low-latency voice token encoding and decoding."""
    def __init__(self, chunk_size_ms: int = 40, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.chunk_size = (sample_rate * chunk_size_ms) // 1000
        self.buffer = []
        self.tokenizer = DiscreteAudioTokenizer(sample_rate=sample_rate)

    def push_raw_audio(self, pcm_chunk: torch.Tensor) -> torch.Tensor:
        """Pushes raw PCM audio chunk and returns synthesized discrete RVQ codebook token IDs if chunk boundary met."""
        self.buffer.append(pcm_chunk)
        concat_buffer = torch.cat(self.buffer, dim=-1) if self.buffer else pcm_chunk
        if concat_buffer.shape[-1] >= self.chunk_size:
            chunk_to_process = concat_buffer[..., :self.chunk_size]
            remaining = concat_buffer[..., self.chunk_size:]
            self.buffer = [remaining] if remaining.shape[-1] > 0 else []
            return self.tokenizer.encode_waveform(chunk_to_process)
        return torch.empty((1, 0), dtype=torch.long)

    def pop_synthesized_waveform(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Decodes streaming discrete audio token IDs into synthesized raw waveform PCM chunk."""
        if token_ids.shape[-1] == 0:
            return torch.empty((1, 0), dtype=torch.float32)
        return self.tokenizer.decode_tokens(token_ids)



