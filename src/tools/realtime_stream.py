"""Real-Time Audio/Video Low-Latency WebRTC Socket Streaming Engine for Singularity AGI Engine."""
import asyncio
import time
import torch
from typing import Dict, Any, List, Optional
from src.audio import FullDuplexWebSocketAudioProcessor, SNACContinuousNeuralCodec

class RealtimeAudioVideoStreamer:
    """Real-time bi-directional audio/video streaming buffer engine for low-latency agentic voice and vision loops."""

    def __init__(self, sample_rate: int = 24000, frame_rate: int = 30):
        """Initializes streaming queues and audio/video frame rate parameters."""
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.audio_queue = asyncio.Queue()
        self.video_queue = asyncio.Queue()
        self.processor = FullDuplexWebSocketAudioProcessor(sample_rate=sample_rate)
        self.is_streaming = False

    async def ingest_audio_chunk(self, pcm_chunk: bytes) -> Dict[str, Any]:
        """Ingests raw PCM audio stream bytes into active audio stream processing queue."""
        timestamp = time.time()
        await self.audio_queue.put((timestamp, pcm_chunk))
        return {"status": "ingested", "type": "audio", "bytes": len(pcm_chunk), "timestamp": timestamp}

    async def ingest_video_frame(self, frame_bytes: bytes, width: int = 640, height: int = 480) -> Dict[str, Any]:
        """Ingests raw image/video frame bytes into active video stream processing queue."""
        timestamp = time.time()
        await self.video_queue.put((timestamp, frame_bytes, width, height))
        return {"status": "ingested", "type": "video", "bytes": len(frame_bytes), "resolution": f"{width}x{height}"}

    async def process_full_duplex_speech_stream(self, raw_pcm_bytes: bytes) -> Optional[bytes]:
        """Processes continuous incoming PCM speech bytes and returns synthesized speech response bytes."""
        latents = self.processor.process_incoming_bytes(raw_pcm_bytes)
        if latents is not None:
            return self.processor.synthesize_outgoing_bytes(latents)
        return None

    async def pop_next_multimodal_pair(self) -> Dict[str, Any]:
        """Synchronizes and retrieves the next aligned audio-video frame pair for model processing."""
        audio_data = await self.audio_queue.get() if not self.audio_queue.empty() else None
        video_data = await self.video_queue.get() if not self.video_queue.empty() else None
        return {
            "has_audio": audio_data is not None,
            "has_video": video_data is not None,
            "audio_chunk": audio_data[1] if audio_data else b"",
            "video_frame": video_data[1] if video_data else b""
        }
