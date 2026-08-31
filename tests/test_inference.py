import os
import sys
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import AGIInferenceEngine, ModelArgs, generate_text

def test_inference_engine_initialization():
    """Validates AGIInferenceEngine setup and default configuration bounds."""
    engine = AGIInferenceEngine()
    assert engine.model is not None
    assert engine.config.vocab_size >= 50257

def test_inference_engine_generation():
    """Validates full text response generation pass."""
    engine = AGIInferenceEngine()
    text = engine.generate_response("Test prompt", max_new_tokens=5)
    assert isinstance(text, str)

def test_async_streaming_token_generator():
    """Validates real-time streaming token generator performance."""
    engine = AGIInferenceEngine()
    prompt = "Hello, world!"
    stream_gen = engine.generate_response_stream(prompt, max_new_tokens=5)
    
    tokens = list(stream_gen)
    assert len(tokens) > 0, "Streaming generator yielded zero tokens!"
    assert isinstance(tokens[0], str), "Yielded token is not a string!"

def test_generate_text_helper():
    """Validates module-level generate_text entry point."""
    res = generate_text("Ping", variant=None)
    assert isinstance(res, str)
