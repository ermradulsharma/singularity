import os
import sys
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prm import StepProcessRewardModel
from src.inference import AGIInferenceEngine

def test_prm_cot_reasoning_extraction_and_scoring():
    """
    Validates DeepSeek-R1 style Process Reward Model (PRM) step extraction and trajectory scoring.
    """
    prm = StepProcessRewardModel(d_model=64)
    
    trajectory = """
<think>
Thought: We need to calculate 2 + 2 step-by-step.
Code: Write python code to calculate 2 + 2 inside ```python print(2+2) ``` block.
Observation: 4
</think>
Final Answer: The answer is 4.
"""
    steps = prm.extract_reasoning_steps(trajectory)
    assert len(steps) >= 2, "Failed to extract reasoning steps from trajectory!"
    
    score = prm.score_trajectory(trajectory)
    assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"
    assert score > 0.6, f"Expected high quality score for valid trajectory, got {score}"

def test_async_streaming_token_generator():
    """
    Validates real-time token streaming generator performance.
    """
    engine = AGIInferenceEngine()
    prompt = "Hello, world!"
    stream_gen = engine.generate_response_stream(prompt, max_new_tokens=5)
    
    tokens = list(stream_gen)
    assert len(tokens) > 0, "Streaming generator yielded zero tokens!"
    assert isinstance(tokens[0], str), "Yielded token is not a string!"

def test_openai_api_schema_integrity():
    """
    Validates OpenAI-compatible request and response models in src/api.py.
    """
    from src.api import ChatCompletionRequest, ChatMessage
    
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Explain quantum computing.")]
    )
    assert req.model == "singularity-agi"
    assert req.messages[0].content == "Explain quantum computing."
