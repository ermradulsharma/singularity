import torch
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prm import StepProcessRewardModel
from src.model import GPTLanguageModel, QuantizedLinear

def test_process_reward_model():
    prm = StepProcessRewardModel(d_model=64)
    sample_steps = [
        "Thought: Let's calculate 2 + 2",
        "Code: ```python\nprint(2 + 2)\n```",
        "Observation: Error: SyntaxError",
        "Final Answer: 4"
    ]
    scores = prm.score_reasoning_steps(sample_steps)
    assert len(scores) == 4
    assert scores[0] > 0.5
    assert scores[1] > 0.7
    assert scores[2] < 0.5

def test_quantized_linear_forward():
    q_layer = QuantizedLinear(in_features=32, out_features=16)
    x = torch.randn(2, 4, 32)
    out = q_layer(x)
    assert out.shape == (2, 4, 16)
    assert not torch.isnan(out).any()
