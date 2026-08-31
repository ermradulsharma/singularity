import os
import sys
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prm import StepProcessRewardModel
from src.evaluator import promote_best_checkpoint

def test_process_reward_model():
    """Validates Process Reward Model step scoring heuristics."""
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

def test_prm_cot_reasoning_extraction_and_scoring():
    """Validates DeepSeek-R1 style Process Reward Model CoT step extraction and trajectory scoring."""
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

def test_checkpoint_evaluator_promotion():
    """Validates automated benchmark checkpoint evaluator promotion logic."""
    res = promote_best_checkpoint(models_dir="models")
    assert isinstance(res, str)
