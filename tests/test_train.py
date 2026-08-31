import os
import sys
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import setup_fsdp_model

def test_fsdp_wrapper():
    """Validates FSDP multi-GPU wrapper initialization."""
    dummy = torch.nn.Linear(10, 10)
    wrapped = setup_fsdp_model(dummy, rank=0, world_size=1)
    assert wrapped is not None
