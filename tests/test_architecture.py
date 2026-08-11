import torch
import pytest
import os
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import GPTLanguageModel
from src.inference import ModelArgs

def test_model_forward_pass_stability():
    """
    Architectural CI Test: 
    Validates that the causal transformer topology initializes correctly
    and the forward pass dimensions (MoE / GQA) are mathematically stable.
    """
    config = ModelArgs()
    # Downscale configuration for fast headless CI execution
    model = GPTLanguageModel(
        vocab_size=config.vocab_size, 
        n_embd=64, 
        n_head=2, 
        n_kv_head=1,
        n_layer=2, 
        block_size=128, 
        num_experts=2, 
        num_experts_per_tok=1
    )
    
    # Generate random sequence of tokens
    batch_size = 2
    seq_len = 16
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Execute forward pass
    model.eval()
    with torch.no_grad():
        logits, _ = model(idx)
        
    # Validation 1: Dimensionality Check (B, T, V)
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Dimension crash detected in forward pass!"
    
    # Validation 2: NaN / Stability Check
    assert not torch.isnan(logits).any(), "NaN values detected! Activation or MoE router instability."
    assert logits.abs().max() < 1e4, "Gradient explosion logic detected in raw logits."
