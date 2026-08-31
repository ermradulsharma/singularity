import os
import sys
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sandbox import SecureSandbox
from src.p2p import P2PTensorShardNode
from src.evaluator import evaluate_humaneval_sample, evaluate_gsm8k_sample

def test_multilang_sandbox():
    """Validates multi-language compiled code execution sandbox filter."""
    sandbox = SecureSandbox(use_docker=False)
    res = sandbox.execute_compiled_lang("int main() { return 0; }", lang="cpp")
    assert "CPP" in res

def test_p2p_tensor_shard():
    """Validates P2P Tensor Sharding across micro-nodes."""
    node = P2PTensorShardNode(node_id="test_node_0", dim=64, shard_rank=0, total_shards=2)
    x = torch.randn(1, 8, 64)
    out = node(x)
    assert out.shape == (1, 8, 32)
    status = node.get_node_status()
    assert status["node_id"] == "test_node_0"
    assert status["active"] is True

def test_benchmark_evaluators():
    """Validates HumanEval and GSM8K benchmark execution accuracy evaluators."""
    code_pass = "def add(a, b):\n    return a + b\nprint(add(2, 3))"
    assert evaluate_humaneval_sample(code_pass) == 1.0
    
    code_fail = "def add(a, b)\n    return a + b"
    assert evaluate_humaneval_sample(code_fail) == 0.0
    
    assert evaluate_gsm8k_sample("Final Answer: 42", "42") == 0.8
    assert evaluate_gsm8k_sample("42", "42") == 1.0
