import torch
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluator import promote_best_checkpoint
from src.p2p import P2PTensorShardNode

def test_p2p_tensor_shard():
    node = P2PTensorShardNode(node_id="test_node_0", dim=64, shard_rank=0, total_shards=2)
    x = torch.randn(1, 8, 64)
    out = node(x)
    assert out.shape == (1, 8, 32)
    status = node.get_node_status()
    assert status["node_id"] == "test_node_0"
    assert status["active"] is True

def test_checkpoint_evaluator_promotion():
    res = promote_best_checkpoint(models_dir="models")
    assert isinstance(res, str)
