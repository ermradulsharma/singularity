import torch
import torch.nn as nn

class P2PTensorShardNode(nn.Module):
    """Micro-Node Peer-to-Peer tensor sharding protocol for offloading matrix computations across local devices."""
    def __init__(self, node_id: str, dim: int = 128, shard_rank: int = 0, total_shards: int = 2):
        super().__init__()
        self.node_id = node_id
        self.dim = dim
        self.shard_rank = shard_rank
        self.total_shards = total_shards
        self.shard_dim = dim // total_shards
        self.linear_shard = nn.Linear(dim, self.shard_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes local tensor slice matrix multiplication."""
        partial_output = self.linear_shard(x)
        return partial_output

    def get_node_status(self) -> dict:
        return {
            "node_id": self.node_id,
            "shard_rank": self.shard_rank,
            "total_shards": self.total_shards,
            "shard_dim": self.shard_dim,
            "active": True
        }
