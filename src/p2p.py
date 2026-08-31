import io
import socket
import torch
import torch.nn as nn

class P2PTensorShardNode(nn.Module):
    """Micro-Node Peer-to-Peer tensor sharding protocol for offloading matrix computations across network endpoints."""
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

    def send_tensor_over_socket(self, tensor: torch.Tensor, host: str, port: int) -> bool:
        """Serializes and transmits tensor payload to remote P2P micro-node over TCP socket stream."""
        try:
            buffer = io.BytesIO()
            torch.save(tensor.detach().cpu(), buffer)
            payload = buffer.getvalue()
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5.0)
                s.connect((host, port))
                s.sendall(len(payload).to_bytes(8, byteorder='big') + payload)
            return True
        except Exception:
            return False

    @staticmethod
    def receive_tensor_from_socket(sock: socket.socket) -> torch.Tensor:
        """Deserializes tensor payload received from TCP socket connection."""
        raw_size = sock.recv(8)
        if not raw_size:
            return None
        size = int.from_bytes(raw_size, byteorder='big')
        data = bytearray()
        while len(data) < size:
            packet = sock.recv(min(4096, size - len(data)))
            if not packet:
                break
            data.extend(packet)
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=True)

    def get_node_status(self) -> dict:
        return {
            "node_id": self.node_id,
            "shard_rank": self.shard_rank,
            "total_shards": self.total_shards,
            "shard_dim": self.shard_dim,
            "active": True
        }
