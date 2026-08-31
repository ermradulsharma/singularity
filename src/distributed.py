"""Industrial Multi-Node / Multi-GPU Distributed Orchestration System for Singularity AGI Engine."""
import os
import torch
import torch.distributed as dist

class DistributedClusterManager:
    """Manages multi-node and multi-GPU distributed cluster initialization, FSDP process groups, and rank topologies."""

    def __init__(self):
        self.is_distributed = False
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_cluster(self, backend: str = None) -> bool:
        """Initializes PyTorch torch.distributed process group for DDP and FSDP execution across nodes."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

            if backend is None:
                backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"

            if not dist.is_initialized():
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.local_rank)
                    self.device = torch.device(f"cuda:{self.local_rank}")
                
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    world_size=self.world_size,
                    rank=self.rank
                )
            
            self.is_distributed = True
            return True
        else:
            return False

    def barrier(self):
        """Executes a global synchronization barrier across all distributed ranks."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier()

    def cleanup(self):
        """Destroys distributed process group upon training completion."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
            self.is_distributed = False

    @staticmethod
    def get_world_size() -> int:
        """Returns current global world size across all nodes."""
        if dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def get_rank() -> int:
        """Returns global rank of current node/GPU."""
        if dist.is_initialized():
            return dist.get_rank()
        return 0

cluster_manager = DistributedClusterManager()
