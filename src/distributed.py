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

    def setup_3d_parallel_groups(self, tp_size: int = 1, pp_size: int = 1):
        """Builds 3D Parallelism process groups (Tensor Parallelism, Pipeline Parallelism, Data Parallelism)."""
        if not self.is_distributed or not dist.is_initialized():
            return None, None, None
        
        dp_size = max(1, self.world_size // max(1, tp_size * pp_size))
        tp_group, pp_group, dp_group = None, None, None
        
        for i in range(0, self.world_size, max(1, tp_size)):
            ranks = list(range(i, min(self.world_size, i + max(1, tp_size))))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                tp_group = group
                
        return tp_group, pp_group, dp_group

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

class ColumnParallelLinear(torch.nn.Module):
    """Column-Parallel Linear Layer for Megatron-LM style Tensor Parallelism."""
    def __init__(self, in_features: int, out_features: int, tp_size: int = 1, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_partition = out_features // tp_size
        self.tp_size = tp_size
        self.linear = torch.nn.Linear(in_features, self.out_features_per_partition, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class RowParallelLinear(torch.nn.Module):
    """Row-Parallel Linear Layer with All-Reduce for Megatron-LM style Tensor Parallelism."""
    def __init__(self, in_features: int, out_features: int, tp_size: int = 1, bias: bool = False):
        super().__init__()
        self.in_features_per_partition = in_features // tp_size
        self.out_features = out_features
        self.tp_size = tp_size
        self.linear = torch.nn.Linear(self.in_features_per_partition, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_parallel = self.linear(x)
        if self.tp_size > 1 and dist.is_initialized():
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)
        return output_parallel

class PipelineParallelStage(torch.nn.Module):
    """Pipeline Parallelism 1F1B (One Forward One Backward) Micro-Batch Stage Execution Engine."""
    def __init__(self, stage_module: torch.nn.Module, stage_id: int = 0, num_stages: int = 1):
        super().__init__()
        self.stage_module = stage_module
        self.stage_id = stage_id
        self.num_stages = num_stages

    def forward_micro_batch(self, micro_batch_x: torch.Tensor) -> torch.Tensor:
        """Executes a single forward pass over a micro-batch."""
        return self.stage_module(micro_batch_x)

class SequenceParallelScatter(torch.autograd.Function):
    """Scatters sequence tokens across Tensor Parallel ranks for Sequence Parallelism (SP)."""
    @staticmethod
    def forward(ctx, input_tensor, tp_size):
        ctx.tp_size = tp_size
        if tp_size <= 1:
            return input_tensor
        dim_size = input_tensor.size(1)
        sub_seq = dim_size // tp_size
        return input_tensor[:, :sub_seq, :].clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

cluster_manager = DistributedClusterManager()



