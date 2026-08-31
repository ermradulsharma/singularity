import pytest
import torch
from src.kernels import GPUArchitectureProfile, CUDAKernelAccelerator
from src.model import TritonFusedKernels

def test_gpu_architecture_profile():
    profile = GPUArchitectureProfile.get_profile()
    assert "arch" in profile
    assert "sm" in profile
    assert "num_warps" in profile
    assert "num_stages" in profile
    print(f"Detected GPU Profile: {profile}")

def test_cuda_kernel_accelerator_profile():
    profile = CUDAKernelAccelerator.get_device_profile()
    assert profile is not None
    assert "arch" in profile

def test_fused_rmsnorm_execution():
    x = torch.randn(2, 16, 64)
    weight = torch.ones(64)
    out = TritonFusedKernels.fused_rmsnorm(x, weight)
    assert out.shape == x.shape

def test_fused_moe_routing_execution():
    flat_nx = torch.randn(16, 64)
    w1_stack = torch.randn(4, 128, 64)
    w2_stack = torch.randn(4, 64, 128)
    token_expert_weights = torch.softmax(torch.randn(16, 4), dim=-1)
    out = TritonFusedKernels.fused_moe_routing(flat_nx, w1_stack, w2_stack, token_expert_weights)
    assert out.shape == (16, 64)
