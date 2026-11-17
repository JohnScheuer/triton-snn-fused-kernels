import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton_snn import fused_lif, fused_stdp_update, fused_causal_spiking_attention

def test_fused_lif_correctness():
    print("\n[Test 1/3] Testing Fused LIF Kernel vs Native PyTorch Eager...")
    device = torch.device("cuda")
    N = 100000

    v = torch.randn(N, device=device, requires_grad=True)
    x = torch.randn(N, device=device, requires_grad=True)
    alpha = 0.85
    v_th = 1.0
    v_reset = 0.0

    # Triton Fused LIF
    s_triton, v_next_triton = fused_lif(v, x, alpha=alpha, v_th=v_th, v_reset=v_reset)

    # Native PyTorch Eager
    v_temp_pytorch = alpha * v + x
    s_pytorch = (v_temp_pytorch >= v_th).float()
    v_next_pytorch = torch.where(s_pytorch > 0, v_reset, v_temp_pytorch)

    # Compare Forward Outputs
    assert torch.allclose(s_triton, s_pytorch), "Spike mismatch between Triton and PyTorch!"
    assert torch.allclose(v_next_triton, v_next_pytorch, atol=1e-5), "Membrane mismatch between Triton and PyTorch!"
    print("  ✅ Fused LIF Forward Pass Match: 100% identical outputs!")

    # Test Backward Pass Gradients
    loss_triton = s_triton.sum() + v_next_triton.sum()
    loss_triton.backward()

    print("  ✅ Fused LIF Backward Pass (Surrogate Gradient) executed cleanly!")


def test_fused_stdp_correctness():
    print("\n[Test 2/3] Testing Fused STDP Kernel vs Native PyTorch Outer Product...")
    device = torch.device("cuda")
    N_post = 1024
    N_pre = 1024

    torch.manual_seed(42)
    W_triton = torch.rand(N_post, N_pre, device=device)
    W_pytorch = W_triton.clone()

    conn_mask = (torch.rand(N_post, N_pre, device=device) < 0.20).float()
    s_pre = (torch.rand(N_pre, device=device) < 0.10).float()
    s_post = (torch.rand(N_post, device=device) < 0.10).float()
    tr_pre = torch.rand(N_pre, device=device)
    tr_post = torch.rand(N_post, device=device)

    a_ltp = 0.005
    a_ltd = 0.002
    w_max = 1.5

    # Triton Fused STDP Update
    fused_stdp_update(W_triton, conn_mask, s_pre, s_post, tr_pre, tr_post, a_ltp, a_ltd, w_max)

    # Native PyTorch Outer Product STDP Update
    delta_ltp = a_ltp * torch.outer(s_post, tr_pre)
    delta_ltd = a_ltd * torch.outer(tr_post, s_pre)
    dW = (delta_ltp - delta_ltd) * conn_mask
    W_pytorch = torch.clamp(W_pytorch + dW, min=0.0, max=w_max)

    diff = torch.abs(W_triton - W_pytorch).max().item()
    assert diff < 1e-4, f"STDP weight mismatch! Max diff: {diff}"
    print(f"  ✅ Fused STDP Kernel Match: Max difference = {diff:.6f}!")


def test_fused_causal_spiking_attention_correctness():
    print("\n[Test 3/3] Testing Fused Causal Spiking Attention Kernel...")
    device = torch.device("cuda")
    B_H, T, D = 8, 128, 64

    # Float16 tensors for Tensor Core GEMM
    q = (torch.rand(B_H, T, D, device=device) < 0.15).to(torch.float16)
    k = (torch.rand(B_H, T, D, device=device) < 0.15).to(torch.float16)
    v = torch.randn(B_H, T, D, device=device, dtype=torch.float16)

    out_triton = fused_causal_spiking_attention(q, k, v)

    # Reference Causal Attention in PyTorch
    scores = torch.matmul(q, k.transpose(-1, -2))
    causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.float16)).unsqueeze(0)
    scores = scores * causal_mask
    out_pytorch = torch.matmul(scores, v)

    diff = torch.abs(out_triton - out_pytorch).max().item()
    assert diff < 1e-2, f"Attention mismatch! Max diff: {diff}"
    print(f"  ✅ Fused Causal Spiking Attention Match: Max difference = {diff:.6f}!")


if __name__ == "__main__":
    print("=" * 70)
    print("  Triton SNN Fused Kernels — Correctness Verification Suite")
    print("=" * 70)
    test_fused_lif_correctness()
    test_fused_stdp_correctness()
    test_fused_causal_spiking_attention_correctness()
    print("\n" + "=" * 70)
    print("  ALL KERNEL CORRECTNESS TESTS PASSED SUCCESSFULLY!")
    print("=" * 70 + "\n")
