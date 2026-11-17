import torch
import triton
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton_snn import fused_lif, fused_stdp_update, fused_causal_spiking_attention

def benchmark_lif():
    print("\n[Benchmark 1/3] Fused LIF Neuron Kernel Speed & VRAM Scaling")
    print("-" * 75)
    print(f"{'Batch Size / Elements':<25} | {'PyTorch Eager (ms)':<20} | {'Triton Fused (ms)':<20} | {'Speedup':<10}")
    print("-" * 75)

    sizes = [100000, 1000000, 10000000]
    device = torch.device("cuda")

    for N in sizes:
        v = torch.randn(N, device=device)
        x = torch.randn(N, device=device)
        alpha, v_th, v_reset = 0.85, 1.0, 0.0

        # Warmup
        for _ in range(10):
            _ = fused_lif(v, x, alpha, v_th, v_reset)
            _ = torch.where((alpha * v + x) >= v_th, v_reset, alpha * v + x)

        torch.cuda.synchronize()

        # PyTorch Eager Timing
        t0 = time.perf_counter()
        for _ in range(100):
            v_temp = alpha * v + x
            s = (v_temp >= v_th).float()
            v_next = torch.where(s > 0, v_reset, v_temp)
        torch.cuda.synchronize()
        t_pytorch = (time.perf_counter() - t0) * 10.0 # ms per call

        # Triton Fused Timing
        t0 = time.perf_counter()
        for _ in range(100):
            _ = fused_lif(v, x, alpha, v_th, v_reset)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0) * 10.0 # ms per call

        speedup = t_pytorch / t_triton
        print(f"N = {N:<21,d} | {t_pytorch:<20.4f} | {t_triton:<20.4f} | {speedup:<9.2f}x")


def benchmark_stdp():
    print("\n[Benchmark 2/3] Fused STDP Plasticity Kernel Speedup (2D Weight Matrices)")
    print("-" * 75)
    print(f"{'Matrix Dimensions (N_post x N_pre)':<30} | {'PyTorch Outer (ms)':<20} | {'Triton Fused (ms)':<18} | {'Speedup':<10}")
    print("-" * 75)

    dims = [(512, 512), (2048, 2048), (4096, 4096)]
    device = torch.device("cuda")

    for N_post, N_pre in dims:
        W_triton = torch.rand(N_post, N_pre, device=device)
        W_py = W_triton.clone()
        conn = (torch.rand(N_post, N_pre, device=device) < 0.20).float()
        s_pre = (torch.rand(N_pre, device=device) < 0.10).float()
        s_post = (torch.rand(N_post, device=device) < 0.10).float()
        tr_pre = torch.rand(N_pre, device=device)
        tr_post = torch.rand(N_post, device=device)

        # Warmup
        for _ in range(5):
            fused_stdp_update(W_triton, conn, s_pre, s_post, tr_pre, tr_post)

        torch.cuda.synchronize()

        # PyTorch Eager Outer Product Timing
        t0 = time.perf_counter()
        for _ in range(50):
            delta_ltp = 0.002 * torch.outer(s_post, tr_pre)
            delta_ltd = 0.001 * torch.outer(tr_post, s_pre)
            dW = (delta_ltp - delta_ltd) * conn
            W_py = torch.clamp(W_py + dW, min=0.0, max=2.0)
        torch.cuda.synchronize()
        t_py = (time.perf_counter() - t0) * 20.0 # ms per call

        # Triton Fused Timing
        t0 = time.perf_counter()
        for _ in range(50):
            fused_stdp_update(W_triton, conn, s_pre, s_post, tr_pre, tr_post)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0) * 20.0 # ms per call

        speedup = t_py / t_triton
        dim_str = f"{N_post} x {N_pre}"
        print(f"{dim_str:<32} | {t_py:<20.4f} | {t_triton:<18.4f} | {speedup:<9.2f}x")


def benchmark_spiking_attention():
    print("\n[Benchmark 3/3] Fused Causal Spiking Attention Speedup & Memory Scaling")
    print("-" * 75)
    print(f"{'Seq Length (T)':<20} | {'PyTorch Attention (ms)':<22} | {'Triton Spiking Attn (ms)':<22} | {'Speedup':<10}")
    print("-" * 75)

    seq_lengths = [256, 1024, 2048]
    device = torch.device("cuda")

    for T in seq_lengths:
        B_H, D = 8, 64
        q = (torch.rand(B_H, T, D, device=device) < 0.15).to(torch.float16)
        k = (torch.rand(B_H, T, D, device=device) < 0.15).to(torch.float16)
        v = torch.randn(B_H, T, D, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(5):
            _ = fused_causal_spiking_attention(q, k, v)

        torch.cuda.synchronize()

        # PyTorch Attention Timing
        t0 = time.perf_counter()
        for _ in range(50):
            scores = torch.matmul(q, k.transpose(-1, -2))
            causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.float16)).unsqueeze(0)
            scores = scores * causal_mask
            out_py = torch.matmul(scores, v)
        torch.cuda.synchronize()
        t_py = (time.perf_counter() - t0) * 20.0

        # Triton Spiking Attention Timing
        t0 = time.perf_counter()
        for _ in range(50):
            _ = fused_causal_spiking_attention(q, k, v)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0) * 20.0

        speedup = t_py / t_triton
        print(f"T = {T:<16d} | {t_py:<22.4f} | {t_triton:<22.4f} | {speedup:<9.2f}x")


if __name__ == "__main__":
    print("=" * 75)
    print("  Triton SNN Fused Kernels — Speedup & Latency Benchmark Suite")
    print("  Hardware: NVIDIA GeForce RTX 5060 (WSL2 / CUDA 13.0)")
    print("=" * 75)
    benchmark_lif()
    benchmark_stdp()
    benchmark_spiking_attention()
    print("\n" + "=" * 75)
    print("  BENCHMARK SUITE COMPLETED SUCCESSFULLY!")
    print("=" * 75 + "\n")
