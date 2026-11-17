import torch
import triton
import triton.language as tl

# =====================================================================
# Triton JIT Kernel: 2D Block Fused Event-Driven STDP Plasticity
# =====================================================================
@triton.jit
def _fused_stdp_kernel(
    w_ptr,             # Pointer to Weight matrix W [N_post, N_pre]
    conn_mask_ptr,    # Pointer to Connectivity mask M [N_post, N_pre]
    s_pre_ptr,        # Pointer to Pre-synaptic spikes S_pre [N_pre]
    s_post_ptr,       # Pointer to Post-synaptic spikes S_post [N_post]
    tr_pre_ptr,       # Pointer to Pre-synaptic trace A_pre [N_pre]
    tr_post_ptr,      # Pointer to Post-synaptic trace A_post [N_post]
    stride_w_post,    # Stride row W
    stride_w_pre,     # Stride col W
    stride_m_post,    # Stride row M
    stride_m_pre,     # Stride col M
    n_post,           # Number of post-synaptic neurons
    n_pre,            # Number of pre-synaptic neurons
    a_ltp,            # LTP learning rate float
    a_ltd,            # LTD learning rate float
    w_max,            # Maximum weight ceiling
    BLOCK_SIZE_M: tl.constexpr, # Post-synaptic block size
    BLOCK_SIZE_N: tl.constexpr, # Pre-synaptic block size
):
    pid_m = tl.program_id(0) # Row block index
    pid_n = tl.program_id(1) # Col block index

    # Compute 2D row and col offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < n_post
    mask_n = offs_n < n_pre

    mask_2d = mask_m[:, None] & mask_n[None, :]

    # Load 1D Spikes and Traces for Pre and Post populations
    s_pre = tl.load(s_pre_ptr + offs_n, mask=mask_n, other=0.0)
    s_post = tl.load(s_post_ptr + offs_m, mask=mask_m, other=0.0)
    tr_pre = tl.load(tr_pre_ptr + offs_n, mask=mask_n, other=0.0)
    tr_post = tl.load(tr_post_ptr + offs_m, mask=mask_m, other=0.0)

    # Load 2D Weight Matrix block & Connectivity Mask block from VRAM
    w_ptrs = w_ptr + (offs_m[:, None] * stride_w_post + offs_n[None, :] * stride_w_pre)
    m_ptrs = conn_mask_ptr + (offs_m[:, None] * stride_m_post + offs_n[None, :] * stride_m_pre)

    w = tl.load(w_ptrs, mask=mask_2d, other=0.0)
    conn = tl.load(m_ptrs, mask=mask_2d, other=0.0)

    # Fused LTP: Post spiked -> strengthen by pre-synaptic trace
    # LTP term: A_LTP * (S_post [M, 1] * A_pre [1, N])
    delta_ltp = a_ltp * (s_post[:, None] * tr_pre[None, :])

    # Fused LTD: Pre spiked -> weaken by post-synaptic trace
    # LTD term: A_LTD * (S_pre [1, N] * A_post [M, 1])
    delta_ltd = a_ltd * (s_pre[None, :] * tr_post[:, None])

    # Total weight update: W_new = W + (delta_ltp - delta_ltd) * Conn_Mask
    dw = (delta_ltp - delta_ltd) * conn
    w_new = w + dw

    # Clamp weights in SRAM: [0.0, w_max]
    w_clamped = tl.minimum(tl.maximum(w_new, 0.0), w_max)

    # Store updated weight block back to VRAM in single memory transaction
    tl.store(w_ptrs, w_clamped, mask=mask_2d)


# =====================================================================
# PyTorch Wrapper Function
# =====================================================================
def fused_stdp_update(
    w: torch.Tensor,
    conn_mask: torch.Tensor,
    s_pre: torch.Tensor,
    s_post: torch.Tensor,
    tr_pre: torch.Tensor,
    tr_post: torch.Tensor,
    a_ltp: float = 0.002,
    a_ltd: float = 0.001,
    w_max: float = 2.0,
):
    """
    Executes Fused STDP Weight Update on GPU using Triton 2D Block Kernel.
    Updates tensor w in-place.
    """
    assert w.is_cuda and conn_mask.is_cuda
    assert w.shape == conn_mask.shape

    n_post, n_pre = w.shape

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (
        triton.cdiv(n_post, BLOCK_M),
        triton.cdiv(n_pre, BLOCK_N),
    )

    _fused_stdp_kernel[grid](
        w, conn_mask,
        s_pre, s_post,
        tr_pre, tr_post,
        w.stride(0), w.stride(1),
        conn_mask.stride(0), conn_mask.stride(1),
        n_post, n_pre,
        float(a_ltp), float(a_ltd), float(w_max),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
    )
    return w
