import torch
import triton
import triton.language as tl

# =====================================================================
# Triton JIT Kernel: Fused Causal Spiking Attention (Spiking Transformer)
# Q, K, V are Binary / Low-Bit Spike Tensors
# Memory Complexity: O(T) SRAM Streaming (FlashAttention-style for SNNs)
# =====================================================================
@triton.jit
def _fused_spiking_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_ob, stride_ot, stride_od,
    seq_len, head_dim,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_head_idx = tl.program_id(0)
    i_idx = tl.program_id(1) # Query time block index

    offs_i = i_idx * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    mask_i = offs_i < seq_len
    mask_d = offs_d < head_dim

    # Base pointers for current Batch-Head
    q_base = q_ptr + batch_head_idx * stride_qb
    k_base = k_ptr + batch_head_idx * stride_kb
    v_base = v_ptr + batch_head_idx * stride_vb
    out_base = out_ptr + batch_head_idx * stride_ob

    # Load Query block [BLOCK_T, BLOCK_D]
    q_ptrs = q_base + (offs_i[:, None] * stride_qt + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_i[:, None] & mask_d[None, :], other=0.0)

    # Accumulator for Output [BLOCK_T, BLOCK_D]
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    # Stream Key and Value blocks causally (j <= i)
    for j_start in range(0, (i_idx + 1) * BLOCK_T, BLOCK_T):
        offs_j = j_start + tl.arange(0, BLOCK_T)
        mask_j = offs_j < seq_len

        # Load Key block [BLOCK_T, BLOCK_D]
        k_ptrs = k_base + (offs_j[:, None] * stride_kt + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=mask_j[:, None] & mask_d[None, :], other=0.0)

        # Load Value block [BLOCK_T, BLOCK_D]
        v_ptrs = v_base + (offs_j[:, None] * stride_vt + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=mask_j[:, None] & mask_d[None, :], other=0.0)

        # Spike Add-Attention Score: S_ij = Q_i @ K_j^T
        scores = tl.dot(q, tl.trans(k))

        # Apply Causal Masking: mask out j > i
        causal_mask = offs_i[:, None] >= offs_j[None, :]
        scores = tl.where(causal_mask, scores, 0.0)

        # Accumulate Value: Convert scores to fp16 for tensor core dot product with v
        scores_fp16 = scores.to(tl.float16)
        acc += tl.dot(scores_fp16, v)

    # Write output to VRAM
    out_ptrs = out_base + (offs_i[:, None] * stride_ot + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc, mask=mask_i[:, None] & mask_d[None, :])


def fused_causal_spiking_attention(q, k, v):
    """
    Fused Causal Spiking Attention for Spiking Transformers.
    Inputs: q, k, v of shape [Batch * Heads, Seq_Len, Head_Dim] (dtype=torch.float16)
    Returns: out of shape [Batch * Heads, Seq_Len, Head_Dim]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.float16, "Inputs must be torch.float16 for Tensor Core attention"
    batch_heads, seq_len, head_dim = q.shape

    out = torch.empty_like(q)

    BLOCK_T = 32
    BLOCK_D = min(64, head_dim)

    grid = (
        batch_heads,
        triton.cdiv(seq_len, BLOCK_T),
    )

    _fused_spiking_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        seq_len, head_dim,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
    )

    return out
