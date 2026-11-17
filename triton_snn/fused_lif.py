import torch
import triton
import triton.language as tl

# =====================================================================
# 1. Triton JIT Kernel: Fused LIF Forward Pass
# =====================================================================
@triton.jit
def _fused_lif_fwd_kernel(
    v_ptr,          # Pointer to input membrane potential v_in
    x_ptr,          # Pointer to input current/tensor x
    s_ptr,          # Pointer to output spike tensor s
    v_next_ptr,     # Pointer to output membrane potential v_out
    alpha,          # Decay factor float (exp(-dt/tau))
    v_th,           # Spike threshold float (e.g. 1.0)
    v_reset,        # Reset potential float (e.g. 0.0)
    n_elements,     # Total tensor elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load state variables from VRAM into SRAM
    v = tl.load(v_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    # Fused leaky integration: v_temp = alpha * v + x
    v_temp = alpha * v + x

    # Fused threshold condition: spike = 1.0 if v_temp >= v_th else 0.0
    spike = tl.where(v_temp >= v_th, 1.0, 0.0)

    # Fused soft/hard reset: if spiked -> v_reset, else -> v_temp
    v_next = tl.where(spike > 0.0, v_reset, v_temp)

    # Write output states back to VRAM in single memory transaction
    tl.store(s_ptr + offsets, spike, mask=mask)
    tl.store(v_next_ptr + offsets, v_next, mask=mask)


# =====================================================================
# 2. Triton JIT Kernel: Fused LIF Backward Pass (Surrogate Gradient)
# Fast Sigmoid Surrogate Derivative: dS/dv = 1 / (1 + scale * |v - v_th|)^2
# =====================================================================
@triton.jit
def _fused_lif_bwd_kernel(
    grad_s_ptr,        # Pointer to incoming gradient dL/ds
    grad_v_next_ptr,   # Pointer to incoming gradient dL/dv_next
    v_temp_ptr,        # Pointer to pre-reset membrane potential (v_temp)
    grad_x_ptr,        # Pointer to output gradient dL/dx
    grad_v_ptr,        # Pointer to output gradient dL/dv_in
    alpha,             # Decay factor float
    v_th,              # Spike threshold float
    surrogate_scale,   # Fast Sigmoid scale (k)
    n_elements,        # Total elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_s = tl.load(grad_s_ptr + offsets, mask=mask)
    grad_v_next = tl.load(grad_v_next_ptr + offsets, mask=mask)
    v_temp = tl.load(v_temp_ptr + offsets, mask=mask)

    # Fast Sigmoid surrogate gradient calculation
    # ds_dv = 1.0 / (1.0 + scale * abs(v_temp - v_th))^2
    v_diff = tl.abs(v_temp - v_th)
    denom = 1.0 + surrogate_scale * v_diff
    ds_dv = 1.0 / (denom * denom)

    # Chain rule fusion: dL/dx = (dL/ds * ds/dv) + dL/dv_next
    grad_x = grad_s * ds_dv + grad_v_next
    grad_v = grad_x * alpha

    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
    tl.store(grad_v_ptr + offsets, grad_v, mask=mask)


# =====================================================================
# 3. PyTorch Autograd Function Wrapper
# =====================================================================
class FusedLIFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, x, alpha=0.9, v_th=1.0, v_reset=0.0, surrogate_scale=2.0):
        v = v.contiguous()
        x = x.contiguous()

        s = torch.empty_like(x)
        v_next = torch.empty_like(v)
        n_elements = x.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        _fused_lif_fwd_kernel[grid](
            v, x, s, v_next,
            float(alpha), float(v_th), float(v_reset),
            n_elements,
            BLOCK_SIZE=1024
        )

        v_temp = alpha * v + x
        ctx.save_for_backward(v_temp)
        ctx.alpha = alpha
        ctx.v_th = v_th
        ctx.surrogate_scale = surrogate_scale

        return s, v_next

    @staticmethod
    def backward(ctx, grad_s, grad_v_next):
        (v_temp,) = ctx.saved_tensors
        grad_s = grad_s.contiguous()
        grad_v_next = grad_v_next.contiguous()

        grad_x = torch.empty_like(grad_s)
        grad_v = torch.empty_like(grad_s)
        n_elements = grad_s.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        _fused_lif_bwd_kernel[grid](
            grad_s, grad_v_next, v_temp,
            grad_x, grad_v,
            float(ctx.alpha), float(ctx.v_th), float(ctx.surrogate_scale),
            n_elements,
            BLOCK_SIZE=1024
        )

        return grad_v, grad_x, None, None, None, None


def fused_lif(v, x, alpha=0.9, v_th=1.0, v_reset=0.0, surrogate_scale=2.0):
    return FusedLIFFunction.apply(v, x, alpha, v_th, v_reset, surrogate_scale)
