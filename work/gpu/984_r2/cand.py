import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# Fused forward: y = x if x > lambd else 0
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _hardshrink_relu_forward_kernel(x_ptr, y_ptr, n_elements, lambd, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = tl.where(x > lambd, x, 0)
    tl.store(y_ptr + offs, y, mask=mask)


# Fused backward: grad_in = grad_out if x > lambd else 0
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _hardshrink_relu_backward_kernel(x_ptr, gout_ptr, gin_ptr, n_elements, lambd, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    g_out = tl.load(gout_ptr + offs, mask=mask, other=0)
    g_in = tl.where(x > lambd, g_out, 0)
    tl.store(gin_ptr + offs, g_in, mask=mask)


class _HardshrinkReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        # CPU fallback / non-Triton environment
        if (not x.is_cuda) or (not _TRITON_AVAILABLE):
            y = torch.nn.functional.hardshrink(x, lambd=lambd)
            y = torch.relu(y)
            # Only save for backward if needed
            if x.requires_grad:
                ctx.save_for_backward(x)
                ctx.lambd = lambd
            else:
                ctx.save_for_backward(None)
                ctx.lambd = lambd
            return y

        # Ensure contiguity for optimal memory access
        x_contig = x if x.is_contiguous() else x.contiguous()
        y = torch.empty_like(x_contig)

        n = x_contig.numel()
        # Adaptive grid: for small tensors, use as few programs as possible
        # Try to process everything in one program up to BLOCK_SIZE=4096
        if n <= 4096:
            grid = (1,)
        else:
            # Default: enough programs to cover n
            # The autotuner will pick a good BLOCK_SIZE to minimize grid size
            # and maximize efficiency.
            # We still use cdiv to ensure full coverage.
            grid = (triton.cdiv(n, 1024),)

        _hardshrink_relu_forward_kernel[grid](
            x_contig, y, n, lambd,
            # num_warps is controlled by autotune configs
        )

        # Save for backward only if needed
        if x.requires_grad:
            ctx.save_for_backward(x_contig)
            ctx.lambd = lambd
        else:
            ctx.save_for_backward(None)
            ctx.lambd = lambd
        return y.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors[0]
        lambd = ctx.lambd

        if saved is None:
            # No grad required during forward; nothing to propagate
            return torch.zeros_like(grad_out), None

        x = saved

        if (not grad_out.is_cuda) or (not _TRITON_AVAILABLE):
            gin = torch.where(x > lambd, grad_out, torch.zeros_like(grad_out))
            return gin, None

        gout_contig = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
        gin = torch.empty_like(gout_contig)

        n = x.numel()
        if n <= 4096:
            grid = (1,)
        else:
            grid = (triton.cdiv(n, 1024),)

        _hardshrink_relu_backward_kernel[grid](
            x, gout_contig, gin, n, lambd,
        )

        return gin.view_as(grad_out), None


class ModelNew(nn.Module):
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _HardshrinkReLUFn.apply(x, self.lambd)


# Example usage / quick check
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(64, 128, device=dev)

    baseline = nn.Sequential(nn.Hardshrink(), nn.ReLU())(x)
    fused = ModelNew()(x)

    print("Max abs diff:", (baseline - fused).abs().max().item())

    xb = x.clone().detach().requires_grad_(True)
    bf = nn.Sequential(nn.Hardshrink(), nn.ReLU())(xb).sum()
    xf = ModelNew()(xb).sum()
    bf.backward()
    xf.backward()
    print("Grad max abs diff:", (xb.grad - xb.grad).abs().max().item())
