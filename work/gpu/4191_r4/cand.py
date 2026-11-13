import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def std_reduce_dim1_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om,
    unbiased,  # 0 or 1
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row, reduce across columns (dim=1)
    row = tl.program_id(0)

    # Scalar accumulators in fp32
    s = tl.zeros((), dtype=tl.float32)
    ss = tl.zeros((), dtype=tl.float32)

    # Iterate across columns in blocks
    for col_start in range(0, N, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        vals = tl.load(x_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        s += tl.sum(vals, axis=0)
        ss += tl.sum(vals * vals, axis=0)

    # Counts as scalar tensors
    n_f = tl.full((), N, dtype=tl.float32)
    n_i = tl.full((), N, dtype=tl.int32)

    # Population variance
    mean = s / n_f
    var_pop = ss / n_f - mean * mean

    # Unbiased variance
    var_unb = (ss - (s * s) / n_f) / (n_f - 1.0)

    # Masks for selection
    mask_unbiased = unbiased != 0
    mask_n_gt1 = n_i > 1
    mask_make_nan = mask_unbiased & (~mask_n_gt1)

    # Select variance and compute std
    var_sel = tl.where(mask_unbiased & mask_n_gt1, var_unb, var_pop)
    var_sel = tl.maximum(var_sel, 0.0)
    std_non_nan = tl.sqrt(var_sel)

    nan_val = tl.full((), float("nan"), dtype=tl.float32)
    std = tl.where(mask_make_nan, nan_val, std_non_nan)

    # Store scalar result per row
    tl.store(out_ptr + row * stride_om, std)


@triton.jit
def std_reduce_dim0_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_on,
    unbiased,  # 0 or 1
    BLOCK_SIZE: tl.constexpr,
):
    # One program per column, reduce across rows (dim=0)
    col = tl.program_id(0)

    s = tl.zeros((), dtype=tl.float32)
    ss = tl.zeros((), dtype=tl.float32)

    for row_start in range(0, M, BLOCK_SIZE):
        offs = row_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < M
        vals = tl.load(x_ptr + offs * stride_xm + col * stride_xn, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        s += tl.sum(vals, axis=0)
        ss += tl.sum(vals * vals, axis=0)

    n_f = tl.full((), M, dtype=tl.float32)
    n_i = tl.full((), M, dtype=tl.int32)

    mean = s / n_f
    var_pop = ss / n_f - mean * mean
    var_unb = (ss - (s * s) / n_f) / (n_f - 1.0)

    mask_unbiased = unbiased != 0
    mask_n_gt1 = n_i > 1
    mask_make_nan = mask_unbiased & (~mask_n_gt1)

    var_sel = tl.where(mask_unbiased & mask_n_gt1, var_unb, var_pop)
    var_sel = tl.maximum(var_sel, 0.0)
    std_non_nan = tl.sqrt(var_sel)

    nan_val = tl.full((), float("nan"), dtype=tl.float32)
    std = tl.where(mask_make_nan, nan_val, std_non_nan)

    tl.store(out_ptr + col * stride_on, std)


def _choose_launch_config(reduce_len: int):
    # Simple heuristic for H100
    if reduce_len >= 2048:
        block = 1024
        warps = 8
    elif reduce_len >= 1024:
        block = 512
        warps = 8
    else:
        block = 256
        warps = 4
    return block, warps


class ModelNew(nn.Module):
    def __init__(self, dim, unbiased):
        super().__init__()
        self.dim = dim
        self.unbiased = unbiased

    def forward(self, x: torch.Tensor):
        # Fallback for unsupported cases
        if (x.device.type != "cuda") or (x.ndim != 2):
            std = torch.std(x, dim=self.dim, unbiased=self.unbiased)
            return torch.round(std)

        # Normalize dim
        dim = self.dim
        if dim < 0:
            dim = x.ndim + dim
        if dim not in (0, 1):
            std = torch.std(x, dim=self.dim, unbiased=self.unbiased)
            return torch.round(std)

        M, N = x.shape
        unbiased_flag = 1 if self.unbiased else 0

        # Decide whether to use original or transposed view for better coalescing
        # We prefer reducing along a contiguous axis (stride == 1).
        if dim == 1:
            # Reduce across columns; if stride(1) != 1, use transposed view and dim0 kernel
            if x.stride(1) == 1:
                out = torch.empty((M,), device=x.device, dtype=torch.float32)
                grid = (M,)
                block, warps = _choose_launch_config(N)
                std_reduce_dim1_kernel[grid](
                    x, out,
                    M, N,
                    x.stride(0), x.stride(1),
                    out.stride(0),
                    unbiased_flag,
                    BLOCK_SIZE=block,
                    num_warps=warps,
                    num_stages=4,
                )
            else:
                # Use transposed view and dim0 kernel to read contiguous rows of x.T
                x_t = x.transpose(0, 1)
                Mt, Nt = x_t.shape  # Nt == M, Mt == N
                out = torch.empty((M,), device=x.device, dtype=torch.float32)
                grid = (Mt,)  # N
                block, warps = _choose_launch_config(Nt)  # M
                std_reduce_dim0_kernel[grid](
                    x_t, out,
                    Mt, Nt,
                    x_t.stride(0), x_t.stride(1),
                    out.stride(0),
                    unbiased_flag,
                    BLOCK_SIZE=block,
                    num_warps=warps,
                    num_stages=4,
                )
        else:
            # dim == 0: reduce across rows; use transposed view if stride(0) != 1
            if x.stride(0) == 1:
                out = torch.empty((N,), device=x.device, dtype=torch.float32)
                grid = (N,)
                block, warps = _choose_launch_config(M)
                std_reduce_dim0_kernel[grid](
                    x, out,
                    M, N,
                    x.stride(0), x.stride(1),
                    out.stride(0),
                    unbiased_flag,
                    BLOCK_SIZE=block,
                    num_warps=warps,
                    num_stages=4,
                )
            else:
                # Use transposed view so that we read contiguous rows (axis 0)
                x_t = x.transpose(0, 1)
                Mt, Nt = x_t.shape  # Nt == M, Mt == N
                out = torch.empty((N,), device=x.device, dtype=torch.float32)
                grid = (Nt,)  # M
                block, warps = _choose_launch_config(Mt)  # N
                std_reduce_dim0_kernel[grid](
                    x_t, out,
                    Mt, Nt,
                    x_t.stride(0), x_t.stride(1),
                    out.stride(0),
                    unbiased_flag,
                    BLOCK_SIZE=block,
                    num_warps=warps,
                    num_stages=4,
                )

        # Cast to input dtype and round with PyTorch to preserve exact rounding semantics
        out = out.to(x.dtype)
        out = torch.round(out)
        return out


# Helpers for local testing
def get_inputs():
    x = torch.randn(64, 128, device="cuda")
    return [x]


def get_init_inputs():
    return [0, True]
