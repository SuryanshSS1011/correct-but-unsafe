import torch
import triton
import triton.language as tl


@triton.jit
def copy_block(x_ptr, y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask and col_mask  # the compiler makes this elementwise
    offs = rows[:, None] * N + cols[None, :]
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask, other=0.0), mask=mask)


M, N = 3, 8
x = torch.arange(M * N, dtype=torch.float32)
buf = torch.full((4 * M * N,), -1.0)   # the output is the first M*N elements of a larger buffer
y = buf[: M * N]
copy_block[(1,)](x, y, M, N, BLOCK_M=8, BLOCK_N=8)
print("elements written past the output:", int((buf[M * N:] != -1.0).sum()))
