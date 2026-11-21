"""vLLM plugin that swaps SiluAndMul's kernel for the study's fixed-grid Triton kernel.

vLLM's SiluAndMul.forward_cuda allocates its output with torch.empty and calls
self.op(out, x). This plugin points self.op at cbu::silu_and_mul, which has the
same signature, so the output buffer still comes from vLLM's own allocation.
It does nothing unless CBU_KERNEL is set:

    CBU_KERNEL=defective  launches a grid computed for a block of 1024 with
                          BLOCK=CBU_BLOCK (default 256), the pattern of the
                          accepted corpus kernels, so with 256 three quarters
                          of the output stay unwritten
    CBU_KERNEL=floor      computes the grid with floor division, n // BLOCK,
                          so the last n % BLOCK elements stay unwritten; with
                          CBU_BLOCK=1024, which does not divide the MLP widths
                          of the models used, that is part of the last
                          token's row, 768 elements for Qwen2.5-0.5B
    CBU_KERNEL=tail       leaves only the last CBU_SKIP elements (default 1)
                          of each output unwritten, like 2487_r3
    CBU_KERNEL=rowtail    leaves the last CBU_SKIP elements (default 1) of
                          every row unwritten, an off-by-one in the column
                          mask, so each token's row has a hole even when the
                          engine pads the batch with extra rows
    CBU_KERNEL=correct    the same kernel with the grid computed from BLOCK

The op only runs when silu_and_mul is enabled in compilation_config.custom_ops,
which gpu/vllm_leak/run_leak.py does. When run_leak.py sets RECORD, which needs
eager mode in one process, each call's output is copied to the host with its
address and the phase that run_leak.py sets in PHASE. When CBU_SHAPES names a
file, each call that runs in Python appends a line with whether the stream was
being captured into a CUDA graph and how many rows the output had.
"""

import os

PHASE = ""
RECORD: list = []
RECORDING = False
_REGISTERED = False
KINDS = ("defective", "floor", "tail", "rowtail", "correct")


def written(kind: str, n: int, block: int, skip: int) -> int:
    """How many leading elements of an n-element output the kernel writes."""
    if kind == "defective":
        return min(n, -(-n // 1024) * block)
    if kind == "floor":
        return n // block * block
    if kind == "tail":
        return max(n - skip, 0)
    return n


def unwritten(kind: str, n: int, d: int, block: int, skip: int):
    """Indices of the elements of an n-element output with rows of d that the kernel leaves."""
    import numpy as np

    if kind == "rowtail":
        cols = np.arange(max(d - skip, 0), d)
        return (np.arange(n // d)[:, None] * d + cols).ravel()
    return np.arange(written(kind, n, block, skip), n)


def launcher(kind: str):
    """Return launch(out, x), which runs the kernel and returns the indices it left unwritten."""
    import triton
    import triton.language as tl

    block = int(os.environ.get("CBU_BLOCK", "256"))
    skip = int(os.environ.get("CBU_SKIP", "1"))

    @triton.jit
    def _silu_and_mul(x_ptr, out_ptr, n, d, d_write, x_row_stride, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row = offs // d
        col = offs % d
        mask = (offs < n) & (col < d_write)
        gate = tl.load(x_ptr + row * x_row_stride + col, mask=mask).to(tl.float32)
        up = tl.load(x_ptr + row * x_row_stride + d + col, mask=mask).to(tl.float32)
        y = gate / (1.0 + tl.exp(-gate)) * up
        tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)

    def launch(out, x):
        x2 = x.reshape(-1, x.shape[-1])
        d = x2.shape[-1] // 2
        n = written(kind, out.numel(), block, skip)
        d_write = d - skip if kind == "rowtail" else d
        if n:
            _silu_and_mul[(triton.cdiv(n, block),)](
                x2, out, n, d, d_write, x2.stride(0), BLOCK=block
            )
        return lambda: unwritten(kind, out.numel(), d, block, skip)

    return launch


def register() -> None:
    global _REGISTERED
    kind = os.environ.get("CBU_KERNEL", "")
    if kind not in KINDS or _REGISTERED:
        return
    _REGISTERED = True

    import torch
    from vllm.model_executor.layers import activation

    launch = launcher(kind)
    shapes = os.environ.get("CBU_SHAPES")

    @torch.library.custom_op("cbu::silu_and_mul", mutates_args=("out",))
    def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        holes = launch(out, x)
        if shapes:
            with open(shapes, "a") as f:
                f.write(
                    f"{int(torch.cuda.is_current_stream_capturing())} {x.numel() // x.shape[-1]}\n"
                )
        if RECORDING:
            import numpy as np

            x2 = x.reshape(-1, x.shape[-1])
            torch.cuda.synchronize()
            # The input first: the gate and up projection wrote all of it before this call.
            if x2.is_contiguous():
                RECORD.append(
                    {
                        "phase": PHASE,
                        "kind": "in",
                        "ptr": x2.data_ptr(),
                        "itemsize": x2.element_size(),
                        "unwritten": np.arange(0),
                        "data": x2.reshape(-1).view(torch.int16).cpu().numpy(),
                    }
                )
            RECORD.append(
                {
                    "phase": PHASE,
                    "kind": "out",
                    "ptr": out.data_ptr(),
                    "itemsize": out.element_size(),
                    "unwritten": holes(),
                    "data": out.reshape(-1).view(torch.int16).cpu().numpy(),
                }
            )

    @silu_and_mul.register_fake
    def _(out: torch.Tensor, x: torch.Tensor) -> None:
        return None

    original_init = activation.SiluAndMul.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.op = torch.ops.cbu.silu_and_mul

    activation.SiluAndMul.__init__ = init
