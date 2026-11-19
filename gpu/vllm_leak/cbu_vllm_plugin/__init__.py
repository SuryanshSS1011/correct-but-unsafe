"""vLLM plugin that swaps SiluAndMul's kernel for the study's fixed-grid Triton kernel.

vLLM's SiluAndMul.forward_cuda allocates its output with torch.empty and calls
self.op(out, x). This plugin points self.op at cbu::silu_and_mul, which has the
same signature, so the output buffer still comes from vLLM's own allocation.
It does nothing unless CBU_KERNEL is set:

    CBU_KERNEL=defective  launches a grid computed for a block of 1024 with
                          BLOCK=CBU_BLOCK (default 256), the pattern of the
                          accepted corpus kernels, so with 256 three quarters
                          of the output stay unwritten
    CBU_KERNEL=correct    the same kernel with the grid computed from BLOCK

The op only runs when silu_and_mul is enabled in compilation_config.custom_ops,
which gpu/vllm_leak/run_leak.py does. When run_leak.py sets RECORD, which needs
eager mode in one process, each call's output is copied to the host with its
address and the phase that run_leak.py sets in PHASE.
"""

import os

PHASE = ""
RECORD: list = []
RECORDING = False
_REGISTERED = False


def register() -> None:
    global _REGISTERED
    kind = os.environ.get("CBU_KERNEL", "")
    if kind not in ("defective", "correct") or _REGISTERED:
        return
    _REGISTERED = True

    import torch
    import triton
    import triton.language as tl
    from vllm.model_executor.layers import activation

    block = int(os.environ.get("CBU_BLOCK", "256"))
    grid_block = 1024 if kind == "defective" else block

    @triton.jit
    def _silu_and_mul(x_ptr, out_ptr, n, d, x_row_stride, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        row = offs // d
        col = offs % d
        gate = tl.load(x_ptr + row * x_row_stride + col, mask=mask).to(tl.float32)
        up = tl.load(x_ptr + row * x_row_stride + d + col, mask=mask).to(tl.float32)
        y = gate / (1.0 + tl.exp(-gate)) * up
        tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)

    @torch.library.custom_op("cbu::silu_and_mul", mutates_args=("out",))
    def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        x2 = x.reshape(-1, x.shape[-1])
        d = x2.shape[-1] // 2
        n = out.numel()
        grid = triton.cdiv(n, grid_block)
        _silu_and_mul[(grid,)](x2, out, n, d, x2.stride(0), BLOCK=block)
        if RECORDING:
            torch.cuda.synchronize()
            # The input first: the gate and up projection wrote all of it before this call.
            if x2.is_contiguous():
                RECORD.append(
                    {
                        "phase": PHASE,
                        "kind": "in",
                        "ptr": x2.data_ptr(),
                        "itemsize": x2.element_size(),
                        "written": x2.numel(),
                        "data": x2.reshape(-1).view(torch.int16).cpu().numpy(),
                    }
                )
            RECORD.append(
                {
                    "phase": PHASE,
                    "kind": "out",
                    "ptr": out.data_ptr(),
                    "itemsize": out.element_size(),
                    "written": min(n, grid * block),
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
