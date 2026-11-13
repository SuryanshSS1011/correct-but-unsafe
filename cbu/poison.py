"""Fill every tensor that PyTorch hands out uninitialized with a recognizable bit pattern.

`torch.empty` and its relatives return whatever the allocator had. On a GPU that
can be another request's data. A kernel that does not write every output element
returns that memory to the caller. Under this shim the unwritten elements carry a
fixed pattern, so they can be counted exactly and the result does not depend on
what the heap happened to contain.

Floating-point patterns are NaNs with a fixed payload. A NaN in the output that
the unpoisoned run did not have means the kernel computed from uninitialized
values. Eight-bit and boolean tensors are left alone because one byte cannot be
told apart from data.
"""
from __future__ import annotations

import functools

_STATE = {"installed": False, "allocations": 0, "elements": 0}


def _patterns():
    import torch

    return {
        torch.float16: (torch.int16, 0x7E5A),
        torch.bfloat16: (torch.int16, 0x7FDA),
        torch.float32: (torch.int32, 0x7FC5A5A5),
        torch.float64: (torch.int64, 0x7FF85A5A5A5A5A5A),
        torch.int16: (torch.int16, 0x5A5A),
        torch.int32: (torch.int32, 0x5A5A5A5A),
        torch.int64: (torch.int64, 0x5A5A5A5A5A5A5A5A),
    }


def _fill(t):
    import torch

    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return t
    pat = _patterns().get(t.dtype)
    if pat is None:
        return t
    try:
        with torch.no_grad():
            t.detach().view(pat[0]).fill_(pat[1])
        _STATE["allocations"] += 1
        _STATE["elements"] += t.numel()
    except Exception:  # noqa: BLE001 - a tensor that cannot be viewed stays as the allocator left it
        pass
    return t


def _wrap(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return _fill(fn(*args, **kwargs))

    return inner


def install() -> None:
    import torch

    if _STATE["installed"]:
        return
    for name in ("empty", "empty_like", "empty_strided"):
        setattr(torch, name, _wrap(getattr(torch, name)))
    for name in ("new_empty", "new_empty_strided"):
        setattr(torch.Tensor, name, _wrap(getattr(torch.Tensor, name)))
    _STATE["installed"] = True


def count(tensors) -> dict:
    """Count elements that still hold the poison pattern, and NaNs, across tensors."""
    import torch

    total = unwritten = nan = 0
    per = []
    pats = _patterns()
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        t = t.detach().to("cpu").contiguous()
        n = t.numel()
        total += n
        u = 0
        pat = pats.get(t.dtype)
        if pat is not None and n:
            u = int((t.view(pat[0]) == pat[1]).sum())
        if t.is_floating_point():
            nan += int(torch.isnan(t).sum())
        unwritten += u
        per.append({"shape": list(t.shape), "dtype": str(t.dtype), "elements": n, "unwritten": u})
    return {"elements": total, "unwritten": unwritten, "nan": nan, "tensors": per}


def stats() -> dict:
    return {"allocations": _STATE["allocations"], "elements": _STATE["elements"]}
