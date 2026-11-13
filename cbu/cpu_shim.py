"""Redirect CUDA device requests to the CPU so kernels written for a GPU run under the Triton interpreter.

Benchmark and model-generated kernels usually hard-code `device='cuda'` or call
`.cuda()`. On a CPU-only runner those calls raise before any kernel executes.
The shim rewrites device arguments on the common tensor factories and on
`Tensor.to`, `Tensor.cuda`, `Module.to` and `Module.cuda`.

Most model-generated wrappers also fall back to plain PyTorch when the input is
not a CUDA tensor. In that case no Triton kernel launches and nothing is
audited. With `pretend_cuda=True` the shim reports CPU tensors as CUDA tensors
(`is_cuda`, `device`, `get_device`, `torch.cuda.is_available`) while every
allocation still lands on the CPU, so the wrapper takes its Triton path. The
target runner counts interpreter launches, and a run with zero launches is an
exclusion, never a clean result.
"""
from __future__ import annotations

import functools
import os

_INSTALLED = False
# Largest number of autotune configs seen on any kernel in this process.
AUTOTUNE_SEEN = {"max_configs": 0}

_FACTORIES = [
    "randn", "rand", "randint", "randperm", "zeros", "ones", "empty", "full", "tensor", "as_tensor",
    "arange", "linspace", "logspace", "eye", "zeros_like", "ones_like", "empty_like", "full_like",
    "randn_like", "rand_like", "randint_like", "tril_indices", "triu_indices", "normal", "scalar_tensor",
    "from_numpy", "empty_strided", "hann_window", "hamming_window", "bernoulli", "multinomial",
]
_NEW_METHODS = ["new_empty", "new_zeros", "new_ones", "new_full", "new_tensor"]


def _is_cuda(dev) -> bool:
    import torch

    if dev is None:
        return False
    if isinstance(dev, str):
        return dev.startswith("cuda")
    if isinstance(dev, torch.device):
        return dev.type == "cuda"
    if isinstance(dev, int) and not isinstance(dev, bool):
        return True  # bare ordinal means a CUDA device index
    return False


def _fix_kwargs(kwargs: dict) -> dict:
    if "device" in kwargs and _is_cuda(kwargs["device"]):
        kwargs = dict(kwargs)
        kwargs["device"] = "cpu"
    if kwargs.get("pin_memory"):
        kwargs = dict(kwargs)
        kwargs["pin_memory"] = False
    return kwargs


def _wrap_factory(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **_fix_kwargs(kwargs))

    return inner


class _AutotuneProxy:
    """Stand-in for triton.autotune under the interpreter.

    The real autotuner benchmarks every config on a GPU driver, which does not
    exist on a CPU runner. The proxy launches the kernel once with one config.
    CBU_AUTOTUNE_INDEX selects the config, so the caller can audit every config.
    Any of them can win on some GPU, so each one has to be safe.
    """

    def __init__(self, fn, configs):
        self.fn = fn
        self.configs = list(configs or [])
        AUTOTUNE_SEEN["max_configs"] = max(AUTOTUNE_SEEN["max_configs"], len(self.configs))
        index = int(os.environ.get("CBU_AUTOTUNE_INDEX", "0") or 0)
        self.best_config = self.configs[index % len(self.configs)] if self.configs else None

    def _config_kwargs(self) -> dict:
        cfg = self.best_config
        if cfg is None:
            return {}
        if hasattr(cfg, "all_kwargs"):
            try:
                return dict(cfg.all_kwargs())
            except Exception:  # noqa: BLE001
                pass
        return dict(getattr(cfg, "kwargs", {}) or {})

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            merged = self._config_kwargs()
            merged.update(kwargs)
            return self.fn[grid](*args, **merged)

        return launch

    def run(self, *args, **kwargs):
        merged = self._config_kwargs()
        merged.update(kwargs)
        return self.fn.run(*args, **merged)

    def __getattr__(self, name):
        return getattr(self.fn, name)


def _fake_autotune(configs=None, key=None, **_ignored):
    def decorator(fn):
        return _AutotuneProxy(fn, configs)

    return decorator


def install(pretend_cuda: bool = True) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import torch

    for name in _FACTORIES:
        fn = getattr(torch, name, None)
        if fn is not None:
            setattr(torch, name, _wrap_factory(fn))
    for name in _NEW_METHODS:
        fn = getattr(torch.Tensor, name, None)
        if fn is not None:
            setattr(torch.Tensor, name, _wrap_factory(fn))

    orig_tensor_to = torch.Tensor.to

    def tensor_to(self, *args, **kwargs):
        args = tuple("cpu" if (not isinstance(a, torch.Tensor) and not isinstance(a, torch.dtype) and _is_cuda(a) and not isinstance(a, int)) else a for a in args)
        return orig_tensor_to(self, *args, **_fix_kwargs(kwargs))

    torch.Tensor.to = tensor_to
    torch.Tensor.cuda = lambda self, *a, **k: self

    orig_module_to = torch.nn.Module.to

    def module_to(self, *args, **kwargs):
        args = tuple("cpu" if (not isinstance(a, (torch.Tensor, torch.dtype)) and _is_cuda(a) and not isinstance(a, int)) else a for a in args)
        return orig_module_to(self, *args, **_fix_kwargs(kwargs))

    torch.nn.Module.to = module_to
    torch.nn.Module.cuda = lambda self, *a, **k: self

    cuda = torch.cuda
    cuda.synchronize = lambda *a, **k: None
    cuda.empty_cache = lambda *a, **k: None
    cuda.reset_peak_memory_stats = lambda *a, **k: None
    cuda.max_memory_allocated = lambda *a, **k: 0
    cuda.memory_allocated = lambda *a, **k: 0
    cuda.current_device = lambda *a, **k: 0
    cuda.device_count = lambda *a, **k: 1
    cuda.get_device_name = lambda *a, **k: "cpu-shim"
    cuda.manual_seed = lambda *a, **k: None
    cuda.manual_seed_all = lambda *a, **k: None
    cuda.set_device = lambda *a, **k: None
    try:
        import triton

        triton.autotune = _fake_autotune
        triton.runtime.autotune = _fake_autotune
    except Exception:  # noqa: BLE001 - triton is optional for the shim itself
        pass
    if pretend_cuda:
        fake = torch.device("cuda", 0)
        cuda.is_available = lambda *a, **k: True
        torch.Tensor.is_cuda = property(lambda self: True)
        torch.Tensor.device = property(lambda self: fake)
        torch.Tensor.get_device = lambda self, *a, **k: 0
    _INSTALLED = True
