"""Run one Triton operator on generated inputs in this process and print a JSON result line.

This module is the unit of work for every detector. A detector launches it in a
subprocess with a different environment (plain interpreter, TileLens sanitizer,
ASan preload, malloc perturbation) and reads the line that starts with CBU_JSON.

Corpus kernels are untrusted code. Run this only inside a throwaway process on a
machine that holds no secrets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

MARK = "CBU_JSON:"


def _emit(payload: dict) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    print(MARK + json.dumps(payload, default=str), flush=True)


def _load_module(path: str, name: str = "cbu_target_module"):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _is_kernel(obj) -> bool:
    name = type(obj).__name__
    return name in {"JITFunction", "InterpretedFunction", "Autotuner", "Trace", "TraceInterface"} or hasattr(obj, "interpreter_fn")


def _find_callable(mod, op: str, func: str | None):
    if func:
        return getattr(mod, func)
    for cand in (f"triton_{op}", f"triton{op}", op, "forward", "op"):
        f = getattr(mod, cand, None)
        if callable(f) and not _is_kernel(f) and not isinstance(f, type):
            return f
    for name, val in vars(mod).items():
        if name.startswith("_") or isinstance(val, type) or _is_kernel(val):
            continue
        if callable(val) and getattr(val, "__module__", None) == mod.__name__ and name.startswith("triton"):
            return val
    raise LookupError(f"no launch function found for op {op!r}")


def _flatten(obj, out: list) -> None:
    import torch

    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _flatten(x, out)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            _flatten(obj[k], out)


def _digest(tensors) -> dict:
    import torch

    h = hashlib.sha256()
    shapes, nan, inf = [], 0, 0
    for t in tensors:
        t = t.detach().to("cpu").contiguous()
        shapes.append([str(t.dtype), list(t.shape)])
        if t.is_floating_point():
            nan += int(torch.isnan(t).sum())
            inf += int(torch.isinf(t).sum())
        h.update(t.numpy().tobytes())
    return {"sha256": h.hexdigest(), "shapes": shapes, "nan": nan, "inf": inf}


def _install_tilelens():
    """Patch triton.jit so every kernel runs under the TileLens symbolic sanitizer without aborting."""
    import triton
    import tilelens
    from tilelens.clients import Sanitizer

    clients = []
    orig_jit, orig_autotune = triton.jit, triton.autotune

    def wrap(kernel):
        client = Sanitizer(abort_on_error=False)
        clients.append(client)
        return tilelens.trace(client=client, frontend="triton")(kernel)

    def patched_jit(fn=None, **kw):
        if fn is None:
            return lambda f: wrap(orig_jit(**kw)(f))
        return wrap(orig_jit(fn))

    def patched_autotune(fn=None, **kw):
        if fn is None:
            return lambda f: wrap(orig_autotune(**kw)(f))
        return wrap(orig_autotune(fn))

    triton.jit = patched_jit
    triton.language.jit = patched_jit
    import triton.runtime.interpreter as _interp

    _interp.jit = patched_jit
    triton.autotune = patched_autotune
    return clients


def _tilelens_records(clients) -> list:
    recs = []
    for c in clients:
        for r in getattr(c, "records", []) or []:
            tb = []
            for t in getattr(r, "user_code_tracebacks", []) or []:
                tb.append({k: getattr(t, k, None) for k in ("filename", "lineno", "func_name", "line_of_code")})
            tensor = getattr(r, "tensor", None)
            recs.append(
                {
                    "op": getattr(getattr(r, "op_type", None), "__name__", str(getattr(r, "op_type", None))),
                    "tensor_name": getattr(r, "tensor_name", None),
                    "tensor_shape": list(tensor.shape) if tensor is not None else None,
                    "tensor_dtype": str(tensor.dtype) if tensor is not None else None,
                    "violation_address": getattr(r, "violation_address", None),
                    "traceback": tb,
                }
            )
    return recs


def _asan_log_bytes() -> int | None:
    opts = os.environ.get("ASAN_OPTIONS", "")
    for part in opts.split(":"):
        if part.startswith("log_path="):
            path = f"{part[len('log_path='):]}.{os.getpid()}"
            try:
                return os.path.getsize(path)
            except OSError:
                return 0
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="python file that defines the launch function")
    ap.add_argument("--op", required=True)
    ap.add_argument("--func", default=None)
    ap.add_argument("--shapes", required=True, help="JSON list of tensor shapes for one case")
    ap.add_argument("--root", default=None, help="directory to put on sys.path (repo checkout)")
    ap.add_argument("--mode", choices=["plain", "tilelens"], default="plain")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shim", action="store_true", help="do not redirect CUDA device requests to the CPU")
    a = ap.parse_args()

    res: dict = {"op": a.op, "mode": a.mode, "shapes": json.loads(a.shapes), "status": "start"}
    t0 = time.time()
    clients = []
    try:
        if a.root:
            sys.path.insert(0, a.root)
        import torch

        if not a.no_shim and not torch.cuda.is_available():
            from cbu import cpu_shim

            cpu_shim.install()
            res["cpu_shim"] = True
        if a.mode == "tilelens":
            clients = _install_tilelens()
        res["status"] = "importing"
        mod = _load_module(a.file)
        # Reports raised while the module body runs come from the file's own
        # self-test calls, not from the benchmark inputs. Record the boundary.
        res["tilelens_records_at_import"] = sum(len(getattr(c, "records", []) or []) for c in clients)
        res["asan_log_bytes_at_import"] = _asan_log_bytes()
        fn = _find_callable(mod, a.op, a.func)
        res["func"] = getattr(fn, "__name__", str(fn))
        torch.manual_seed(a.seed)
        dtype = getattr(torch, a.dtype)
        args = tuple(torch.randn(*shape, dtype=dtype) for shape in res["shapes"])
        res["status"] = "running"
        out = fn(*args)
        outs: list = []
        _flatten(out, outs)
        res["output"] = _digest(outs)
        res["inputs_after"] = _digest(list(args))
        res["status"] = "ok"
    except SystemExit as e:  # some tools call sys.exit on error
        res["status"] = "sysexit"
        res["exit_code"] = e.code
    except BaseException as e:  # noqa: BLE001 - untrusted target code
        res["status"] = "exception"
        res["exception"] = f"{type(e).__name__}: {e}"[:2000]
        res["traceback"] = traceback.format_exc()[-4000:]
    res["seconds"] = round(time.time() - t0, 3)
    if a.mode == "tilelens":
        try:
            res["tilelens_records"] = _tilelens_records(clients)
        except BaseException as e:  # noqa: BLE001
            res["tilelens_records_error"] = repr(e)
    _emit(res)
    return 0


if __name__ == "__main__":
    os._exit(main())
