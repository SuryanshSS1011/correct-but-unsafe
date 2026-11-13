"""Run one kernel on generated inputs in this process and print a JSON result line.

This module is the unit of work for every detector. A detector launches it in a
subprocess with a different environment (plain interpreter, TileLens sanitizer,
ASan preload, malloc perturbation) and reads the line that starts with CBU_JSON.

Two adapters are supported.

tritongym    A file that defines a launch function. Inputs are random tensors
             built from a list of shapes.
kernelbench  A candidate file that defines `ModelNew` and a reference file that
             defines `Model`, `get_inputs` and `get_init_inputs`. This is the
             format used by KernelBench and by the Dr. Kernel corpus.

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


def _load_module(path: str, name: str):
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


def _poison_report(res: dict, outs: list, inputs: list) -> None:
    if "poison" not in res:
        return
    from cbu import poison

    res["poison"].update({"output": poison.count(outs), "inputs_after": poison.count(inputs), **poison.stats()})


TILELENS_LAUNCHES = {"n": 0}


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
        traced = tilelens.trace(client=client, frontend="triton")(kernel)
        try:
            orig_run = traced.run

            def counted_run(*args, **kwargs):
                if not kwargs.get("warmup", False):
                    TILELENS_LAUNCHES["n"] += 1
                return orig_run(*args, **kwargs)

            traced.run = counted_run
        except Exception:  # noqa: BLE001 - counting is best effort
            TILELENS_LAUNCHES["n"] = -1
        return traced

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
    # When the CPU shim has replaced the autotuner with its launch-once proxy,
    # keep the proxy. It wraps the traced kernel that patched_jit returns.
    if getattr(orig_autotune, "__name__", "") != "_fake_autotune":
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


def _count_kernel_launches() -> dict:
    """Count interpreter kernel launches so a run that never reaches a Triton kernel is visible."""
    counter = {"n": 0}
    try:
        import triton.runtime.interpreter as interp

        orig = interp.GridExecutor.__call__

        def counted(self, *args, **kwargs):
            counter["n"] += 1
            return orig(self, *args, **kwargs)

        interp.GridExecutor.__call__ = counted
    except Exception:  # noqa: BLE001
        counter["n"] = -1
    return counter


def _count_compiled_launches(counter: dict) -> None:
    """Count launches of compiled kernels, for runs on a real GPU."""
    try:
        from triton.runtime.jit import JITFunction

        orig = JITFunction.run

        def run(self, *args, **kwargs):
            if not kwargs.get("warmup"):
                counter["n"] = (counter.get("n") or 0) + 1
            return orig(self, *args, **kwargs)

        JITFunction.run = run
    except Exception:  # noqa: BLE001
        pass


def _expose_atomics() -> dict:
    """Route the addresses of atomic operations through a masked load before the operation runs.

    The interpreter does loads and stores with memcpy, which a preloaded ASan
    intercepts. Its atomic operations write through raw pointers, which ASan
    cannot see in an uninstrumented library. Loading the same addresses first
    makes an out-of-bounds atomic access visible as an out-of-bounds read.
    """
    seen = {"n": 0}
    try:
        import numpy as np
        import triton.language as tl
        import triton.runtime.interpreter as interp

        builder = interp.InterpreterBuilder
        orig_rmw, orig_cas = builder.create_atomic_rmw, builder.create_atomic_cas

        def rmw(self, rmwOp, ptr, val, mask, sem, scope):
            seen["n"] += 1
            self.create_masked_load(ptr, mask, None, None, None, False)
            return orig_rmw(self, rmwOp, ptr, val, mask, sem, scope)

        def cas(self, ptr, cmp, val, sem, scope):
            seen["n"] += 1
            mask = interp.TensorHandle(np.ones_like(ptr.data, dtype=bool), tl.int1)
            self.create_masked_load(ptr, mask, None, None, None, False)
            return orig_cas(self, ptr, cmp, val, sem, scope)

        builder.create_atomic_rmw, builder.create_atomic_cas = rmw, cas
    except Exception:  # noqa: BLE001
        seen["n"] = -1
    return seen


def _match_compiler_boolops() -> dict:
    """Make `and` and `or` on block tensors behave under the interpreter as they do when compiled.

    The compiler turns `a and b` on tensors into an elementwise logical AND. The
    interpreter leaves Python semantics in place and treats every non-scalar
    tensor as true, so `row_mask and col_mask` silently becomes `col_mask`. A
    kernel that is correct on a GPU then makes out-of-bounds accesses on CPU.
    The interpreter's source rewriter is extended to route boolean operators
    through a helper that follows the compiler's rule.
    """
    seen = {"n": 0}
    try:
        import ast

        import triton.language as tl
        import triton.runtime.interpreter as interp

        sem = interp.interpreter_semantic

        def boolop(kind, *thunks):
            blocks, last = [], None
            for thunk in thunks:
                value = last = thunk()
                is_block = isinstance(value, tl.core.tensor) and getattr(value.handle.data, "size", 1) != 1
                if is_block:
                    blocks.append(value)
                    continue
                truth = bool(value)
                if (kind == "and" and not truth) or (kind == "or" and truth):
                    return value
            if not blocks:
                return last
            seen["n"] += 1
            out = blocks[0]
            for other in blocks[1:]:
                out = sem.logical_and(out, other) if kind == "and" else sem.logical_or(out, other)
            return out

        type(sem)._cbu_boolop = staticmethod(boolop)

        def visit_BoolOp(self, node):
            self.generic_visit(node)
            kind = "and" if isinstance(node.op, ast.And) else "or"
            no_args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
            thunks = [ast.Lambda(args=no_args, body=v) for v in node.values]
            func = ast.Attribute(value=ast.Name(id="interpreter_semantic", ctx=ast.Load()), attr="_cbu_boolop", ctx=ast.Load())
            return ast.copy_location(ast.Call(func=func, args=[ast.Constant(value=kind)] + thunks, keywords=[]), node)

        orig_assign = interp.ASTTransformer.visit_Assign

        def visit_Assign(self, node):
            # The stock rewriter wraps the right-hand side without descending into it.
            node.value = self.visit(node.value)
            return orig_assign(self, node)

        interp.ASTTransformer.visit_BoolOp = visit_BoolOp
        interp.ASTTransformer.visit_Assign = visit_Assign
    except Exception:  # noqa: BLE001
        seen["n"] = -1
    return seen


_STATE: dict = {}


def _bounds_oracle() -> dict:
    """Check every masked-in address of every load and store against the kernel's own tensors.

    A Triton kernel can only reach memory through its pointer arguments. At each
    launch the storage ranges of the tensor arguments are recorded, and each
    access is compared with them. This is exact for the executed inputs and does
    not depend on the allocator, so it also sees an access that lands in another
    live allocation, which a redzone-based sanitizer misses.
    """
    state = {"ranges": [], "violations": [], "n": 0, "accesses": 0, "active": False}
    try:
        import numpy as np
        import torch
        import triton.runtime.interpreter as interp

        builder = interp.InterpreterBuilder
        orig_call = interp.GridExecutor.__call__
        orig_load, orig_store = builder.create_masked_load, builder.create_masked_store

        def call(self, *args, **kwargs):
            ranges = []
            for v in list(args) + list(kwargs.values()):
                if isinstance(v, torch.Tensor) and v.numel():
                    st = v.untyped_storage()
                    ranges.append((int(st.data_ptr()), int(st.data_ptr()) + int(st.nbytes())))
            prev, state["ranges"] = state["ranges"], sorted(set(ranges))
            try:
                return orig_call(self, *args, **kwargs)
            finally:
                state["ranges"] = prev

        def check(ptr, mask, kind):
            ranges = state["ranges"]
            if not ranges:
                return
            try:
                size = max(1, int(ptr.dtype.element_ty.primitive_bitwidth) // 8)
                p, m = np.broadcast_arrays(np.asarray(ptr.data, dtype=np.int64), np.asarray(mask.data).astype(bool))
                state["accesses"] += int(m.sum())
                ok = np.zeros(p.shape, dtype=bool)
                for lo, hi in ranges:
                    ok |= (p >= lo) & (p + size <= hi)
                bad = m & ~ok
                if not bad.any():
                    return
                state["n"] += int(bad.sum())
                if len(state["violations"]) < 8:
                    addr = int(p[bad].ravel()[0])
                    lo, hi = min(ranges, key=lambda r: min(abs(addr - r[0]), abs(addr - r[1])))
                    state["violations"].append({
                        "kind": kind, "size": size, "lanes": int(bad.sum()), "region_bytes": hi - lo,
                        "side": "before" if addr < lo else "after",
                        "offset_bytes": (lo - addr) if addr < lo else (addr - hi),
                    })
            except Exception as e:  # noqa: BLE001 - the oracle must never change the kernel's behaviour
                state.setdefault("errors", []).append(f"{type(e).__name__}: {e}"[:160])

        def load(self, ptrs, mask, *rest, **kw):
            check(ptrs, mask, "load")
            return orig_load(self, ptrs, mask, *rest, **kw)

        def store(self, ptrs, value, mask, *rest, **kw):
            check(ptrs, mask, "store")
            return orig_store(self, ptrs, value, mask, *rest, **kw)

        interp.GridExecutor.__call__ = call
        builder.create_masked_load, builder.create_masked_store = load, store
        state["active"] = True
    except Exception as e:  # noqa: BLE001
        state["install_error"] = f"{type(e).__name__}: {e}"[:200]
    return state


def _run_replay(a, res: dict) -> None:
    """Replay KernelGYM's correctness protocol in one process and record every trial.

    The loop follows kernelgym/toolkit/kernelbench/correctness.py. The models are
    built once, the trial seeds are drawn from seed 42, the reference runs before
    the candidate and the outputs are compared with allclose at 1e-2. KernelGYM
    leaves the models in training mode, and --model-mode selects that or eval.
    """
    import torch

    res["status"] = "reference"
    torch.manual_seed(a.seed)
    ref = _load_module(a.ref_file, "cbu_reference_module")
    torch.manual_seed(a.seed)
    get_init = getattr(ref, "get_init_inputs", None)
    init_args = list(get_init() or []) if get_init is not None else []
    torch.manual_seed(a.seed)
    ref_model = ref.Model(*init_args)
    res["status"] = "importing"
    torch.manual_seed(a.seed)
    cand = _load_module(a.file, "cbu_target_module")
    torch.manual_seed(a.seed)
    cand_model = cand.ModelNew(*init_args)
    if a.model_mode == "eval":
        ref_model.eval()
        cand_model.eval()
    on_gpu = bool(torch.cuda.is_available())
    if on_gpu:
        # A no-op under the CPU shim. On a real GPU this is what KernelGYM does.
        ref_model, cand_model = ref_model.cuda(), cand_model.cuda()
    torch.manual_seed(42)
    seeds = [int(torch.randint(0, 2**32 - 1, (1,)).item()) for _ in range(a.replay)]
    launches = _STATE.get("launches") or {}
    trials: list = []
    res["replay"] = {"model_mode": a.model_mode, "trials": trials}
    res["status"] = "running"
    with torch.no_grad():
        for seed in seeds:
            torch.manual_seed(seed)
            inputs = list(ref.get_inputs())
            if on_gpu:
                inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
            torch.manual_seed(seed)
            before = launches.get("n") or 0
            output = ref_model(*inputs)
            trial: dict = {}
            trials.append(trial)
            try:
                output_new = cand_model(*inputs)
            except Exception as e:  # noqa: BLE001
                trial["error"] = f"{type(e).__name__}: {e}"[:200]
                break
            trial["launches"] = (launches.get("n") or 0) - before
            o: list = []
            c: list = []
            _flatten(output, o)
            _flatten(output_new, c)
            same = [list(x.shape) for x in o] == [list(x.shape) for x in c]
            trial["shape_match"] = same
            trial["allclose"] = bool(same and all(torch.allclose(x, y, atol=1e-2, rtol=1e-2) for x, y in zip(o, c)))
            if "poison" in res:
                from cbu import poison

                pc = poison.count(c)
                trial.update({"elements": pc["elements"], "unwritten": pc["unwritten"], "nan": pc["nan"]})
    res["input_shapes"] = [[str(x.dtype), list(x.shape)] if isinstance(x, torch.Tensor) else type(x).__name__ for x in inputs]
    # Which config did each autotuned kernel settle on? On a GPU this is the real timing winner.
    tuned = []
    for name, obj in vars(cand).items():
        configs, best = getattr(obj, "configs", None), getattr(obj, "best_config", None)
        if not isinstance(configs, list) or len(configs) < 2:
            continue
        idx = next((i for i, c in enumerate(configs) if c is best or str(c) == str(best)), None)
        timings = getattr(obj, "configs_timings", None)
        tuned.append({"kernel": name, "n_configs": len(configs), "winner_index": idx, "winner": str(best)[:200],
                      "timings": {str(k)[:120]: v for k, v in timings.items()} if isinstance(timings, dict) else None})
    res["replay"]["autotuned"] = tuned
    res["replay"]["device"] = torch.cuda.get_device_name(0) if on_gpu and not res.get("cpu_shim") else "cpu-interpreter"


def _vary_batch(inputs: list, variant: str) -> list:
    """Change the leading dimension shared by the tensor inputs. Used as a cheap shape perturbation."""
    import torch

    if variant == "native":
        return inputs
    target = {"batch1": 1, "batch3": 3, "batch7": 7}[variant]
    tensors = [x for x in inputs if isinstance(x, torch.Tensor) and x.dim() >= 1]
    if not tensors:
        raise ValueError("no tensor inputs to vary")
    b = tensors[0].shape[0]
    out = []
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.dim() >= 1 and x.shape[0] == b:
            reps = -(-target // b)
            y = x.repeat(*([reps] + [1] * (x.dim() - 1)))[:target].contiguous()
            out.append(y)
        else:
            out.append(x)
    return out


SKIP_KEYWORDS = {"dim", "axis", "dtype", "p", "ord", "device", "start_dim", "end_dim"}


def _int_constants(tree) -> list:
    """Integer literals that define the benchmark shapes, as (label, node) pairs.

    Two sources are collected in order: module-level `name = <int>` assignments,
    and integer literals inside `get_inputs` and `get_init_inputs`. Only values
    of at least 2 are kept, and literals passed as axis-like keywords are skipped.
    """
    import ast

    slots = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = node.value
            if isinstance(v, ast.Constant) and isinstance(v.value, int) and not isinstance(v.value, bool) and v.value >= 2:
                slots.append((node.targets[0].id, v))

    def visit(n, fname):
        for field, value in ast.iter_fields(n):
            children = value if isinstance(value, list) else [value]
            for child in children:
                if not isinstance(child, ast.AST):
                    continue
                if isinstance(child, ast.keyword) and child.arg in SKIP_KEYWORDS:
                    continue
                if isinstance(child, ast.Constant) and isinstance(child.value, int) and not isinstance(child.value, bool) and child.value >= 2:
                    slots.append((f"{fname}:{child.lineno}:{child.col_offset}", child))
                else:
                    visit(child, fname)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("get_inputs", "get_init_inputs"):
            visit(node, node.name)
    return slots


def _perturbed_reference(path: str, variant: str, res: dict) -> str:
    """Rewrite shape constants in the reference file and return the path of the new file.

    `const:i` adds one to every literal that has the i-th distinct value, and
    `const:all` adds one to every collected literal.
    The reference defines its inputs and constructor arguments from these
    constants, so the perturbed file describes the same module at a nearby shape.
    """
    import ast
    import tempfile

    src = open(path).read()
    tree = ast.parse(src)
    # Literals with the same value move together, so a size that appears in both
    # get_inputs and get_init_inputs stays consistent after the change.
    groups: dict = {}
    for label, node in _int_constants(tree):
        groups.setdefault(node.value, []).append((label, node))
    values = list(groups)
    res["n_int_constants"] = len(values)
    if not variant.startswith("const:"):
        return path
    which = variant.split(":", 1)[1]
    chosen = values if which == "all" else [values[int(which)]]
    changed = {}
    for value in chosen:
        for label, node in groups[value]:
            changed[label] = [value, value + 1]
            node.value = value + 1
    res["perturbed_constants"] = changed
    fd, new_path = tempfile.mkstemp(suffix=".py", prefix="cbu_ref_")
    with os.fdopen(fd, "w") as f:
        f.write(ast.unparse(tree))
    return new_path


def _run_tritongym(a, res: dict) -> None:
    import torch

    mod = _load_module(a.file, "cbu_target_module")
    res["tilelens_records_at_import"] = None
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
    _poison_report(res, outs, list(args))


def _run_kernelbench(a, res: dict) -> None:
    import torch

    res["status"] = "reference"
    ref_path = _perturbed_reference(a.ref_file, a.variant, res)
    # Some references build their inputs at import time, so the seed has to come first.
    torch.manual_seed(a.seed)
    ref = _load_module(ref_path, "cbu_reference_module")
    torch.manual_seed(a.seed)
    get_init = getattr(ref, "get_init_inputs", None)
    init_args = list(get_init() or []) if get_init is not None else []
    torch.manual_seed(a.seed)
    ref_model = ref.Model(*init_args)
    if a.model_mode == "eval":
        ref_model.eval()
    torch.manual_seed(a.seed)
    inputs = list(ref.get_inputs())
    if a.variant.startswith("const:"):
        # The perturbed shape only counts when the reference itself accepts it.
        with torch.no_grad():
            ref_model(*[x.clone() if isinstance(x, torch.Tensor) else x for x in inputs])
    else:
        inputs = _vary_batch(inputs, a.variant)
    res["input_shapes"] = [[str(x.dtype), list(x.shape)] if isinstance(x, torch.Tensor) else type(x).__name__ for x in inputs]

    res["inputs_before"] = _digest([x for x in inputs if isinstance(x, torch.Tensor)])
    if "poison" in res:
        from cbu import poison

        # A benchmark input that is itself torch.empty carries the pattern before any kernel runs.
        res["poison"]["inputs_before"] = poison.count(inputs)
    res["status"] = "importing"
    torch.manual_seed(a.seed)
    cand = _load_module(a.file, "cbu_target_module")
    res["asan_log_bytes_at_import"] = _asan_log_bytes()
    torch.manual_seed(a.seed)
    cand_model = cand.ModelNew(*init_args)
    if a.model_mode == "eval":
        cand_model.eval()
    try:
        cand_model.load_state_dict(ref_model.state_dict(), strict=False)
    except Exception as e:  # noqa: BLE001 - parameter layouts may differ by design
        res["state_dict_note"] = f"{type(e).__name__}: {e}"[:200]

    cand_inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
    ref_inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
    res["status"] = "running"
    # Candidate first, so it cannot pick up memory the reference just freed.
    with torch.no_grad():
        out = cand_model(*cand_inputs)
    outs: list = []
    _flatten(out, outs)
    res["output"] = _digest(outs)
    res["inputs_after"] = _digest([x for x in cand_inputs if isinstance(x, torch.Tensor)])
    _poison_report(res, outs, cand_inputs)
    res["status"] = "comparing"
    try:
        with torch.no_grad():
            ref_out = ref_model(*ref_inputs)
        refs: list = []
        _flatten(ref_out, refs)
        if "poison" in res:
            from cbu import poison

            res["poison"]["reference"] = poison.count(refs)
        cmp = {"n_out": len(outs), "n_ref": len(refs), "shape_match": [list(o.shape) for o in outs] == [list(r.shape) for r in refs]}
        if cmp["shape_match"] and outs:
            diffs, close = [], True
            for o, r in zip(outs, refs):
                o64, r64 = o.detach().double(), r.detach().double()
                d = (o64 - r64).abs()
                diffs.append(float(torch.nan_to_num(d, nan=float("inf")).max()) if d.numel() else 0.0)
                close = close and bool(torch.allclose(o64, r64, rtol=1e-2, atol=1e-2, equal_nan=False))
            cmp["max_abs_diff"] = max(diffs) if diffs else 0.0
            cmp["allclose_1e-2"] = close
        else:
            cmp["allclose_1e-2"] = False
        res["reference"] = cmp
    except Exception as e:  # noqa: BLE001
        res["reference"] = {"error": f"{type(e).__name__}: {e}"[:300]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["tritongym", "kernelbench"], default="tritongym")
    ap.add_argument("--file", required=True, help="tritongym: file with the launch function. kernelbench: candidate file with ModelNew")
    ap.add_argument("--ref-file", default=None, help="kernelbench: reference file with Model, get_inputs, get_init_inputs")
    ap.add_argument("--op", default="kernel")
    ap.add_argument("--func", default=None)
    ap.add_argument("--shapes", default="[]", help="tritongym: JSON list of tensor shapes for one case")
    ap.add_argument("--variant", default="native", help="native, batch1, batch3, batch7, const:<i> or const:all")
    ap.add_argument("--root", default=None, help="directory to put on sys.path")
    ap.add_argument("--mode", choices=["plain", "tilelens"], default="plain")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-mode", choices=["eval", "train"], default="eval", help="KernelGYM leaves models in training mode")
    ap.add_argument("--replay", type=int, default=0, help="kernelbench: replay this many KernelGYM correctness trials in one process")
    ap.add_argument("--no-shim", action="store_true", help="do not redirect CUDA device requests to the CPU")
    ap.add_argument("--no-pretend-cuda", action="store_true", help="keep is_cuda and device truthful, so CUDA-guarded wrappers fall back")
    a = ap.parse_args()

    res: dict = {"op": a.op, "adapter": a.adapter, "mode": a.mode, "shapes": json.loads(a.shapes), "variant": a.variant, "status": "start"}
    t0 = time.time()
    clients = []
    launches = {"n": None}
    atomics = {"n": None}
    bounds: dict = {}
    boolops = {"n": None}
    try:
        if a.root:
            sys.path.insert(0, a.root)
        import torch

        if not a.no_shim and not torch.cuda.is_available():
            from cbu import cpu_shim

            cpu_shim.install(pretend_cuda=not a.no_pretend_cuda)
            res["cpu_shim"] = "pretend_cuda" if not a.no_pretend_cuda else "redirect_only"
        if os.environ.get("CBU_POISON") == "1":
            from cbu import poison

            poison.install()
            res["poison"] = {}
        launches = _count_kernel_launches()
        if os.environ.get("TRITON_INTERPRET") != "1":
            _count_compiled_launches(launches)
        _STATE["launches"] = launches
        bounds = _bounds_oracle()
        atomics = _expose_atomics()
        boolops = _match_compiler_boolops()
        if a.mode == "tilelens":
            clients = _install_tilelens()
        res["status"] = "importing"
        if a.adapter == "tritongym":
            _run_tritongym(a, res)
        elif a.replay:
            _run_replay(a, res)
        else:
            _run_kernelbench(a, res)
        res["status"] = "ok"
    except SystemExit as e:  # some tools call sys.exit on error
        res["status_at_failure"] = res["status"]
        res["status"] = "sysexit"
        res["exit_code"] = e.code
    except BaseException as e:  # noqa: BLE001 - untrusted target code
        res["status_at_failure"] = res["status"]
        res["status"] = "exception"
        res["exception"] = f"{type(e).__name__}: {e}"[:2000]
        res["traceback"] = traceback.format_exc()[-3000:]
    res["seconds"] = round(time.time() - t0, 3)
    res["kernel_launches"] = launches.get("n")
    res["atomic_ops"] = atomics.get("n") if isinstance(atomics, dict) else None
    res["bounds"] = {k: bounds.get(k) for k in ("active", "n", "accesses", "violations", "errors", "install_error") if bounds.get(k) is not None}
    res["model_mode"] = a.model_mode
    res["tensor_boolops"] = boolops.get("n") if isinstance(boolops, dict) else None
    try:
        from cbu import cpu_shim as _shim

        res["autotune_max_configs"] = _shim.AUTOTUNE_SEEN["max_configs"]
        res["autotune_index"] = os.environ.get("CBU_AUTOTUNE_INDEX", "0") or "0"
        res["autotune_tuning_passes"] = _shim.AUTOTUNE_SEEN.get("tuning_passes", 0)
    except Exception:  # noqa: BLE001
        pass
    if a.mode == "tilelens":
        res["kernel_launches"] = max(launches.get("n") or 0, TILELENS_LAUNCHES["n"])
        try:
            res["tilelens_records"] = _tilelens_records(clients)
        except BaseException as e:  # noqa: BLE001
            res["tilelens_records_error"] = repr(e)
    _emit(res)
    return 0


if __name__ == "__main__":
    os._exit(main())
