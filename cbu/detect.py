"""CPU-only detectors for memory-unsafe Triton kernels.

Every detector runs `cbu.target` in a fresh subprocess under the Triton
interpreter, where `tl.load` and `tl.store` become unchecked `memcpy` calls on
real host addresses for every lane whose mask is true. That makes ordinary CPU
memory tooling applicable to Triton kernels without a GPU.

Detectors
---------
plain     Baseline run. Records whether the kernel runs at all on CPU.
tilelens  Symbolic out-of-bounds check from TileLens (Triton-Sanitizer).
asan      AddressSanitizer runtime preloaded into an uninstrumented Python. Its
          memcpy interceptor checks each interpreter load and store against the
          heap redzones around every tensor allocation.
perturb   Differential fill of uninitialized and freed heap memory through
          glibc MALLOC_PERTURB_. Output that changes with the fill byte depends
          on uninitialized or stale memory.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MARK = "CBU_JSON:"
REPO_ROOT = Path(__file__).resolve().parent.parent

ASAN_HEAD = re.compile(r"ERROR: AddressSanitizer: (\S+) on (?:unknown )?address (0x[0-9a-f]+)")
ASAN_SIGNAL = re.compile(r"The signal is caused by a (READ|WRITE) memory access")
ASAN_ACCESS = re.compile(r"^(READ|WRITE) of size (\d+) at (0x[0-9a-f]+)", re.M)
ASAN_WHERE = re.compile(r"is located (\d+) bytes (after|before|inside|to the right of|to the left of)(?: of)? (\d+)-byte region")


def _gcc_lib(name: str) -> str | None:
    for cc in ("gcc", "cc"):
        try:
            p = subprocess.run([cc, f"-print-file-name={name}"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            continue
        cand = p.stdout.strip()
        if cand and os.path.sep in cand and os.path.exists(cand):
            return os.path.realpath(cand)
    return None


def find_asan_preload() -> str | None:
    """Return the LD_PRELOAD value for an uninstrumented Python.

    libstdc++ must be preloaded next to the ASan runtime. The Python binary does
    not link it, so ASan cannot resolve __cxa_throw at start-up and aborts the
    first time a C++ extension throws.
    """
    asan = _gcc_lib("libasan.so")
    if asan is None:
        return None
    std = _gcc_lib("libstdc++.so")
    return f"{asan} {std}" if std else asan


def _target_cmd(spec: dict, mode: str) -> list[str]:
    cmd = [sys.executable, "-m", "cbu.target", "--file", spec["file"], "--op", spec.get("op", "kernel"), "--mode", mode]
    cmd += ["--adapter", spec.get("adapter", "tritongym")]
    cmd += ["--shapes", json.dumps(spec.get("shapes", []))]
    cmd += ["--dtype", spec.get("dtype", "float32")]
    cmd += ["--variant", spec.get("variant", "native")]
    for key, flag in (("func", "--func"), ("root", "--root"), ("ref_file", "--ref-file"), ("model_mode", "--model-mode"), ("replay", "--replay")):
        if spec.get(key):
            cmd += [flag, str(spec[key])]
    return cmd


def _run(spec: dict, mode: str, env_extra: dict, timeout: int) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["CBU_AUTOTUNE_INDEX"] = str(spec.get("autotune_index", 0))
    env.update(env_extra)
    out: dict = {"returncode": None, "result": None, "timeout": False}
    try:
        p = subprocess.run(_target_cmd(spec, mode), capture_output=True, text=True, timeout=timeout, env=env, cwd=spec.get("root") or None)
        out["returncode"] = p.returncode
        stdout, stderr = p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        out["timeout"] = True
        stdout = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    for line in stdout.splitlines():
        if line.startswith(MARK):
            try:
                out["result"] = json.loads(line[len(MARK):])
            except json.JSONDecodeError:
                out["result"] = {"status": "bad_json"}
    out["stderr_tail"] = stderr[-1500:]
    out["_stderr"] = stderr
    return out


# SIGSEGV, SIGABRT and SIGBUS. Under the interpreter every load and store is a raw
# memcpy, so these signals mean an access left mapped memory or corrupted the heap.
CRASH_SIGNALS = {-11: "SIGSEGV", -6: "SIGABRT", -7: "SIGBUS"}


def run_plain(spec: dict, timeout: int = 600) -> dict:
    """Baseline run. A fatal memory signal under the interpreter counts as a detection."""
    r = _run(spec, "plain", {"TRITON_INTERPRET": "1"}, timeout)
    r.pop("_stderr", None)
    r["crash_signal"] = CRASH_SIGNALS.get(r.get("returncode"))
    r["flagged"] = r["crash_signal"] is not None
    return r


def run_tilelens(spec: dict, timeout: int = 900) -> dict:
    r = _run(spec, "tilelens", {"TRITON_INTERPRET": "1"}, timeout)
    r.pop("_stderr", None)
    recs = (r.get("result") or {}).get("tilelens_records") or []
    r["flagged"] = len(recs) > 0
    r["n_reports"] = len(recs)
    r["tool_crash_signal"] = CRASH_SIGNALS.get(r.get("returncode"))
    return r


def parse_asan(text: str) -> list[dict]:
    reports = []
    for chunk in text.split("=================================================================")[1:]:
        head = ASAN_HEAD.search(chunk)
        if not head:
            continue
        acc = ASAN_ACCESS.search(chunk)
        sig = ASAN_SIGNAL.search(chunk)
        where = ASAN_WHERE.search(chunk)
        reports.append(
            {
                "kind": head.group(1),
                "address": head.group(2),
                "access": acc.group(1) if acc else (sig.group(1) if sig else None),
                "size": int(acc.group(2)) if acc else None,
                "offset_bytes": int(where.group(1)) if where else None,
                "side": where.group(2) if where else None,
                "region_bytes": int(where.group(3)) if where else None,
                "in_interpreter": ("init_triton_interpreter" in chunk) or ("interpreter.cc" in chunk),
            }
        )
    return reports


def run_asan(spec: dict, timeout: int = 1800, redzone: int = 2048) -> dict:
    lib = find_asan_preload()
    if lib is None:
        return {"error": "libasan not found", "flagged": None}
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "asan")
        opts = [
            "detect_leaks=0",
            "halt_on_error=0",
            "abort_on_error=0",
            "exitcode=0",
            "alloc_dealloc_mismatch=0",
            "detect_odr_violation=0",
            "new_delete_type_mismatch=0",
            "allocator_may_return_null=1",
            f"redzone={redzone}",
            f"max_redzone={max(redzone, 2048)}",
            "quarantine_size_mb=64",
            "fast_unwind_on_malloc=1",
            "malloc_context_size=2",
            f"log_path={log}",
        ]
        r = _run(spec, "plain", {"TRITON_INTERPRET": "1", "LD_PRELOAD": lib, "ASAN_OPTIONS": ":".join(opts)}, timeout)
        text = ""
        for f in sorted(Path(d).glob("asan*")):
            text += f.read_text(errors="replace")
        text += r.pop("_stderr", "")
    reports = parse_asan(text)
    r["asan_reports"] = reports[:50]
    r["n_reports"] = len(reports)
    r["flagged"] = any(x["kind"] in {"heap-buffer-overflow", "heap-use-after-free", "SEGV", "unknown-crash", "global-buffer-overflow", "stack-buffer-overflow", "negative-size-param"} for x in reports)
    if reports:
        r["asan_excerpt"] = text[text.find("ERROR: AddressSanitizer"):][:1800]
    return r


def run_perturb(spec: dict, timeout: int = 600, fills: tuple[int, int] = (85, 170)) -> dict:
    """Run with two different fill bytes for fresh and freed heap memory and compare the outputs."""
    runs = {}
    plan = [("base_a", {}), ("base_b", {}), (f"fill_{fills[0]}", {"MALLOC_PERTURB_": str(fills[0])}), (f"fill_{fills[1]}", {"MALLOC_PERTURB_": str(fills[1])})]
    baseline = spec.get("baseline")
    if baseline and baseline.get("out"):
        # The caller already has one unperturbed run. Reuse it and save a subprocess.
        runs["base_a"] = {"status": "ok", "returncode": 0, "timeout": False, "out": baseline.get("out"), "inp": baseline.get("inp"), "inp0": baseline.get("inp0"), "nan": baseline.get("nan"), "exception": None, "reused": True}
        plan = plan[1:]
    for label, env in plan:
        env = dict(env, TRITON_INTERPRET="1")
        x = _run(spec, "plain", env, timeout)
        x.pop("_stderr", None)
        res = x.get("result") or {}
        runs[label] = {
            "status": res.get("status"),
            "returncode": x.get("returncode"),
            "timeout": x.get("timeout"),
            "out": (res.get("output") or {}).get("sha256"),
            "inp": (res.get("inputs_after") or {}).get("sha256"),
            "inp0": (res.get("inputs_before") or {}).get("sha256"),
            "nan": (res.get("output") or {}).get("nan"),
            "exception": res.get("exception"),
        }
    a, b = runs["base_a"], runs["base_b"]
    fa, fb = runs[f"fill_{fills[0]}"], runs[f"fill_{fills[1]}"]
    statuses = sorted({str(v["status"]) for v in runs.values()})
    ok = all(v["status"] == "ok" for v in runs.values())
    # The comparison only means something when every run started from the same inputs.
    starts = {v.get("inp0") for v in runs.values() if v.get("inp0")}
    same_inputs = len(starts) <= 1
    # Identical runs that disagree mean the output depends on memory the inputs do not define.
    unstable = ok and same_inputs and (a["out"] != b["out"] or a["inp"] != b["inp"])
    fill_dependent = ok and same_inputs and not unstable and (fa["out"] != fb["out"] or fa["inp"] != fb["inp"] or fa["out"] != a["out"])
    crashed = any(v["returncode"] not in (0, None) for v in runs.values())
    return {
        "runs": runs,
        "all_ok": ok,
        "same_inputs": bool(same_inputs),
        "unstable": bool(unstable),
        "fill_dependent": bool(fill_dependent),
        "crashed": bool(crashed),
        "flagged": bool(unstable or fill_dependent or (crashed and len(statuses) > 1)),
        "statuses": statuses,
    }


def run_poison(spec: dict, timeout: int = 600) -> dict:
    """Run with every uninitialized PyTorch allocation filled with a fixed pattern and count what survives."""
    r = _run(spec, "plain", {"TRITON_INTERPRET": "1", "CBU_POISON": "1"}, timeout)
    r.pop("_stderr", None)
    res = r.get("result") or {}
    p = res.get("poison") or {}
    out, inp, ref = p.get("output") or {}, p.get("inputs_after") or {}, p.get("reference") or {}
    base_nan = (spec.get("baseline") or {}).get("nan")
    # A reference that itself returns poison or NaN cannot serve as the clean side of the comparison.
    ref_clean = not ref or (ref.get("unwritten", 0) == 0 and ref.get("nan", 0) == 0)
    before = p.get("inputs_before") or {}
    # Pattern that was already in the inputs is the benchmark's doing, not the kernel's.
    inputs_added = max(0, int(inp.get("unwritten", 0)) - int(before.get("unwritten", 0)))
    unwritten = int(out.get("unwritten", 0)) + inputs_added
    r["inputs_uninitialized"] = bool(before.get("unwritten", 0))
    derived_nan = max(0, int(out.get("nan", 0)) - int(base_nan or 0)) if base_nan is not None else 0
    r["unwritten"] = unwritten
    r["unwritten_fraction"] = round(out.get("unwritten", 0) / out["elements"], 6) if out.get("elements") else None
    r["derived_nan"] = derived_nan
    r["reference_clean"] = bool(ref_clean)
    r["flagged"] = bool(res.get("status") == "ok" and ref_clean and (unwritten > 0 or derived_nan > 0))
    return r


def bounds_from(plain: dict) -> dict:
    """Read the exact bounds oracle out of a finished plain run. It costs no extra run."""
    res = plain.get("result") or {}
    b = res.get("bounds") or {}
    return {
        "active": bool(b.get("active")),
        "violating_lanes": int(b.get("n") or 0),
        "accesses": int(b.get("accesses") or 0),
        "violations": b.get("violations") or [],
        "flagged": bool(b.get("n")),
        "result": {"status": res.get("status"), "kernel_launches": res.get("kernel_launches")},
    }


def run_bounds(spec: dict, timeout: int = 600) -> dict:
    return bounds_from(run_plain(spec, timeout))


def run_replay(spec: dict, winner: int, trials: int = 5, model_mode: str = "train", timeout: int = 900) -> dict:
    """Replay KernelGYM's correctness trials with poisoned allocations, assuming config `winner` won autotuning."""
    sp = dict(spec, replay=trials, model_mode=model_mode, autotune_index=f"tune:{winner}")
    r = _run(sp, "plain", {"TRITON_INTERPRET": "1", "CBU_POISON": "1"}, timeout)
    r.pop("_stderr", None)
    res = r.get("result") or {}
    tr = (res.get("replay") or {}).get("trials") or []
    r["trials"] = tr
    r["launches"] = [t.get("launches") for t in tr]
    r["unwritten"] = [t.get("unwritten") for t in tr]
    r["pass"] = [bool(t.get("allclose")) and not t.get("unwritten") and "error" not in t for t in tr]
    return r


DETECTORS = {"bounds": run_bounds, "plain": run_plain, "tilelens": run_tilelens, "asan": run_asan, "perturb": run_perturb, "poison": run_poison}
