"""Does NVIDIA compute-sanitizer initcheck see the unwritten output, with and without PyTorch's caching allocator?

initcheck reports a read of device memory that nothing has written. A
defective kernel never reads its own unwritten output, but the harness's
comparison does. With the caching allocator the output block is usually a
recycled block that an earlier tensor wrote, so the read looks initialized.
With PYTORCH_NO_CUDA_MEMORY_CACHING=1 every allocation is a fresh cudaMalloc.

For each kernel a warm-up process without the sanitizer runs the five
KernelGYM trials with the poison fill, tunes, and writes the winner to a
private Triton cache. Two sanitizer processes then load that winner from the
cache, so they skip tuning and use the same config, and run the trials without
the poison fill, once with each allocator setting.

    python gpu/initcheck.py --data data/k.jsonl --out results/raw/initcheck.json
    python gpu/initcheck.py --minimal --out results/raw/initcheck_minimal.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SANITIZER = ["compute-sanitizer", "--tool", "initcheck", "--print-limit", "200"]
ALLOCATORS = {"caching": {}, "no_caching": {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}}


def parse_sanitizer(out: str) -> dict:
    """Count initcheck errors and the kernels that made the reads."""
    m = re.search(r"ERROR SUMMARY: (\d+) error", out)
    kernels: Counter = Counter()
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if "Uninitialized __global__ memory read" in ln:
            at = next(
                (x for x in lines[i + 1 : i + 4] if x.strip("= ").startswith("at ")), ""
            )
            kernels[re.sub(r"\+0x[0-9a-f]+$", "", at.strip("= ")[3:])[:120]] += 1
    return {
        "errors": int(m.group(1)) if m else None,
        "printed_by_kernel": dict(kernels.most_common(8)),
    }


def cbu_json(out: str) -> dict | None:
    line = next((ln for ln in out.splitlines() if ln.startswith("CBU_JSON:")), None)
    return json.loads(line[len("CBU_JSON:") :]) if line else None


def trials_summary(res: dict | None) -> dict:
    rp = (res or {}).get("replay") or {}
    return {
        "status": (res or {}).get("status"),
        "allclose": [t.get("allclose") for t in rp.get("trials") or []],
        "unwritten": [t.get("unwritten") for t in rp.get("trials") or []],
        "first_call_launches": ((rp.get("trials") or [{}])[0]).get("launches"),
        "winners": [
            (a.get("kernel"), a.get("winner_index")) for a in rp.get("autotuned") or []
        ],
        "timings": [a.get("timings") for a in rp.get("autotuned") or []],
    }


def run(cmd: list[str], env: dict, timeout: int) -> tuple[str, int | None]:
    """Run cmd and return its output and exit code, or None for the code on timeout.

    timeout <= 0 means no limit. On a timeout the whole process group is killed,
    because compute-sanitizer runs the target in a child that would otherwise
    outlive it and keep the output pipe open.
    """
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(REPO),
        start_new_session=True,
    )
    try:
        out, _ = p.communicate(timeout=timeout if timeout > 0 else None)
        return out, p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return out, None


def check_kernel(row: dict, timeout: int) -> dict:
    kid = f"{row['uuid']}_r{row['round']}"
    d = REPO / "work" / "initcheck" / kid
    d.mkdir(parents=True, exist_ok=True)
    (d / "ref.py").write_text(row["ref_code"])
    (d / "cand.py").write_text(row["code"])
    cache = tempfile.mkdtemp(prefix=f"triton-{kid}-")
    base = dict(
        os.environ,
        PYTHONPATH=str(REPO),
        PYTHONDONTWRITEBYTECODE="1",
        TRITON_CACHE_DIR=cache,
        TRITON_CACHE_AUTOTUNING="1",
    )
    target = [
        sys.executable,
        "-m",
        "cbu.target",
        "--adapter",
        "kernelbench",
        "--file",
        str(d / "cand.py"),
        "--ref-file",
        str(d / "ref.py"),
        "--model-mode",
        "eval",
        "--replay",
        "5",
    ]
    entry: dict = {"id": kid}
    out, rc = run(target, dict(base, CBU_POISON="1"), timeout)
    entry["warmup_poison"] = {"returncode": rc, **trials_summary(cbu_json(out))}
    for name, extra in ALLOCATORS.items():
        out, rc = run(SANITIZER + target, dict(base, CBU_POISON="0", **extra), timeout)
        entry[name] = {
            "returncode": rc,
            "timeout": rc is None,
            **parse_sanitizer(out),
            **trials_summary(cbu_json(out)),
        }
        if entry[name]["errors"] is None:
            entry[name]["tail"] = out[-1500:]
    return entry


def check_minimal(timeout: int, sizes: str = "") -> dict:
    entry: dict = {"id": "minimal_repro"}
    for name, extra in ALLOCATORS.items():
        out, rc = run(
            SANITIZER
            + [sys.executable, str(REPO / "gpu" / "minimal_repro.py"), "--child"]
            + (["--sizes", sizes] if sizes else []),
            dict(os.environ, **extra),
            timeout,
        )
        line = next(
            (ln for ln in out.splitlines() if ln.startswith("REPRO_JSON:")), None
        )
        res = json.loads(line[len("REPRO_JSON:") :]) if line else {}
        entry[name] = {
            "returncode": rc,
            "timeout": rc is None,
            **parse_sanitizer(out),
            "sizes": [
                {
                    k: r[k]
                    for k in ("n", "winner_block", "unwritten_marked", "allclose_plain")
                }
                for r in res.get("sizes", [])
            ],
        }
        if entry[name]["errors"] is None:
            entry[name]["tail"] = out[-1500:]
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="JSONL from tools/dump_kernels.py")
    ap.add_argument(
        "--minimal",
        action="store_true",
        help="check gpu/minimal_repro.py instead of corpus kernels",
    )
    ap.add_argument("--minimal-sizes", default="", help="sizes for --minimal, default all four")
    ap.add_argument("--timeout", type=int, default=0, help="seconds per sanitizer process, 0 for no limit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sv = subprocess.run(
        SANITIZER[:1] + ["--version"], check=False, capture_output=True, text=True
    )
    results = {
        "sanitizer": sv.stdout.strip().splitlines()[-1:] or [sv.stderr.strip()[-200:]],
        "kernels": [],
    }
    if a.minimal:
        results["kernels"].append(check_minimal(a.timeout, a.minimal_sizes))
    else:
        for line in Path(a.data).read_text().splitlines():
            results["kernels"].append(check_kernel(json.loads(line), a.timeout))
            out.write_text(json.dumps(results, indent=1))
    out.write_text(json.dumps(results, indent=1))
    for k in results["kernels"]:
        print(
            k["id"],
            {n: (k[n].get("errors"), k[n].get("allclose", "")) for n in ALLOCATORS},
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
