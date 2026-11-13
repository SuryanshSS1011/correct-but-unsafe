"""Minimal demonstration of a partial write that autotuning hides on the first call.

The kernel is autotuned over its block size. The wrapper launches it with a
grid computed for a block of 1024, which is the pattern found in accepted
Dr. Kernel kernels. A config with a smaller block writes only part of the
output. Run it on any CUDA GPU:

    python gpu/minimal_repro.py --repeats 5

Each repeat runs in a fresh process, because the autotuner caches its winner
per process. The script needs only torch and triton.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

BLOCKS = (256, 512, 1024, 2048)
SIZES = (8192, 1 << 16, 1 << 20, 1 << 24)


def child() -> dict:
    import torch
    import triton
    import triton.language as tl

    @triton.autotune(configs=[triton.Config({"BLOCK": b}, num_warps=4) for b in BLOCKS], key=["n"])
    @triton.jit
    def add_one(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1.0, mask=mask)

    def launch(x, y):
        grid = (triton.cdiv(x.numel(), 1024),)  # fixed grid, complete only when BLOCK >= 1024
        add_one[grid](x, y, x.numel())

    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": triton.__version__, "sizes": []}
    for n in SIZES:
        rec: dict = {"n": n, "unwritten_marked": [], "allclose_plain": []}
        # Part A. The output is pre-filled with NaN, so every unwritten element is visible.
        for _ in range(4):
            x = torch.randn(n, device="cuda")
            y = torch.full_like(x, float("nan"))
            launch(x, y)
            torch.cuda.synchronize()
            rec["unwritten_marked"].append(int(torch.isnan(y).sum()))
        best = add_one.best_config
        rec["winner_block"] = int(best.kwargs["BLOCK"])
        timings = getattr(add_one, "configs_timings", None) or {}
        rec["timings_ms"] = {str(c.kwargs["BLOCK"]): (t[0] if isinstance(t, (list, tuple)) else t) for c, t in timings.items()}
        # Part B. Plain torch.empty, fresh inputs per trial and the reference computed first,
        # which is how a benchmark harness calls the kernel.
        for _ in range(5):
            x = torch.randn(n, device="cuda")
            ref = x + 1.0
            y = torch.empty_like(x)
            launch(x, y)
            torch.cuda.synchronize()
            rec["allclose_plain"].append(bool(torch.allclose(ref, y, atol=1e-2, rtol=1e-2)))
        out["sizes"].append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="gpu_minimal_repro.json")
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        print("REPRO_JSON:" + json.dumps(child()))
        return 0
    runs = []
    for i in range(a.repeats):
        p = subprocess.run([sys.executable, __file__, "--child"], capture_output=True, text=True)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("REPRO_JSON:")), None)
        if line is None:
            print(f"repeat {i} failed:\n{p.stderr[-1500:]}")
            continue
        runs.append(json.loads(line[len("REPRO_JSON:"):]))
        for rec in runs[-1]["sizes"]:
            print(f"repeat {i} n={rec['n']}: winner BLOCK={rec['winner_block']} unwritten per call={rec['unwritten_marked']} plain trials pass={rec['allclose_plain']}")
    json.dump({"runs": runs}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    print("Reading the output: the first call always shows 0 unwritten elements, because tuning ran every config on the buffer.")
    print("If the winner block is below 1024, later calls leave elements unwritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
