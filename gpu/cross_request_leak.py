"""Does a partial write return an earlier request's data?

This is the step between the CPU marker test and a full vLLM experiment. One
long-lived process plays a server. Request A, the victim, computes on data
filled with a recognisable pattern and then frees everything, as a finished
request does. Request B, an ordinary caller, then runs a kernel that is
autotuned over its block size but launched with a fixed grid, so a small
winning block leaves part of B's `torch.empty` output unwritten. The script
counts how many of B's unwritten elements hold A's values.

    python gpu/cross_request_leak.py --out gpu_leak.json

It needs only torch and triton and runs on any CUDA GPU. Each scenario runs in
a fresh process, so the autotuner and the caching allocator start clean.

Scenarios:
1. same_size: A's output and B's output have the same size.
2. larger_victim: A's tensor is four times B's, so B gets part of a split block.
3. interleaved: unrelated allocations happen between A and B.
4. no_caching: PYTORCH_NO_CUDA_MEMORY_CACHING=1, so every tensor is a fresh cudaMalloc.
5. expandable: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
6. cross_process: A runs in one process that exits, and B runs in the next one.
   This checks whether memory handed to a new process still holds the old data.

In every scenario B's kernel is forced to the partial config, which is what a
process sees after a restart when the disk autotune cache holds a small winner.
`--tuned` instead lets the autotuner pick, and reports which block won.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

SCENARIOS = ("same_size", "larger_victim", "interleaved", "no_caching", "expandable", "cross_process")
ENV = {
    "no_caching": {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"},
    "expandable": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
}
N = int(os.environ.get("CBU_N", 1 << 20))
DEVICE = os.environ.get("CBU_DEVICE", "cuda")  # "cpu" with TRITON_INTERPRET=1 only checks the script's logic
SECRET = 4242.0  # A's values are SECRET + i / N, so each element is recognisable and position-coded


def _sync():
    import torch

    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _kernel(tuned: bool):
    import triton
    import triton.language as tl

    blocks = (256, 512, 1024, 2048) if tuned else (256,)

    @triton.autotune(configs=[triton.Config({"BLOCK": b}, num_warps=4) for b in blocks], key=["n"])
    @triton.jit
    def add_one(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1.0, mask=mask)

    def launch(x, y):
        add_one[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel())  # fixed grid

    return add_one, launch


def victim(n: int):
    import torch

    x = torch.arange(n, device=DEVICE, dtype=torch.float32) / n + SECRET
    y = x * 1.0  # A's output, the kind of tensor a finished request leaves behind
    _sync()
    del x, y  # the request ends and its memory returns to the allocator


def caller(n: int, tuned: bool, victim_n: int) -> dict:
    import torch

    kern, launch = _kernel(tuned)
    if tuned:
        # The first call tunes on a scratch buffer, as a warm server would have done earlier.
        s = torch.randn(n, device=DEVICE)
        launch(s, torch.empty_like(s))
        _sync()
        del s
    x = torch.randn(n, device=DEVICE)
    y = torch.empty_like(x)
    launch(x, y)
    _sync()
    expected = x + 1.0
    unwritten = ~torch.isclose(y, expected, atol=1e-3, rtol=1e-3)
    idx = torch.arange(n, device=DEVICE, dtype=torch.float32)
    a_same_pos = torch.isclose(y, idx / victim_n + SECRET, atol=0, rtol=0) & unwritten
    a_any = (y >= SECRET) & (y <= SECRET + 1.0) & unwritten
    zeros = (y == 0) & unwritten
    return {
        "n": n,
        "winner_block": int(kern.best_config.kwargs["BLOCK"]),
        "unwritten": int(unwritten.sum()),
        "victim_value_same_position": int(a_same_pos.sum()),
        "victim_value_any_position": int(a_any.sum()),
        "zero": int(zeros.sum()),
    }


def child(scenario: str, role: str, tuned: bool) -> dict:
    import torch

    out = {"scenario": scenario, "role": role, "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE, "torch": torch.__version__,
           "env": {k: os.environ.get(k) for k in ("PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_CUDA_ALLOC_CONF")}}
    victim_n = 4 * N if scenario == "larger_victim" else N
    if role in ("victim", "both"):
        victim(victim_n)
    if role == "victim":
        return out
    if scenario == "interleaved":
        junk = [torch.randn(N // 8, device=DEVICE) for _ in range(6)]
        _sync()
        del junk
    out.update(caller(N, tuned, victim_n))
    return out


def run(scenario: str, role: str, tuned: bool) -> dict | None:
    env = dict(os.environ, **ENV.get(scenario, {}))
    args = [sys.executable, __file__, "--child", scenario, "--role", role] + (["--tuned"] if tuned else [])
    p = subprocess.run(args, capture_output=True, text=True, env=env)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("LEAK_JSON:")), None)
    if line is None:
        return {"scenario": scenario, "role": role, "error": (p.stderr or "")[-1500:]}
    return json.loads(line[len("LEAK_JSON:"):])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gpu_leak.json")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tuned", action="store_true", help="let the autotuner pick instead of forcing the partial config")
    ap.add_argument("--child")
    ap.add_argument("--role", default="both")
    a = ap.parse_args()
    if a.child:
        print("LEAK_JSON:" + json.dumps(child(a.child, a.role, a.tuned)))
        return 0
    results = []
    for _ in range(a.repeats):
        for sc in SCENARIOS:
            if sc == "cross_process":
                results.append(run(sc, "victim", a.tuned))
                results.append(run(sc, "caller", a.tuned))
            else:
                results.append(run(sc, "both", a.tuned))
    with open(a.out, "w") as f:
        json.dump({"tuned": a.tuned, "runs": results}, f, indent=1)
    for r in results:
        if r and "unwritten" in r:
            share = r["victim_value_same_position"] / r["unwritten"] if r["unwritten"] else 0.0
            print(f"{r['scenario']:14s} block {r['winner_block']:5d} unwritten {r['unwritten']:8d} "
                  f"victim data at the same position {r['victim_value_same_position']:8d} ({share:.1%}) "
                  f"anywhere {r['victim_value_any_position']:8d} zero {r['zero']:8d}")
        elif r and "error" in r:
            print(f"{r['scenario']:14s} {r['role']} error: {r['error'][-200:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
