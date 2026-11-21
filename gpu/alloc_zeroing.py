"""Does a new allocation without PyTorch's cache come back zeroed, and does it depend on size?

The synthetic leak check found the unwritten region zeroed with
PYTORCH_NO_CUDA_MEMORY_CACHING=1 at 4 MiB, while the vLLM runs without the cache
found holes holding earlier data. CUDA documents cudaMalloc memory as
uninitialized and promises no zeroing. This script measures what a freshly
allocated tensor holds after an earlier tensor of the same size was filled with
a marker and freed, across sizes from 4 KiB to 64 MiB, with and without the
cache, and across processes.

    python gpu/alloc_zeroing.py --out alloc_zeroing.json

Needs only torch and a CUDA GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

SIZES = [4 << 10, 64 << 10, 512 << 10, 1 << 20, 2 << 20, (2 << 20) + (4 << 10), 8 << 20, 64 << 20]
MARK = 1234.5


def child(role: str) -> list[dict]:
    import torch

    out = []
    for size in SIZES:
        n = size // 4
        rec = {"bytes": size, "role": role}
        if role in ("victim", "both"):
            a = torch.full((n,), MARK, device="cuda")
            torch.cuda.synchronize()
            del a
            torch.cuda.synchronize()
        if role in ("caller", "both"):
            b = torch.empty(n, device="cuda")
            torch.cuda.synchronize()
            rec.update(marker=int((b == MARK).sum()), zero=int((b == 0).sum()), elements=n)
            del b
        out.append(rec)
    return out


def run(env_extra: dict, role: str) -> list[dict]:
    env = dict(os.environ, **env_extra)
    p = subprocess.run([sys.executable, __file__, "--child", role], capture_output=True, text=True, env=env)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("ZERO_JSON:")), None)
    return json.loads(line[len("ZERO_JSON:"):]) if line else [{"error": p.stderr[-800:]}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="alloc_zeroing.json")
    ap.add_argument("--child")
    a = ap.parse_args()
    if a.child:
        print("ZERO_JSON:" + json.dumps(child(a.child)))
        return 0
    import torch

    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "settings": {}}
    for name, env in (("default", {}), ("no_caching", {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"})):
        res["settings"][name] = run(env, "both")
        run(env, "victim")
        res["settings"][name + "_next_process"] = run(env, "caller")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for name, rows in res["settings"].items():
        for r in rows:
            if "elements" in r:
                print(f"{name:26s} {r['bytes']:>10d} B  marker {r['marker'] / r['elements']:6.1%}  zero {r['zero'] / r['elements']:6.1%}")
            else:
                print(name, r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
