"""Positive control on TritonGym reference kernels.

TritonGym issue #1 reported 13 reference kernels with out-of-bounds accesses.
The maintainer confirmed all 13 and fixed them in commit 6998be6. Running the
detectors on the commit before the fix measures recall on known bugs. Running
them on the fixed commit measures how often they still fire on kernels a
maintainer considers repaired.

Usage:
    python controls/tritongym/run_control.py --checkout ext/tritongym_pre \
        --label pre --detectors plain,bounds,tilelens,asan,perturb,poison --out results/pre.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cbu.detect import DETECTORS  # noqa: E402

KNOWN = json.loads((Path(__file__).parent / "expected.json").read_text())


def to_cases(raw) -> list[list[list[int]]]:
    """Normalize TritonGym input_shapes the same way its evaluator does."""
    if not raw:
        return []
    if all(isinstance(x, int) for x in raw):
        return [[list(raw)]]
    if all(isinstance(s, (list, tuple)) and all(isinstance(d, int) for d in s) for s in raw):
        return [[list(s) for s in raw]]
    return [[list(s) for s in case] for case in raw]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--dataset", default="data_v2")
    ap.add_argument("--ops", default="known", help="'known', 'all', or a comma-separated list")
    ap.add_argument("--detectors", default="plain,bounds,tilelens,asan,perturb,poison")
    ap.add_argument("--shard", default="0/1", help="i/n: run ops whose index mod n equals i")
    ap.add_argument("--max-cases", type=int, default=2)
    ap.add_argument("--dtypes", default="float16,float32", help="TritonGym feeds float16 on GPU and float32 on CPU")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    checkout = Path(a.checkout).resolve()
    data = checkout / "benchmark" / a.dataset
    bench = json.loads((data / "benchmark.json").read_text())
    known_ops = sorted(KNOWN["ops"])
    if a.ops == "known":
        ops = [o for o in known_ops if o in bench]
    elif a.ops == "all":
        ops = sorted(bench)
    else:
        ops = [o for o in a.ops.split(",") if o]
    i, n = (int(x) for x in a.shard.split("/"))
    ops = [o for k, o in enumerate(ops) if k % n == i]
    detectors = [d for d in a.detectors.split(",") if d]

    results = {"label": a.label, "checkout": str(checkout), "dataset": a.dataset, "detectors": detectors, "ops": {}}
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for op in ops:
        f = data / "triton" / f"{op}.py"
        entry = {"known_bug": op in KNOWN["ops"], "cases": []}
        if not f.exists():
            entry["error"] = "triton file missing"
            results["ops"][op] = entry
            continue
        combos = [(ci, shapes, dt) for ci, shapes in enumerate(to_cases(bench[op].get("input_shapes"))[: a.max_cases]) for dt in a.dtypes.split(",")]
        for ci, shapes, dt in combos:
            spec = {"file": str(f), "op": op, "shapes": shapes, "root": str(checkout), "dtype": dt}
            case = {"case": ci, "dtype": dt, "shapes": shapes, "detectors": {}}
            for d in detectors:
                t0 = time.time()
                try:
                    r = DETECTORS[d](spec, timeout=a.timeout)
                except Exception as e:  # noqa: BLE001
                    r = {"error": repr(e), "flagged": None}
                r["wall_seconds"] = round(time.time() - t0, 2)
                case["detectors"][d] = r
                status = (r.get("result") or {}).get("status") if isinstance(r.get("result"), dict) else None
                print(f"[{a.label}] {op} case{ci} {dt} {d}: flagged={r.get('flagged')} status={status} t={r['wall_seconds']}s", flush=True)
            entry["cases"].append(case)
        results["ops"][op] = entry
        out_path.write_text(json.dumps(results, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
