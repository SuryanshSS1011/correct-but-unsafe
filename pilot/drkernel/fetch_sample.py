"""Fetch the Dr. Kernel cold-start corpus, join it with released verifier verdicts, and write a stratified sample.

The corpus has no correctness column. The contract-grade verifier (RishiShah99/lethe,
MIT) released per-row verdicts for 3,134 kernels, keyed by an integer id. This
script checks that the id is the dataset row index by comparing `final_speedup`,
then samples accepted rows by operator class and speedup quartile.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import statistics
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lethe", required=True, help="path to a lethe checkout")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-accepted", action="store_true", help="write every accepted row instead of a sample")
    a = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("hkust-nlp/drkernel-coldstart-8k", split="train")
    print("rows", len(ds))
    print("features", ds.features)
    info = getattr(ds, "info", None)
    print("fingerprint", getattr(ds, "_fingerprint", None))
    print("best_round", collections.Counter(ds["best_round"]))
    sp = ds["final_speedup"]
    print("final_speedup >0:", sum(1 for x in sp if x and x > 0), ">1:", sum(1 for x in sp if x and x > 1), "median:", statistics.median(sp))
    print("enable_thinking", collections.Counter(ds["enable_thinking"]))

    verdicts = [json.loads(line) for line in gzip.open(Path(a.lethe) / "results" / "audit_rows.jsonl.gz", "rt")]
    print("verifier rows", len(verdicts), collections.Counter(v["status"] for v in verdicts))
    match = mismatch = 0
    for v in verdicts:
        i = int(v["id"])
        if i < len(ds) and abs(float(ds[i]["final_speedup"]) - float(v["final_speedup"])) < 1e-9:
            match += 1
        else:
            mismatch += 1
    print(f"id-as-row-index check: {match} match, {mismatch} mismatch")

    accepted = [v for v in verdicts if v["status"] == "gated" and float(v["final_speedup"]) > 0]
    print("accepted (gated and final_speedup > 0):", len(accepted))
    speeds = sorted(float(v["final_speedup"]) for v in accepted)
    q = [speeds[int(len(speeds) * k / 4)] for k in (1, 2, 3)]

    def quartile(x: float) -> int:
        return sum(x >= t for t in q)

    strata = collections.defaultdict(list)
    for v in accepted:
        strata[(v["op_class"], quartile(float(v["final_speedup"])))].append(v)
    rng = random.Random(a.seed)
    if a.all_accepted:
        chosen = accepted
    else:
        chosen = []
        keys = sorted(strata)
        for k in keys:
            rng.shuffle(strata[k])
        per = max(1, a.n // len(keys))
        for k in keys:
            chosen += strata[k][:per]
        rest = [v for k in keys for v in strata[k][per:]]
        rng.shuffle(rest)
        chosen += rest[: max(0, a.n - len(chosen))]
        chosen = chosen[: a.n]
    print("chosen", len(chosen), collections.Counter(v["op_class"] for v in chosen))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for v in chosen:
            row = ds[int(v["id"])]
            gates = v.get("gates") or {}
            rec = {
                "id": int(v["id"]),
                "uuid": row.get("uuid"),
                "entry_point": row.get("entry_point"),
                "repo_name": row.get("repo_name"),
                "module_name": row.get("module_name"),
                "final_speedup": row.get("final_speedup"),
                "best_round": row.get("best_round"),
                "num_rounds": row.get("num_rounds"),
                "original_python_code": row.get("original_python_code"),
                "messages": row.get("messages"),
                "verifier": {
                    "op_class": v.get("op_class"),
                    "native_shape": v.get("native_shape"),
                    "output_aliasing": v.get("output_aliasing"),
                    "gates_failed": sorted(g for g, r in gates.items() if isinstance(r, dict) and r.get("status") == "fail"),
                    "gates_passed_all": all(isinstance(r, dict) and r.get("status") != "fail" for r in gates.values()) if gates else None,
                },
            }
            f.write(json.dumps(rec) + "\n")
    print("wrote", out, out.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
