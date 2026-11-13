"""Build the audit population from the Dr. Kernel cold-start corpus and write a random sample.

Population: one kernel per trajectory, taken from the trajectory's best round,
kept only when that round has recorded KernelGYM feedback saying the kernel
compiled, passed the correctness trials, and was not a decoy. Round 5 has no
recorded feedback, so trajectories whose best round is 5 are left out.

The contract-grade verifier (RishiShah99/lethe, MIT) released per-row verdicts
keyed by the corpus `uuid` column. They are attached when available.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lethe", default=None, help="path to a lethe checkout (optional)")
    ap.add_argument("--n", type=int, default=300, help="sample size, or 0 for the whole population")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("hkust-nlp/drkernel-coldstart-8k", split="train")
    print("rows", len(ds))

    verdicts = {}
    if a.lethe:
        for line in gzip.open(Path(a.lethe) / "results" / "audit_rows.jsonl.gz", "rt"):
            v = json.loads(line)
            verdicts[int(v["id"])] = v
        by_uuid = {int(u): float(s) for u, s in zip(ds["uuid"], ds["final_speedup"])}
        match = sum(1 for u, v in verdicts.items() if u in by_uuid and abs(by_uuid[u] - float(v["final_speedup"])) < 1e-9)
        print(f"verifier rows {len(verdicts)}; id matches corpus uuid with equal final_speedup: {match}")

    stats = collections.Counter()
    population = []
    for row in ds:
        rs = extract.rounds(row["messages"])
        for x in rs:
            fb = x["feedback"]
            if fb is not None:
                stats["rounds_with_feedback"] += 1
                if fb.get("compiled") and fb.get("correctness") and not fb.get("decoy_kernel"):
                    stats["rounds_accepted"] += 1
        br = int(row["best_round"])
        stats[f"best_round_{br}"] += 1
        if br > len(rs) or br > 4:
            stats["skip_best_round_without_feedback"] += 1
            continue
        best = rs[br - 1]
        fb = best["feedback"] or {}
        if not best["code"]:
            stats["skip_no_code"] += 1
            continue
        if not (fb.get("compiled") and fb.get("correctness") and not fb.get("decoy_kernel")):
            stats["skip_best_round_not_accepted"] += 1
            continue
        v = verdicts.get(int(row["uuid"]))
        ver = None
        if v is not None:
            gates = v.get("gates") or {}
            ver = {
                "status": v.get("status"),
                "op_class": v.get("op_class"),
                "native_shape": v.get("native_shape"),
                "output_aliasing": v.get("output_aliasing"),
                "gates_failed": sorted(g for g, r in gates.items() if isinstance(r, dict) and r.get("status") == "fail"),
            }
        population.append(
            {
                "uuid": int(row["uuid"]),
                "round": br,
                "best_round": br,
                "final_speedup": row["final_speedup"],
                "entry_point": row["entry_point"],
                "ref_code": row["original_python_code"],
                "code": best["code"],
                "feedback": fb,
                "verifier": ver,
            }
        )
    stats["population"] = len(population)
    stats["population_with_verifier_verdict"] = sum(1 for p in population if p["verifier"])
    for k in sorted(stats):
        print(f"{k}: {stats[k]}")

    population.sort(key=lambda p: p["uuid"])
    rng = random.Random(a.seed)
    chosen = population if a.n <= 0 else rng.sample(population, min(a.n, len(population)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(out).endswith(".gz") else open
    with opener(out, "wt") as f:
        for p in chosen:
            f.write(json.dumps(p) + "\n")
    (out.parent / "population_stats.json").write_text(json.dumps(dict(stats), indent=1))
    print("wrote", len(chosen), "kernels to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
