"""Run the CPU detectors over sampled Dr. Kernel kernels that the KernelGYM harness accepted.

Input is a JSONL file produced by fetch_sample.py. Each line holds one kernel:
the reference source, the candidate source and the recorded harness feedback.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cbu.detect import DETECTORS, bounds_from  # noqa: E402


def read_jsonl(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", default="work/drkernel")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--detectors", default="plain,bounds,tilelens,asan,perturb,poison")
    ap.add_argument("--variants", default="native,batch3")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-configs", type=int, default=6, help="audit at most this many autotune configs per kernel")
    ap.add_argument("--budget", type=int, default=1500, help="seconds per kernel before remaining variants are skipped")
    ap.add_argument("--const-variants", type=int, default=3, help="perturb up to this many shape constants one at a time, plus all at once")
    ap.add_argument("--model-mode", choices=["eval", "train"], default="eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    i, n = (int(x) for x in a.shard.split("/"))
    items = [r for k, r in enumerate(read_jsonl(a.data)) if k % n == i]
    if a.limit:
        items = items[: a.limit]
    detectors = [d for d in a.detectors.split(",") if d]
    variants = [v for v in a.variants.split(",") if v]
    work = Path(a.work).resolve()
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"detectors": detectors, "variants": variants, "shard": a.shard, "model_mode": a.model_mode, "kernels": []}
    for item in items:
        kid = f"{item['uuid']}_r{item['round']}"
        d = work / kid
        d.mkdir(parents=True, exist_ok=True)
        (d / "ref.py").write_text(item["ref_code"])
        (d / "cand.py").write_text(item["code"])
        entry = {k: item.get(k) for k in ("uuid", "round", "best_round", "final_speedup", "feedback", "verifier", "entry_point")}
        entry["id"] = kid
        entry["runs"] = []
        # Each work item is (shape variant, autotune config index). Config indices
        # beyond 0 are added after the first run reports how many configs exist.
        queue = [(v, 0) for v in variants]
        done = set()
        native_failed = False
        kernel_t0 = time.time()
        while queue:
            variant, cfg = queue.pop(0)
            if (variant, cfg) in done or (native_failed and (variant, cfg) != ("native", 0)):
                continue
            if time.time() - kernel_t0 > a.budget:
                entry.setdefault("skipped_for_budget", []).append([variant, cfg])
                continue
            done.add((variant, cfg))
            spec = {"adapter": "kernelbench", "file": str(d / "cand.py"), "ref_file": str(d / "ref.py"), "op": kid, "variant": variant, "autotune_index": cfg, "model_mode": a.model_mode}
            run = {"variant": variant if cfg == 0 else f"{variant}/cfg{cfg}", "shape_variant": variant, "autotune_index": cfg, "detectors": {}}
            for det in detectors:
                if det == "tilelens" and (cfg > 0 or variant.startswith("const:")):
                    continue  # the symbolic check runs on the benchmark shapes; other variants use the concrete detectors
                if det == "bounds":
                    continue  # read from the plain run below, which already carries the oracle's result
                t0 = time.time()
                try:
                    r = DETECTORS[det](spec, timeout=a.timeout)
                except Exception as e:  # noqa: BLE001
                    r = {"error": repr(e), "flagged": None}
                r["wall_seconds"] = round(time.time() - t0, 2)
                run["detectors"][det] = r
                res = r.get("result") if isinstance(r.get("result"), dict) else {}
                print(f"{kid} {run['variant']} {det}: flagged={r.get('flagged')} status={res.get('status')} launches={res.get('kernel_launches')} t={r['wall_seconds']}s", flush=True)
                if det == "plain":
                    if "bounds" in detectors:
                        run["detectors"]["bounds"] = bounds_from(r)
                    if res.get("status") == "ok":
                        spec["baseline"] = {"out": (res.get("output") or {}).get("sha256"), "inp": (res.get("inputs_after") or {}).get("sha256"), "inp0": (res.get("inputs_before") or {}).get("sha256"), "nan": (res.get("output") or {}).get("nan")}
                    failed = res.get("status") != "ok" and r.get("crash_signal") is None
                    if (variant, cfg) == ("native", 0):
                        # A kernel that cannot run natively on CPU is an exclusion. Skip its other variants.
                        native_failed = failed
                        n_cfg = int(res.get("autotune_max_configs") or 0)
                        entry["autotune_max_configs"] = n_cfg
                        for extra in range(1, min(n_cfg, a.max_configs)):
                            queue.append(("native", extra))
                        n_const = int(res.get("n_int_constants") or 0)
                        entry["n_int_constants"] = n_const
                        if a.const_variants > 0 and n_const > 0:
                            for ci in range(min(n_const, a.const_variants)):
                                queue.append((f"const:{ci}", 0))
                            if n_const > 1:
                                queue.append(("const:all", 0))
                    if failed:
                        break  # other detectors would hit the same non-memory failure
            entry["runs"].append(run)
        results["kernels"].append(entry)
        out_path.write_text(json.dumps(results, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
