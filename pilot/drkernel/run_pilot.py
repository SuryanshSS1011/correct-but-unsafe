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

from cbu.detect import DETECTORS  # noqa: E402


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
    ap.add_argument("--detectors", default="plain,tilelens,asan,perturb")
    ap.add_argument("--variants", default="native,batch3")
    ap.add_argument("--timeout", type=int, default=600)
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
    results = {"detectors": detectors, "variants": variants, "shard": a.shard, "kernels": []}
    for item in items:
        kid = f"{item['uuid']}_r{item['round']}"
        d = work / kid
        d.mkdir(parents=True, exist_ok=True)
        (d / "ref.py").write_text(item["ref_code"])
        (d / "cand.py").write_text(item["code"])
        entry = {k: item.get(k) for k in ("uuid", "round", "best_round", "final_speedup", "feedback", "verifier", "entry_point")}
        entry["id"] = kid
        entry["runs"] = []
        skip_variants = False
        for variant in variants:
            if skip_variants and variant != "native":
                continue
            spec = {"adapter": "kernelbench", "file": str(d / "cand.py"), "ref_file": str(d / "ref.py"), "op": kid, "variant": variant}
            run = {"variant": variant, "detectors": {}}
            for det in detectors:
                t0 = time.time()
                try:
                    r = DETECTORS[det](spec, timeout=a.timeout)
                except Exception as e:  # noqa: BLE001
                    r = {"error": repr(e), "flagged": None}
                r["wall_seconds"] = round(time.time() - t0, 2)
                run["detectors"][det] = r
                res = r.get("result") if isinstance(r.get("result"), dict) else {}
                print(f"{kid} {variant} {det}: flagged={r.get('flagged')} status={res.get('status')} launches={res.get('kernel_launches')} t={r['wall_seconds']}s", flush=True)
                if det == "plain" and variant == "native":
                    # A kernel that cannot run natively on CPU is an exclusion. Do not spend time on its variants.
                    if res.get("status") != "ok" and r.get("crash_signal") is None:
                        skip_variants = True
                if det == "plain" and res.get("status") != "ok" and r.get("crash_signal") is None:
                    break  # other detectors would hit the same non-memory failure
            entry["runs"].append(run)
        results["kernels"].append(entry)
        out_path.write_text(json.dumps(results, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
