"""Deep recheck of named Dr. Kernel kernels.

For every kernel and for both model modes it audits every autotune config with
the concrete detectors, and then replays KernelGYM's five correctness trials
once per assumed autotune winner. Input is the JSONL written by
tools/dump_kernels.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cbu import detect  # noqa: E402


def slim(r: dict) -> dict:
    """Keep what the analysis needs and drop bulky fields."""
    res = r.get("result") or {}
    out = {k: r.get(k) for k in ("flagged", "crash_signal", "timeout", "unwritten", "unwritten_fraction", "derived_nan", "pass", "launches", "wall_seconds") if k in r}
    out["status"] = res.get("status")
    out["exception"] = (str(res.get("exception"))[:160] if res.get("exception") else None)
    out["kernel_launches"] = res.get("kernel_launches")
    out["allclose"] = (res.get("reference") or {}).get("allclose_1e-2")
    out["tuning_passes"] = res.get("autotune_tuning_passes")
    if r.get("asan_reports"):
        out["asan"] = [{k: x.get(k) for k in ("access", "size", "offset_bytes", "side", "region_bytes", "in_interpreter")} for x in r["asan_reports"][:2]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", default="work/recheck")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--modes", default="eval,train")
    ap.add_argument("--max-configs", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--budget", type=int, default=3000, help="seconds per kernel")
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    i, n = (int(x) for x in a.shard.split("/"))
    rows = [json.loads(line) for line in open(a.data) if line.strip()]
    rows = [r for k, r in enumerate(rows) if k % n == i]
    work = Path(a.work).resolve()
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"kernels": []}
    for row in rows:
        kid = f"{row['uuid']}_r{row['round']}"
        d = work / kid
        d.mkdir(parents=True, exist_ok=True)
        (d / "ref.py").write_text(row["ref_code"])
        (d / "cand.py").write_text(row["code"])
        entry = {"id": kid, "feedback": row.get("feedback"), "modes": {}}
        t0 = time.time()
        for mode in a.modes.split(","):
            base = {"adapter": "kernelbench", "file": str(d / "cand.py"), "ref_file": str(d / "ref.py"), "op": kid, "variant": "native", "model_mode": mode}
            m = {"configs": [], "replay": []}
            entry["modes"][mode] = m
            first = detect.run_plain(dict(base, autotune_index=0), timeout=a.timeout)
            fres = first.get("result") or {}
            n_cfg = max(1, min(int(fres.get("autotune_max_configs") or 0), a.max_configs))
            m["autotune_configs"] = int(fres.get("autotune_max_configs") or 0)
            m["input_shapes"] = fres.get("input_shapes")
            for cfg in range(n_cfg):
                if time.time() - t0 > a.budget:
                    m["skipped_for_budget"] = True
                    break
                spec = dict(base, autotune_index=cfg)
                plain = first if cfg == 0 else detect.run_plain(spec, timeout=a.timeout)
                pres = plain.get("result") or {}
                c = {"index": cfg, "plain": slim(plain), "bounds": {k: v for k, v in detect.bounds_from(plain).items() if k != "result"}}
                if pres.get("status") == "ok":
                    spec["baseline"] = {"nan": (pres.get("output") or {}).get("nan")}
                    c["poison"] = slim(detect.run_poison(spec, timeout=a.timeout))
                    c["asan"] = slim(detect.run_asan(spec, timeout=a.timeout))
                m["configs"].append(c)
                print(f"{kid} {mode} cfg{cfg}: launches={pres.get('kernel_launches')} bounds={c['bounds'].get('flagged')} poison={(c.get('poison') or {}).get('unwritten_fraction')} asan={(c.get('asan') or {}).get('flagged')}", flush=True)
            if not a.no_replay:
                for winner in range(n_cfg):
                    if time.time() - t0 > a.budget:
                        m["skipped_for_budget"] = True
                        break
                    r = detect.run_replay(base, winner=winner, model_mode=mode, timeout=a.timeout)
                    m["replay"].append(dict(slim(r), winner=winner))
                    print(f"{kid} {mode} replay winner={winner}: launches={r.get('launches')} unwritten={r.get('unwritten')} pass={r.get('pass')}", flush=True)
        entry["seconds"] = round(time.time() - t0, 1)
        results["kernels"].append(entry)
        out_path.write_text(json.dumps(results, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
