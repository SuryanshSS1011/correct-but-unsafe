"""GPU confirmation of the CPU findings on accepted Dr. Kernel kernels.

For each kernel it replays KernelGYM's five correctness trials natively, with
the real autotuner, in a fresh process per repeat. It does so in training and
eval mode, with and without the poison fill. It records which autotune config
won, how many output elements each trial left unwritten and whether each
trial passed.

    pip install datasets
    python gpu/confirm.py --kernels flagged --repeats 3

`--interpret` runs the same plumbing on the CPU interpreter for a dry run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def flagged() -> str:
    t = json.loads((REPO / "results" / "triage-census.json").read_text())
    ids = list(t["out_of_bounds_read"]) + list(t["uninitialized_output_source_read"]) + t["uninitialized_output_pattern_only"]
    return ",".join(i.replace("_r", ":") for i in sorted(ids))


def run(cand: Path, ref: Path, mode: str, poison: bool, interpret: bool, timeout: int, leak: bool = False) -> dict:
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    env["CBU_POISON"] = "1" if poison else "0"
    if interpret:
        env.update(TRITON_INTERPRET="1", CBU_AUTOTUNE_INDEX="0" if leak else "tune:0")
    cmd = [sys.executable, "-m", "cbu.target", "--adapter", "kernelbench", "--file", str(cand), "--ref-file", str(ref), "--model-mode", mode]
    cmd += ["--leak-demo", "3"] if leak else ["--replay", "5"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(REPO))
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("CBU_JSON:")), None)
    if line is None:
        return {"returncode": p.returncode, "stderr": p.stderr[-600:]}
    res = json.loads(line[len("CBU_JSON:"):])
    if leak:
        lk = res.get("leak") or {}
        return {"status": res.get("status"), "exception": (str(res.get("exception"))[:200] if res.get("exception") else None),
                "device": lk.get("device"), "leak_trials": lk.get("trials")}
    rp = res.get("replay") or {}
    return {"status": res.get("status"), "exception": (str(res.get("exception"))[:200] if res.get("exception") else None),
            "device": rp.get("device"), "trials": rp.get("trials"), "autotuned": rp.get("autotuned")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="flagged", help="'flagged' or uuid:round,...")
    ap.add_argument("--data", default=None, help="existing JSONL from tools/dump_kernels.py; skips the download")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--modes", default="train,eval")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--interpret", action="store_true")
    ap.add_argument("--out", default="results/raw/gpu_confirmation.json")
    a = ap.parse_args()
    data = Path(a.data) if a.data else REPO / "data" / "gpu_kernels.jsonl"
    if not a.data:
        subprocess.run([sys.executable, str(REPO / "tools" / "dump_kernels.py"), "--kernels", flagged() if a.kernels == "flagged" else a.kernels, "--out", str(data)], check=True)
    out_path = REPO / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for line in data.read_text().splitlines():
        row = json.loads(line)
        kid = f"{row['uuid']}_r{row['round']}"
        d = REPO / "work" / "gpu" / kid
        d.mkdir(parents=True, exist_ok=True)
        (d / "ref.py").write_text(row["ref_code"])
        (d / "cand.py").write_text(row["code"])
        entry = {"id": kid, "runs": []}
        for mode in a.modes.split(","):
            for poison in (True, False):
                for rep in range(a.repeats):
                    r = run(d / "cand.py", d / "ref.py", mode, poison, a.interpret, a.timeout)
                    r.update(mode=mode, poison=poison, repeat=rep)
                    entry["runs"].append(r)
                    tr = r.get("trials") or []
                    win = [(t.get("kernel"), t.get("winner_index")) for t in (r.get("autotuned") or [])]
                    print(f"{kid} {mode} poison={int(poison)} rep={rep}: launches={[t.get('launches') for t in tr]} unwritten={[t.get('unwritten') for t in tr]} "
                          f"allclose={[t.get('allclose') for t in tr]} winner={win} {r.get('exception') or ''}", flush=True)
            # Marker test: does the unwritten part of the output return memory that held other data?
            for rep in range(a.repeats):
                r = run(d / "cand.py", d / "ref.py", mode, False, a.interpret, a.timeout, leak=True)
                r.update(mode=mode, leak=True, repeat=rep)
                entry["runs"].append(r)
                lt = r.get("leak_trials") or []
                print(f"{kid} {mode} marker test rep={rep}: marker elements in output={[t.get('leaked') for t in lt]} of {[t.get('elements') for t in lt][:1]} "
                      f"allclose={[t.get('allclose') for t in lt]} {r.get('exception') or ''}", flush=True)
        results.append(entry)
        out_path.write_text(json.dumps({"kernels": results}, indent=1))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
