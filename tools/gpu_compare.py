"""Compare GPU kit results across GPU types and read the disk-cache run.

Usage: python tools/gpu_compare.py [--raw results/raw]
It reads the A100 run from results/raw/ and the other runs from
results/raw/t4/, h100/ and a100-cache/, as written by gpu/modal_app.py, and
prints the numbers reported in results/2026-11-17-gpu.md.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(d: Path, name: str) -> dict | None:
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def plain_signature(run: dict) -> str | None:
    ok = tuple(t.get("allclose") for t in run.get("trials") or [])
    if not ok:
        return None
    if all(ok):
        return "5 of 5"
    if ok == (True, False, False, False, False):
        return "1 of 5"
    return "0 of 5" if not any(ok) else "other"


def summarize(k: dict) -> dict:
    s = {
        "winners": set(),
        "leak": 0,
        "poison": 0,
        "plain_fail": 0,
        "sig": Counter(),
        "error": k.get("error"),
    }
    for r in k["runs"]:
        if r.get("leak"):
            continue
        tr = r.get("trials") or []
        s["winners"] |= {
            t["winner_index"]
            for t in r.get("autotuned") or []
            if t["winner_index"] is not None
        }
        sig = plain_signature(r)
        s["sig"][sig] += 1
        if r.get("poison"):
            s["poison"] += 1
            s["leak"] += any((t.get("unwritten") or 0) > 0 for t in tr)
        else:
            s["plain_fail"] += sig != "5 of 5"
    return s


def hardware(sets: dict[str, Path]) -> None:
    data = {}
    for g, d in sets.items():
        c = load(d, "gpu_confirmation.json")
        if c:
            data[g] = {k["id"]: summarize(k) for k in c["kernels"]}
    gpus = list(data)
    print("== poison runs with unwritten elements, of 6, and autotune winners")
    for i in sorted(set().union(*data.values())):
        row = {g: data[g].get(i) for g in gpus}
        leaks = {g: row[g]["leak"] > 0 for g in gpus if row[g] and not row[g]["error"]}
        cells = " | ".join(
            f"{g} {row[g]['leak']}/{row[g]['poison']} w{sorted(row[g]['winners'])}"
            + (" ERROR" if row[g]["error"] else "")
            for g in gpus
            if row[g]
        )
        print(
            f"{i:9s} {cells}"
            + ("  <-- differs" if len(set(leaks.values())) > 1 else "")
        )
    print()
    for g in gpus:
        v = data[g].values()
        print(
            f"{g}: kernels with unwritten elements {sum(s['leak'] > 0 for s in v)}/{len(v)}, "
            f"poison runs {sum(s['leak'] for s in v)}/{sum(s['poison'] for s in v)}, "
            f"kernels with the 1-of-5 signature {sum(s['sig']['1 of 5'] > 0 for s in v)}, "
            f"unwritten but every plain trial passes {sum(s['leak'] > 0 and s['plain_fail'] == 0 for s in v)}, "
            f"errors {sum(bool(s['error']) for s in v)}"
        )
    print("\n== minimal demonstration")
    for g, d in sets.items():
        m = load(d, "gpu_minimal.json")
        if not m or not m.get("runs"):
            continue
        by_n: dict[int, list[int]] = {}
        for run in m["runs"]:
            for rec in run["sizes"]:
                by_n.setdefault(rec["n"], []).append(rec["winner_block"])
        part = sum(b < 1024 for bs in by_n.values() for b in bs)
        print(
            f"{g} ({m['runs'][0]['device']}): partial winner in {part}/{sum(map(len, by_n.values()))} cells; "
            + "; ".join(f"n={n}: {bs}" for n, bs in by_n.items())
        )
    print("\n== crash check")
    for g, d in sets.items():
        c = load(d, "gpu_crash_check.json")
        for k in (c or {}).get("kernels", []):
            plain = [r for r in k["runs"] if not r.get("leak")]
            exc = {r.get("exception") for r in plain} - {None}
            ok = all(t.get("allclose") for r in plain for t in r.get("trials") or [])
            print(
                f"{g} {k['id']}: {'all trials match' if ok else 'some trials do not match'}"
                + (f", exceptions {exc}" if exc else "")
                + (f", {k['error']}" if k.get("error") else "")
            )


def cache(base: Path, cached: Path) -> None:
    c = load(cached, "gpu_confirmation.json")
    if not c:
        return
    print(
        "\n== disk cache. The first process that launches the kernel tunes, later ones should load the result."
    )
    hits = launched = first_call_leaks = 0
    for k in c["kernels"]:
        runs = [r for r in k["runs"] if not r.get("leak")]
        ran = [
            r
            for r in runs
            if r.get("trials")
            and (r["trials"][0].get("launches") or 0) > 0
            and r.get("autotuned")
        ]
        if not ran:
            continue
        t0 = {a["kernel"]: a["timings"] for a in ran[0]["autotuned"]}
        for r in ran[1:]:
            launched += 1
            hits += all(a["timings"] == t0.get(a["kernel"]) for a in r["autotuned"])
            first_call_leaks += bool(
                r["poison"] and (r["trials"][0].get("unwritten") or 0) > 0
            )
    print(
        f"later processes that launched an autotuned kernel: {launched}, loaded the cached result: {hits}, "
        f"poison runs whose first call left elements unwritten: {first_call_leaks}"
    )
    for name, d in (("without cache", base), ("with cache", cached)):
        sig: Counter = Counter()
        for k in load(d, "gpu_confirmation.json")["kernels"]:
            sig.update(
                plain_signature(r)
                for r in k["runs"]
                if not r.get("leak") and not r.get("poison")
            )
        sig.pop(None, None)
        print(f"plain runs {name}: {dict(sig)}")
    m = load(cached, "gpu_minimal.json")
    for i, run in enumerate((m or {}).get("runs", [])):
        print(
            f"minimal with cache, process {i}: "
            + ", ".join(
                f"n={r['n']} block {r['winner_block']} first call unwritten {r['unwritten_marked'][0]}"
                for r in run["sizes"]
            )
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    raw = Path(ap.parse_args().raw)
    hardware({"T4": raw / "t4", "A100": raw, "H100": raw / "h100"})
    cache(raw, raw / "a100-cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
