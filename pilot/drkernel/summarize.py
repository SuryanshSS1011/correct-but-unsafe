"""Merge pilot shard results and print a Markdown summary with Wilson intervals."""
from __future__ import annotations

import collections
import glob
import json
import math
import sys
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k} of {n} ({100 * k / n:.1f}%, 95% CI {100 * lo:.1f} to {100 * hi:.1f})" if n else "0 of 0"


def classify(kernel: dict) -> dict:
    out = {"id": kernel["id"], "auditable": False, "exclusion": None, "flags": {}, "details": [], "matches_ref_native": None}
    native = next((r for r in kernel["runs"] if r["variant"] == "native"), None)
    out["config_mismatch"] = []
    for r in kernel["runs"]:
        pres = (r["detectors"].get("plain", {}).get("result") or {})
        if r.get("autotune_index") and pres.get("status") == "ok" and (pres.get("reference") or {}).get("allclose_1e-2") is False:
            out["config_mismatch"].append(r["variant"])
    if native is None:
        out["exclusion"] = "no native run"
        return out
    plain = native["detectors"].get("plain", {})
    res = plain.get("result") or {}
    crash = plain.get("crash_signal")
    if res.get("status") == "ok" and (res.get("kernel_launches") or 0) > 0:
        out["auditable"] = True
        out["matches_ref_native"] = (res.get("reference") or {}).get("allclose_1e-2")
    elif crash:
        out["auditable"] = True
    elif res.get("status") == "ok":
        out["exclusion"] = "no Triton kernel launched on CPU"
    elif plain.get("timeout"):
        out["exclusion"] = "timeout"
    else:
        exc = (res.get("exception") or "unknown failure").split("\n")[0]
        out["exclusion"] = exc[:70]
    if not out["auditable"]:
        return out
    out["nearby"] = collections.Counter()
    for run in kernel["runs"]:
        if not str(run.get("shape_variant", "")).startswith("const:"):
            continue
        p = run["detectors"].get("plain", {})
        pres = p.get("result") or {}
        if p.get("crash_signal"):
            out["nearby"]["crash"] += 1
        elif pres.get("status") == "exception" and pres.get("status_at_failure") == "reference":
            out["nearby"]["reference rejects shape"] += 1
        elif pres.get("status") == "exception":
            out["nearby"]["candidate raises"] += 1
        elif pres.get("status") == "ok" and (pres.get("kernel_launches") or 0) == 0:
            out["nearby"]["no launch"] += 1
        elif pres.get("status") == "ok" and (pres.get("reference") or {}).get("allclose_1e-2"):
            out["nearby"]["matches reference"] += 1
        elif pres.get("status") == "ok":
            out["nearby"]["silently wrong"] += 1
    for run in kernel["runs"]:
        for det, r in run["detectors"].items():
            if not r.get("flagged"):
                continue
            out["flags"].setdefault(det, []).append(run["variant"])
            d = {"detector": det, "variant": run["variant"]}
            if det == "asan":
                d["reports"] = [{k: x.get(k) for k in ("kind", "access", "size", "offset_bytes", "side", "region_bytes", "in_interpreter")} for x in (r.get("asan_reports") or [])[:4]]
            elif det == "tilelens":
                d["reports"] = [{"op": x.get("op"), "tensor": x.get("tensor_name"), "line": (x.get("traceback") or [{}])[0].get("line_of_code")} for x in ((r.get("result") or {}).get("tilelens_records") or [])[:4]]
            elif det == "perturb":
                d["unstable"], d["fill_dependent"], d["crashed"] = r.get("unstable"), r.get("fill_dependent"), r.get("crashed")
            elif det == "poison":
                d["unwritten"], d["unwritten_fraction"], d["derived_nan"] = r.get("unwritten"), r.get("unwritten_fraction"), r.get("derived_nan")
            elif det == "plain":
                d["signal"] = r.get("crash_signal")
            out["details"].append(d)
    return out


def main(argv: list[str]) -> int:
    files = sorted(f for pat in argv[1:] for f in glob.glob(pat, recursive=True))
    kernels = []
    for f in files:
        kernels += json.loads(Path(f).read_text()).get("kernels", [])
    rows = [dict(classify(k), meta=k) for k in kernels]
    n = len(rows)
    aud = [r for r in rows if r["auditable"]]
    flagged = [r for r in aud if r["flags"]]
    print(f"## Dr. Kernel pilot: {n} accepted kernels from {len(files)} shard files\n")
    print(f"- Auditable on the CPU interpreter: {pct(len(aud), n)}.")
    excl = collections.Counter(r["exclusion"] for r in rows if not r["auditable"])
    for reason, c in excl.most_common(8):
        print(f"  - Excluded, {reason}: {c}")
    m = [r for r in aud if r["matches_ref_native"] is not None]
    print(f"- Auditable kernels whose CPU output matches the reference within 1e-2: {pct(sum(1 for r in m if r['matches_ref_native']), len(m))}.")
    print(f"- Flagged by at least one detector: {pct(len(flagged), len(aud))}.")
    for det in ("plain", "bounds", "tilelens", "asan", "perturb", "poison"):
        k = sum(1 for r in aud if det in r["flags"])
        print(f"  - {det}: {pct(k, len(aud))}")
    both = sum(1 for r in aud if "asan" in r["flags"] and "bounds" in r["flags"])
    only_a = [r["id"] for r in aud if "asan" in r["flags"] and "bounds" not in r["flags"]]
    only_b = [r["id"] for r in aud if "bounds" in r["flags"] and "asan" not in r["flags"]]
    print(f"- asan and the exact bounds oracle agree on {both} kernels. asan only: {only_a}. bounds only: {only_b}.")
    concrete = [r for r in aud if any(d in r["flags"] for d in ("plain", "bounds", "asan", "perturb", "poison"))]
    symbolic_only = [r for r in flagged if set(r["flags"]) == {"tilelens"}]
    uninit = [r for r in aud if any(d in r["flags"] for d in ("perturb", "poison"))]
    print(f"- Flagged by a concrete detector (plain crash, asan, perturb or poison): {pct(len(concrete), len(aud))}.")
    print(f"- Flagged by the symbolic detector alone, which needs manual triage: {len(symbolic_only)}.")
    print(f"- Uninitialized-memory signal (perturb or poison): {pct(len(uninit), len(aud))}.")
    oob = [r for r in aud if any(d in r["flags"] for d in ("tilelens", "asan", "plain", "bounds"))]
    both = [r for r in aud if "tilelens" in r["flags"] and "asan" in r["flags"]]
    print(f"- Out-of-bounds signal (plain crash, tilelens, asan or bounds): {pct(len(oob), len(aud))}.")
    print(f"- Flagged by both tilelens and asan: {len(both)}.")
    native_only = [r for r in flagged if any("native" in v for v in r["flags"].values())]
    print(f"- Flagged at the benchmark's own input shape: {pct(len(native_only), len(aud))}.")
    latent = [r for r in flagged if r["matches_ref_native"]]
    print(f"- Flagged while the native output still matches the reference: {len(latent)} of {len(flagged)}.")
    nearby = collections.Counter()
    for r in aud:
        nearby.update(r.get("nearby") or {})
    if nearby:
        print("- Runs at nearby shapes (one shape constant plus one, or all plus one): " + ", ".join(f"{k} {v}" for k, v in nearby.most_common()) + ".")
        wrong = [r for r in aud if (r.get("nearby") or {}).get("silently wrong") or (r.get("nearby") or {}).get("crash")]
        print(f"- Kernels that are silently wrong or crash at a nearby shape: {pct(len(wrong), len(aud))}.")
    cfgm = [r for r in aud if r.get("config_mismatch")]
    print(f"- Kernels whose output stops matching under another autotune config: {pct(len(cfgm), len(aud))}.")
    print("\nThese are detector flags. A flag becomes a confirmed bug only after a minimal reproduction and a root-cause label.\n")
    if flagged:
        print("| kernel | speedup | flags | first evidence |")
        print("|---|---|---|---|")
        for r in flagged[:60]:
            ev = r["details"][0] if r["details"] else {}
            evs = json.dumps({k: v for k, v in ev.items() if k not in ("detector",)}, default=str)[:170].replace("|", "/")
            fl = ", ".join(f"{d}:{'+'.join(sorted(set(v)))}" for d, v in sorted(r["flags"].items()))
            print(f"| {r['id']} | {r['meta'].get('final_speedup'):.2f} | {fl} | {evs} |")
    summary = {
        "n": n,
        "auditable": len(aud),
        "flagged": len(flagged),
        "exclusions": dict(excl),
        "flagged_kernels": [{"id": r["id"], "uuid": r["meta"].get("uuid"), "round": r["meta"].get("round"), "final_speedup": r["meta"].get("final_speedup"), "flags": r["flags"], "details": r["details"], "matches_ref_native": r["matches_ref_native"], "verifier": r["meta"].get("verifier")} for r in flagged],
        "concrete_flagged": [r["id"] for r in concrete],
        "symbolic_only": [r["id"] for r in symbolic_only],
        "nearby_shape_outcomes": dict(nearby),
        "config_mismatch_kernels": [r["id"] for r in cfgm],
        "nearby_wrong_kernels": [r["id"] for r in aud if (r.get("nearby") or {}).get("silently wrong") or (r.get("nearby") or {}).get("crash")],
    }
    Path("pilot_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
