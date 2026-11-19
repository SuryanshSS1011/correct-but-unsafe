"""Evaluate check strategies as defenses, from the census raw results.

Reads the census shard JSON (run 43387362408), the census triage and the lint
summary, and reports, for the 34 census defects, which configs each detector
flags. It then compares the recall and cost of a config-0 check, a per-config
check, and a lint-gated per-config check. Executes no kernel code.

Usage: gh run download 43387362408 -D census && python -I tools/defense_eval.py census . results/defense_eval.json
"""
import glob
import json
import statistics
import sys
from pathlib import Path

census_dir, repo, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
triage = json.loads((repo / "results/triage-census.json").read_text())
lint = json.loads((repo / "results/lint_summary.json").read_text())
l1 = set(lint["l1_best_ids"])

oob = set(triage["out_of_bounds_read"])
uninit = set(triage["uninitialized_output_source_read"]) | set(triage["uninitialized_output_pattern_only"])
fp = set(triage["false_positive"])
defects = oob | uninit

kernels = {}
for p in sorted(glob.glob(str(census_dir / "census-shard-*/results/raw/*.json"))):
    for k in json.loads(Path(p).read_text())["kernels"]:
        kernels[k["id"]] = k


def det_flags(run):
    d = run["detectors"]
    po = d.get("poison") or {}
    asan = d.get("asan") or {}
    plain = d.get("plain") or {}
    pres = (plain.get("result") or {})
    ref = pres.get("reference") or {}
    return {
        "poison": bool(po.get("flagged")),
        "asan": bool(asan.get("flagged")),
        "plain_mismatch": pres.get("status") == "ok" and ref.get("allclose_1e-2") is False,
        "seconds": sum((d.get(n) or {}).get("wall_seconds") or 0 for n in ("plain", "asan", "poison")),
        "poison_seconds": (po.get("wall_seconds") or 0) + (plain.get("wall_seconds") or 0),
        "n_cfg": pres.get("autotune_max_configs") or 0,
    }


rows = {}
for kid, k in kernels.items():
    runs = [r for r in k["runs"] if r.get("shape_variant", "native") == "native"]
    per = {r["autotune_index"]: det_flags(r) for r in runs}
    rows[kid] = per

summary = {"defects": {}, "strategies": {}, "cost": {}}
for kid in sorted(defects):
    per = rows.get(kid)
    if not per:
        summary["defects"][kid] = {"missing": True}
        continue
    kind = "oob" if kid in oob else "uninit"
    flagged = sorted(i for i, f in per.items() if f["poison"] or f["asan"])
    summary["defects"][kid] = {
        "kind": kind,
        "lint_L1": kid in l1,
        "configs_available": max(f["n_cfg"] for f in per.values()),
        "configs_audited": sorted(per),
        "flagged_configs": flagged,
        "cfg0_poison": per.get(0, {}).get("poison", False),
        "cfg0_asan": per.get(0, {}).get("asan", False),
        "cfg0_plain_mismatch": per.get(0, {}).get("plain_mismatch", False),
        "any_plain_mismatch": any(f["plain_mismatch"] for f in per.values()),
    }

D = summary["defects"]


def recall(pred, kinds=("oob", "uninit")):
    sel = [k for k, v in D.items() if not v.get("missing") and v["kind"] in kinds]
    hit = [k for k in sel if pred(D[k])]
    return {"hit": len(hit), "of": len(sel), "missed": sorted(set(sel) - set(hit))}


summary["strategies"] = {
    "plain_allclose_cfg0": recall(lambda v: v["cfg0_plain_mismatch"]),
    "plain_allclose_any_cfg": recall(lambda v: v["any_plain_mismatch"]),
    "poison_cfg0": recall(lambda v: v["cfg0_poison"]),
    "poison_asan_cfg0": recall(lambda v: v["cfg0_poison"] or v["cfg0_asan"]),
    "poison_asan_per_config": recall(lambda v: bool(v["flagged_configs"])),
    "uninit_poison_cfg0": recall(lambda v: v["cfg0_poison"], ("uninit",)),
    "uninit_poison_per_config": recall(lambda v: bool(v["flagged_configs"]), ("uninit",)),
    "uninit_lint_L1": recall(lambda v: v["lint_L1"], ("uninit",)),
    "uninit_lint_gated_per_config": recall(lambda v: v["lint_L1"] and bool(v["flagged_configs"]), ("uninit",)),
    "config_dependent": recall(lambda v: 0 < len(v["flagged_configs"]) < len(v["configs_audited"])),
}

# False positives of the dynamic check across the whole census.
flagged_any = [k for k, per in rows.items() if any(f["poison"] or f["asan"] for f in per.values())]
summary["dynamic_flagged_total"] = len(flagged_any)
summary["dynamic_flagged_true"] = len(set(flagged_any) & defects)
summary["dynamic_flagged_fp_triaged"] = sorted(set(flagged_any) & fp)
summary["dynamic_flagged_other"] = sorted(set(flagged_any) - defects - fp)

# Lint as a static gate over the best-round population.
n_best = lint["best"]
summary["lint"] = {
    "marked": len(l1),
    "population": n_best,
    "marked_fraction": round(len(l1) / n_best, 4),
    "uninit_in_marked": len(uninit & l1),
    "uninit_unmarked": len(uninit - l1),
    "precision_uninit": round(len(uninit & l1) / len(l1), 4),
    "rate_unmarked": round(len(uninit - l1) / (n_best - len(l1)), 5),
}

# Cost: seconds of a config-0 check against a per-config check.
auto = [per for per in rows.values() if max((f["n_cfg"] for f in per.values()), default=0) > 1]
c0 = [per[0]["poison_seconds"] for per in auto if 0 in per]
call = [sum(f["poison_seconds"] for f in per.values()) for per in auto]
ratio = [a / b for a, b in zip(call, c0) if b > 0]
summary["cost"] = {
    "autotuned_kernels": len(auto),
    "median_configs_audited": statistics.median(len(per) for per in auto) if auto else None,
    "median_cfg0_seconds": round(statistics.median(c0), 2) if c0 else None,
    "median_per_config_seconds": round(statistics.median(call), 2) if call else None,
    "median_overhead_factor": round(statistics.median(ratio), 2) if ratio else None,
    "total_cfg0_hours": round(sum(c0) / 3600, 2),
    "total_per_config_hours": round(sum(call) / 3600, 2),
    "lint_gated_per_config_hours": round(sum(sum(f["poison_seconds"] for f in rows[k].values()) for k in l1 if k in rows) / 3600, 2),
}

out.write_text(json.dumps(summary, indent=1))
for k, v in summary["strategies"].items():
    print(f"{k}: {v['hit']}/{v['of']} missed={v['missed']}")
print("lint", summary["lint"])
print("cost", summary["cost"])
print("dynamic flagged", summary["dynamic_flagged_total"], "true", summary["dynamic_flagged_true"], "fp", summary["dynamic_flagged_fp_triaged"], "other", summary["dynamic_flagged_other"][:20])
