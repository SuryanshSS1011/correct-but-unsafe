"""Static lint over every round of the Dr. Kernel corpus, joined with the recorded harness feedback.

The main rule, L1, marks an autotuned kernel that is launched with a grid
that is not a function of the tuned parameters. Under such a launch a config
with a smaller block covers only part of the data. The script also tests two
predictions against the corpus's own GPU records.

1. If the tuning pass hides a partial write on the first call, rejected L1
   rounds should pass exactly one of five correctness trials unusually often.
2. If identical code is accepted in one evaluation and rejected in another,
   acceptance depends on something other than the code.
"""
from __future__ import annotations

import argparse
import ast
import collections
import gzip
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pilot" / "drkernel"))
import extract  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NON_SHAPE = {"num_warps", "num_stages", "num_ctas", "maxnreg"}


def features(code: str) -> dict:
    f: dict = {"parse_ok": True, "autotune": False, "n_configs": 0, "tunables": {}, "launch_static": 0, "launch_callable": 0,
               "L1": False, "grid_src": None, "empty": 0, "training_guard": False, "is_cuda_guard": False, "boolop_in_kernel": False}
    if not code:
        f["parse_ok"] = False
        return f
    f["empty"] = len(re.findall(r"\b(?:torch\.empty|empty_like|new_empty)\b", code))
    f["training_guard"] = bool(re.search(r"self\.training", code))
    f["is_cuda_guard"] = bool(re.search(r"\.is_cuda\b", code))
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        f["parse_ok"] = False
        return f
    autotuned: dict = {}
    kernels: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        decs = [ast.unparse(d) for d in node.decorator_list]
        if any("jit" in d for d in decs):
            kernels.add(node.name)
            if any(isinstance(n, ast.BoolOp) for n in ast.walk(node)):
                f["boolop_in_kernel"] = True
        for dec in node.decorator_list:
            if "autotune" not in ast.unparse(dec):
                continue
            cfgs = [c for c in ast.walk(dec) if isinstance(c, ast.Call) and ast.unparse(c.func).endswith("Config")]
            tun: dict = {}
            for c in cfgs:
                if c.args and isinstance(c.args[0], ast.Dict):
                    for k, v in zip(c.args[0].keys, c.args[0].values):
                        if isinstance(k, ast.Constant):
                            tun.setdefault(str(k.value), set()).add(ast.unparse(v))
            autotuned[node.name] = {"n": len(cfgs), "tun": {k: sorted(v) for k, v in tun.items()}}
            kernels.add(node.name)
    if autotuned:
        f["autotune"] = True
        f["n_configs"] = max(v["n"] for v in autotuned.values())
        f["tunables"] = {k: v for a in autotuned.values() for k, v in a["tun"].items()}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        assigns = {n.targets[0].id: n.value for n in ast.walk(fn) if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
        for n in ast.walk(fn):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Name) and n.func.value.id in kernels):
                continue
            g = n.func.slice
            if isinstance(g, ast.Name) and g.id in assigns:
                g = assigns[g.id]
            callable_grid = isinstance(g, (ast.Lambda, ast.Name, ast.Attribute))
            f["launch_callable" if callable_grid else "launch_static"] += 1
            name = n.func.value.id
            if name in autotuned and not callable_grid:
                shape_tun = {k: v for k, v in autotuned[name]["tun"].items() if k not in NON_SHAPE and len(v) > 1}
                if shape_tun:
                    f["L1"] = True
                    f["grid_src"] = ast.unparse(g)[:120]
    return f


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k} of {n}, {100 * p:.1f}% ({100 * (c - h):.1f} to {100 * (c + h):.1f})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/raw/lint")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from datasets import load_dataset

    ds = load_dataset("hkust-nlp/drkernel-coldstart-8k", split="train")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with gzip.open(out_dir / "lint_rounds.jsonl.gz", "wt") as fo:
        for i, row in enumerate(ds):
            if a.limit and i >= a.limit:
                break
            ref_sha = hashlib.sha1(row["original_python_code"].encode()).hexdigest()[:12]
            for x in extract.rounds(row["messages"]):
                code = x["code"] or ""
                fb = x["feedback"] or {}
                f = features(code)
                norm = "\n".join(line.rstrip() for line in code.strip().splitlines())
                rec = {"uuid": int(row["uuid"]), "round": x["round"], "best_round": row["best_round"], "ref": ref_sha,
                       "code": hashlib.sha1(norm.encode()).hexdigest()[:12], "has_feedback": bool(x["feedback"]),
                       "compiled": fb.get("compiled"), "correct": fb.get("correctness"), "decoy": fb.get("decoy_kernel"),
                       "speedup": fb.get("speedup"), "trials": fb.get("correct_trials"), "status": fb.get("status"), **f}
                rec["accepted"] = bool(rec["compiled"] and rec["correct"] and rec["decoy"] is False)
                rows.append(rec)
                fo.write(json.dumps(rec) + "\n")

    fb_rows = [r for r in rows if r["has_feedback"]]
    S: dict = {"rounds": len(rows), "rounds_with_feedback": len(fb_rows), "tasks": len({r["ref"] for r in rows})}
    L = []

    def rate(sel, label):
        acc = sum(1 for r in sel if r["accepted"])
        L.append(f"| {label} | {len(sel)} | {wilson(acc, len(sel))} |")
        return {"n": len(sel), "accepted": acc}

    L.append(f"## Static lint over {len(rows)} rounds, {len(fb_rows)} with feedback, {S['tasks']} distinct tasks\n")
    L.append("| Rounds with feedback | Count | Accepted |\n|---|---|---|")
    S["all"] = rate(fb_rows, "All")
    S["autotune"] = rate([r for r in fb_rows if r["autotune"]], "Uses triton.autotune")
    S["L1"] = rate([r for r in fb_rows if r["L1"]], "L1: autotuned kernel with a fixed grid")
    S["autotune_not_L1"] = rate([r for r in fb_rows if r["autotune"] and not r["L1"]], "Autotuned, grid is a function of the config")
    S["no_autotune"] = rate([r for r in fb_rows if not r["autotune"]], "No autotune")

    # Prediction 1: the number of passed trials among rounds that compiled and failed correctness.
    L.append("\n### Passed trials among rounds that compiled, ran every trial and failed correctness\n")
    L.append("| Group | Rounds | Passed exactly 1 of 5 | Passed 0 of 5 | Passed 2 to 4 |\n|---|---|---|---|---|")
    S["trials"] = {}
    for label, sel in (("L1", [r for r in fb_rows if r["L1"]]), ("Autotuned, not L1", [r for r in fb_rows if r["autotune"] and not r["L1"]]), ("No autotune", [r for r in fb_rows if not r["autotune"]])):
        bad = [r for r in sel if r["compiled"] and r["correct"] is False and r["trials"] and r["trials"][1] == 5]
        c = collections.Counter(r["trials"][0] for r in bad)
        mid = sum(v for k, v in c.items() if 2 <= k <= 4)
        L.append(f"| {label} | {len(bad)} | {wilson(c.get(1, 0), len(bad))} | {c.get(0, 0)} | {mid} |")
        S["trials"][label] = {"n": len(bad), "dist": {str(k): v for k, v in sorted(c.items())}}

    # Prediction 2: identical code evaluated more than once.
    groups: dict = collections.defaultdict(list)
    for r in fb_rows:
        if r["compiled"]:
            groups[(r["ref"], r["code"])].append(r)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    L.append("\n### Identical code for the same task evaluated more than once\n")
    L.append("| Group | Duplicate groups | Groups with both a pass and a fail |\n|---|---|---|")
    S["dups"] = {}
    for label, pred in (("L1", lambda r: r["L1"]), ("Autotuned, not L1", lambda r: r["autotune"] and not r["L1"]), ("No autotune", lambda r: not r["autotune"])):
        g = [v for v in multi.values() if pred(v[0])]
        mixed = [v for v in g if len({bool(r["correct"]) for r in v}) > 1]
        L.append(f"| {label} | {len(g)} | {wilson(len(mixed), len(g))} |")
        S["dups"][label] = {"groups": len(g), "mixed": len(mixed), "examples": [[f"{r['uuid']}:{r['round']}:{r['trials']}" for r in v] for v in mixed[:12]]}

    # Which mode did the harness run in? A kernel guarded by self.training never launches in training mode.
    L.append("\n### Decoy rate by training guard, among rounds that compiled\n")
    L.append("| Group | Rounds | Recorded as decoy |\n|---|---|---|")
    S["decoy"] = {}
    for label, sel in (("Mentions self.training", [r for r in fb_rows if r["compiled"] and r["training_guard"]]), ("No mention", [r for r in fb_rows if r["compiled"] and not r["training_guard"]])):
        d = sum(1 for r in sel if r["decoy"])
        L.append(f"| {label} | {len(sel)} | {wilson(d, len(sel))} |")
        S["decoy"][label] = {"n": len(sel), "decoy": d}

    # Accepted best-round population and the census triage.
    best = [r for r in rows if r["has_feedback"] and r["accepted"] and r["round"] == r["best_round"]]
    tri = json.loads((REPO / "results" / "triage-census.json").read_text())
    un = set(tri["uninitialized_output_source_read"]) | set(tri["uninitialized_output_pattern_only"])
    oob = set(tri["out_of_bounds_read"])
    byid = {f"{r['uuid']}_r{r['round']}": r for r in rows}
    L.append(f"\n### Accepted best-round population: {len(best)} kernels\n")
    for key in ("autotune", "L1", "training_guard", "is_cuda_guard", "boolop_in_kernel"):
        k = sum(1 for r in best if r[key])
        L.append(f"- {key}: {wilson(k, len(best))}")
        S[f"best_{key}"] = k
    S["best"] = len(best)
    hit = [k for k in un if k in byid and byid[k]["L1"]]
    L.append(f"- Census uninitialized-output kernels that L1 marks: {len(hit)} of {len(un)}. Not marked: {sorted(k for k in un if k not in hit)}")
    L.append(f"- Census out-of-bounds kernels that use autotune: {sorted(k for k in oob if k in byid and byid[k]['autotune'])}")
    l1_best = sorted(f"{r['uuid']}_r{r['round']}" for r in best if r["L1"])
    L.append(f"- Accepted best-round kernels marked by L1: {len(l1_best)}. Of these the census flagged {len([k for k in l1_best if k in un])}.")
    S["l1_best_ids"] = l1_best
    S["l1_hit"] = sorted(hit)
    S["grid_examples"] = [r["grid_src"] for r in best if r["L1"]][:15]
    # every accepted round marked by L1, for a targeted dynamic recheck
    S["l1_accepted_rounds"] = sorted({f"{r['uuid']}:{r['round']}" for r in fb_rows if r["accepted"] and r["L1"]})
    S["l1_rejected_1of5"] = sorted({f"{r['uuid']}:{r['round']}" for r in fb_rows if r["L1"] and r["correct"] is False and r["trials"] == [1, 5]})[:400]
    L.append(f"- Accepted rounds of any round index marked by L1: {len(S['l1_accepted_rounds'])}")
    print("\n".join(L))
    (out_dir / "lint_summary.json").write_text(json.dumps(S, indent=1))
    (out_dir / "lint_summary.md").write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
