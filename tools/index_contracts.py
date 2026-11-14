"""Static feasibility count for dropped safety contracts.

A PyTorch operator that takes an index rejects an index outside its range. A
Triton kernel that replaces it can do pointer arithmetic with the same index
and read or write outside the buffer. This script counts, without executing
anything, how many reference programs use an index-style operator and how many
candidate kernels compute an address from loaded data.

Usage: python tools/index_contracts.py --out-dir results/raw/index_contracts
       python tools/index_contracts.py --jsonl FILE   (local file with the corpus fields)
"""
from __future__ import annotations

import argparse
import ast
import collections
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pilot" / "drkernel"))
import extract  # noqa: E402

# Operators whose index argument PyTorch checks against the indexed dimension.
INDEX_OPS = {
    "embedding": r"\bnn\.Embedding\b|\bF\.embedding\b|\bembedding_bag\b|\bnn\.EmbeddingBag\b",
    "gather": r"\.gather\(|\btorch\.gather\(|\btake_along_dim\(",
    "index_select": r"\bindex_select\(",
    "take": r"\btorch\.take\(|\.take\(",
    "scatter": r"\.scatter_?\(|\btorch\.scatter\(|\.scatter_add_?\(|\bscatter_reduce",
    "index_put": r"\.index_put_?\(|\.index_add_?\(|\.index_copy_?\(|\.index_fill_?\(",
    "nll_or_ce": r"\bnll_loss\(|\bcross_entropy\(|\bnn\.NLLLoss\b|\bnn\.CrossEntropyLoss\b",
    "one_hot": r"\bone_hot\(",
    "bincount": r"\bbincount\(",
    "advanced_index": r"\[[^\]\n]*\.long\(\)[^\]\n]*\]",
}
WRITE_OPS = {"scatter", "index_put"}
INT_INPUT = re.compile(r"torch\.randint\(|\.long\(\)|dtype\s*=\s*torch\.(int64|long|int32|int)\b|torch\.arange\(|torch\.randperm\(")


def ref_features(ref: str) -> dict:
    ops = sorted(k for k, pat in INDEX_OPS.items() if re.search(pat, ref))
    m = re.search(r"def get_inputs\(\):(.*?)(?:\ndef |\Z)", ref, re.S)
    return {"index_ops": ops, "int_inputs": bool(m and INT_INPUT.search(m.group(1)))}


def kernel_features(code: str) -> dict:
    """Does a jit kernel compute a load or store address from a value it loaded?"""
    out = {"parse_ok": True, "kernels": 0, "data_dep_load": False, "data_dep_store": False}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        out["parse_ok"] = False
        return out
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or not any("jit" in ast.unparse(d) for d in fn.decorator_list):
            continue
        out["kernels"] += 1
        tainted: set = set()
        assigns = [n for n in ast.walk(fn) if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and getattr(n, "value", None) is not None]

        def is_load(e):
            return isinstance(e, ast.Call) and ast.unparse(e.func).split(".")[-1] == "load"

        def mentions(e, names):
            return any(isinstance(m, ast.Name) and m.id in names for m in ast.walk(e))

        for _ in range(4):
            for n in sorted(assigns, key=lambda x: x.lineno):
                src = n.value
                if any(is_load(c) for c in ast.walk(src)) or mentions(src, tainted):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    for t in targets:
                        for m in ast.walk(t):
                            if isinstance(m, ast.Name):
                                tainted.add(m.id)
        for c in ast.walk(fn):
            if not isinstance(c, ast.Call) or not c.args:
                continue
            name = ast.unparse(c.func).split(".")[-1]
            ptr = c.args[0]
            nested = any(is_load(x) for x in ast.walk(ptr))
            if name == "load" and (mentions(ptr, tainted) or nested):
                out["data_dep_load"] = True
            if (name == "store" or name.startswith("atomic_")) and (mentions(ptr, tainted) or nested):
                out["data_dep_store"] = True
    return out


def rows_from(args):
    if args.jsonl:
        op = gzip.open if args.jsonl.endswith(".gz") else open
        for line in op(args.jsonl, "rt"):
            row = json.loads(line)
            if isinstance(row.get("messages"), str):
                row["messages"] = ast.literal_eval(row["messages"])
            yield row
    else:
        from datasets import load_dataset

        yield from load_dataset("hkust-nlp/drkernel-coldstart-8k", split="train")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/raw/index_contracts")
    ap.add_argument("--jsonl", default=None)
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    S = collections.Counter()
    by_op = collections.Counter()
    best_candidates, write_candidates, accepted_round_candidates = [], [], []
    for row in rows_from(a):
        S["tasks"] += 1
        rf = ref_features(row["original_python_code"])
        if rf["index_ops"]:
            S["tasks_with_index_op"] += 1
            S["tasks_with_index_op_and_int_inputs"] += rf["int_inputs"]
            for o in rf["index_ops"]:
                by_op[o] += 1
        best = int(row["best_round"]) if str(row.get("best_round", "")).isdigit() else None
        for x in extract.rounds(row["messages"]):
            fb = x["feedback"]
            if not fb:
                continue
            accepted = bool(fb.get("compiled") and fb.get("correctness") and fb.get("decoy_kernel") is False)
            if not accepted:
                continue
            S["accepted_rounds"] += 1
            kf = kernel_features(x["code"] or "")
            dep = kf["data_dep_load"] or kf["data_dep_store"]
            S["accepted_rounds_data_dep_address"] += dep
            is_best = best == x["round"]
            S["best_round_kernels"] += is_best
            if is_best:
                S["best_data_dep_address"] += dep
                S["best_data_dep_store"] += kf["data_dep_store"]
            if rf["index_ops"] and dep:
                S["accepted_rounds_index_op_and_data_dep"] += 1
                accepted_round_candidates.append(f"{int(row['uuid'])}:{x['round']}")
                if is_best:
                    S["best_index_op_and_data_dep"] += 1
                    best_candidates.append({"id": f"{int(row['uuid'])}:{x['round']}", "ops": rf["index_ops"], "int_inputs": rf["int_inputs"],
                                            "store": kf["data_dep_store"]})
                    if kf["data_dep_store"] or (set(rf["index_ops"]) & WRITE_OPS):
                        write_candidates.append(f"{int(row['uuid'])}:{x['round']}")
    res = {"counts": dict(S), "tasks_by_index_op": dict(by_op), "best_candidates": best_candidates,
           "write_candidates": sorted(set(write_candidates)), "accepted_round_candidates": sorted(set(accepted_round_candidates))}
    (out_dir / "index_contracts.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["counts"], indent=1))
    print("tasks by index operator:", dict(by_op))
    print("best-round candidates:", len(best_candidates), "| with a store or a write operator:", len(res["write_candidates"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
