"""Sample a second corpus of generated Triton kernels for the CPU audit.

Two public corpora are supported. The script applies the static lint and
writes a sample in the JSONL layout that pilot/drkernel/recheck.py reads. It
executes no corpus code.

makora  makora-ai/triton-gpu-latency. Each row is one program that defines a
        PyTorch reference `Model` and a Triton `ModelNew`, with the measured
        GPU runtime of `ModelNew` when it ran. There are no correctness
        labels, so the audit computes correctness itself.
traces  ppbhatt500/kernelbook-triton-multiturn-reasoning-traces. Each row has
        a PyTorch reference, the model's final Triton code and a stop reason.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lint_rounds import features, wilson  # noqa: E402

NAMES = {"makora": "makora-ai/triton-gpu-latency", "traces": "ppbhatt500/kernelbook-triton-multiturn-reasoning-traces"}
CODE_COLUMNS = ("triton_code", "final_triton_code", "final_code", "generated_code", "code", "answer", "solution", "response")


def pick_split(dd):
    name = "train" if "train" in dd else list(dd.keys())[0]
    return name, dd[name]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(NAMES), required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--lint-n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out-dir", default="data/corpus2")
    a = ap.parse_args()
    from datasets import load_dataset

    # Only the train split is read. The Makora test split has a different schema and fails to load with it.
    split, ds = "train", load_dataset(NAMES[a.corpus], split="train")
    n = len(ds)
    print(f"{NAMES[a.corpus]} split {split}: {n} rows, columns {ds.column_names}")
    first = ds[0]
    print({k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) for k, v in first.items()})
    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if a.corpus == "makora":
        def view(row):
            return row["x"] or "", row["x"] or "", row["y"] is not None, {"y": row["y"]}, None
    else:
        probe = [ds[i] for i in range(min(n, 200))]
        strings = [c for c in ds.column_names if isinstance(first[c], str) and c != "pytorch_code"]
        code_col = next((c for c in CODE_COLUMNS if c in strings), None)
        if code_col is None:
            code_col = max(strings, key=lambda c: sum("triton.jit" in (r[c] or "") for r in probe))
        entry_col = next((c for c in ("entry_point", "module_name", "class_name") if c in ds.column_names), None)
        print(f"candidate code column: {code_col}; entry column: {entry_col}")
        print("stop reasons:", collections.Counter(r.get("stop_reason") for r in probe).most_common(8))

        def view(row):
            ok = str(row.get("stop_reason") or "").startswith("success")
            fb = {k: row.get(k) for k in ("stop_reason", "num_turns", "source", "origin", "dataset") if k in row}
            return row.get("pytorch_code") or "", row[code_col] or "", ok, fb, (row.get(entry_col) if entry_col else None)

    lint_idx = sorted(rng.sample(range(n), min(a.lint_n, n)))
    c: collections.Counter = collections.Counter()
    marked = []
    for i in lint_idx:
        ref, cand, ok, _, _ = view(ds[i])
        f = features(cand)
        c["rows"] += 1
        c["usable"] += ok
        c["parse_ok"] += f["parse_ok"]
        c["has_get_inputs"] += "def get_inputs" in ref
        c["autotune"] += f["autotune"]
        c["L1"] += f["L1"]
        c["empty"] += bool(f["empty"])
        c["usable_autotune"] += ok and f["autotune"]
        c["usable_L1"] += ok and f["L1"]
        if ok and f["L1"]:
            marked.append(i)
    label = "a measured runtime" if a.corpus == "makora" else "a success stop reason"
    lines = [f"## Static lint on {c['rows']} of {n} programs from {NAMES[a.corpus]}\n",
             f"- With {label}: {wilson(c['usable'], c['rows'])}",
             f"- Candidate parses as Python: {wilson(c['parse_ok'], c['rows'])}",
             f"- Reference defines get_inputs: {wilson(c['has_get_inputs'], c['rows'])}",
             f"- Use triton.autotune: {wilson(c['autotune'], c['rows'])}",
             f"- L1, an autotuned kernel with a fixed grid: {wilson(c['L1'], c['rows'])}",
             f"- L1 among programs with {label}: {wilson(c['usable_L1'], c['usable'])}",
             f"- Allocate with torch.empty or a relative: {wilson(c['empty'], c['rows'])}"]
    print("\n".join(lines))

    pool = rng.sample(range(n), min(n, a.n * 5))
    uniform = [i for i in pool if view(ds[i])[2]][: a.n]
    chosen = set(uniform)
    extra = [i for i in marked if i not in chosen][:400]  # the rare class, so it is represented
    with open(out / "sample.jsonl", "w") as fo:
        for stratum, idxs in (("uniform", uniform), ("l1", extra)):
            for i in idxs:
                ref, cand, _, fb, entry = view(ds[i])
                rec = {"uuid": f"{a.corpus[:2]}{i}", "round": 0, "ref_code": ref, "code": cand, "feedback": dict(fb, stratum=stratum)}
                if entry:
                    rec["ref_class"], rec["cand_class"] = entry, f"{entry}New"
                fo.write(json.dumps(rec) + "\n")
    summary = {"dataset": NAMES[a.corpus], "rows": n, "lint": dict(c), "sample_uniform": len(uniform), "sample_l1_extra": len(extra)}
    (out / "lint.json").write_text(json.dumps(summary, indent=1))
    (out / "lint.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
