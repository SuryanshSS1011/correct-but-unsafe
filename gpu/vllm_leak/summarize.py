"""Summarize a directory of run_leak.py results.

    python gpu/vllm_leak/summarize.py RESULTS_DIR

For sequential runs, B's output in a trial is its top-5 log-probabilities at
every position before the first NaN, which catches a leak that moves the
probabilities but not the tokens. For batched runs it is B's tokens, because
batching changes the control's log-probabilities slightly but never its tokens. For each configuration and victim
length the verdict is one of:

    determined by victim   repeats with the same victim give the same output,
                           and no output after V1 equals one after V2
    varies with history    B's output varies, but not as a function of the last
                           victim alone, so earlier requests matter too
    no variation           B's output is the same in every trial
    NaN                    B's log-probabilities are NaN, so the output carries
                           no information either way
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys


def signature(trial: dict, basis: str) -> tuple:
    """B's output in a trial: its tokens, or its top-5 log-probabilities up to the first NaN."""
    if basis == "tokens":
        return tuple(trial["token_ids"])
    sig = []
    for pos in trial["top_logprobs"]:
        if any(math.isnan(lp) for _, lp in pos):
            break
        sig.append(tuple((tok, round(lp, 6)) for tok, lp in pos))
    return tuple(sig)


def verdict(trials: list[dict], basis: str) -> tuple[str, int]:
    by: dict[str, list[tuple]] = {}
    for t in trials:
        by.setdefault(t["victim"], []).append(signature(t, basis))
    if basis == "logprobs" and all(s == () for sigs in by.values() for s in sigs):
        return "NaN", 1
    distinct = len({s for sigs in by.values() for s in sigs})
    if distinct == 1:
        return "no variation", 1
    agree = all(len(set(sigs)) == 1 for sigs in by.values())
    sets = [set(sigs) for sigs in by.values()]
    disjoint = all(a.isdisjoint(b) for i, a in enumerate(sets) for b in sets[i + 1 :])
    return (
        "determined by victim" if agree and disjoint else "varies with history"
    ), distinct


def main() -> int:
    rows = []
    for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        c = d["config"]
        for ln in d["summary"]:
            trials = [t for t in d["trials"] if t["length"] == ln]
            # Batching moves the control's log-probabilities but never its tokens.
            basis = "tokens" if c["order"] == "batched" else "logprobs"
            v, distinct = verdict(trials, basis)
            rec = [t["record"] for t in trials if t.get("record")]
            rows.append(
                {
                    "kernel": c["kernel"],
                    "mode": c["mode"],
                    "alloc": "expandable" if c["alloc_conf"] else "default",
                    "order": c["order"] + ("+record" if c["record"] else ""),
                    "length": ln,
                    "basis": basis,
                    "verdict": v,
                    "distinct": distinct,
                    "bit_equal": sum(r["bit_equal_to_victim"] for r in rec)
                    if rec
                    else "",
                    "baseline": sum(r["bit_equal_shifted_baseline"] for r in rec)
                    if rec
                    else "",
                }
            )
    widths = {
        "kernel": 9,
        "mode": 6,
        "alloc": 10,
        "order": 17,
        "length": 6,
        "basis": 8,
        "verdict": 20,
        "distinct": 8,
        "bit_equal": 9,
        "baseline": 8,
    }
    print(" ".join(f"{h:<{w}s}" for h, w in widths.items()))
    for r in rows:
        print(" ".join(f"{r[h]!s:<{w}s}" for h, w in widths.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
