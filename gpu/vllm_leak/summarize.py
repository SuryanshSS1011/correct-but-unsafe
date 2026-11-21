"""Summarize a directory of run_leak.py results.

    python gpu/vllm_leak/summarize.py RESULTS_DIR [RESULTS_DIR ...]

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

The as_control column counts B's trials whose tokens equal those the correct
kernel gave in the same configuration, when that run is among the inputs. A
defect that leaves B's tokens as the control's but still varies the verdict
gives normal-looking text that depends on another request.
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


def alloc(c: dict) -> str:
    if c.get("no_caching"):
        return "nocache"
    return "expandable" if c["alloc_conf"] else "default"


def engine(c: dict) -> str:
    return c.get("engine", "vllm").split()[0]


def main() -> int:
    files = []
    for d in sys.argv[1:]:
        files += sorted(glob.glob(os.path.join(d, "*.json")))
    runs = []
    for f in files:
        with open(f) as fh:
            runs.append(json.load(fh))

    def cell(c: dict) -> tuple:
        return (
            engine(c),
            c["model"],
            c["device"],
            c["mode"],
            alloc(c),
            c["order"],
            c["record"],
            c.get("chat", False),
        )

    control = {cell(d["config"]): d for d in runs if d["config"]["kernel"] == "correct"}
    rows = []
    for d in runs:
        c = d["config"]
        ctl = control.get(cell(c))
        for ln in d["summary"]:
            trials = [t for t in d["trials"] if t["length"] == ln]
            # Batching moves the control's log-probabilities but never its tokens.
            basis = "tokens" if c["order"] == "batched" else "logprobs"
            v, distinct = verdict(trials, basis)
            rec = [t["record"] for t in trials if t.get("record")]
            same = ""
            if ctl:
                ref = {
                    tuple(t["token_ids"]) for t in ctl["trials"] if t["length"] == ln
                }
                if len(ref) == 1:
                    same = f"{sum(tuple(t['token_ids']) in ref for t in trials)}/{len(trials)}"
            rows.append(
                {
                    "engine": engine(c),
                    "model": c["model"].split("/")[-1][:12]
                    + ("+chat" if c.get("chat") else ""),
                    "device": c["device"].replace("NVIDIA ", "")[:9],
                    "kernel": c["kernel"],
                    "mode": c["mode"],
                    "alloc": alloc(c),
                    "order": c["order"] + ("+record" if c["record"] else ""),
                    "length": ln,
                    "basis": basis,
                    "verdict": v,
                    "distinct": distinct,
                    "as_control": same,
                    "bit_equal": sum(r["bit_equal_to_victim"] for r in rec)
                    if rec
                    else "",
                    "baseline": sum(r["bit_equal_shifted_baseline"] for r in rec)
                    if rec
                    else "",
                }
            )
    widths = {
        "engine": 6,
        "model": 17,
        "device": 9,
        "kernel": 9,
        "mode": 6,
        "alloc": 10,
        "order": 17,
        "length": 6,
        "basis": 8,
        "verdict": 20,
        "distinct": 8,
        "as_control": 10,
        "bit_equal": 9,
        "baseline": 8,
    }
    print(" ".join(f"{h:<{w}s}" for h, w in widths.items()))
    for r in rows:
        print(" ".join(f"{r[h]!s:<{w}s}" for h, w in widths.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
