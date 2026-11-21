"""Can B's output name the previous request's secret?

    python gpu/vllm_leak/analyze_infer.py infer_*.json

For each file, every trial is classified by leave-one-out nearest neighbour on
B's top-20 log-probabilities at the recorded positions, with each trial's
nearest other trial voting its secret. Accuracy is compared with chance, the share
of the other trials that carry the same secret, and with a permutation test that shuffles the secret
labels. Trials whose features are identical to many others count as a tie
broken at random, so a control whose output never varies scores at chance.
Standard library only.
"""

from __future__ import annotations

import json
import math
import random
import sys

FLOOR = -50.0  # log-probability given to a token missing from a trial's top-20


def features(trial: dict, positions: int) -> dict:
    f = {}
    for p, pos in enumerate(trial["top_logprobs"][:positions]):
        for tok, lp in pos:
            f[(p, tok)] = lp if not math.isnan(lp) else FLOOR
    return f


def dist(a: dict, b: dict) -> float:
    keys = a.keys() | b.keys()
    return sum(abs(a.get(k, FLOOR) - b.get(k, FLOOR)) for k in keys)


def loo_accuracy(feats: list, labels: list, rng: random.Random) -> float:
    hit = 0.0
    n = len(feats)
    for i in range(n):
        best, near = math.inf, []
        for j in range(n):
            if j == i:
                continue
            d = dist(feats[i], feats[j])
            if d < best - 1e-12:
                best, near = d, [j]
            elif abs(d - best) <= 1e-12:
                near.append(j)
        hit += sum(labels[j] == labels[i] for j in near) / len(near)
    return hit / n


def main() -> int:
    rng = random.Random(0)
    print(f"{'file':40s} {'kernel':9s} {'len':>4s} {'gap':>3s} {'trials':>6s} {'distinct':>8s} {'acc':>6s} {'chance':>6s} {'p':>7s}")
    for path in sys.argv[1:]:
        d = json.load(open(path))
        c = d["config"]
        trials = d["trials"]
        feats = [features(t, c["positions"]) for t in trials]
        labels = [t["secret"] for t in trials]
        acc = loo_accuracy(feats, labels, rng)
        distinct = len({json.dumps(sorted(f.items())) for f in feats})
        null = []
        for _ in range(200):
            perm = labels[:]
            rng.shuffle(perm)
            null.append(loo_accuracy(feats, perm, rng))
        p = (1 + sum(x >= acc for x in null)) / (1 + len(null))
        # Chance under leave-one-out: the share of the other trials that carry the same secret.
        chance = sum((labels.count(x) - 1) / (len(labels) - 1) for x in labels) / len(labels)
        print(f"{path[-40:]:40s} {c['kernel']:9s} {c['length']:>4d} {c['gap']:>3d} {len(trials):>6d} {distinct:>8d} {acc:>6.2f} {chance:>6.2f} {p:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
