"""Combine rejected-sample recheck shards and compute partial-write and OOB rates.
Usage: gh run download 43498603107 and 44431107642, then
python -I tools/rejected_rate.py results/rejected_sample.json <recheck_shard*.json from both runs>
A round is auditable when some config runs with plain status ok. A partial
write is a config whose poison run leaves output unwritten. An out-of-bounds
access is a config the bounds oracle flags."""
import json, sys
from math import sqrt

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (c - h) / d, 100 * (c + h) / d)

out, paths = sys.argv[1], sys.argv[2:]
rows, per_shard = {}, {}
for p in paths:
    d = json.load(open(p))
    per_shard[p] = len(d["kernels"])
    for k in d["kernels"]:
        cfgs = [c for m in k["modes"].values() for c in m.get("configs", [])]
        ok = any((c.get("plain") or {}).get("status") == "ok" for c in cfgs)
        partial = any(((c.get("poison") or {}).get("unwritten") or 0) > 0 for c in cfgs)
        oob = any((c.get("bounds") or {}).get("flagged") for c in cfgs)
        rows[k["id"]] = {"auditable": ok, "partial": partial and ok, "oob": oob and ok, "configs": len(cfgs), "shard": p}
aud = [r for r in rows.values() if r["auditable"]]
pw = [k for k, r in rows.items() if r["partial"]]
ob = [k for k, r in rows.items() if r["oob"]]
either = [k for k, r in rows.items() if r["partial"] or r["oob"]]
s = {"rounds": len(rows), "auditable": len(aud), "partial_write": len(pw), "oob": len(ob), "either": len(either),
     "partial_pct": round(100 * len(pw) / len(aud), 2), "partial_ci": [round(x, 2) for x in wilson(len(pw), len(aud))],
     "either_pct": round(100 * len(either) / len(aud), 2), "either_ci": [round(x, 2) for x in wilson(len(either), len(aud))],
     "per_shard_rounds": per_shard, "partial_ids": sorted(pw), "oob_ids": sorted(ob)}
json.dump(s, open(out, "w"), indent=1)
print({k: v for k, v in s.items() if not k.endswith("_ids")})
