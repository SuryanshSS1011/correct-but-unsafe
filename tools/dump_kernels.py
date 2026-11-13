"""Write the reference and candidate source of chosen Dr. Kernel kernels, for manual triage.

Usage: python tools/dump_kernels.py --kernels 1918:2,4636:4 --out data/dump.jsonl
Each item is uuid:round.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pilot" / "drkernel"))
import extract  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want = {}
    for item in a.kernels.split(","):
        if item.strip():
            u, r = item.split(":")
            want.setdefault(int(u), set()).add(int(r))

    from datasets import load_dataset

    ds = load_dataset("hkust-nlp/drkernel-coldstart-8k", split="train")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for row in ds:
            u = int(row["uuid"])
            if u not in want:
                continue
            rs = extract.rounds(row["messages"])
            for r in sorted(want[u]):
                x = rs[r - 1]
                f.write(json.dumps({"uuid": u, "round": r, "best_round": row["best_round"], "final_speedup": row["final_speedup"], "entry_point": row["entry_point"], "ref_code": row["original_python_code"], "code": x["code"], "feedback": x["feedback"], "analysis": x["analysis"], "all_feedback": [y["feedback"] for y in rs]}) + "\n")
                n += 1
    print("wrote", n, "kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
