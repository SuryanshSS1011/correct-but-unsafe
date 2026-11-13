"""Print a Markdown summary of one or more positive-control result files."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def op_row(entry: dict, detectors: list[str]) -> dict:
    row = {}
    for d in detectors:
        flagged, statuses = False, set()
        for case in entry.get("cases", []):
            r = case["detectors"].get(d, {})
            if r.get("flagged"):
                flagged = True
            res = r.get("result") or {}
            if res.get("status"):
                statuses.add(res["status"])
            if d == "plain" and r.get("returncode") not in (0, None):
                statuses.add(f"rc{r.get('returncode')}")
        row[d] = (flagged, statuses)
    return row


def main(paths: list[str]) -> None:
    for p in paths:
        data = json.loads(Path(p).read_text())
        dets = list(data["detectors"])
        print(f"### {data['label']} ({len(data['ops'])} ops, dataset {data['dataset']})\n")
        print("The plain column flags a fatal memory signal under the interpreter.\n")
        print("| op | known bug | plain status | " + " | ".join(dets) + " | any |")
        print("|---|---|---|" + "---|" * (len(dets) + 1))
        counts = {d: 0 for d in dets}
        any_known = any_other = n_known = n_other = 0
        for op in sorted(data["ops"]):
            e = data["ops"][op]
            row = op_row(e, data["detectors"])
            plain = ",".join(sorted(row.get("plain", (False, set()))[1])) or "-"
            cells, anyf = [], False
            for d in dets:
                f = row[d][0]
                anyf = anyf or f
                counts[d] += int(f and e.get("known_bug", False))
                cells.append("FLAG" if f else ".")
            if e.get("known_bug"):
                n_known += 1
                any_known += int(anyf)
            else:
                n_other += 1
                any_other += int(anyf)
            if e.get("known_bug") or anyf:
                print(f"| {op} | {'yes' if e.get('known_bug') else 'no'} | {plain} | " + " | ".join(cells) + f" | {'FLAG' if anyf else '.'} |")
        print()
        print(f"Known-bug ops flagged by any detector: {any_known} of {n_known}.")
        for d in dets:
            print(f"Known-bug ops flagged by {d}: {counts[d]} of {n_known}.")
        if n_other:
            print(f"Other ops flagged by any detector: {any_other} of {n_other}.")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
