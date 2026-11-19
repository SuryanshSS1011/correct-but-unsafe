"""Apply the per-kernel grid rewrites in tools/grid_fixes.json to a kernel dump.

Usage: python tools/apply_grid_fix.py IN_JSONL FIXES_JSON OUT_JSONL
Each fix replaces one exact text span on one line. A kernel whose spans do not
all match is left unchanged and reported, so a failed rewrite cannot pass as a
fixed kernel. Kernels without an entry are copied unchanged. Executes no code.
"""
import json
import sys

src, fixes_path, dst = sys.argv[1:4]
fixes = {k: v for k, v in json.load(open(fixes_path)).items() if not k.startswith("_")}
report = {"patched": [], "unpatched": [], "failed": []}
out = []
for line in open(src):
    if not line.strip():
        continue
    row = json.loads(line)
    kid = f"{row['uuid']}_r{row['round']}"
    if kid not in fixes:
        report["unpatched"].append(kid)
        out.append(row)
        continue
    lines = row["code"].split("\n")
    ok = True
    for lineno, old, new in fixes[kid]:
        if old not in lines[lineno - 1]:
            ok = False
            break
        lines[lineno - 1] = lines[lineno - 1].replace(old, new, 1)
    if ok:
        row = dict(row, code="\n".join(lines), grid_fix=True)
        report["patched"].append(kid)
    else:
        report["failed"].append(kid)
    out.append(row)
with open(dst, "w") as f:
    for row in out:
        f.write(json.dumps(row) + "\n")
print(json.dumps({k: len(v) for k, v in report.items()}), json.dumps(report))
