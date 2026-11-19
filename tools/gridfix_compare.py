"""Compare the grid-fix run against the unpatched control.
Usage: gh run download 44474680089 -D fixed; gh run download 44474692558 -D control
python -I tools/gridfix_compare.py tools/grid_fixes.json results/grid_fix.json fixed control"""
import glob, json, sys

fixes_path, out, fdir, cdir = sys.argv[1:5]
patched = {k for k in json.load(open(fixes_path)) if not k.startswith("_")}

def load(d):
    rows = {}
    for p in glob.glob(f"{d}/*/recheck_shard*.json"):
        for k in json.load(open(p))["kernels"]:
            per = []
            for mode, m in k["modes"].items():
                for c in m.get("configs", []):
                    pl = c.get("plain") or {}
                    po = c.get("poison") or {}
                    per.append({"mode": mode, "cfg": c["index"], "status": pl.get("status"), "allclose": pl.get("allclose"),
                                "unwritten": po.get("unwritten"), "bounds": (c.get("bounds") or {}).get("flagged"),
                                "skipped": m.get("skipped_for_budget", False)})
            rows[k["id"]] = per
    return rows

F, C = load(fdir), load(cdir)
res = {}
for kid in sorted(set(F) | set(C)):
    def summ(per):
        ok = [x for x in per if x["status"] == "ok"]
        return {"configs": len(per), "ran": len(ok),
                "partial": sorted({(x["mode"], x["cfg"]) for x in ok if (x["unwritten"] or 0) > 0}),
                "mismatch": sorted({(x["mode"], x["cfg"]) for x in ok if x["allclose"] is False}),
                "bounds": sorted({(x["mode"], x["cfg"]) for x in per if x["bounds"]}),
                "not_ok": sorted({(x["mode"], x["cfg"], x["status"]) for x in per if x["status"] != "ok"}),
                "skipped": any(x["skipped"] for x in per)}
    f, c = summ(F.get(kid, [])), summ(C.get(kid, []))
    clean = f["ran"] > 0 and f["ran"] == f["configs"] and not f["partial"] and not f["mismatch"] and not f["bounds"]
    res[kid] = {"patched": kid in patched, "fixed": f, "control": c, "clean_after_fix": clean}
json.dump(res, open(out, "w"), indent=1, default=list)
for kid, r in res.items():
    f, c = r["fixed"], r["control"]
    print(f"{kid:8s} {'P' if r['patched'] else '-'} control partial {len(c['partial'])}/{c['ran']} mism {len(c['mismatch'])} | fixed partial {len(f['partial'])}/{f['ran']} of {f['configs']} mism {len(f['mismatch'])} notok {f['not_ok'][:2]} skip {f['skipped']} -> {'CLEAN' if r['clean_after_fix'] else ''}")
