"""Static scan of checked-out repositories for rule L1 in hand-written Triton code.

Rule L1 marks a kernel that is autotuned over a size-like parameter and is
launched with a grid that cannot depend on the chosen config. Such a kernel
writes a different share of its output under different configs unless the
tuned parameter is consumed by a loop inside the kernel, so every mark needs a
manual read and a dynamic check. The script parses source files and executes
none of them.

Usage: python -I tools/scan_repos.py --root DIR_WITH_ONE_SUBDIR_PER_REPO --out results/wild_scan.json
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

NON_SHAPE = {"num_warps", "num_stages", "num_ctas", "maxnreg"}


def _name(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _config_calls(node):
    return [c for c in ast.walk(node) if isinstance(c, ast.Call) and (ast.unparse(c.func).endswith("Config"))]


def _tunables_from(node, scope_assigns, scope_funcs, depth=0):
    """Return ({key: set(value sources)}, n_config_sites, resolved) for a configs expression."""
    tun: dict = {}
    sites = 0
    resolved = False
    for c in _config_calls(node):
        sites += 1
        resolved = True
        if c.args and isinstance(c.args[0], ast.Dict):
            for k, v in zip(c.args[0].keys, c.args[0].values):
                if isinstance(k, ast.Constant):
                    tun.setdefault(str(k.value), set()).add(ast.unparse(v))
        elif c.args and isinstance(c.args[0], (ast.Name, ast.Call, ast.DictComp)):
            tun.setdefault("<dynamic dict>", set()).update({"a", "b"})
    # A dict value that is a comprehension variable takes several values even though it has one source form.
    for comp in [n for n in ast.walk(node) if isinstance(n, (ast.ListComp, ast.GeneratorExp))]:
        targets = set()
        for gen in comp.generators:
            for t in ast.walk(gen.target):
                if isinstance(t, ast.Name):
                    targets.add(t.id)
        for c in _config_calls(comp):
            if c.args and isinstance(c.args[0], ast.Dict):
                for k, v in zip(c.args[0].keys, c.args[0].values):
                    if isinstance(k, ast.Constant) and any(isinstance(n, ast.Name) and n.id in targets for n in ast.walk(v)):
                        tun.setdefault(str(k.value), set()).add("<comprehension>")
    if depth < 3:
        for n in ast.walk(node):
            ref = None
            if isinstance(n, ast.Name):
                ref = scope_assigns.get(n.id)
            elif isinstance(n, ast.Call):
                ref = scope_funcs.get(_name(n.func) or "")
            if ref is not None and ref is not node:
                t2, s2, r2 = _tunables_from(ref, scope_assigns, scope_funcs, depth + 1)
                for k, v in t2.items():
                    tun.setdefault(k, set()).update(v)
                sites += s2
                resolved = resolved or r2
    return tun, sites, resolved


def _kernel_usage(fn: ast.FunctionDef, tunable: str) -> dict:
    """How the kernel body uses a tuned parameter."""
    pid_vars = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func).endswith("program_id"):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    pid_vars.add(t.id)
    # one level of propagation: x = pid * something, row = pid // n
    for _ in range(2):
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and any(isinstance(m, ast.Name) and m.id in pid_vars for m in ast.walk(n.value)):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id != tunable:
                        pid_vars.add(t.id)
    # Names computed from the tuned parameter, so that `n = cdiv(T, BT)` followed by `for i in range(n)` counts as a loop over it.
    derived = {tunable}
    for _ in range(3):
        for n in ast.walk(fn):
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None and any(isinstance(m, ast.Name) and m.id in derived for m in ast.walk(n.value)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Name):
                        derived.add(t.id)
    use = {"pid_scaled": False, "loop_step": False, "loop_derived": False, "arange": False, "mentions": 0}
    for n in ast.walk(fn):
        if isinstance(n, ast.For) and isinstance(n.iter, ast.Call) and ast.unparse(n.iter.func).split(".")[-1] in {"range", "static_range"}:
            if any(isinstance(m, ast.Name) and m.id in derived for a in n.iter.args for m in ast.walk(a)):
                use["loop_derived"] = True
        if isinstance(n, ast.While) and any(isinstance(m, ast.Name) and m.id in derived for m in ast.walk(n.test)):
            use["loop_derived"] = True
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id == tunable:
            use["mentions"] += 1
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            names_l = {m.id for m in ast.walk(n.left) if isinstance(m, ast.Name)}
            names_r = {m.id for m in ast.walk(n.right) if isinstance(m, ast.Name)}
            calls = ast.unparse(n)
            if (tunable in names_l and (names_r & pid_vars or "program_id" in ast.unparse(n.right))) or (
                tunable in names_r and (names_l & pid_vars or "program_id" in ast.unparse(n.left))
            ):
                use["pid_scaled"] = True
            del calls
        if isinstance(n, ast.For) and isinstance(n.iter, ast.Call) and ast.unparse(n.iter.func).split(".")[-1] in {"range", "static_range"}:
            if any(isinstance(m, ast.Name) and m.id == tunable for a in n.iter.args for m in ast.walk(a)):
                use["loop_step"] = True
        if isinstance(n, ast.While) and any(isinstance(m, ast.Name) and m.id == tunable for m in ast.walk(n)):
            use["loop_step"] = True
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("arange"):
            if any(isinstance(m, ast.Name) and m.id == tunable for a in n.args for m in ast.walk(a)):
                use["arange"] = True
    return use


def scan_repo(root: Path) -> dict:
    files = [p for p in root.rglob("*.py") if ".git" not in p.parts]
    trees = {}
    for p in files:
        try:
            src = p.read_text(errors="replace")
        except OSError:
            continue
        if "triton" not in src:
            continue
        try:
            trees[p] = ast.parse(src)
        except (SyntaxError, ValueError, RecursionError):
            continue
    kernels: dict = {}  # name -> list of records
    for p, tree in trees.items():
        mod_assigns = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        mod_funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            decs = [ast.unparse(d) for d in fn.decorator_list]
            if not any("jit" in d for d in decs):
                continue
            rec = {"file": str(p.relative_to(root)), "line": fn.lineno, "name": fn.name, "autotune": False, "tunables": {},
                   "configs_resolved": None, "n_config_sites": 0, "node": fn}
            for dec in fn.decorator_list:
                if "autotune" not in ast.unparse(dec):
                    continue
                rec["autotune"] = True
                tun, sites, resolved = _tunables_from(dec, mod_assigns, mod_funcs)
                rec["tunables"] = {k: sorted(v) for k, v in tun.items()}
                rec["n_config_sites"] = sites
                rec["configs_resolved"] = resolved
                rec["decorator"] = ast.unparse(dec)[:300]
            kernels.setdefault(fn.name, []).append(rec)
    launches = []
    for p, tree in trees.items():
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.Module)):
                continue
            if isinstance(fn, ast.Module):
                body_nodes = [n for n in fn.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
                walk = [m for n in body_nodes for m in ast.walk(n)]
            else:
                walk = list(ast.walk(fn))
            assigns: dict = {}
            local_defs = set()
            for n in walk:
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    assigns.setdefault(n.targets[0].id, []).append((n.lineno, n.value))
                if isinstance(n, ast.FunctionDef) and n is not fn:
                    local_defs.add(n.name)
                    assigns.setdefault(n.name, []).append((n.lineno, n))

            def latest(name, line):
                # The nearest assignment above the use. A branch-insensitive choice, but it does not
                # confuse a lambda grid with a tuple assigned to the same name in a later branch.
                before = [(ln, v) for ln, v in assigns.get(name, []) if ln <= line]
                return max(before, key=lambda t: t[0])[1] if before else None
            for n in walk:
                if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)):
                    continue
                kname = _name(n.func.value)
                if kname not in kernels:
                    continue
                g = n.func.slice
                hops = 0
                while isinstance(g, ast.Name) and hops < 3:
                    nxt = latest(g.id, n.lineno)
                    if nxt is None:
                        break
                    g = nxt
                    hops += 1
                    if isinstance(g, ast.FunctionDef):
                        break
                if isinstance(g, (ast.Lambda, ast.FunctionDef)):
                    args = {a.arg for a in g.args.args}
                    body = [g.body] if isinstance(g, ast.Lambda) else g.body
                    uses = any(isinstance(m, ast.Name) and m.id in args for b in body for m in ast.walk(b))
                    kind = "callable" if uses else "callable_ignores_meta"
                elif isinstance(g, ast.Name) and g.id in local_defs:
                    kind = "callable"
                elif isinstance(g, (ast.Name, ast.Attribute, ast.Call)) and not isinstance(g, ast.Call):
                    kind = "unknown"
                else:
                    kind = "static"
                kw = sorted(k.arg for k in n.keywords if k.arg)
                launches.append({"file": str(p.relative_to(root)), "line": n.lineno, "kernel": kname, "grid_kind": kind,
                                 "grid_src": ast.unparse(g)[:160], "kwargs": kw})
    out = {"files_with_triton": len(trees), "kernels": 0, "autotuned": 0, "autotuned_shape": 0, "autotuned_unresolved": 0,
           "launch_sites": len(launches), "candidates": []}
    for name, recs in kernels.items():
        for rec in recs:
            out["kernels"] += 1
            if not rec["autotune"]:
                continue
            out["autotuned"] += 1
            shape = {k: v for k, v in rec["tunables"].items() if k not in NON_SHAPE and len(v) > 1}
            if not rec["configs_resolved"]:
                out["autotuned_unresolved"] += 1
            if shape:
                out["autotuned_shape"] += 1
            same_file = [l for l in launches if l["kernel"] == name and l["file"] == rec["file"]]
            mine = same_file or ([l for l in launches if l["kernel"] == name] if len(recs) == 1 else [])
            bad = [l for l in mine if l["grid_kind"] in ("static", "callable_ignores_meta")]
            if not bad:
                continue
            if shape:
                usage = {k: _kernel_usage(rec["node"], k) for k in shape}
                passed = sorted({k for l in bad for k in l["kwargs"]} & set(shape))
                out["candidates"].append({"kernel": name, "file": rec["file"], "line": rec["line"], "tunables": shape, "usage": usage,
                                          "launches": bad[:4], "tunable_also_passed_at_launch": passed, "class": "L1"})
            elif not rec["configs_resolved"]:
                out["candidates"].append({"kernel": name, "file": rec["file"], "line": rec["line"], "tunables": rec["tunables"],
                                          "decorator": rec.get("decorator"), "launches": bad[:2], "class": "unresolved_configs"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for d in sorted(Path(a.root).iterdir()):
        if d.is_dir():
            res[d.name] = scan_repo(d)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    tot = {"kernels": 0, "autotuned": 0, "autotuned_shape": 0, "L1": 0, "L1_pid_scaled": 0}
    for name, r in res.items():
        l1 = [c for c in r["candidates"] if c["class"] == "L1"]
        strong = [c for c in l1 if any(u["pid_scaled"] and not u["loop_step"] for u in c["usage"].values())]
        unres = sum(1 for c in r["candidates"] if c["class"] == "unresolved_configs")
        tot["kernels"] += r["kernels"]; tot["autotuned"] += r["autotuned"]; tot["autotuned_shape"] += r["autotuned_shape"]
        tot["L1"] += len(l1); tot["L1_pid_scaled"] += len(strong)
        print(f"{name[:38]:38s} kern={r['kernels']:4d} auto={r['autotuned']:4d} shape={r['autotuned_shape']:4d} unres={r['autotuned_unresolved']:4d} "
              f"L1={len(l1):3d} strong={len(strong):3d} unresolved_static={unres}")
    print(tot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
