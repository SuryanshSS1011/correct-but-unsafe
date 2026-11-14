"""Run the GPU confirmation kit on Modal.

    pip install modal && modal setup
    modal run gpu/modal_app.py                 # A100 by default
    CBU_GPU=T4 modal run gpu/modal_app.py      # any GPU type Modal offers
    modal run gpu/modal_app.py --kernels 984:2 --repeats 1 --no-minimal --crash ""   # smoke test

Each container clones the public repository, so nothing is uploaded from this
machine and no secrets are passed. The kernels it executes are model-generated
code from a public corpus. One CPU container downloads the corpus rows, then
each kernel runs in its own GPU container, so a slow kernel cannot take the
others down with it. Results are written to results/raw/ as gpu_minimal.json,
gpu_confirmation.json and gpu_crash_check.json.
"""

import os

import modal

GPU = os.environ.get("CBU_GPU", "A100")
REPO = "https://github.com/SuryanshSS1011/correct-but-unsafe"
CRASH = "236:3,481:3,3725:4,6133:2"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "build-essential"
    )  # Triton compiles a small launcher stub with the system C compiler
    .pip_install("torch", "datasets")
)
app = modal.App("correct-but-unsafe-gpu", image=image)


def clone(ref: str):
    import subprocess
    from pathlib import Path

    work = Path("/root/cbu")
    if not work.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref, REPO, str(work)],
            check=True,
        )
    return work


@app.function(timeout=1800)
def dump(kernels: str, ref: str) -> list[str]:
    import subprocess

    work = clone(ref)
    subprocess.run(
        [
            "python",
            "tools/dump_kernels.py",
            "--kernels",
            kernels,
            "--out",
            "data/k.jsonl",
        ],
        cwd=work,
        check=True,
    )
    return (work / "data" / "k.jsonl").read_text().splitlines()


@app.function(gpu=GPU, timeout=3600)
def minimal(ref: str) -> dict | None:
    import json
    import subprocess

    work = clone(ref)
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        check=False,
    )
    subprocess.run(
        [
            "python",
            "gpu/minimal_repro.py",
            "--repeats",
            "5",
            "--out",
            "gpu_minimal.json",
        ],
        cwd=work,
        check=False,
    )
    path = work / "gpu_minimal.json"
    return json.loads(path.read_text()) if path.exists() else None


@app.function(gpu=GPU, timeout=3 * 3600, max_containers=10)
def confirm_one(row: str, repeats: int, blocking: bool, ref: str) -> dict:
    import json
    import os
    import subprocess

    work = clone(ref)
    (work / "data").mkdir(exist_ok=True)
    (work / "data" / "k.jsonl").write_text(row + "\n")
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1") if blocking else None
    out = work / "results" / "raw" / "one.json"
    p = subprocess.run(
        [
            "python",
            "gpu/confirm.py",
            "--data",
            "data/k.jsonl",
            "--repeats",
            str(repeats),
            "--out",
            "results/raw/one.json",
        ],
        cwd=work,
        env=env,
        check=False,
    )
    r = json.loads(row)
    if out.exists():
        return json.loads(out.read_text())["kernels"][0]
    return {
        "id": f"{r['uuid']}_r{r['round']}",
        "runs": [],
        "error": f"confirm.py exited {p.returncode}",
    }


@app.local_entrypoint()
def main(
    repeats: int = 3,
    kernels: str = "flagged",
    crash: str = CRASH,
    ref: str = "audit-dev",
    no_minimal: bool = False,
):
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "gpu"))
    import confirm

    raw = root / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    mini = None if no_minimal else minimal.spawn(ref)
    rows = dump.remote(confirm.flagged() if kernels == "flagged" else kernels, ref)
    crash_rows = dump.remote(crash, ref) if crash else []
    print(f"{GPU}: {len(rows)} kernels, {len(crash_rows)} crash-check kernels")
    jobs = [(r, repeats, False, ref) for r in rows] + [
        (r, 1, True, ref) for r in crash_rows
    ]
    done = {"gpu_confirmation.json": [], "gpu_crash_check.json": []}
    for i, entry in enumerate(confirm_one.starmap(jobs, order_outputs=True, return_exceptions=True)):
        if isinstance(entry, BaseException):
            r = json.loads(jobs[i][0])
            entry = {"id": f"{r['uuid']}_r{r['round']}", "runs": [], "error": repr(entry)[:500]}
        name = "gpu_confirmation.json" if i < len(rows) else "gpu_crash_check.json"
        done[name].append(entry)
        (raw / name).write_text(
            json.dumps({"gpu": GPU, "kernels": done[name]}, indent=1)
        )
        print(f"[{i + 1}/{len(jobs)}] {entry['id']} -> {name} {entry.get('error', '')}")
    if mini is not None:
        res = mini.get()
        (raw / "gpu_minimal.json").write_text(
            json.dumps({"gpu": GPU, **(res or {})}, indent=1)
        )
        print(
            "wrote gpu_minimal.json" if res else "minimal: no result, see the log above"
        )
