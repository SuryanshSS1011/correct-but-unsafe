"""Run gpu/initcheck.py on Modal, one GPU container per kernel.

    modal run gpu/modal_initcheck.py                       # A100, the default kernel set
    modal run gpu/modal_initcheck.py --kernels 7520:1 --no-minimal   # smoke test

The image is a CUDA devel image, so compute-sanitizer is on PATH. Unlike
gpu/modal_app.py it uploads this working tree's cbu/ and gpu/ instead of
cloning the repository, and the corpus rows are downloaded on this machine.
At most 9 kernel containers and the minimal one run at a time, within a 10-GPU
limit. Results go to results/raw/<gpu>-initcheck/.
"""

import os
from pathlib import Path

import modal

GPU = os.environ.get("CBU_GPU", "A100")
ROOT = Path(__file__).resolve().parents[1]
# Controls that never left an element unwritten on any GPU, three kernels without
# autotuning whose partial write is deterministic, and five autotuned kernels that
# leave elements unwritten on most GPUs.
KERNELS = "160:2,1695:2,2118:3,6852:1,7520:1,2121:2,5239:2,7962:1,1586:2,2487:3"

image = (
    modal.Image.from_registry("nvidia/cuda:13.4.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install(
        "build-essential"
    )  # Triton compiles a small launcher stub with the system C compiler
    .pip_install("torch")
    .add_local_dir(ROOT / "cbu", "/root/cbu/cbu", ignore=["__pycache__"])
    .add_local_file(ROOT / "gpu" / "initcheck.py", "/root/cbu/gpu/initcheck.py")
    .add_local_file(ROOT / "gpu" / "minimal_repro.py", "/root/cbu/gpu/minimal_repro.py")
)
app = modal.App("correct-but-unsafe-initcheck", image=image)


def _run(args: list[str]) -> dict:
    import json
    import subprocess

    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], check=False
    )
    out = "/root/cbu/results/raw/one.json"
    p = subprocess.run(
        ["python", "gpu/initcheck.py", *args, "--out", "results/raw/one.json"],
        cwd="/root/cbu",
        check=False,
    )
    if Path(out).exists():
        return json.loads(Path(out).read_text())
    return {"kernels": [], "error": f"initcheck.py exited {p.returncode}"}


@app.function(gpu=GPU, timeout=24 * 3600, max_containers=9)
def kernel(row: str) -> dict:
    Path("/root/cbu/data").mkdir(parents=True, exist_ok=True)
    Path("/root/cbu/data/k.jsonl").write_text(row + "\n")
    return _run(["--data", "data/k.jsonl"])


@app.function(
    gpu=GPU, timeout=24 * 3600, retries=2
)  # a preempted run restarts instead of being lost
def minimal(sizes: str) -> dict:
    return _run(["--minimal", *(["--minimal-sizes", sizes] if sizes else [])])


@app.local_entrypoint()
def main(
    kernels: str = KERNELS,
    no_minimal: bool = False,
    out: str = "",
    minimal_sizes: str = "",
):
    import json
    import subprocess
    import tempfile

    raw = ROOT / (out or f"results/raw/{GPU.lower()}-initcheck")
    raw.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    if kernels:  # --kernels "" runs only the minimal demonstration
        with tempfile.TemporaryDirectory() as tmp:
            rows_file = Path(tmp) / "k.jsonl"
            subprocess.run(
                [
                    "python3",  # not sys.executable, which under `modal run` lacks datasets
                    str(ROOT / "tools" / "dump_kernels.py"),
                    "--kernels",
                    kernels,
                    "--out",
                    str(rows_file),
                ],
                check=True,
            )
            rows = rows_file.read_text().splitlines()
    mini = None if no_minimal else minimal.spawn(minimal_sizes)
    done = []
    for i, res in enumerate(
        kernel.map(rows, order_outputs=False, return_exceptions=True) if rows else []  # map([]) blocks
    ):
        if isinstance(res, BaseException):
            res = {"kernels": [], "error": repr(res)[:500]}
        done.extend(res["kernels"] or [{"id": f"row {i}", "error": res.get("error")}])
        meta = {"gpu": GPU, "sanitizer": res.get("sanitizer")}
        (raw / "initcheck.json").write_text(
            json.dumps({**meta, "kernels": done}, indent=1)
        )
        print(
            f"[{len(done)}/{len(rows)}] "
            + json.dumps(
                [
                    {
                        k["id"]: {
                            n: k.get(n, {}).get("errors")
                            for n in ("caching", "no_caching")
                        }
                    }
                    for k in res["kernels"]
                ]
            )
            + (f" {res.get('error')}" if res.get("error") else ""),
            flush=True,
        )
    if mini is not None:
        res = mini.get()
        (raw / "initcheck_minimal.json").write_text(
            json.dumps({"gpu": GPU, **res}, indent=1)
        )
        print(
            "minimal:",
            json.dumps(
                {
                    n: res["kernels"][0][n].get("errors")
                    for n in ("caching", "no_caching")
                }
            )
            if res.get("kernels")
            else res.get("error"),
            flush=True,
        )
