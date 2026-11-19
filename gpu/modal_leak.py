"""Run gpu/cross_request_leak.py on Modal, one GPU per launch.

    modal run gpu/modal_leak.py                    # A100, partial config forced
    modal run gpu/modal_leak.py --tuned            # let the autotuner pick
    CBU_GPU=H100 modal run gpu/modal_leak.py

The container clones the public repository at the given ref, so nothing is
uploaded and no secrets are passed. The only kernel it runs is the minimal
add-one kernel written for this study. The result is written locally to
results/raw/<gpu>-leak/gpu_leak.json.
"""

import json
import os
from pathlib import Path

import modal

GPU = os.environ.get("CBU_GPU", "A100")
REPO = "https://github.com/SuryanshSS1011/correct-but-unsafe"

image = modal.Image.debian_slim(python_version="3.12").apt_install("git", "build-essential").pip_install("torch")
app = modal.App("correct-but-unsafe-leak", image=image)


@app.function(gpu=GPU, timeout=1800)
def leak(ref: str, tuned: bool, repeats: int) -> dict | None:
    import subprocess

    work = Path("/root/cbu")
    subprocess.run(["git", "clone", "--depth", "1", "--branch", ref, REPO, str(work)], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], check=False)
    args = ["python", "gpu/cross_request_leak.py", "--repeats", str(repeats), "--out", "gpu_leak.json"]
    subprocess.run(args + (["--tuned"] if tuned else []), cwd=work, check=False)
    path = work / "gpu_leak.json"
    return json.loads(path.read_text()) if path.exists() else None


@app.local_entrypoint()
def main(ref: str = "audit-dev", tuned: bool = False, repeats: int = 3):
    res = leak.remote(ref, tuned, repeats)
    out = Path("results/raw") / f"{GPU.lower()}-leak{'-tuned' if tuned else ''}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gpu_leak.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {out / 'gpu_leak.json'}")
