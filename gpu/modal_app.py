"""Run the GPU confirmation kit on Modal.

    pip install modal && modal setup
    modal run gpu/modal_app.py                 # A100 by default
    CBU_GPU=T4 modal run gpu/modal_app.py      # any GPU type Modal offers

The function clones the public repository inside the container, so nothing is
uploaded from this machine and no secrets are passed. The kernels it executes
are model-generated code from a public corpus. Results are written to the
current directory as gpu_minimal.json and gpu_confirmation.json.

This launcher has not been run. Check the first lines of output.
"""
import os

import modal

GPU = os.environ.get("CBU_GPU", "A100")
REPO = "https://github.com/SuryanshSS1011/correct-but-unsafe"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")  # Triton compiles a small launcher stub with the system C compiler
    .pip_install("torch", "datasets")
)
app = modal.App("correct-but-unsafe-gpu", image=image)


@app.function(gpu=GPU, timeout=3 * 3600)
def confirm(repeats: int = 3, ref: str = "audit-dev", kernels: str = "flagged") -> dict:
    import json
    import subprocess
    from pathlib import Path

    work = Path("/root/cbu")
    subprocess.run(["git", "clone", "--depth", "1", "--branch", ref, REPO, str(work)], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    out: dict = {}
    subprocess.run(["python", "gpu/minimal_repro.py", "--repeats", "5", "--out", "gpu_minimal.json"], cwd=work, check=False)
    subprocess.run(["python", "gpu/confirm.py", "--kernels", kernels, "--repeats", str(repeats)], cwd=work, check=False)
    for name, path in (("minimal", work / "gpu_minimal.json"), ("confirmation", work / "results" / "raw" / "gpu_confirmation.json")):
        out[name] = json.loads(path.read_text()) if path.exists() else None
    return out


@app.local_entrypoint()
def main(repeats: int = 3, kernels: str = "flagged"):
    import json

    res = confirm.remote(repeats=repeats, kernels=kernels)
    for name, data in res.items():
        with open(f"gpu_{name}.json", "w") as f:
            json.dump(data, f, indent=1)
        print(f"wrote gpu_{name}.json" if data is not None else f"{name}: no result, see the log above")
