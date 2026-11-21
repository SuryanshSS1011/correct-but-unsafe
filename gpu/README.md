# GPU confirmation kit

Everything in the rest of this repository runs on CPU. These two scripts are
the only part that needs a GPU. They ran on an A100 through Modal on 14 November
2026, and on a T4, an H100 and an A100 with the autotuner's disk cache on 17
November. The results are in `results/2026-11-14-gpu.md` and
`results/2026-11-17-gpu.md`. `python tools/gpu_compare.py` compares the runs.

The kernels they execute are model-generated code from a public corpus. Run
them in a container or a batch job, not in a session that holds credentials.

Any CUDA GPU works, including a free Colab T4. Both scripts together should
take well under one hour.

## Setup

```
git clone https://github.com/SuryanshSS1011/correct-but-unsafe
cd correct-but-unsafe
pip install torch triton datasets
```

## 1. Minimal demonstration

```
python gpu/minimal_repro.py --repeats 5
```

It defines one autotuned kernel and launches it with a fixed grid. It prints,
for four tensor sizes and five fresh processes, which block size the autotuner
chose and how many output elements each call left unwritten.

**What to expect:** the first call always leaves zero elements unwritten, because the autotuner runs every config on the output buffer. When the chosen block is below 1024, every later call leaves a fixed fraction unwritten.

**What it decides:** how often a partial config wins on this GPU, and whether that depends on the tensor size.

## 2. The flagged Dr. Kernel kernels

```
python gpu/confirm.py --kernels flagged --repeats 3
```

It downloads the 34 kernels listed in `results/triage-census.json` and replays
KernelGYM's five correctness trials for each one, in training and eval mode,
with and without the poison fill, in a fresh process per repeat. Results go to
`results/raw/gpu_confirmation.json`.

**What it decides:**

1. Whether the CPU findings hold in compiled form. With the poison fill, a trial with unwritten elements confirms the defect.
2. Which autotune config wins for each kernel on real hardware.
3. Whether stale allocator memory hides the defect. That is the case when a trial has unwritten elements with the poison fill and still passes without it.

4. Whether the unwritten region returns memory that held other data. Before each call the marker test creates and frees a tensor of the output's size filled with the value 1027. Output elements equal to 1027 came from that freed block. On the CPU interpreter, 6,144 of 8,192 output elements of `984_r2` and 64 of 128 of `4191_r4` returned the marker.

## Four kernels that crashed on a GPU and run clean on CPU

```
CUDA_LAUNCH_BLOCKING=1 python gpu/confirm.py --kernels 236:3,481:3,3725:4,6133:2 --repeats 1 --out results/raw/gpu_crash_check.json
```

KernelGYM recorded an illegal memory access for these four. On CPU they run
without any out-of-bounds access and match the reference. If they crash here
too, the interpreter misses something. If they run clean, the recorded error
came from elsewhere in the worker.

## Launchers

1. **Modal:** `modal run gpu/modal_app.py` runs all three parts, one GPU container per kernel, at most 9 at a time plus one for the minimal demonstration, and writes to `results/raw/<gpu>/`. Set `CBU_GPU=T4` or another type to compare GPUs. Add `--cache` to run every process with `TRITON_CACHE_AUTOTUNING=1`, which writes to `results/raw/<gpu>-cache/`. Run one launch at a time, since each launch uses up to 10 GPUs, and keep the local client connected, because Modal stops the app when it disconnects.
2. **Roar:** fill in the account and partition in `gpu/roar.sbatch`, do the setup on a login node as its header describes, then `sbatch gpu/roar.sbatch`.

The Modal launcher has run on a T4, an A100 and an H100. The Roar launcher has not been run. The winning autotune config, and so which kernels leak, differs between the three GPUs.

## Compute Sanitizer initcheck, with and without the caching allocator

```
modal run gpu/modal_initcheck.py
```

`gpu/initcheck.py` runs KernelGYM's five plain trials under `compute-sanitizer --tool initcheck`, once with PyTorch's caching allocator and once with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`. A warm-up process with the poison fill tunes each autotuned kernel and writes the winner to a private Triton cache, so both sanitizer runs use the same config and skip tuning. The launcher uses a CUDA devel image, uploads this working tree's `cbu/` and `gpu/` instead of cloning the repository, and writes to `results/raw/<gpu>-initcheck/`.

**What it decides:** whether the standard tool for uninitialized reads sees the unwritten output under the default allocator.

## Optional: NVIDIA Compute Sanitizer

`compute-sanitizer --tool memcheck python -m cbu.target ...` gives an
independent confirmation of the five out-of-bounds reads. It is slow and is
not needed for the claims above.

## Cross-request leak check

`gpu/cross_request_leak.py` asks whether a partial write returns an earlier
request's data. One process plays a server. Request A fills its tensors with a
position-coded pattern and frees them, and request B then runs the fixed-grid
add-one kernel forced to its 256 block, which is what a restarted process runs
when the disk cache holds a small winner. The script counts how many of B's
unwritten elements hold A's values, in six scenarios: same size, a larger
victim, unrelated allocations in between, `PYTORCH_NO_CUDA_MEMORY_CACHING=1`,
expandable segments, and A and B in separate processes one after the other.

    modal run gpu/modal_leak.py              # one A100, writes results/raw/a100-leak/gpu_leak.json
    modal run gpu/modal_leak.py --tuned      # let the autotuner pick the block
    python gpu/cross_request_leak.py         # any local CUDA GPU

A CPU dry run with `CBU_DEVICE=cpu CBU_N=8192 TRITON_INTERPRET=1` checks the
script's logic only. It leaves 6,144 of 8,192 elements unwritten, as expected,
and shows no victim data, because host memory is not the target.

## Cross-request leak in vLLM

`gpu/vllm_leak/` runs the leak test in vLLM 0.34 and SGLang 0.5.9.
`cbu_vllm_plugin` is a vLLM general plugin that, when `CBU_KERNEL` is set,
replaces the kernel behind `SiluAndMul` with a Triton kernel of the same
signature. `defective` launches the fixed grid of the corpus kernels and leaves
three quarters of the output unwritten, `floor` computes the grid with floor
division, `tail` leaves only the last element unwritten, and `correct` is the
control. `run_leak.py` runs one engine per process, alternates two victim
prompts before the same request and records the request's output, and with
`--record` compares its activation buffers between trials. `run_leak_sglang.py`
does the same in SGLang, where `sglang_patch/sitecustomize.py` swaps the kernel
in SGLang's spawned processes. `summarize.py` gives a verdict per cell.

It ran on Penn State Roar Collab. Copy `gpu/vllm_leak/roar_env.sh` to
`/storage/work/$USER/cbu-vllm/env.sh`, create the venv there with
`uv pip install vllm==0.34.0`, and for SGLang a second venv, `venv-sglang`, with
`uv pip install sglang==0.5.9` from the PyTorch cu130 index. Then:

    sbatch --account=<allocation> gpu/vllm_leak/roar_vllm.sbatch                # 16 configurations and the recording runs
    SET=subtle sbatch --account=<allocation> gpu/vllm_leak/roar_vllm.sbatch     # the floor and tail defects
    SET=nocache sbatch --account=<allocation> gpu/vllm_leak/roar_vllm.sbatch    # allocator cache off
    SET=model MODEL=<hf id> sbatch --account=<allocation> gpu/vllm_leak/roar_vllm.sbatch
    SET=model sbatch --account=<allocation> --gres=gpu:a40:1 gpu/vllm_leak/roar_vllm.sbatch
    sbatch --account=<allocation> gpu/vllm_leak/roar_sglang.sbatch

`roar_env.sh` keeps every cache off the home directory, which has a small
quota. The results are in `results/2026-11-19-vllm.md` and
`results/2026-11-20-vllm-followup.md`.
