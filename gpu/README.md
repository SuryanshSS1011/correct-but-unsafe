# GPU confirmation kit

Everything in the rest of this repository runs on CPU. These two scripts are
the only part that needs a GPU. They have been dry-run on the CPU interpreter
and have not yet been run on a GPU.

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

1. **Modal:** `modal run gpu/modal_app.py`. Set `CBU_GPU=T4` or another type to compare GPUs.
2. **Roar:** fill in the account and partition in `gpu/roar.sbatch`, do the setup on a login node as its header describes, then `sbatch gpu/roar.sbatch`.

Neither launcher has been run. Running on two GPU types is useful, because the winning autotune config can differ between them.

## Optional: NVIDIA Compute Sanitizer

`compute-sanitizer --tool memcheck python -m cbu.target ...` gives an
independent confirmation of the five out-of-bounds reads. It is slow and is
not needed for the claims above.
