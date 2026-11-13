# GPU confirmation kit

Everything in the rest of this repository runs on CPU. These two scripts are
the only part that needs a GPU. They have been dry-run on the CPU interpreter
and have not yet been run on a GPU.

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

## Optional: NVIDIA Compute Sanitizer

`compute-sanitizer --tool memcheck python -m cbu.target ...` gives an
independent confirmation of the five out-of-bounds reads. It is slow and is
not needed for the claims above.
