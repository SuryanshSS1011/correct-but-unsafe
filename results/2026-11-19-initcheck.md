# Compute Sanitizer initcheck, with and without the caching allocator, 19 November 2026

This file asks whether NVIDIA's tool for reads of uninitialized device memory
sees the defect. `compute-sanitizer --tool initcheck` reports a read of device
memory that nothing has written. A defective kernel never reads its own
unwritten output, but the harness's comparison does. The runs used
`gpu/modal_initcheck.py` on an A100-SXM4-40GB, compute-sanitizer 2026.3.0,
torch 2.15.0 and Triton 3.9.0. The raw files are in `results/raw/a100-initcheck/`.

For each kernel, a warm-up process with the poison fill ran KernelGYM's five
trials, tuned any autotuned kernel and wrote the winner to a private Triton
cache. Two processes under initcheck then loaded that winner, so they used the
same config without tuning, and ran the five trials without the poison fill.
One used PyTorch's caching allocator and one set
`PYTORCH_NO_CUDA_MEMORY_CACHING=1`, so that every allocation is a fresh
`cudaMalloc`.

## 1. Results

| Kernel | Unwritten elements per call | Default allocator: plain trials passing | Default allocator: initcheck errors | No caching: initcheck errors |
|---|---|---|---|---|
| `160_r2` (control) | 0 | 5 of 5 | 0 | 0 |
| `1695_r2` (control) | 0 | 5 of 5 | 0 | 0 |
| `2118_r3` | 3,840 | 5 of 5 | **0** | 17,280 |
| `5239_r2` | 3,686,400 | 5 of 5 | **0** | 13,824,000 |
| `2487_r3` | 1 | 0 of 5 | **0** | 15 |
| `7520_r1` | 224,768 | 5 of 5 | 169,728 | 848,640 |
| `6852_r1` | 36,672 | 5 of 5 | 191,424 | 465,600 |
| `7962_r1` | 55,104 | 5 of 5 | 364,992 | 1,394,880 |
| Minimal demonstration, three sizes | up to 786,432 | 0 of 5 | **0** | 3,087,360 |

The two controls never leave an element unwritten on any GPU, and initcheck
reports nothing for them in either setting, so torch and its libraries add no
false reports. Where the report names the reading kernel, it is the
elementwise comparison inside `torch.allclose`, which reads the unwritten
output. For `2487_r3` the 15 errors are 5 trials times the three elementwise
kernels of `allclose` reading its one unwritten element. The report for
`6852_r1` and `7962_r1` did not name the reading kernel in a form the parser
reads.

`1586_r2` and `2121_r2` also ran. Their warm-up drew a covering config, so they
left no element unwritten and initcheck reported nothing, which says nothing
about the defect. The covering and partial configs of these kernels are
compared in `results/2026-11-17-gpu.md`.

## 2. What follows

**With the default allocator, initcheck can miss the defect entirely.** For
`2118_r3` and `5239_r2` the benchmark passed every trial and initcheck reported
nothing, while every call left 3,840 and 3,686,400 output elements unwritten.
The output block is recycled from an earlier tensor, so the sanitizer counts it
as initialized. The same holds for `2487_r3` and the minimal demonstration,
which the benchmark rejects only because the stale values happen to differ.

**It does not always miss it.** For `7520_r1`, `6852_r1` and `7962_r1`
initcheck reported errors under the default allocator too, at a fifth to two
fifths of the rate without caching. For `7520_r1` the ratio is exactly 1 to 5, which
fits one of the five trials receiving a block that had never been written.
Whether initcheck sees the defect therefore depends on where the allocator
places the output, not on the kernel.

**Turning off the cache alone does not make the benchmark catch it.** Without
caching, `2118_r3`, `6852_r1` and `7962_r1` still passed every plain trial and
`7520_r1` passed four of five. Only `5239_r2` failed all five. The three that
pass skip output elements whose correct value is zero: the upper triangle of
`torch.tril` for `2118_r3`, and the padding of `F.unfold` for `6852_r1` and
`7962_r1`. In these runs a fresh allocation without caching came back zeroed, as in the
cross-request check in `results/2026-11-19-leak.md`, so the unwritten elements
happened to hold the right value. CUDA does not promise zeroing, and the vLLM
runs without the cache found non-zero leftovers in the same process, so this
holds for these runs and not in general. Under the caching allocator
they passed as well, so the recycled block held zeros at those positions; which
earlier tensor it came from was not established. Only the poison fill or
initcheck without caching detected the defect in every case.

An earlier version of this paragraph, and the message of the commit that added
it, said the driver returns freed pages with their contents. The data do not
support that, and it is withdrawn.

## 3. Limits

These are ten corpus kernels and the minimal demonstration on one GPU type, chosen to cover the controls, the
kernels without autotuning and the autotuned kernels that leak on most GPUs.
initcheck is slow when it has many reads to record. A first pass limited each
process to 60 minutes and lost three totals without caching. `5239_r2` and
`7962_r1` ran again without a limit, in `results/raw/a100-initcheck/rerun/`,
and took more than an hour and nearly three hours. Both drew the same winner
as in the first pass, and `7962_r1` gave the same count under the default
allocator, 364,992, both times.

The minimal demonstration at all four sizes ran without caching for 3 hours
42 minutes before Modal interrupted its container. Its row uses a rerun at the
three sizes up to 1,048,576, which took 17 minutes, in
`results/raw/a100-initcheck/rerun-minimal/`. At the fourth size, 16,777,216,
the first pass recorded no error under the default allocator with 12,582,912
elements unwritten per call, and no total without caching.

## 4. Corrections, 21 November

1. The minimal demonstration did not load a cached winner. Its first call left nothing unwritten, so both initcheck processes tuned for themselves, and at 65,536 elements they chose different blocks, 256 with caching and 512 without. The corpus kernels did load the warm-up's winner as described.
2. Section 3 said nine kernels. Ten corpus kernels ran, `1586_r2` and `2121_r2` included, plus the minimal demonstration.
3. The zeroing statement in section 2 is narrowed to these runs, because the vLLM runs without the cache contradict it. `gpu/alloc_zeroing.py` measures it by allocation size.
