# Approach

This note records how the audit detects memory-unsafe Triton kernels without a
GPU, what was checked at source level, and what the method cannot see.

## 1. Why a CPU is enough for most of the audit

Triton ships an interpreter that runs kernels on the host when
`TRITON_INTERPRET=1` is set. In Triton 3.8.0 the interpreter implements
`tl.load` and `tl.store` in `python/src/interpreter.cc` as a loop over lanes.
Each lane whose mask is true becomes one `memcpy` to or from an absolute 64-bit
host address. A lane whose mask is false is never dereferenced. There is no
bounds check.

Two consequences follow.

1. An unmasked out-of-bounds access in a kernel touches real process memory under the interpreter. Ordinary CPU memory tooling therefore applies to Triton kernels.
2. A masked-off lane that computes an out-of-range address is not an access, on CPU or on GPU, and no detector here reports it.

## 2. Detectors

Every detector runs one kernel in a fresh subprocess through `cbu/target.py`.

| Detector | Mechanism | Sees | Misses |
|---|---|---|---|
| `plain` | Runs the kernel and records SIGSEGV, SIGABRT or SIGBUS. | Wild accesses and heap corruption severe enough to kill the process. | Anything that stays inside mapped memory. |
| `asan` | The GCC AddressSanitizer runtime is preloaded into an uninstrumented Python. Its `memcpy` interceptor checks every interpreter load and store against the redzones around each heap allocation. | Out-of-bounds reads and writes, with access size, offset and region size. | An access that lands inside another live allocation beyond the redzone. The redzone is set to 2048 bytes. |
| `poison` | Every `torch.empty`, `empty_like`, `empty_strided` and `new_empty` allocation is filled with a fixed bit pattern. After the kernel ran, the output elements that still hold the pattern are counted. | Output elements the kernel never wrote, as an exact count and fraction. A NaN that the unpoisoned run did not have shows that the kernel computed from uninitialized values. | Uninitialized memory that does not come from those four calls. Eight-bit and boolean tensors. |
| `perturb` | Four runs. Two are identical. Two set glibc `MALLOC_PERTURB_` to different bytes, which changes the fill of fresh and freed heap memory. | Output that depends on uninitialized, stale or out-of-bounds memory. Two identical runs that disagree are reported as `unstable`. | Reads that do not reach an output. |
| `tilelens` | TileLens 3.1.1 symbolic sanitizer, the released Triton-Sanitizer. | Out-of-bounds loads and stores, with the source line and tensor name. | Uninitialized reads. It crashes on some wild accesses before it reports. It also reports accesses that a runtime branch or a pointer select makes unreachable. |

Three implementation details matter for correctness of the detectors.

1. The `asan` detector needs `libstdc++` preloaded next to `libasan`. Python does not link `libstdc++`, so the sanitizer cannot resolve `__cxa_throw` at start-up and aborts when the first C++ extension throws.
2. The interpreter performs `tl.atomic_*` operations through raw pointers, not through `memcpy`. A preloaded sanitizer cannot observe them. `cbu/target.py` therefore loads the addresses of every atomic operation through the interpreter's masked load first. Before this change an out-of-bounds atomic write in the TritonGym `bgmv_shrink_kernel` reference corrupted the glibc heap while `asan` reported nothing.
3. The `perturb` comparison is valid only when every run starts from the same inputs. Some references build their inputs at import time, so the seed is set before the reference module is imported, and the detector checks a digest of the inputs before it compares outputs.

## 3. Which detector to trust

The ranking below comes from the positive control and from manual triage of
every flag in the first pilot. The numbers are in `results/`.

1. **`asan` is the primary out-of-bounds oracle.** It found 13 of the 13 known TritonGym bugs on the commit before the fix and none on the commit after it. Every `asan` flag triaged so far was a real access.
2. **`poison` is the primary oracle for unwritten outputs.** It is deterministic, it reports the fraction of the output that was never written, and the same shim would run unchanged on a GPU.
3. **`perturb` is a cross-check.** It catches out-of-bounds reads that reach the output, which `poison` cannot see, and it confirms `poison` findings by a different mechanism.
4. **`plain` adds crashes** that the sanitizer run does not reproduce because the two allocators lay memory out differently.
5. **`tilelens` is a secondary signal.** On hand-written TritonGym kernels it agreed with `asan` on 9 of 13 known bugs. On model-written kernels all three of its solo flags in the first pilot were false positives. Each came from control flow that the symbolic execution does not follow: a runtime `if` on the program id, or a pointer chosen by `tl.where` between two tensors. A flag from `tilelens` alone is queued for manual triage and is not counted as a finding.

## 4. Validity threats of interpreter-based auditing

1. **Fallback wrappers.** Most model-generated modules return a plain PyTorch result when the input is not a CUDA tensor. In a first sample of 24 accepted Dr. Kernel kernels, 22 contained an `is_cuda` check. Twelve of the 24 were run without the workaround below, and none of them launched a Triton kernel on CPU. `cbu/cpu_shim.py` therefore reports CPU tensors as CUDA tensors while every allocation stays on the CPU. `cbu/target.py` counts interpreter launches, and a run with zero launches is an exclusion, never a clean result.
2. **Autotuning.** `triton.autotune` benchmarks configs on a GPU driver and keeps the fastest. The shim replaces it with a proxy that launches once with a chosen config, and the pilot audits up to six configs per kernel. This matters because a config that writes less of the output is also faster, so the autotuner can prefer it.
3. **Zeroed fresh memory.** A fresh CPU heap page and a fresh GPU segment are usually zero. A kernel that leaves output elements unwritten therefore matches any reference whose expected value there is zero. The `poison` detector removes this dependence.
4. **Interpreter fidelity.** The interpreter is stricter than the compiler about operand dtypes, and TileLens documents one divergence for `and` and `or` on tensors. A kernel that fails for such a reason is an exclusion.
5. **Interpreter defects.** An `asan` report counts only when its stack contains the interpreter's load or store.
6. **Import-time self-tests.** Some reference files run a self-test when imported. A report raised during import is recorded separately from the audited call.
7. **Benign-looking accesses.** On a GPU, the PyTorch allocator rounds every block to 512 bytes, so a short overread often stays inside the block and changes nothing. The access is still undefined behaviour in the kernel. The audit reports it and labels whether the output was affected.
8. **Races.** The interpreter runs program instances sequentially, so data races are invisible. Races need TritonSan with ThreadSanitizer or a GPU tool.
9. **Timing.** Speedup and the safety tax can only be measured on a GPU.

## 5. Audit population for Dr. Kernel

The corpus `hkust-nlp/drkernel-coldstart-8k` holds 8,920 five-round
trajectories. It has no correctness column. User turns 2 to 5 carry the KernelGYM
server feedback for the previous round, including `compiled`, `correctness`,
`decoy_kernel` and `speedup`. Rounds 1 to 4 therefore have a recorded verdict and
round 5 does not. For trajectories whose best round is 1 to 4, the feedback
speedup of that round equals the row's `final_speedup`, which confirms the
alignment.

The pilot population is one kernel per trajectory, taken from the best round and
kept only when that round compiled, passed its correctness trials and was not a
decoy. Trajectories whose best round is 5 are left out because that verdict is
not recorded.

The contract-grade verifier's released verdicts are keyed by the corpus `uuid`
column, not by row index. The sampler joins on `uuid`.

Assistant turns contain a written analysis before the code block, even though
`enable_thinking` is false for every row. Intent can therefore be studied from
that text.

## 6. What still needs a GPU

1. Confirmation of flagged kernels in compiled form. For unwritten outputs the `poison` shim runs unchanged on a GPU, and NVIDIA Compute Sanitizer's `initcheck` tool reports reads of uninitialized device memory. CuSafe, published at USENIX Security 2026, is described by its authors as open source with an average slowdown near 13%, which would be far cheaper than NVIDIA Compute Sanitizer for this step. Its availability has not been tested here.
2. The safety tax, which is a timing measurement.
3. The CUDA corpus from the Sakana archive.
4. Data races in compiled kernels.

## 7. Possible extension for races on CPU

TritonSan in `triton-shared` compiles Triton kernels to CPU code and runs them
under AddressSanitizer or ThreadSanitizer. It supports Ubuntu 24.04, which
matches the hosted runner image. It needs a custom LLVM build that its README
says can exceed 20 minutes. The build could be cached as a workflow artifact.
