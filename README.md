# correct-but-unsafe

Memory-safety audit of benchmark-accepted LLM-generated GPU kernels.

Kernel-generation benchmarks accept a kernel when its outputs match a reference
on a few inputs. That check cannot see an out-of-bounds read, an uninitialized
read or a partly written output that happens to leave the compared values
unchanged. This repository measures how often accepted kernels have such bugs.

## Status

Work in progress on the `audit-dev` branch. See `docs/APPROACH.md` for the
detector design and `results/` for run summaries.

## Layout

| Path | Purpose |
|---|---|
| `cbu/target.py` | Runs one kernel on generated inputs in a throwaway process. |
| `cbu/detect.py` | CPU-only detectors that wrap the target runner. |
| `cbu/cpu_shim.py` | Redirects CUDA device requests to the CPU for interpreter runs. |
| `controls/tritongym/` | Positive control on 13 confirmed bugs in TritonGym reference kernels. |
| `.github/workflows/` | GitHub Actions workflows that run the detectors on hosted CPU runners. |

## Detectors

All detectors run kernels under the Triton interpreter (`TRITON_INTERPRET=1`),
where every unmasked `tl.load` and `tl.store` lane is a raw `memcpy` on a host
address. No GPU is needed.

1. `plain` runs the kernel and flags a fatal memory signal.
2. `tilelens` runs the symbolic out-of-bounds check from TileLens.
3. `asan` preloads the AddressSanitizer runtime so each interpreter access is checked against heap redzones.
4. `perturb` varies the fill byte of fresh and freed heap memory and flags output that changes.

## Safety

Corpus kernels are untrusted code. Workflows that execute them hold no secrets
and use a read-only token.
