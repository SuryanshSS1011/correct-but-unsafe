# Why defective kernels pass, 13 November 2026

This file extends `2026-11-13-census.md`. It reports the analyses that explain
how 34 defective kernels were accepted, and the checks made on the detectors.
`docs/DECISIONS.md` states what follows from them.

## 1. The corpus's own records confirm the autotuning mechanism

Run 43472129700 applied a static lint to all 44,600 rounds of the corpus and
joined it with the recorded KernelGYM feedback. Rule L1 marks an autotuned
kernel that is launched with a grid that does not depend on the tuned
parameters. No corpus code was executed.

| Rounds with feedback | Count | Accepted |
|---|---|---|
| All | 35,680 | 61.5% |
| L1: autotuned kernel with a fixed grid | 4,044 | 42.7% (41.1 to 44.2) |
| Autotuned, grid depends on the config | 8,119 | 66.6% (65.5 to 67.6) |
| No autotune | 23,517 | 63.0% (62.4 to 63.7) |

The prediction was that a partial write hidden by the tuning pass shows up as
exactly one passed trial of five. The first trial runs every config on the
output buffer. The other four run only the winner.

| Group, among rounds that compiled, ran five trials and failed | Rounds | Passed exactly 1 of 5 |
|---|---|---|
| L1 | 675 | 453, 67.1% (63.5 to 70.5) |
| Autotuned, not L1 | 301 | 35, 11.6% (8.5 to 15.7) |
| No autotune | 630 | 18, 2.9% (1.8 to 4.5) |

As a share of all rounds in each group, the 1-of-5 outcome is 11.2% for L1,
0.4% for other autotuned rounds and 0.1% for rounds without autotuning.

After a 1-of-5 failure the next round switched to a config-dependent grid in
384 of 453 cases, and 311 of those were accepted.

## 2. One trajectory shows the same code structure failing and then passing

Trajectory 8676 keeps the grid `(triton.cdiv(n, 1024),)` and the same five
configs, with block sizes 512, 1024, 1024, 2048 and 4096, in rounds 3 and 4.
The only code change between them is `x * inv_modulus` becoming `x / modulus`.

| Round | KernelGYM record | CPU replay, winner is block 512 | CPU replay, any other winner |
|---|---|---|---|
| 3 | 1 of 5 trials passed | pass, fail, fail, fail, fail | five passes |
| 4 | 5 of 5 trials passed, accepted | pass, fail, fail, fail, fail | five passes |

The replay shows that round 3 is numerically correct under a covering config.
Its recorded failure therefore has the signature of block 512 winning the
tuning race, and round 4 was accepted with the same exposure.

## 3. Recheck of the flagged census kernels

Run 43472135784 rechecked the flagged kernels under every autotune config, in
eval and training mode, and replayed the five trials once per possible winner.
Twenty-seven of 36 kernels had finished when this was written. The other nine
have large inputs.

| Measure | Result |
|---|---|
| Uninitialized-output kernels with at least one partial config | 21 of 21 |
| Of those, partial under every config | 5 |
| Of those, partial under some configs | 16 |
| Partial configs among the config-dependent kernels | 27 of 84 |
| Replay shows pass, fail, fail, fail, fail exactly for the partial winners | 16 of 17 kernels checked |
| Out-of-bounds kernels confirmed by both the sanitizer and the bounds oracle | 4 of 4 |
| Earlier false positives that are now clean | 2 of 2 |
| Configs where the sanitizer and the bounds oracle disagree | 0 of 104 |
| Kernels that behave differently in training mode | 2 of 27 |

The two mode-dependent kernels, `228_r2` and `4163_r2`, fall back to PyTorch
in training mode, so their Triton kernel never launches there.

## 4. The exact bounds oracle

The oracle records, at each launch, the allocation behind every tensor
argument. It carries that provenance through pointer arithmetic, broadcasts
and selects, and checks each masked-in address of each load, store and atomic
operation against the allocation the pointer came from.

Run 43475919982 compared it with the preloaded sanitizer on the positive control.

| TritonGym commit | Ops flagged by the sanitizer | Ops flagged by the oracle | Cases flagged by both | Cases flagged by one only |
|---|---|---|---|---|
| Before the fix | 13 of 13 | 13 of 13 | 37 | 0 |
| After the fix | 0 of 13 | 0 of 13 | 0 | 0 |

Provenance was resolved for every access. A first version of the oracle
without provenance flagged only 5 of 13. Six of the misses came from the
process dying before it reported, and the oracle now prints a violation when
it finds it.

## 5. Nearby shapes

In the repeat pilot, ten kernels that are correct at the benchmark's shape
gave wrong output when one constant was increased by one.

1. Two leave part of the output unwritten when a size is not a multiple of the block: `3964_r4` and `6912_r4`.
2. Four give wrong values without unwritten elements.
3. Three are artifacts of the perturbation, which changed a literal that is not a shape.
4. One is `7520_r1`, which is already counted.

A population run should perturb named shape constants only.

## 6. Still running

1. Run 43475965294 rechecks the 1,725 accepted rounds that L1 marks.
2. Run 43475957047 rechecks the 453 rejected 1-of-5 rounds.
3. Run 43475920247 repeats the 300-kernel pilot with the bounds oracle.
4. Run 43475950907 repeats the TritonGym sweep with the bounds oracle.
