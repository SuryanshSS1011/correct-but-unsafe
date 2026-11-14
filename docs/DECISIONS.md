# Decision document, 13 November 2026

This document records what the study has established, what is still open, and
which decisions follow. Its purpose is to make sure no large amount of compute
or time is spent before the analysis behind it is done. It supersedes the go
and no-go rules of the December plan. Numbers come from the
files in `results/`. Section 6 records how the numbers that were pending
settled each decision rule. A few runs were still in progress when this was
written, and each is named where it matters.

## 1. What is established

Each claim carries a grade. Grade A means the claim is supported by CPU runs
and by GPU records, from the corpus or from the GPU kit in `gpu/`. Grade B means CPU runs and
source reading only. Grade C means inference.

| # | Claim | Evidence | Grade |
|---|---|---|---|
| 1 | The CPU pipeline finds real memory-safety bugs and does not flag fixed code. | 13 of 13 known TritonGym bugs flagged before the fix and 0 of 13 after it. | A |
| 2 | TritonGym still has seven buggy reference kernels after its fix. | Flagged by concrete detectors among 114 references that ran, and each confirmed by reading the source. | B |
| 3 | 34 of 6,113 accepted Dr. Kernel kernels have a memory-safety defect, which is 0.56% with an interval of 0.40% to 0.78%. | Census of the whole accepted population. 29 return partly uninitialized output and 5 read out of bounds. | B |
| 4 | Triton's autotuner hides a partial write on the first call. | The Triton documentation states that every config runs on the real arguments during tuning. A CPU replay of KernelGYM's five trials shows pass, fail, fail, fail, fail exactly when a partial config is the winner. On an A100, 17 flagged kernels show the same signature natively. | A |
| 5 | Partial configs do win on real GPUs. | In KernelGYM's own records, 453 of 675 failing rounds with the fixed-grid pattern passed exactly 1 of 5 trials. The rate is 67.1%. It is 11.6% for other autotuned kernels and 2.9% for kernels without autotuning. A CPU recheck of those 453 rounds found a partial config in 420 of the 444 that ran, which is 94.6%. On an A100 a partial config won in 17 of 20 size and repeat cells of the minimal demonstration. | A |
| 6 | Acceptance of such a kernel is not determined by its code. | Trajectory 8676 kept the same fixed grid and the same five configs in rounds 3 and 4. Round 3 passed 1 of 5 trials and round 4 passed 5 of 5. The CPU replay gives the same two signatures for the two possible kinds of winner. On an A100, six flagged kernels pass all five trials in some fresh processes and fail 1 of 5 in others. | A |
| 7 | The fixed-grid pattern lowers acceptance. | 42.7% of 4,044 rounds with the pattern were accepted, against 66.6% of 8,119 autotuned rounds whose grid depends on the config. | A |
| 8 | The model usually repairs the pattern after a failure. | After a 1-of-5 failure, 384 of 453 next rounds use a grid that depends on the config, and 311 of those were accepted. | A |
| 9 | The pattern is not rewarded with speed. | Accepted rounds with the pattern have a median speedup of 0.63, against 0.60 and 0.68 for the other two groups. | A |
| 10 | AddressSanitizer on the interpreter and an exact bounds oracle agree. | They agree on all 37 flagged cases of the positive control, on all 141 configs of the 36 flagged kernels, and on a random sample of 298 accepted kernels, where neither flags anything. | B |
| 11 | A static rule finds the config-dependent class. | Rule L1 marks 23 of the 24 config-dependent census kernels. It also marks 281 accepted best-round kernels that the census did not flag, so it over-approximates. | B |
| 12 | A kernel with the defect is rejected about three times in four. | The recheck found 420 rejected rounds with a partial config and 150 accepted rounds, both lists complete. Partial configs are 30% to 40% of the configs in these kernels, so they win about twice as often as a uniform draw would give. | B |
| 13 | The CPU detectors miss some errors that crash on a GPU. | Of the 24 rounds that recorded an illegal memory access or a device-side assert, 17 ran on CPU. The bounds oracle flagged 9 of them. Four of the misses run clean and correct on CPU, and also on an A100 with `CUDA_LAUNCH_BLOCKING=1`, so their recorded errors came from elsewhere in the worker. | A |
| 14 | Stale allocator memory is a second way a partial write passes. | On an A100, 7 flagged kernels leave elements unwritten under the poison fill while every plain trial passes. Three of them have no autotuning, so their partial write is deterministic. 18 kernels return a marker value from a freed block in the unwritten region. | A |

## 2. What changed in the reading of the results

1. The first reading was that defective kernels pass because of luck with stale memory. The corpus records show a sharper mechanism. The first trial always passes, and the other four pass only when a covering config won the timing race.
2. The benchmark's five-trial protocol does catch the defect when a partial config wins. It caught at least 453 rounds that way. What it cannot catch is the same defect when a covering config happened to win.
3. The accepted population therefore under-represents the defect. The 0.56% rate describes what slipped through, and not how often the model writes the pattern. The model wrote the fixed-grid pattern in 4,044 of 35,680 rounds, which is 11.3%.

## 3. What is still open

| # | Open question | What resolves it | Cost |
|---|---|---|---|
| 1 | How often does a partial config win on a given GPU, and does stale allocator memory ever make a partial write pass? | On an A100, a partial config won in 17 of 20 cells of the minimal demonstration, and 19 of 29 uninitialized-output kernels left elements unwritten. Stale memory made the partial write pass for 7 kernels. See `results/2026-11-14-gpu.md`. A second GPU type has not been run. | Answered on one GPU. |
| 2 | How many accepted rounds of any index have at least one partial config? | 150 of the 1,691 that ran, out of 1,725. | Answered. |
| 3 | Is the 1-of-5 signature really the partial-write defect? | Yes. 420 of the 444 that ran have a partial config. | Answered. |
| 4 | Does the rate hold outside this corpus? | Two corpora were approved. The traces corpus has 207 rows, never uses autotuning and calls kernels through a wrapper function, so it was not audited. The Makora corpus has 544,028 programs in the same format as KernelBench. In a static scan of 40,000 of them, 25.6% use autotuning and 3.2% match rule L1, against 11.3% of Dr. Kernel rounds. A dynamic audit of 1,500 sampled programs and 400 marked ones is running. | Free CI. No generation cost. |
| 5 | How many more kernels fail at a nearby shape? | A population run over all 6,230 kernels at the native shape and four constant-shifted shapes found about 27 more kernels with a partial write at a nearby shape, which roughly doubles the native-shape count. The figure is a lower bound and still needs the census triage. | Answered in part. |
| 6 | Did the corpus harness run models in training or eval mode? | The released KernelGYM code never calls `eval()`. The decoy rate is 5.7% with and without a `self.training` guard, which fits eval mode. Of 36 rechecked kernels, 3 behave differently between modes. | Low impact. No action. |
| 7 | Why do four kernels crash on a GPU and run clean on CPU? | `236:3`, `481:3`, `3725:4` and `6133:2` ran clean and correct on an A100 with `CUDA_LAUNCH_BLOCKING=1`. The recorded errors most likely came from another kernel in the same worker, not from a gap in the interpreter. | Answered. |

## 4. Decisions

**D1. Headline.** The paper leads with the autotuning blind spot. The claim is that a correctness check after a tuned first call cannot see a config-dependent partial write, and that accepted kernels in a public corpus carry it. The census, the benchmark-reference bugs and the CPU tooling support that claim. The original headline, that accepted generated kernels are often memory-unsafe, is dropped because the census rate is 0.56%.

**D2. GPU spending.** No large GPU budget is justified now. The only GPU work that changes a decision is the one-hour kit in `gpu/`. Its outcomes map to actions as follows.

1. A partial config wins in some repeats. The mechanism is confirmed on hardware and the paper states it without qualification.
2. A partial config never wins on that GPU at any size. The claim narrows to the GPU model used for the corpus, and the paper leans on the corpus records for evidence.
3. A partial write passes without the poison fill. Stale memory is a second mechanism and deserves its own paragraph.

On an A100, outcomes 1 and 3 occurred. The mechanism is stated without qualification, and stale memory gets its own paragraph.

**D3. No audit of all 21,956 accepted rounds.** The lint selects the rounds that can carry the config-dependent class, and those are being rechecked. The remaining classes occur at about 0.1% to 0.2%, so a full audit would add roughly 20 to 40 kernels at four times the census cost. It is deferred unless a reviewer asks for it.

**D4. A second corpus before any generation spend.** Generating kernels from other models costs API or GPU money. The static L1 scan costs nothing. The order is a static scan of public Triton code first, then a dynamic check of marked kernels, and only then a decision on generation.

**D5. Upstream reports, in this order, each after review by the author.**

1. TritonGym: seven reference bugs. The draft is in `reports/tritongym-reference-bugs.md`.
2. Triton: the interpreter treats `and` and `or` on tensors differently from the compiler.
3. Triton: a warning when an autotuned kernel is launched with a grid that is not callable.
4. KernelGYM: run each autotune config separately and fill fresh allocations with a pattern.

**D6. The mitigation is part of the contribution.** Two checks would have rejected all 34 census kernels. The poison fill with one run per autotune config catches the 29 partial writes, and the bounds oracle catches the 5 out-of-bounds reads. Both run on CPU.

**D7. Scope of claims.** Every number is stated for the audited inputs, which are the benchmark's own shape and up to four autotune configs in the census. The paper does not claim the absence of defects elsewhere.

## 5. Threats to validity and their status

| Threat | Status |
|---|---|
| The interpreter differs from compiled code. | Three divergences were found and handled: `and` and `or` on tensors, atomic operations invisible to the sanitizer, and import-time self-tests. On an A100 the compiled kernels reproduce the partial writes, and the four kernels the CPU detectors missed run clean there too. |
| A preloaded sanitizer misses accesses that land in another allocation. | The bounds oracle tracks which allocation each pointer came from and does not depend on the allocator. |
| Nineteen census kernels were classified by signature and not read. | The recheck confirmed a partial config for all 29 uninitialized-output kernels, so the classification no longer rests on the signature alone. |
| The harness ran in a different mode from the audit. | Rechecked in both modes. Three of 36 differ, because their kernel does not launch in training mode. |
| The static rule over-approximates. | It marks kernels whose tuned parameter is consumed by a loop inside the kernel, including four human-written TritonBench references that are safe. Only the dynamic check per config counts as a finding. |
| 117 kernels could not be audited. | 102 are timeouts on large inputs. They are 1.9% of the population and are reported as exclusions. |
| One corpus from one model family. | Open. The Makora audit is running. See D4. |
| The CPU detectors have limited recall. | Against 17 runnable kernels that crashed on a GPU, the bounds oracle flagged 9. The out-of-bounds count of 5 among accepted kernels is therefore a lower bound. |
| Related work on stale memory in kernel evaluation exists. | An industry report from Wafer describes a kernel that read uninitialized memory and passed a tolerance check. It proposes a bitwise determinism check and clears the allocator cache between timing iterations. Whether that catches the config-dependent class has not been tested here, and the paper has to position against it. |

## 6. What the numbers that were pending decided

1. **Accepted rounds with a partial config.** The rule was that a count above 50 keeps D1. The final count is 150, so D1 stands.
2. **Rejected 1-of-5 rounds with a partial config.** The rule was that a share above 80% validates the signature. The share is 94.6%, so claim 5 stands.
3. **Sanitizer against the bounds oracle on the pilot sample.** The rule was that any kernel flagged by the oracle alone triggers a second census. There is none among 298 kernels, so the census numbers stand. A population run with the oracle at nearby shapes is running for a different reason, which is open question 5.
4. **GPU kit.** The rules were outcomes 1 to 3 of D2. Outcomes 1 and 3 occurred on an A100, so claims 4 to 6 gain direct GPU evidence, claim 13 moves to grade A and claim 14 is added.

## 7. Order of work for the next two weeks

1. Done on an A100. A second GPU type, such as a T4, would show whether the winning config differs between GPUs.
2. Fold in the Makora audit, the nearby-shape population run and the sample of rejected rounds when they finish.
3. Review and file the TritonGym report and the two Triton drafts in `reports/`.
4. Decide on a further corpus only if the Makora result is ambiguous.
5. Write the paper around D1, with the census and the reference bugs as supporting sections.
