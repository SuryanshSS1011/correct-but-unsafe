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
| 5 | Partial configs do win on real GPUs. | In KernelGYM's own records, 453 of 675 failing rounds with the fixed-grid pattern passed exactly 1 of 5 trials. The rate is 67.1%. It is 11.6% for other autotuned kernels and 2.9% for kernels without autotuning. A CPU recheck of those 453 rounds found a partial config in 420 of the 444 that ran, which is 94.6%. A partial config won in 15 of 20 size and repeat cells of the minimal demonstration on a T4, 16 of 20 on an A100 and 18 of 20 on an H100. | A |
| 6 | Acceptance of such a kernel is not determined by its code. | Trajectory 8676 kept the same fixed grid and the same five configs in rounds 3 and 4. Round 3 passed 1 of 5 trials and round 4 passed 5 of 5. The CPU replay gives the same two signatures for the two possible kinds of winner. On an A100, six flagged kernels pass all five trials in some fresh processes and fail 1 of 5 in others. | A |
| 7 | The fixed-grid pattern lowers acceptance. | 42.7% of 4,044 rounds with the pattern were accepted, against 66.6% of 8,119 autotuned rounds whose grid depends on the config. | A |
| 8 | The model usually repairs the pattern after a failure. | After a 1-of-5 failure, 384 of 453 next rounds use a grid that depends on the config, and 311 of those were accepted. | A |
| 9 | The pattern is not rewarded with speed. | Accepted rounds with the pattern have a median speedup of 0.63, against 0.60 and 0.68 for the other two groups. | A |
| 10 | AddressSanitizer on the interpreter and an exact bounds oracle agree. | They agree on all 37 flagged cases of the positive control, on all 141 configs of the 36 flagged kernels, and on a random sample of 298 accepted kernels, where neither flags anything. | B |
| 11 | A static rule finds the config-dependent class. | Rule L1 marks 23 of the 24 config-dependent census kernels. It also marks 281 accepted best-round kernels that the census did not flag, so it over-approximates. | B |
| 12 | Among lint-marked kernels with a partial config, about three in four were rejected. | The recheck found 420 rejected one-of-five rounds with a partial config and 150 accepted rounds, both lists complete. The ratio covers lint-marked rounds and one-of-five rejections only. Partial configs are 30% to 40% of the configs in these kernels, so they win about twice as often as a uniform draw would give. | B |
| 13 | The CPU detectors miss some errors that crash on a GPU. | Of the 24 rounds that recorded an illegal memory access or a device-side assert, 17 ran on CPU. The bounds oracle flagged 9 of them. Four of the misses run clean on CPU, and also on a T4, an A100 and an H100 with `CUDA_LAUNCH_BLOCKING=1`, so their recorded errors came from elsewhere in the worker. On the H100 one of them, `6133_r2`, fails the comparison because the TF32 reference convolution flips the sign of near-zero outputs. The kernel itself matches a float64 reference. | A |
| 14 | Stale allocator memory is a second way a partial write passes. | On an A100, 7 flagged kernels leave elements unwritten under the poison fill while every plain trial passes, and 5 do so on a T4 and on an H100. Three of them have no autotuning, so their partial write is deterministic. 18 kernels return a marker value from a freed block in the unwritten region on an A100. | A |
| 15 | The defect is far more common among rejected rounds. | A random sample of 1,200 compiled-and-rejected Dr. Kernel rounds, all eight shards, had a partial write in 75 of the 467 rounds that run under the interpreter, which is 16.1% with an interval of 13.0% to 19.7%, and either defect in 80, which is 17.1%, against 0.56% in the accepted population. See `results/rejected_sample.json`. | B |
| 16 | The defect appears in a second corpus, and the lint selects it there too. | A dynamic audit of Makora found a partial write in 3 of 230 uniformly sampled programs that ran, which is 1.3%, and in 28 of 100 lint-marked programs that ran, which is 28.0% with an interval of 20.1% to 37.5%. The enrichment is about 20 times, the same direction as in Dr. Kernel. | B |
| 17 | Whether a kernel leaks depends on the GPU type. | With the same code, inputs and software, 9 flagged kernels never left an element unwritten in 12 fresh processes on a T4 and did on an A100 or an H100. Two others leak only on the T4. The autotune winner differs between the GPUs. The out-of-bounds read of `7350_r1` faults on the A100 and the H100 and passes every trial on the T4. See `results/2026-11-17-gpu.md`. | A |
| 18 | The autotuner's disk cache removes the covered first call. | With `TRITON_CACHE_AUTOTUNING=1` on an A100, all 273 later processes loaded the cached winner without tuning. Plain runs that passed only the first trial fell from 46 to 0, and runs that passed no trial rose from 6 to 60. After a restart, the first call of a deployed kernel is already partial. | A |
| 19 | Compute Sanitizer's initcheck can miss the defect under PyTorch's default allocator. | On an A100, `2118_r3` and `5239_r2` left 3,840 and 3,686,400 output elements unwritten on every call, passed every plain trial and drew no initcheck error with the caching allocator. With `PYTORCH_NO_CUDA_MEMORY_CACHING=1` initcheck reported 17,280 and 13,824,000 errors in the harness's comparison. For three other kernels it reported errors in both settings, and the controls drew none. See `results/2026-11-19-initcheck.md`. | A |
| 20 | Inside one process, a partial write returns the previous request's data. | On an A100, the minimal kernel at its partial block left 786,432 of 1,048,576 output elements unwritten, and all of them held a freed victim request's values, in 3 of 3 repeats each with the caching allocator, with unrelated allocations in between and with expandable segments. Without the cache and across processes the 4 MiB region came back zeroed in this test, but the vLLM runs without the cache found non-zero leftovers, so zeroing is not guaranteed and may depend on allocation size. The test is a synthetic server, not a serving engine. See `results/2026-11-19-leak.md`. | A |
| 21 | In vLLM, a kernel with the defect makes one user's request depend on another's. | With the corpus's fixed grid placed in vLLM 0.34's SiluAndMul on an A100, a request's output was a function of the last victim in 4 of the 8 sequential cells, 4 configurations at two victim lengths, and NaN in 2. Batched cells are not counted: they run the victim in the same batch as B, and batching moves the correct kernel's log-probabilities by up to 0.44 nats. The correct kernel varied in none of its sequential cells. Victims alternate in a fixed order, so the last victim is confounded with the one two requests back; `gpu/vllm_leak/run_infer.py` randomizes the order. In eager mode its unwritten activations were identical after the same victim and 99.5% different after another, so they are a function of the previous user's input. The defect was placed for the experiment. See `results/2026-11-19-vllm.md`. | A |
| 22 | The cross-request dependence survives small defects, other models, another GPU and another engine. | In vLLM on an A100, a kernel that left only the last element of its output unwritten made a request's fluent output a function of the previous user's prompt in eager mode, and its one unwritten element per layer differed in 72 of 72 comparisons after different victims and 0 of 144 after the same one. CUDA graphs pad the batch so that tail falls in padding, but a hole in every row leaked in 8 of 8 graph cells. It also leaked on an A40, on Qwen2.5-7B and TinyLlama, and in SGLang 0.5.9, where the request's text matched the correct kernel's while its log-probabilities varied. Turning the allocator cache off left a non-zero, history-dependent hole and disables CUDA graphs. The correct kernel varied in none of 44 control cells. The SGLang result rests on variation alone, with no recorded buffers. See `results/2026-11-20-vllm-followup.md` and its corrections. | A |

## 2. What changed in the reading of the results

1. The first reading was that defective kernels pass because of luck with stale memory. The corpus records show a sharper mechanism. The first trial always passes, and the other four pass only when a covering config won the timing race.
2. The benchmark's five-trial protocol does catch the defect when a partial config wins. It caught at least 453 rounds that way. What it cannot catch is the same defect when a covering config happened to win.
3. The accepted population therefore under-represents the defect. The 0.56% rate describes what slipped through, and not how often the model writes the pattern. The model wrote the fixed-grid pattern in 4,044 of 35,680 rounds, which is 11.3%.

## 3. What is still open

| # | Open question | What resolves it | Cost |
|---|---|---|---|
| 1 | How often does a partial config win on a given GPU, and does stale allocator memory ever make a partial write pass? | A partial config won in 15, 16 and 18 of 20 cells of the minimal demonstration on a T4, an A100 and an H100. 15, 19 and 21 of the 34 flagged kernels left elements unwritten, and stale memory made the partial write pass for 5, 7 and 5. See `results/2026-11-14-gpu.md` and `results/2026-11-17-gpu.md`. | Answered on three GPUs. |
| 2 | How many accepted rounds of any index have at least one partial config? | 150 of the 1,691 that ran, out of 1,725. | Answered. |
| 3 | Is the 1-of-5 signature really the partial-write defect? | Yes. 420 of the 444 that ran have a partial config. | Answered. |
| 4 | Does the rate hold outside this corpus? | Yes. The defect appears in Makora, which is `makora-ai/triton-gpu-latency` with 544,028 programs. A static scan of 40,000 found 25.6% autotuned and 3.2% matching rule L1, against 11.3% of Dr. Kernel rounds. A dynamic audit found a partial write in 3 of 230 uniformly sampled programs that ran, which is 1.3%, and in 28 of 100 lint-marked programs, which is 28.0%, an enrichment of about 20 times. The other approved corpus, the traces set, has 207 rows, never autotunes and uses a wrapper function, so it was not audited. | Answered. Free CI. |
| 5 | How many more kernels fail at a nearby shape? | A population run over all 6,230 kernels at the native shape and four constant-shifted shapes found about 27 more kernels with a partial write at a nearby shape, which roughly doubles the native-shape count. The figure is a lower bound and still needs the census triage. | Answered in part. |
| 6 | Did the corpus harness run models in training or eval mode? | The released KernelGYM code never calls `eval()`. The decoy rate is 5.7% with and without a `self.training` guard, which fits eval mode. Of 36 rechecked kernels, 3 behave differently between modes. | Low impact. No action. |
| 7 | Why do four kernels crash on a GPU and run clean on CPU? | `236:3`, `481:3`, `3725:4` and `6133:2` ran clean and correct on an A100 with `CUDA_LAUNCH_BLOCKING=1`. The recorded errors most likely came from another kernel in the same worker, not from a gap in the interpreter. | Answered. |

## 4. Decisions

**D1. Headline.** The paper leads with the autotuning blind spot. The claim is that a correctness check after a tuned first call cannot see a config-dependent partial write, and that accepted kernels in a public corpus carry it. The census, the benchmark-reference bugs and the CPU tooling support that claim. The original headline, that accepted generated kernels are often memory-unsafe, is dropped because the census rate is 0.56%.

**D2. GPU spending.** No large GPU budget is justified now. The only GPU work that changes a decision is the one-hour kit in `gpu/`. Its outcomes map to actions as follows.

1. A partial config wins in some repeats. The mechanism is confirmed on hardware and the paper states it without qualification.
2. A partial config never wins on that GPU at any size. The claim narrows to the GPU model used for the corpus, and the paper leans on the corpus records for evidence.
3. A partial write passes without the poison fill. Stale memory is a second mechanism and deserves its own paragraph.

On a T4, an A100 and an H100, outcomes 1 and 3 occurred. The mechanism is stated without qualification, and stale memory gets its own paragraph. The kernels that leak differ between the three GPUs, which claim 17 records.

**D3. No audit of all 21,956 accepted rounds.** The lint selects the rounds that can carry the config-dependent class, and those are being rechecked. The remaining classes occur at about 0.1% to 0.2%, so a full audit would add roughly 20 to 40 kernels at four times the census cost. It is deferred unless a reviewer asks for it.

**D4. A second corpus before any generation spend.** Generating kernels from other models costs API or GPU money. The static L1 scan costs nothing. The order is a static scan of public Triton code first, then a dynamic check of marked kernels, and only then a decision on generation.

**D5. Upstream reports, in this order, each after review by the author.**

1. TritonGym: seven reference bugs. Filed by the user as TritonGym issue 5 on 16 November 2026, with no maintainer reply as of 21 November.
2. Triton: the interpreter treats `and` and `or` on tensors differently from the compiler.
3. Triton: a warning when an autotuned kernel is launched with a grid that is not callable.
4. KernelGYM: run each autotune config separately and fill fresh allocations with a pattern.

**D6. The mitigation is part of the contribution.** Two checks would have rejected all 34 census kernels. The poison fill with one run per autotune config catches the 29 partial writes, and the bounds oracle catches the 5 out-of-bounds reads. Both run on CPU. Because the first check runs every config, it does not depend on which config wins on a given GPU, and claim 17 shows that this varies. A check on one CI GPU type does not have that property. Compute Sanitizer's initcheck is not a substitute under the default allocator, which hands the output a recycled block, as claim 19 records.

**D7. Scope of claims.** Every number is stated for the audited inputs, which are the benchmark's own shape and up to four autotune configs in the census. The paper does not claim the absence of defects elsewhere.

## 5. Threats to validity and their status

| Threat | Status |
|---|---|
| The interpreter differs from compiled code. | Three divergences were found and handled: `and` and `or` on tensors, atomic operations invisible to the sanitizer, and import-time self-tests. On an A100 the compiled kernels reproduce the partial writes, and the four kernels the CPU detectors missed run clean there too. |
| A preloaded sanitizer misses accesses that land in another allocation. | The bounds oracle tracks which allocation each pointer came from and does not depend on the allocator. |
| Nineteen census kernels were classified by signature and not read. | The recheck confirmed a partial config for all 29 uninitialized-output kernels, so the classification no longer rests on the signature alone. |
| The harness ran in a different mode from the audit. | Rechecked in both modes. Three of 36 differ, because their kernel does not launch in training mode. |
| The static rule over-approximates. | It marks kernels whose tuned parameter is consumed by a loop inside the kernel, including four human-written TritonBench references that are safe. Only the dynamic check per config counts as a finding. |
| 117 kernels could not be audited. | 102 are timeouts on large inputs. They are 1.6% of the population, and all 117 exclusions are 1.9% and are reported as exclusions. |
| One corpus from one model family. | Addressed. The defect appears in Makora at 1.3% of uniform programs and 28.0% of lint-marked ones, the same enrichment direction as Dr. Kernel. See D4. |
| The CPU detectors have limited recall. | Against 17 runnable kernels that crashed on a GPU, the bounds oracle flagged 9. The out-of-bounds count of 5 among accepted kernels is therefore a lower bound. |
| Related work on stale memory in kernel evaluation exists. | An industry report from Wafer describes a kernel that read uninitialized memory and passed a tolerance check. It proposes a bitwise determinism check and clears the allocator cache between timing iterations. Whether that catches the config-dependent class has not been tested here, and the paper has to position against it. |

## 6. What the numbers that were pending decided

1. **Accepted rounds with a partial config.** The rule was that a count above 50 keeps D1. The final count is 150, so D1 stands.
2. **Rejected 1-of-5 rounds with a partial config.** The rule was that a share above 80% validates the signature. The share is 94.6%, so claim 5 stands.
3. **Sanitizer against the bounds oracle on the pilot sample.** The rule was that any kernel flagged by the oracle alone triggers a second census. There is none among 298 kernels, so the census numbers stand. A population run with the oracle at nearby shapes is running for a different reason, which is open question 5.
4. **GPU kit.** The rules were outcomes 1 to 3 of D2. Outcomes 1 and 3 occurred on an A100, so claims 4 to 6 gain direct GPU evidence, claim 13 moves to grade A and claim 14 is added. They occurred again on a T4 and an H100, which added claims 17 and 18.

## 7. Order of work for the next two weeks

1. Done on a T4, an A100 and an H100, and with the disk cache on an A100. The winning config, and so the leak, differs between GPUs.
2. The enriched Makora comparison is done, at 28.0% of lint-marked programs against 1.3% of uniform ones.
3. Review and file the two Triton drafts in `reports/`. The TritonGym report was filed as issue 5.
4. No further corpus is needed. The Makora result is clear.
5. Write the paper around D1, with the census and the reference bugs as supporting sections.
