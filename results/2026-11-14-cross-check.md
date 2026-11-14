# Cross-check against a published verifier, 14 November 2026

This file compares the census of this study with the per-kernel results that
another group published for the same corpus. The question was whether an
existing verifier already catches the kernels that return uninitialized
output.

## 1. The other audit

"A Contract-Grade Verifier for LLM-Generated GPU Kernels" is arXiv 2608.12700.
Its public repository, `RishiShah99/lethe`, holds `results/audit_drkernel.json`
and `results/audit_rows.jsonl.gz`. The audit covers 3,134 trajectories of
`hkust-nlp/drkernel-coldstart-8k` and runs twelve gates on each kernel. One
gate, ORD-02, calls the kernel five times on the same input and requires
byte-identical outputs, with earlier outputs kept alive.

## 2. Matching the two audits

1. Every one of the 3,134 ids is a `uuid` of the corpus.
2. For 2,330 ids the `final_speedup` in their rows equals the speedup of the best round in the corpus, so both audits looked at the same kernel. For the other 804 no round matches, and they are left out of nothing below, because every kernel used in section 3 was matched individually.

## 3. Results

**The 29 uninitialized-output kernels of the census:**

| Step | Kernels |
|---|---|
| In the other audit's sample | 17 |
| Failed to run in its harness | 4 |
| Reached its gates | 13 |
| Failed its determinism gate ORD-02 | 1 |
| Failed its value-parity gate CMP-01 | 5 |
| Failed at least one gate | 9 |
| Passed all twelve gates | 4 |

The four that passed every gate are `4270`, `6897`, `7485` and `3278`. All 13
were confirmed as the same kernel by speedup and round.

**The 5 out-of-bounds kernels of the census:** Four reached the gates. Three
failed at least one gate, and `160` passed all twelve.

**The other direction:** The other audit reports 137 determinism failures. One
is in the uninitialized-output set of the census and one is in its
out-of-bounds set.

**By kernel group, among accepted best-round kernels that reached the gates:**

| Group | ORD-02 failures | Rate | Interval |
|---|---|---|---|
| Fixed-grid pattern | 5 of 128 | 3.9% | 1.7% to 8.8% |
| Other autotuned | 9 of 280 | 3.2% | 1.7% to 6.0% |
| No autotuning | 81 of 1,362 | 5.9% | 4.8% to 7.3% |

## 4. Reading

1. A repeated-call determinism check does not reliably catch a config-dependent partial write. It caught 1 of 13 here.
2. The determinism failures that the other audit reports are a different population. They are not concentrated in the fixed-grid kernels, and 135 of 137 are outside the census defects.
3. A twelve-gate verifier accepted 4 of 13 kernels that return uninitialized output and 1 of 4 that read out of bounds. A check of every autotune config on a pre-filled output would have rejected all of them.

## 5. Limits

1. The sample is small. Thirteen kernels support a direction and not a rate.
2. The other audit ran on its own hardware and software, and its harness changes the input shape for some gates. Why its determinism gate passed a given kernel is not known. A covering config may have won there.
3. The other audit's driver script is not in its public repository, so its exact call sequence was inferred from the gate code and the result rows.
