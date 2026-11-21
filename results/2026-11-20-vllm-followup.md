# Cross-request leak in vLLM and SGLang: follow-up, 20 and 21 November 2026

This file extends `results/2026-11-19-vllm.md`. That run placed a severe
version of the corpus defect, three quarters of each activation unwritten, in
vLLM's SiluAndMul and found that one request's output depended on earlier
requests. Its limits were that B's output was nonsense, so a user would
notice, and that it covered one model, one GPU type, one engine and no run with
the allocator cache off. This run addresses each.

Everything ran on Penn State Roar Collab: vLLM 0.34.0 with torch
2.14.0+cu130, SGLang 0.5.9 with torch 2.9.1+cu130, driver 580.178, on an
A100-PCIE-40GB unless the GPU is named. The raw files are in
`results/raw/roar-vllm-followup/`, one directory per job, and
`python gpu/vllm_leak/summarize.py results/raw/roar-vllm-followup/<set>`
prints the verdicts. The method, victims, request B and verdicts are those of
the first file, with three victims-then-B trials per victim and length.

## 1. The defects

| Kernel | What it leaves unwritten | Corpus analogue |
|---|---|---|
| `defective` | Three quarters of every output (grid for a block of 1024, block of 256) | The fixed-grid kernels of the first run |
| `floor` | The last `n % 1024` elements, part of the last token's row, 768 for Qwen2.5-0.5B | A grid computed with floor division |
| `tail` | The last element of the output | `2487_r3`, which leaves one element |
| `rowtail` | The last element of every row | An off-by-one in a column mask |
| `correct` | Nothing | Control |

## 2. One unwritten element is enough

With Qwen2.5-0.5B in eager mode with sequential requests, `tail` and `floor`
made B's output a function of the last victim in all 8 cells, 4 each, with
either allocator setting. With `tail`, B's output was fluent:

| Victim before B | B's output with `tail` |
|---|---|
| V1, 24 tokens | " Explain in one sentence how rain forms. Rain forms when water vapor from the atmosphere" |
| V2, 24 tokens | " and describe the process of evaporation. Rain forms when water vapor from the atmosphere" |
| V1, 400 tokens | " Rain forms when water from the atmosphere falls from the sky and falls to the ground" |
| Correct kernel | " Rain forms when water vapor in the atmosphere condenses into liquid d…" |

B's one unwritten element per layer was identical in every pair of trials
after the same victim and different in every pair after different victims, 72
of 72 across the prefill, for both lengths. The A40 gave the same result.
`rowtail`, one element per row, behaved the same, 648 of 648 different and 0 of
1,296 same.

## 3. CUDA graphs pad the batch, which hides a tail defect but not a row defect

With CUDA graphs, `tail` and `floor` showed no variation in any of 16 cells.
Logging the shapes the op was called with showed why. In eager mode B's 9-token
prefill calls the op with 9 rows. With graphs it never does: it runs at the
captured size of 16, so B's last real row is row 9 of 16 and the unwritten tail
falls in padding. `rowtail` has a hole in every row, real or padding, and with
graphs it made B's output depend on earlier requests in all 8 cells, in 3 of
them as a function of the last victim alone. A defect at the end of a buffer is
therefore masked by padding in graph mode, and a defect inside each row is not.

## 4. Other models, another GPU and another engine

Cells where B's output depended on earlier requests, out of the cells that
were not NaN, with the default allocator:

| Set | `defective` | `floor` | `tail` | `rowtail` | `correct` |
|---|---|---|---|---|---|
| Qwen2.5-0.5B, A40 | 8 of 8 | | 2 of 8 | pending | 0 of 8 |
| Qwen2.5-7B | 5 of 6 | | 1 of 8 | 8 of 8 | 0 of 8 |
| TinyLlama-1.1B, raw prompt | 7 of 8 | | 1 of 8 | | 0 of 8 |
| TinyLlama-1.1B, chat template | 6 of 8 | | 5 of 8 | 6 of 8 | 0 of 8 |
| SGLang, Qwen2.5-0.5B | 6 of 6 | 7 of 7 | 8 of 8 | 8 of 8 | 0 of 8 |

The correct kernel varied in none of the 44 control cells. Without the chat
template TinyLlama answers B with the end-of-sequence token, so only its first
log-probabilities carry a signal. The chat-template runs fix that.

Where a defect did not leak, the recording runs show that B's unwritten
elements held the same values after both victims. This happened for TinyLlama
with the chat template in eager mode with sequential requests, with every
defect, and for Qwen2.5-7B with `tail` and after long victims with
`defective`. Whether a hole holds a
victim's data depends on which buffer the allocator hands back, which differs
between models.

**In SGLang the leak is invisible in the text.** With `tail` and `rowtail` in
eager mode with sequential requests, B's tokens equaled the correct kernel's in
6 of 6 trials, yet its top-5 log-probabilities varied between trials, with 6
distinct sets in 6 trials for `rowtail` at both lengths and 3 and 5 for `tail`. B's
first-token log-probability with `rowtail` was −0.200, −0.141, −0.160, −0.128,
−0.144 and −0.138 across trials alternating V1 and V2. A user sees the right
answer, and a client that requests log-probabilities still sees another
request's influence.

## 5. Turning the allocator cache off does not remove the dependence

With `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, vLLM cannot capture CUDA graphs and
fails at startup (`cudaErrorStreamCaptureInvalidated`), so the setting only
applies in eager mode. There, B's output still varied with earlier requests in
all 4 `floor` cells, all 4 `rowtail` cells and 2 `defective` cells, and never
as a function of the last victim alone. The correct kernel varied in none.
B's buffers moved between trials, so the recording runs compared them call by
call. Most unwritten elements were not zero, and they differed between trials
even after the same victim, 47 of 54 at layer 0 with `rowtail`. Without the
cache, a hole receives memory from the driver that earlier work in the process
left behind, not a clean page and not a predictable copy of the last request.

## 6. Limits

The defects were placed in the engines for the experiment. Their kernels do not
have them. The runs show that one request's input changes another's output and
the memory it reads, not that B can read a specific value of the victim's back.
Two `rowtail` cells with expandable segments, and several `defective` cells,
were NaN from the first position. The shapes runs used one repeat, so their
verdicts only show that the two outputs differed. The A40 `rowtail` set is
still running and is pending in the table above.
