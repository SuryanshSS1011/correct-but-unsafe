# Track 2 feasibility: dropped index-bounds contracts, 14 November 2026

This file records the feasibility count for the second research track. The
idea is that a PyTorch operator rejects an index outside its range, while a
generated kernel that replaces it can compute an address from that index and
read or write out of bounds. The count below is static. No kernel was executed.

## 1. PyTorch rejects every out-of-range index on CPU

Each of these raised on an out-of-range or negative index with torch 2.14.1.

| Operator | Result on a bad index |
|---|---|
| embedding | IndexError |
| index_select | IndexError |
| gather | RuntimeError, out of bounds |
| take | IndexError |
| scatter_ (a write) | RuntimeError, out of bounds |
| index_put_ and index_add_ (writes) | IndexError or RuntimeError |
| nll_loss and cross_entropy | IndexError, target out of bounds |
| one_hot | RuntimeError |
| advanced indexing | IndexError |

The contract exists, so a kernel that drops it would turn a rejected input into
an out-of-bounds access. The matching CUDA behaviour still needs a GPU check,
which is part of the GPU work below.

## 2. Static count over the whole corpus

The count ran in CI over all 8,920 tasks, which is run 43580342224. An earlier
run over the 2,576 accepted-kernel tasks held locally gave 58 candidates and 23
with a write, and the full corpus scales that up.

| Quantity | Count |
|---|---|
| Tasks whose reference uses an index-style operator | 437 |
| Of those, tasks that also feed integer inputs | 281 |
| Accepted rounds whose kernel computes an address from a loaded value | 1,245 |
| Best-round kernels that do so | 355 |
| Best-round kernels that do so in a store or atomic | 126 |
| Best-round candidates that both replace an index operator and compute a data-dependent address | 276 |
| Of those, candidates with a data-dependent write or a write operator | 87 |

Index operators in the references were embedding 211, gather 92, index_select
63, one_hot 47, nll_loss or cross_entropy 15, scatter 13 and index_put 1.

## 3. Decision

The gate was that at least 20 confirmed cases make a section and fewer than 5
make a paragraph. There are 276 static candidates and 87 with a write across the
corpus, so the track clears the bar to pursue. These are candidates and not
confirmed accesses. Confirmation needs the differential oracle that feeds an
index the reference rejects and checks whether the candidate's bounds oracle
flags an access. That step is the next piece of work.

## 4. What is not done

1. No candidate has been confirmed to go out of bounds on an adversarial index. That is the dynamic differential, and it was not run.
2. The 87 write candidates are the priority, because an out-of-bounds write is a stronger primitive than a read.
3. The CUDA half of section 1 is unchecked. Whether a device-side assert fires for an out-of-range index, and whether it is compiled out in some builds, needs a GPU.
