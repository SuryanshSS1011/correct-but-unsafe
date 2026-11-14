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

## 2. Static count over the local corpus subset

The count ran over the 2,576 accepted-kernel tasks held locally. The full
corpus has 8,920 tasks, so these figures are a floor.

| Quantity | Count |
|---|---|
| Tasks whose reference uses an index-style operator | 96 |
| Of those, tasks that also feed integer inputs | 70 |
| Accepted rounds whose kernel computes an address from a loaded value | 298 |
| Best-round kernels that do so | 79 |
| Best-round kernels that do so in a store or atomic | 30 |
| Best-round candidates that both replace an index operator and compute a data-dependent address | 58 |
| Of those, candidates with a data-dependent write or a write operator | 23 |

Index operators in the references were embedding 39, gather 27, one_hot 13,
nll_loss or cross_entropy 8, index_select 7 and scatter 4.

## 3. Decision

The gate was that at least 20 confirmed cases make a section and fewer than 5
make a paragraph. There are 58 static candidates and 23 with a write, on a
subset, so the track clears the bar to pursue. These are candidates and not
confirmed accesses. Confirmation needs the differential oracle that feeds an
index the reference rejects and checks whether the candidate's bounds oracle
flags an access. That step is the next piece of work.

## 4. What is not done

1. The count is on the local subset. The full corpus needs one static pass in CI, which executes nothing.
2. No candidate has been confirmed to go out of bounds on an adversarial index. That is the dynamic differential, and it was not run.
3. The write candidates are the priority, because an out-of-bounds write is a stronger primitive than a read.
