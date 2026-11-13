## Static lint over 44600 rounds, 35680 with feedback, 8920 distinct tasks

| Rounds with feedback | Count | Accepted |
|---|---|---|
| All | 35680 | 21956 of 35680, 61.5% (61.0 to 62.0) |
| Uses triton.autotune | 12163 | 7129 of 12163, 58.6% (57.7 to 59.5) |
| L1: autotuned kernel with a fixed grid | 4044 | 1725 of 4044, 42.7% (41.1 to 44.2) |
| Autotuned, grid is a function of the config | 8119 | 5404 of 8119, 66.6% (65.5 to 67.6) |
| No autotune | 23517 | 14827 of 23517, 63.0% (62.4 to 63.7) |

### Passed trials among rounds that compiled, ran every trial and failed correctness

| Group | Rounds | Passed exactly 1 of 5 | Passed 0 of 5 | Passed 2 to 4 |
|---|---|---|---|---|
| L1 | 675 | 453 of 675, 67.1% (63.5 to 70.5) | 134 | 88 |
| Autotuned, not L1 | 301 | 35 of 301, 11.6% (8.5 to 15.7) | 128 | 138 |
| No autotune | 630 | 18 of 630, 2.9% (1.8 to 4.5) | 534 | 78 |

### Identical code for the same task evaluated more than once

| Group | Duplicate groups | Groups with both a pass and a fail |
|---|---|---|
| L1 | 0 | n/a |
| Autotuned, not L1 | 0 | n/a |
| No autotune | 0 | n/a |

### Decoy rate by training guard, among rounds that compiled

| Group | Rounds | Recorded as decoy |
|---|---|---|
| Mentions self.training | 563 | 32 of 563, 5.7% (4.1 to 7.9) |
| No mention | 34618 | 1959 of 34618, 5.7% (5.4 to 5.9) |

### Errors recorded by the harness, among rounds with feedback

| Recorded error | Rounds | Of those accepted | With L1 | With autotune |
|---|---|---|---|---|
| illegal_memory_access | 21 of 35680, 0.1% (0.0 to 0.1) | 0 | 4 | 9 |
| device_side_assert | 23 of 35680, 0.1% (0.0 to 0.1) | 0 | 4 | 9 |
| index_out_of_range | 47 of 35680, 0.1% (0.1 to 0.2) | 0 | 1 | 2 |
| cuda_error_other | 23 of 35680, 0.1% (0.0 to 0.1) | 0 | 4 | 9 |
| segfault | 0 of 35680, 0.0% (0.0 to 0.0) | 0 | 0 | 0 |
| out_of_memory | 12 of 35680, 0.0% (0.0 to 0.1) | 0 | 0 | 3 |
| timeout | 429 of 35680, 1.2% (1.1 to 1.3) | 0 | 47 | 148 |
| shape_mismatch | 45 of 35680, 0.1% (0.1 to 0.2) | 0 | 3 | 3 |
| output_mismatch | 1606 of 35680, 4.5% (4.3 to 4.7) | 0 | 675 | 976 |
| nan_or_inf | 562 of 35680, 1.6% (1.5 to 1.7) | 0 | 46 | 96 |

Rounds where the GPU run recorded an illegal memory access, a device-side assert or a segmentation fault: 24.

### Accepted best-round population: 6230 kernels

- autotune: 1418 of 6230, 22.8% (21.7 to 23.8)
- L1: 304 of 6230, 4.9% (4.4 to 5.4)
- training_guard: 109 of 6230, 1.7% (1.5 to 2.1)
- is_cuda_guard: 4901 of 6230, 78.7% (77.6 to 79.7)
- boolop_in_kernel: 38 of 6230, 0.6% (0.4 to 0.8)
- Census uninitialized-output kernels that L1 marks: 23 of 29. Not marked: ['1797_r3', '2118_r3', '4191_r4', '6852_r1', '7520_r1', '7962_r1']
- Census out-of-bounds kernels that use autotune: ['160_r2', '825_r3']
- Accepted best-round kernels marked by L1: 304. Of these the census flagged 23.
- Accepted rounds of any round index marked by L1: 1725
