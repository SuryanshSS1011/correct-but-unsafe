# Draft report: memory-safety bugs in TritonGym reference kernels after commit 6998be6

**Status:** draft for review. Nothing has been filed upstream.

This draft is written for the TritonGym maintainers. It covers reference
kernels under `benchmark/data_v2/triton/` at commit `5d64634`, which already
contains the fix for the 13 bugs reported in issue #1.

## Summary

Seven reference kernels still have a memory-safety bug. Four access memory out
of bounds and three return uninitialized memory. All seven were found by
running the references on CPU under the Triton interpreter with
AddressSanitizer and with a fill pattern for uninitialized allocations, and
all seven were then confirmed by reading the source.

| Op | Class | Root cause |
|---|---|---|
| `embedding` | Out-of-bounds read | The wrapper draws ids from `torch.randint(10, 1000, ...)` and indexes a weight table with 500 rows. |
| `index_select_cat` | Out-of-bounds read | The wrapper hard-codes the index `[0, 2, 4, 6, 8]` and the sources have 5 rows. |
| `token_attn_llama2` | Out-of-bounds read and write | `B_Loc` holds values up to `max_input_len` and is multiplied by `k.stride(0)`, but `K` has only `batch` entries on that axis. The output offset uses `att_out.stride(0)` as the head stride and `att_out.stride(1)` as the position stride, so writes go past the output. |
| `bgmv_shrink_kernel` | Out-of-bounds atomic write | `output_tensor` is sized from `lora_a_weights.size(1)` before the 4D weights are squeezed. With the documented 4D layout the output has one column and the kernel writes `N` columns. The self-test at the bottom of the file triggers this on import. |
| `quantize_kv_transform` | Uninitialized output | Destination indices come from `torch.randint`, so some output slots of a `torch.empty` tensor are never written. |
| `vector_addition_custom` | Uninitialized output | The kernel writes `a.size(0)` elements. The test passes 1024 by 1024 inputs, so 1,024 of 1,048,576 elements of a `torch.empty_like` output are written. The torch reference mirrors this and also returns `torch.empty_like`. |
| `attn_fwd_causal` | Uninitialized output | The store mask keeps the first 96 of 128 head-dimension columns of a `torch.empty_like` output. |

## Why it matters for the benchmark

1. For `vector_addition_custom` the correctness oracle compares uninitialized memory with uninitialized memory. The result of the comparison depends on allocator state.
2. For the out-of-bounds cases, a generated kernel that is correct and in bounds can disagree with the reference.
3. `bgmv_shrink_kernel` writes out of bounds every time the reference file is imported.

## How to reproduce

```
git clone https://github.com/SuryanshSS1011/correct-but-unsafe
pip install torch==2.14.1 triton==3.8.0 tilelens==3.1.1
python controls/tritongym/run_control.py --checkout <TritonGym at 5d64634> --label post \
    --ops embedding,index_select_cat,token_attn_llama2,bgmv_shrink_kernel,quantize_kv_transform,vector_addition_custom,attn_fwd_causal \
    --detectors plain,bounds,asan,perturb,poison --out out.json
```

No GPU is needed.

## Three reports that were not bugs

`argmax`, `masked_select` and `mean_reduction` combine masks with Python `and`.
The compiler makes that elementwise. The Triton interpreter does not, so these
kernels read out of bounds under `TRITON_INTERPRET=1` only. They are correct on
a GPU. Writing `&` would make them behave the same in both modes.
