# Draft reports for the Triton project

**Status:** drafts for review. Nothing has been filed upstream. Both were checked against Triton 3.8.0 on CPU only.

## 1. The interpreter treats `and` and `or` on tensors differently from the compiler

**Summary:** under `TRITON_INTERPRET=1`, `a and b` on two non-scalar tensors evaluates to `b`. The compiler turns the same expression into an elementwise logical AND. A kernel that combines masks with `and` is therefore correct when compiled and accesses memory out of bounds when interpreted.

**Reproduction:** `reports/triton_boolop_repro.py` copies a 3 by 8 tensor with 8 by 8 blocks and the mask `row_mask and col_mask`. Under the interpreter it prints `elements written past the output: 40`. The 40 elements are rows 3 to 7, which the row mask should exclude. The compiled path was not run here. Its code generator maps a boolean operator on tensors to `logical_and` in `visit_BoolOp`, so the expected count there is 0.

**Cause:** the interpreter patches `tensor.__bool__` to return true for any tensor with more than one element, and its source rewriter does not rewrite boolean operators. Python then evaluates `a and b` by truthiness.

**Effect seen in practice:** three TritonGym reference kernels, `argmax`, `masked_select` and `mean_reduction`, read or write out of bounds under the interpreter for this reason alone. Tools built on the interpreter report these as memory errors.

**Possible fix:** rewrite `ast.BoolOp` in the interpreter's `ASTTransformer` to the same `logical_and` and `logical_or` calls the compiler emits. `cbu/target.py` contains a working version in `_match_compiler_boolops`.

## 2. A warning for an autotuned kernel launched with a fixed grid

**Summary:** when a kernel is autotuned over a block size and launched with a grid that is a plain tuple, a config with a smaller block covers only part of the data. The first call does not show this, because tuning runs every config on the caller's buffers. Later calls run only the cached config.

**Evidence:** in a public corpus of model-generated kernels, 4,044 of 35,680 evaluated rounds have this shape. Among the rounds of that kind that failed correctness after running five trials, 453 of 675 passed exactly the first trial.

**Proposal:** in `Autotuner.run`, emit a warning when all of the following hold.

1. More than one config remains after pruning.
2. The configs differ in at least one keyword argument other than `num_warps`, `num_stages`, `num_ctas` and `maxnreg`.
3. The `grid` argument is not callable.

The warning would say that the grid does not depend on the tuned parameters and would suggest `grid = lambda meta: ...`. Kernels that consume the tuned parameter in an inner loop would see a false warning, so the text should say how to silence it.

**Minimal demonstration:** `gpu/minimal_repro.py`. It needs a GPU and has not been run yet.
