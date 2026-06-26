---
name: tilelang-scalar-optimization
description: Use when optimizing TileLang-Ascend kernels with scalar-heavy hot paths, serial gather/index loops, scalar max/min/reduce chains, broadcast-to-tile patterns, or vector-scalar arithmetic that should be converted to T.tile operations. 适用于把 serial/标量逻辑改成 tile gather、tile broadcast、tile max/add/sub/mul 等向量化模式，并要求基于已通过精度的历史提交验证。
---

# TileLang Scalar Optimization

## When To Use

Use this skill when a TileLang-Ascend kernel is correct but profiling or generated Ascend C shows scalar-heavy work in the hot path:

- `T.serial` loops copying one row/column/scalar at a time.
- Scalar `T.alloc_var`, `if`, `T.max`, or per-row reduce chains replacing what could be tile ops.
- Serial gather/index remap from UB before vector compute.
- Manual column expansion before `cast/mul/store` instead of broadcast-to-tile.
- Fast-path opportunities for fixed layout, block shape, or reduction arity.

Do not use uncommitted experiments or failed precision/segfault attempts as success evidence. Prefer committed versions whose tests were known to pass.

## Workflow

1. Establish the correctness baseline first. Do not optimize scalar code while the kernel is already failing precision.
2. Inspect successful history before changing code:
   - `git log --oneline -- <kernel_file>`
   - `git diff <old_good> <new_good> -- <kernel_file>`
3. Classify the scalar hotspot using [scalar-patterns.md](references/scalar-patterns.md).
4. Apply the smallest matching vectorization pattern, keeping fallback paths intact.
5. Re-run precision before trusting performance. If precision fails, debug the transformation; do not silently replace it with a different optimization.
6. Profile generated Ascend C and confirm scalar work moved to tile/vector instructions rather than just relocating the scalar loop.

## References

- For reusable scalar-to-tile patterns, read [scalar-patterns.md](references/scalar-patterns.md).
- For concrete evidence from `per_block_cast_lossless_kernel.py`, read [per-block-cast-evidence.md](references/per-block-cast-evidence.md).

## Guardrails

- Keep scalar arithmetic for compile-time layout choice, loop bounds, offsets, and fast-path guards.
- Move runtime per-element or per-row scalar work into UB tile operations when the data shape is regular.
- Verify API usage from existing examples or TileLang source before inventing signatures.
- Keep generic fallback paths when adding shape/layout-specific fast paths.
