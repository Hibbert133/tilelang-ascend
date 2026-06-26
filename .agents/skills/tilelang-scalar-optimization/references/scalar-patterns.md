# Scalar-To-Tile Optimization Patterns

These patterns are generalized from successful TileLang-Ascend kernel commits. Use them when scalar code is in the vector-core hot path and the data layout is regular enough for tile operations.

## Pattern 1: Serial Gather -> Tile Gather

**Scene:** A `T.serial(vector_data_tile_m)` loop reads one row's scalar from a grouped UB layout and writes a temporary vector.

**Replace with:**

```python
T.reinterpretcast(offset_u32_ub, offset_i32_ub, "uint32_t")
T.tile.arith_progression(offset_i32_ub, base_offset, stride_bytes, vector_data_tile_m)
T.tile.gather(dst_ub, src_ub, offset_u32_ub, 0)
```

**Why it helps:** It converts row-by-row scalar loads into one vector gather. This reduces scalar issue pressure and makes downstream tile ops consume contiguous UB vectors.

**Check carefully:** Offsets are byte offsets for the generated gather path in the observed kernel. For `int32`/`float32`, use `element_index * 4` and `row_stride_elements * 4` when following the same API pattern.

## Pattern 2: Scalar Max Chain -> Tile Max Tree

**Scene:** Per-row scalar code computes max4 with `T.alloc_var` and `if` chains.

**Scalar form:**

```python
e0 = x[row, base + 0]
e1 = x[row, base + 1]
e2 = x[row, base + 2]
e3 = x[row, base + 3]
max_exp = e0
if e1 > max_exp:
    max_exp = e1
if e2 > max_exp:
    max_exp = e2
if e3 > max_exp:
    max_exp = e3
```

**Tile form:**

```python
T.tile.max(max01_ub, e0_ub, e1_ub)
T.tile.max(max23_ub, e2_ub, e3_ub)
T.tile.max(max_exp_ub, max01_ub, max23_ub)
T.tile.add(out_exp_ub, max_exp_ub, -6)
T.tile.max(out_exp_ub, out_exp_ub, 0)
```

**Why it helps:** The reduction arity is fixed and row-parallel, so max4 should be a small vector tree rather than many scalar branches.

## Pattern 3: Serial Relative-Exp Math -> Tile Arithmetic

**Scene:** For every row, code computes `relative_exp = input_exp - out_exp + 127`, then builds float scale bits.

**Replace with:**

```python
T.tile.sub(relative_exp_ub, input_exp_ub, out_exp_ub)
T.tile.add(relative_exp_ub, relative_exp_ub, 127)
T.tile.bitwise_lshift(relative_bits_ub, relative_exp_ub, 23)
T.reinterpretcast(relative_sf_ub, relative_bits_ub, "float")
```

**Why it helps:** This keeps exponent math in vector lanes and avoids scalar per-row register work.

## Pattern 4: Column Seed -> Broadcast-To-Tile -> Tile Compute

**Scene:** A `(M,)` or `(M, 1)` scale vector must multiply a `(M, K)` data tile.

**Use:**

```python
T.tile.broadcast(relative_sf_tile_ub, relative_sf_col_ub, axis=1)
T.tile.cast(x_out_ub, x_in_ub, mode="CAST_NONE", count=tile_elem_count)
T.tile.mul(x_out_ub, x_out_ub, relative_sf_tile_ub)
```

**Why it helps:** Broadcasting once lets `cast/mul/store` operate on full tiles. Avoid manually filling each column in `T.serial` unless the broadcast API cannot express the layout.

## Pattern 5: Row-Major Or Transposed UB Layout To Match Vector Access

**Scene:** The hot path repeatedly remaps `(group, inner)` into row/column scalar indices before each compute tile.

**Use:** Store intermediate exponent or scale data in the layout consumed by vector ops, for example `(vector_data_tile_m, num_in_sf_per_block_k)` for row-wise gather and per-row broadcast.

**Why it helps:** A one-time layout choice can remove scalar remap loops from every output tile.

## Pattern 6: Shape-Specific Fast Path With Safe Fallback

**Scene:** The common case has fixed arity/layout, such as `in_sf_block=(1,32)`, `out_sf_block=(1,128)`, row-major scaling factors, and max4 reduction.

**Use:** Add a guarded fast path for the common shape and leave the generic path unchanged:

```python
use_fast_path = (
    use_vid_local_sf
    and not in_config.use_tma_aligned_col_major_sf
    and not out_config.use_tma_aligned_col_major_sf
    and in_sf_block_m == 1
    and out_sf_block_m == 1
    and num_in_sf_per_out_sf_k == 4
)
```

**Why it helps:** The fast path can use fixed max4, fixed gather stride, and simpler UB layout without risking less common layouts.

## Pattern 7: Scalar Layout Parameter Choice

**Scene:** Runtime performance is dominated by tile count and padding, not a single instruction sequence.

**Use:** Keep this scalar logic outside the kernel body. Prefer layout choices that reduce padded work and tile count, with explicit tie-breaks when needed:

```python
block_k = min(candidates, key=lambda candidate: (align_up(hidden, candidate), -candidate))
```

**Why it helps:** Choosing a better tile geometry can reduce all downstream scalar/vector/MTE work.

## Anti-Patterns

- Do not treat current dirty worktree experiments as proven patterns.
- Do not replace a serial loop with a larger serial loop just because it moved to another macro.
- Do not use destination slicing in `T.tile.broadcast` or combined big copies unless there is a passing precision test for that exact API use.
- Do not add a shape-specific fast path without preserving fallback behavior.
- Do not optimize after a precision failure and call the result successful; fix correctness first.
