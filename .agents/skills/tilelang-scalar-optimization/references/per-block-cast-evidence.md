# per_block_cast_lossless_kernel Evidence

This file records useful scalar optimization evidence from committed `examples/quant/per_block_cast_lossless_kernel.py` history. These commits are used as examples because they were saved as successful performance iterations for the same operator. Re-check precision when porting any pattern.

## Useful Commits

```text
247700c per_block_cast_lossless_kernel V0625 65us
bafb4ac per_block_cast_lossless_kernel V0624 96us
be00477 per_block_cast_lossless_kernel V0623 150us
a81fe05 per_block_cast_lossless_kernel V0623 200us
25882bc per_block_cast_lossless_kernel V0615 230us
8060d42 per_block_cast_lossless_kernel V0615 250us
126afc2 per_block_cast_lossless_kernel V2
3dc7808 per_block_cast_lossless_kernel V1
```

Use these commands to inspect the exact changes:

```bash
git log --oneline -- examples/quant/per_block_cast_lossless_kernel.py
git diff --unified=80 8060d42 25882bc -- examples/quant/per_block_cast_lossless_kernel.py
git diff --unified=80 25882bc a81fe05 -- examples/quant/per_block_cast_lossless_kernel.py
git diff --unified=80 a81fe05 be00477 -- examples/quant/per_block_cast_lossless_kernel.py
git diff --unified=80 be00477 bafb4ac -- examples/quant/per_block_cast_lossless_kernel.py
git diff --unified=80 bafb4ac 247700c -- examples/quant/per_block_cast_lossless_kernel.py
```

## 8060d42 -> 25882bc: Tile Geometry Scalar Tuning

Observed useful change: `block_m = max(out_config.sf_block[0], 64)` became `block_m = max(out_config.sf_block[0], 128)`.

General pattern: scalar layout parameters outside the kernel can reduce per-kernel tile count and amortize scalar/vector overhead. Treat block size changes as scalar optimization when they reduce loop iterations without changing math.

## 25882bc -> a81fe05: Guarded Fast Path

Observed useful change: introduced `use_local_max4_fast_path` for the common non-packed, non-TMA, row-major, `1x32 -> 1x128` case.

General pattern: keep generic logic intact, but add a fast path when shape/layout invariants make scalar reduction arity fixed. This enables specialized gather, max4, and broadcast sequences.

## a81fe05 -> be00477: Better Scalar Expressions In Existing Loop

Observed useful change: inside `load_input_sf_exp_local_max4`, scalar `alloc_var + if` max4 became expression-level `T.max`:

```python
max01 = T.max(e0, e1)
max23 = T.max(e2, e3)
max_exp = T.max(max01, max23)
out_exp = T.max(max_exp - 6, 0)
```

General pattern: when a full tile rewrite is not ready, simplify scalar branch chains first. It reduces generated scalar control flow and prepares the code for tile max conversion.

## be00477 -> bafb4ac: Serial Gather To Tile Gather

Observed useful change: `apply_relative_sf_tiles_pair_local_exp_tile` added offset UB buffers and used `T.tile.arith_progression` plus `T.tile.gather` to replace row-wise scalar indexing.

General pattern:

```python
T.tile.arith_progression(offset_i32_ub, base, stride, vector_data_tile_m)
T.tile.gather(input_exp_ub, x_sf_exp_i32_grouped_ub, offset_u32_ub, 0)
T.tile.sub(relative_exp_ub, input_exp_ub, out_exp_col_ub)
```

This is the core `serial gather -> tile gather` pattern.

## bafb4ac -> 247700c: Row-Major Exp UB And Fused Max4

Observed useful changes:

- Added row-major exponent buffer: `(vector_data_tile_m, num_in_sf_per_block_k)`.
- Loaded input sf exponent once with `reinterpretcast` and `T.tile.bitwise_rshift`.
- Gathered `e0/e1/e2/e3` columns by row with tile gather.
- Replaced grouped scalar max/reduce with tile max4:

```python
T.tile.max(max01_ub, e0_ub, e1_ub)
T.tile.max(max23_ub, e2_ub, e3_ub)
T.tile.max(max_exp_ub, max01_ub, max23_ub)
T.tile.add(out_exp_col_ub, max_exp_ub, -6)
T.tile.max(out_exp_col_ub, out_exp_col_ub, 0)
```

- Built relative scale via tile arithmetic and broadcast:

```python
T.tile.sub(relative_exp_ub, input_exp_ub, out_exp_col_ub)
T.tile.add(relative_exp_ub, relative_exp_ub, 127)
T.tile.bitwise_lshift(relative_bits_ub, relative_exp_ub, 23)
T.reinterpretcast(relative_sf_ub, relative_bits_ub, "float")
T.tile.broadcast(relative_sf_tile_ub, relative_sf_ub, axis=1)
```

General pattern: once the UB layout matches vector access, combine gather, tile arithmetic, broadcast, and tile compute in one fast local path.

## What Not To Generalize From Later Experiments

Several uncommitted experiments tried output double buffering, four-way buffers, broadcast slicing, and larger combined copies. Some produced precision failures, segfaults, or poor performance. Keep them out of success guidance unless a later committed version proves correctness and speed.
