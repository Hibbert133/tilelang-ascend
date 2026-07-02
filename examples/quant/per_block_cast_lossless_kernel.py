import os
from dataclasses import replace

import torch
import tilelang
import tilelang.language as T


from .my_common import *


tilelang.cache.clear_cache()

DEFAULT_IN_SF_BLOCK = (1, 32)
DEFAULT_OUT_SF_BLOCK = (1, 128)
VEC_NUM = 2
NUM_ELEMENTS_PER_VECTOR = 16384
INPUT_DTYPE = "bfloat16"
OUTPUT_DTYPE = "float32"

pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False, tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,}


def _derive_cast_layout(hidden: int, in_config: CastInputConfig, out_config: CastOutputConfig,) -> dict[str, int]:
    assert in_config.dtype == INPUT_DTYPE and out_config.dtype == OUTPUT_DTYPE, ("lossless mode only supports bf16 -> fp32 conversion currently")
    assert in_config.with_sf, "lossless mode requires both input and output scaling factors"
    assert is_power_of_two(in_config.sf_block[1]) and is_power_of_two(out_config.sf_block[1]), ("block_k must be power of 2 for lossless mode")
    assert (
        out_config.sf_block[0] % in_config.sf_block[0] == 0
        and out_config.sf_block[1] % in_config.sf_block[1] == 0
    ), "Output block size must be multiple of input block size"

    block_m = max(out_config.sf_block[0], 128)
    block_k = max(out_config.sf_block[1], 512)
    if (
        not in_config.use_packed_ue8m0
        and not out_config.use_packed_ue8m0
        and not in_config.use_tma_aligned_col_major_sf
        and not out_config.use_tma_aligned_col_major_sf
        and in_config.sf_block == DEFAULT_IN_SF_BLOCK
        and out_config.sf_block == DEFAULT_OUT_SF_BLOCK
    ):
        block_k_candidates = (256, 512)
        block_k = min(
            (
                candidate
                for candidate in block_k_candidates
                if candidate % in_config.sf_block[1] == 0
                and candidate % out_config.sf_block[1] == 0
            ),
            key=lambda candidate: (align_up(hidden, candidate), -candidate),
        )

    assert block_m % out_config.sf_block[0] == 0
    assert block_k % out_config.sf_block[1] == 0
    assert hidden > 0

    num_in_sf_per_block_m = block_m // in_config.sf_block[0]
    num_in_sf_per_block_k = block_k // in_config.sf_block[1]
    num_out_sf_per_block_m = block_m // out_config.sf_block[0]
    num_out_sf_per_block_k = block_k // out_config.sf_block[1]
    num_in_sf_per_out_sf_m = out_config.sf_block[0] // in_config.sf_block[0]
    num_in_sf_per_out_sf_k = out_config.sf_block[1] // in_config.sf_block[1]
    num_in_sf_per_out_sf = num_in_sf_per_out_sf_m * num_in_sf_per_out_sf_k

    return {
        "block_m": block_m,
        "block_k": block_k,
        "num_in_sf_per_block_m": num_in_sf_per_block_m,
        "num_in_sf_per_block_k": num_in_sf_per_block_k,
        "num_out_sf_per_block_m": num_out_sf_per_block_m,
        "num_out_sf_per_block_k": num_out_sf_per_block_k,
        "num_in_sf_per_out_sf_m": num_in_sf_per_out_sf_m,
        "num_in_sf_per_out_sf_k": num_in_sf_per_out_sf_k,
        "num_in_sf_per_out_sf": num_in_sf_per_out_sf,
    }


@tilelang.jit(out_idx=[2, 3], pass_configs=pass_configs)
def get_per_block_cast_lossless_kernel(
    hidden: int,
    block_m: int,
    block_k: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    logical_hidden: int = -1,
    in_sf_block_m: int = DEFAULT_IN_SF_BLOCK[0],
    in_sf_block_k: int = DEFAULT_IN_SF_BLOCK[1],
    out_sf_block_m: int = DEFAULT_OUT_SF_BLOCK[0],
    out_sf_block_k: int = DEFAULT_OUT_SF_BLOCK[1],
):
    assert block_m > 0 and block_k > 0
    assert block_m % out_sf_block_m == 0
    assert block_k % out_sf_block_k == 0
    assert block_m % VEC_NUM == 0
    assert out_sf_block_m % in_sf_block_m == 0
    assert out_sf_block_k % in_sf_block_k == 0
    if logical_hidden < 0:
        logical_hidden = hidden
    assert 0 < logical_hidden <= hidden

    num_tokens = T.symbolic("num_tokens")

    m_num = T.ceildiv(num_tokens, block_m)
    n_num = T.ceildiv(hidden, block_k)
    num_in_sf_per_block_m = block_m // in_sf_block_m
    num_in_sf_per_block_k = block_k // in_sf_block_k
    num_out_sf_per_block_m = block_m // out_sf_block_m
    num_out_sf_per_block_k = block_k // out_sf_block_k
    num_in_sf_per_out_sf_m = out_sf_block_m // in_sf_block_m
    num_in_sf_per_out_sf_k = out_sf_block_k // in_sf_block_k
    num_out_sf_per_block = num_out_sf_per_block_m * num_out_sf_per_block_k
    num_in_sf_per_out_sf = num_in_sf_per_out_sf_m * num_in_sf_per_out_sf_k
    num_in_sf_per_out_sf_aligned = align_up(num_in_sf_per_out_sf, 8)
    compact_sf_row_stride = align_up(num_in_sf_per_block_k, 8)
    compact_sf_row_stride_bytes = compact_sf_row_stride * 4
    num_in_sf_per_out_sf_bytes = num_in_sf_per_out_sf_k * 4
    num_pipeline_pairs = num_out_sf_per_block_k * 2
    num_initial_prefetch_stages = min(4, num_pipeline_pairs)
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    out_sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    vector_data_tile_m = block_m // VEC_NUM
    num_in_sf_per_vector_m = vector_data_tile_m // in_sf_block_m
    use_vid_local_sf = not in_config.use_packed_ue8m0 and not out_config.use_packed_ue8m0
    use_local_max4_fast_path = (
        use_vid_local_sf
        and not in_config.use_tma_aligned_col_major_sf
        and not out_config.use_tma_aligned_col_major_sf
        and in_sf_block_m == 1
        and out_sf_block_m == 1
        and num_in_sf_per_out_sf_m == 1
        and num_in_sf_per_out_sf_k == 4
    )
    use_compact_pattern_pipeline = use_local_max4_fast_path and 1 <= num_out_sf_per_block_k <= 2
    local_num_out_sf_per_block_m = vector_data_tile_m // out_sf_block_m
    local_num_out_sf_per_block = local_num_out_sf_per_block_m * num_out_sf_per_block_k
    if in_config.use_packed_ue8m0:
        assert num_in_sf_per_block_k % 4 == 0
        x_sf_load_shape = (num_in_sf_per_block_k // 4, num_in_sf_per_block_m * 4)
        x_sf_load_dtype = "uint8"
    elif in_config.use_tma_aligned_col_major_sf:
        x_sf_load_shape = (num_in_sf_per_block_k, num_in_sf_per_block_m)
        x_sf_load_dtype = "float32"
    else:
        x_sf_load_shape = (num_in_sf_per_block_m, num_in_sf_per_block_k)
        x_sf_load_dtype = "float32"
    if in_config.use_tma_aligned_col_major_sf:
        x_sf_load_local_shape = (num_in_sf_per_block_k, num_in_sf_per_vector_m)
    else:
        x_sf_load_local_shape = (num_in_sf_per_vector_m, num_in_sf_per_block_k)

    @T.macro
    def load_input_sf_exp_block(x_sf_exp_grouped_ub, x_sf_load_ub, x_sf, sf_m, sf_k):
        if in_config.use_packed_ue8m0:
            T.copy(
                x_sf[
                    sf_k // 4 : sf_k // 4 + num_in_sf_per_block_k // 4,
                    sf_m * 4 : sf_m * 4 + num_in_sf_per_block_m * 4,
                ],
                x_sf_load_ub,
            )
            for flat in T.serial(num_out_sf_per_block * num_in_sf_per_out_sf):
                group = flat // num_in_sf_per_out_sf
                inner = flat % num_in_sf_per_out_sf
                out_i = group // num_out_sf_per_block_k
                out_j = group % num_out_sf_per_block_k
                reduce_i = inner // num_in_sf_per_out_sf_k
                reduce_j = inner % num_in_sf_per_out_sf_k
                in_i = out_i * num_in_sf_per_out_sf_m + reduce_i
                in_j = out_j * num_in_sf_per_out_sf_k + reduce_j
                v = T.Cast("float32", T.Cast("int32", x_sf_load_ub[in_j // 4, in_i * 4 + in_j % 4]))
                x_sf_exp_grouped_ub[group, inner] = v
        else:
            # Gather float32 input SF into the same grouped layout first, then
            # use tile bitwise ops to extract exponent in batch.
            x_sf_value_grouped_ub = T.alloc_ub((num_out_sf_per_block, num_in_sf_per_out_sf_aligned), "float32")
            x_sf_bits_grouped_ub = T.alloc_ub((num_out_sf_per_block, num_in_sf_per_out_sf_aligned), "int32")
            x_sf_exp_mask_ub = T.alloc_ub((num_out_sf_per_block, num_in_sf_per_out_sf_aligned), "int32")

            T.tile.fill(x_sf_value_grouped_ub, 0.0)
            T.tile.fill(x_sf_exp_mask_ub, 0xFF)
            if in_config.use_tma_aligned_col_major_sf:
                T.copy(
                    x_sf[
                        sf_k : sf_k + num_in_sf_per_block_k,
                        sf_m : sf_m + num_in_sf_per_block_m,
                    ],
                    x_sf_load_ub,
                )
                for flat in T.serial(num_out_sf_per_block * num_in_sf_per_out_sf):
                    group = flat // num_in_sf_per_out_sf
                    inner = flat % num_in_sf_per_out_sf
                    out_i = group // num_out_sf_per_block_k
                    out_j = group % num_out_sf_per_block_k
                    reduce_i = inner // num_in_sf_per_out_sf_k
                    reduce_j = inner % num_in_sf_per_out_sf_k
                    in_i = out_i * num_in_sf_per_out_sf_m + reduce_i
                    in_j = out_j * num_in_sf_per_out_sf_k + reduce_j
                    x_sf_value_grouped_ub[group, inner] = x_sf_load_ub[in_j, in_i]
            else:
                T.copy(
                    x_sf[
                        sf_m : sf_m + num_in_sf_per_block_m,
                        sf_k : sf_k + num_in_sf_per_block_k,
                    ],
                    x_sf_load_ub,
                )
                for flat in T.serial(num_out_sf_per_block * num_in_sf_per_out_sf):
                    group = flat // num_in_sf_per_out_sf
                    inner = flat % num_in_sf_per_out_sf
                    out_i = group // num_out_sf_per_block_k
                    out_j = group % num_out_sf_per_block_k
                    reduce_i = inner // num_in_sf_per_out_sf_k
                    reduce_j = inner % num_in_sf_per_out_sf_k
                    in_i = out_i * num_in_sf_per_out_sf_m + reduce_i
                    in_j = out_j * num_in_sf_per_out_sf_k + reduce_j
                    x_sf_value_grouped_ub[group, inner] = x_sf_load_ub[in_i, in_j]

            T.reinterpretcast(x_sf_bits_grouped_ub, x_sf_value_grouped_ub, "int")
            T.tile.bitwise_rshift(x_sf_bits_grouped_ub, x_sf_bits_grouped_ub, 23)
            T.tile.bitwise_and(x_sf_bits_grouped_ub, x_sf_bits_grouped_ub, x_sf_exp_mask_ub)
            T.tile.cast(x_sf_exp_grouped_ub, x_sf_bits_grouped_ub, mode="CAST_NONE", count=num_out_sf_per_block * num_in_sf_per_out_sf_aligned)

    @T.macro
    def load_input_sf_exp_local(x_sf_exp_grouped_ub, x_sf_exp_i32_grouped_ub, x_sf_load_ub, x_sf, sf_m, sf_k):
        x_sf_value_grouped_ub = T.alloc_ub((local_num_out_sf_per_block, num_in_sf_per_out_sf_aligned), "float32")
        x_sf_bits_grouped_ub = T.alloc_ub((local_num_out_sf_per_block, num_in_sf_per_out_sf_aligned), "int32")
        x_sf_exp_mask_ub = T.alloc_ub((local_num_out_sf_per_block, num_in_sf_per_out_sf_aligned), "int32")
        x_sf_value_compact_ub = T.alloc_ub((local_num_out_sf_per_block, num_in_sf_per_out_sf), "float32")
        T.tile.fill(x_sf_value_compact_ub, 0.0)
        T.tile.fill(x_sf_value_grouped_ub, 0.0)
        T.tile.fill(x_sf_exp_mask_ub, 0xFF)

        if in_config.use_tma_aligned_col_major_sf:
            T.copy(
                x_sf[
                    sf_k : sf_k + num_in_sf_per_block_k,
                    sf_m : sf_m + num_in_sf_per_vector_m,
                ],
                x_sf_load_ub,
            )
        else:
            T.copy(
                x_sf[
                    sf_m : sf_m + num_in_sf_per_vector_m,
                    sf_k : sf_k + num_in_sf_per_block_k,
                ],
                x_sf_load_ub,
            )
        T.set_flag("mte2", "v", 0)
        T.wait_flag("mte2", "v", 0)

        for (group, inner) in T.Parallel(local_num_out_sf_per_block, num_in_sf_per_out_sf):
            out_i = group // num_out_sf_per_block_k
            out_j = group - out_i * num_out_sf_per_block_k
            reduce_i = inner // num_in_sf_per_out_sf_k
            reduce_j = inner - reduce_i * num_in_sf_per_out_sf_k
            in_i = out_i * num_in_sf_per_out_sf_m + reduce_i
            in_j = out_j * num_in_sf_per_out_sf_k + reduce_j
            if in_config.use_tma_aligned_col_major_sf:
                x_sf_value_compact_ub[group, inner] = x_sf_load_ub[in_j, in_i]
            else:
                x_sf_value_compact_ub[group, inner] = x_sf_load_ub[in_i, in_j]

        for group in T.serial(local_num_out_sf_per_block):
            v0 = x_sf_value_compact_ub[group, 0]
            v1 = x_sf_value_compact_ub[group, 1]
            v2 = x_sf_value_compact_ub[group, 2]
            v3 = x_sf_value_compact_ub[group, 3]
            x_sf_value_grouped_ub[group, 0] = v0
            x_sf_value_grouped_ub[group, 1] = v1
            x_sf_value_grouped_ub[group, 2] = v2
            x_sf_value_grouped_ub[group, 3] = v3
            x_sf_value_grouped_ub[group, 4] = v0
            x_sf_value_grouped_ub[group, 5] = v1
            x_sf_value_grouped_ub[group, 6] = v2
            x_sf_value_grouped_ub[group, 7] = v3

        T.reinterpretcast(x_sf_bits_grouped_ub, x_sf_value_grouped_ub, "int")
        T.tile.bitwise_rshift(x_sf_exp_i32_grouped_ub, x_sf_bits_grouped_ub, 23)
        T.tile.bitwise_and(x_sf_exp_i32_grouped_ub, x_sf_exp_i32_grouped_ub, x_sf_exp_mask_ub)
        T.tile.cast(x_sf_exp_grouped_ub, x_sf_exp_i32_grouped_ub, mode="CAST_NONE", count=local_num_out_sf_per_block * num_in_sf_per_out_sf_aligned)

    @T.macro
    def load_input_sf_exp_local_max4(
        x_sf_exp_i32_row_ub,
        out_sf_exp_row_ub,
        x_sf_load_ub,
        x_sf,
        sf_m,
        sf_k,
    ):
        x_sf_bits_load_ub = T.alloc_ub(x_sf_load_local_shape, "int32")

        T.copy(
            x_sf[
                sf_m : sf_m + num_in_sf_per_vector_m,
                sf_k : sf_k + num_in_sf_per_block_k,
            ],
            x_sf_load_ub,
        )
        T.set_flag("mte2", "v", 0)
        T.wait_flag("mte2", "v", 0)

        T.reinterpretcast(x_sf_bits_load_ub, x_sf_load_ub, "int")
        T.tile.bitwise_rshift(x_sf_bits_load_ub, x_sf_bits_load_ub, 23)

        for group in T.serial(local_num_out_sf_per_block):
            row = group // num_out_sf_per_block_k
            out_sf_k = group - row * num_out_sf_per_block_k
            sf_k_base = out_sf_k * 4

            e0 = x_sf_bits_load_ub[row, sf_k_base + 0]
            e1 = x_sf_bits_load_ub[row, sf_k_base + 1]
            e2 = x_sf_bits_load_ub[row, sf_k_base + 2]
            e3 = x_sf_bits_load_ub[row, sf_k_base + 3]

            max01 = T.max(e0, e1)
            max23 = T.max(e2, e3)
            max_exp = T.max(max01, max23)

            out_exp = T.max(max_exp - 6, 0)

            out_sf_exp_row_ub[row, out_sf_k] = out_exp
            x_sf_exp_i32_row_ub[row, sf_k_base + 0] = e0
            x_sf_exp_i32_row_ub[row, sf_k_base + 1] = e1
            x_sf_exp_i32_row_ub[row, sf_k_base + 2] = e2
            x_sf_exp_i32_row_ub[row, sf_k_base + 3] = e3

    @T.macro
    def load_input_sf_exp_local_exp_only(
        x_sf_exp_i32_row_ub,
        x_sf_load_ub,
        x_sf,
        sf_m,
        sf_k,
    ):
        x_sf_bits_load_ub = T.alloc_ub(x_sf_load_local_shape, "int32")

        T.copy(
            x_sf[
                sf_m : sf_m + num_in_sf_per_vector_m,
                sf_k : sf_k + num_in_sf_per_block_k,
            ],
            x_sf_load_ub,
        )
        T.set_flag("mte2", "v", 0)
        T.wait_flag("mte2", "v", 0)

        T.reinterpretcast(x_sf_bits_load_ub, x_sf_load_ub, "int")
        T.tile.bitwise_rshift(x_sf_exp_i32_row_ub, x_sf_bits_load_ub, 23)
        T.pipe_barrier("all")

    @T.macro
    def reduce_output_sf_exp(out_sf_exp_ub, x_sf_exp_grouped_ub, out_sf_exp_flat_ub):
        T.reduce_max(x_sf_exp_grouped_ub, out_sf_exp_flat_ub, dim=-1, clear=True, real_shape=[num_out_sf_per_block, num_in_sf_per_out_sf_aligned])

        T.tile.add(out_sf_exp_flat_ub, out_sf_exp_flat_ub, -6.0)
        T.tile.max(out_sf_exp_flat_ub, out_sf_exp_flat_ub, 0.0)
        T.tile.cast(out_sf_exp_ub, out_sf_exp_flat_ub, mode="CAST_ROUND", count=num_out_sf_per_block)

    @T.macro
    def reduce_output_sf_exp_local(out_sf_exp_ub, x_sf_exp_grouped_ub, out_sf_exp_flat_ub):
        T.reduce_max(x_sf_exp_grouped_ub, out_sf_exp_flat_ub, dim=-1, clear=True, real_shape=[local_num_out_sf_per_block, num_in_sf_per_out_sf_aligned])
        T.tile.add(out_sf_exp_flat_ub, out_sf_exp_flat_ub, -6.0)
        T.tile.max(out_sf_exp_flat_ub, out_sf_exp_flat_ub, 0.0)
        T.tile.cast(out_sf_exp_ub, out_sf_exp_flat_ub, mode="CAST_ROUND", count=local_num_out_sf_per_block)

    @T.macro
    def compute_relative_sf(relative_sf_ub, x_sf_exp_grouped_ub, out_sf_exp_ub, sf_exp_row_offset):
        relative_exp_i32_ub = T.alloc_ub((vector_data_tile_m * num_in_sf_per_block_k,), "int32")
        relative_sf_bits_ub = T.alloc_ub((vector_data_tile_m * num_in_sf_per_block_k,), "int32")
        input_exp_i32_ub = T.alloc_ub((vector_data_tile_m * num_in_sf_per_block_k,), "int32")
        out_sf_exp_brd_ub = T.alloc_ub((vector_data_tile_m * num_in_sf_per_block_k,), "int32")
        for flat in T.serial(vector_data_tile_m * num_in_sf_per_block_k):
            sf_k = flat // vector_data_tile_m
            data_i = flat % vector_data_tile_m
            local_sf_i = data_i // in_sf_block_m
            abs_sf_i = sf_exp_row_offset + local_sf_i
            out_sf_m = abs_sf_i // num_in_sf_per_out_sf_m
            out_sf_k = sf_k // num_in_sf_per_out_sf_k
            reduce_j = sf_k % num_in_sf_per_out_sf_k
            inner_i = abs_sf_i % num_in_sf_per_out_sf_m
            inner = inner_i * num_in_sf_per_out_sf_k + reduce_j
            group = out_sf_m * num_out_sf_per_block_k + out_sf_k
            input_exp_i32_ub[flat] = T.Cast("int32", x_sf_exp_grouped_ub[group, inner])
            out_sf_exp_brd_ub[flat] = out_sf_exp_ub[group]

        for flat in T.Parallel(vector_data_tile_m * num_in_sf_per_block_k):
            relative_exp_i32_ub[flat] = input_exp_i32_ub[flat] - out_sf_exp_brd_ub[flat]
        T.tile.add(relative_exp_i32_ub, relative_exp_i32_ub, 127)
        T.tile.bitwise_lshift(relative_sf_bits_ub, relative_exp_i32_ub, 23)
        T.reinterpretcast(relative_sf_ub, relative_sf_bits_ub, "float")

    @T.macro
    def compute_relative_sf_local(relative_sf_ub, x_sf_exp_grouped_ub, out_sf_exp_ub):
        relative_exp_i32_ub = T.alloc_ub((vector_data_tile_m * num_in_sf_per_block_k,), "int32")
        relative_sf_bits_ub = T.alloc_ub((vector_data_tile_m * num_in_sf_per_block_k,), "int32")
        input_exp_i32_ub = T.alloc_ub((vector_data_tile_m * num_in_sf_per_block_k,), "int32")
        out_sf_exp_brd_ub = T.alloc_ub((vector_data_tile_m * num_in_sf_per_block_k,), "int32")
        for flat in T.serial(vector_data_tile_m * num_in_sf_per_block_k):
            sf_k = flat // vector_data_tile_m
            data_i = flat % vector_data_tile_m
            local_sf_i = data_i // in_sf_block_m
            out_sf_m = local_sf_i // num_in_sf_per_out_sf_m
            out_sf_k = sf_k // num_in_sf_per_out_sf_k
            reduce_j = sf_k % num_in_sf_per_out_sf_k
            inner_i = local_sf_i % num_in_sf_per_out_sf_m
            inner = inner_i * num_in_sf_per_out_sf_k + reduce_j
            group = out_sf_m * num_out_sf_per_block_k + out_sf_k
            input_exp_i32_ub[flat] = T.Cast("int32", x_sf_exp_grouped_ub[group, inner])
            out_sf_exp_brd_ub[flat] = out_sf_exp_ub[group]

        for flat in T.Parallel(vector_data_tile_m * num_in_sf_per_block_k):
            relative_exp_i32_ub[flat] = input_exp_i32_ub[flat] - out_sf_exp_brd_ub[flat]
        T.tile.add(relative_exp_i32_ub, relative_exp_i32_ub, 127)
        T.tile.bitwise_lshift(relative_sf_bits_ub, relative_exp_i32_ub, 23)
        T.reinterpretcast(relative_sf_ub, relative_sf_bits_ub, "float")

    @T.macro
    def store_output_sf_local(out_sf, out_sf_fp32_ub, out_sf_row_offset, out_sf_col_offset):
        if not out_config.use_tma_aligned_col_major_sf:
            out_sf_store_ub = T.alloc_ub((local_num_out_sf_per_block_m, num_out_sf_per_block_k), "float32")
            for flat in T.serial(local_num_out_sf_per_block):
                i = flat // num_out_sf_per_block_k
                j = flat % num_out_sf_per_block_k
                out_sf_store_ub[i, j] = out_sf_fp32_ub[flat]
            T.set_flag("v", "mte3", 0)
            T.wait_flag("v", "mte3", 0)
            T.copy(
                out_sf_store_ub,
                out_sf[
                    out_sf_row_offset:out_sf_row_offset + local_num_out_sf_per_block_m,
                    out_sf_col_offset:out_sf_col_offset + num_out_sf_per_block_k,
                ],
            )
        else:
            out_sf_store_ub = T.alloc_ub((num_out_sf_per_block_k, local_num_out_sf_per_block_m), "float32")
            for flat in T.serial(local_num_out_sf_per_block):
                i = flat // num_out_sf_per_block_k
                j = flat % num_out_sf_per_block_k
                out_sf_store_ub[j, i] = out_sf_fp32_ub[flat]
            T.set_flag("v", "mte3", 0)
            T.wait_flag("v", "mte3", 0)
            T.copy(
                out_sf_store_ub,
                out_sf[
                    out_sf_col_offset:out_sf_col_offset + num_out_sf_per_block_k,
                    out_sf_row_offset:out_sf_row_offset + local_num_out_sf_per_block_m,
                ],
            )

    @T.macro
    def apply_relative_sf_tiles_transposed(relative_sf_ub, x, out, row_offset, col_offset):
        relative_sf_col_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf_tile_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")
        x_in_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_out_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        for sf_k in T.serial(num_in_sf_per_block_k):
            for row in T.serial(vector_data_tile_m):
                rel_idx = sf_k * vector_data_tile_m + row
                relative_sf_col_ub[row, 0] = relative_sf_ub[rel_idx]
            T.tile.broadcast(relative_sf_tile_ub, relative_sf_col_ub, axis=1)
            col_tile_offset = col_offset + sf_k * in_sf_block_k
            T.copy(
                x[
                    row_offset : row_offset + vector_data_tile_m,
                    col_tile_offset : col_tile_offset + in_sf_block_k,
                ],
                x_in_ub,
            )
            T.tile.cast(x_out_ub, x_in_ub, mode="CAST_NONE", count=vector_data_tile_m * in_sf_block_k)
            T.tile.mul(x_out_ub, x_out_ub, relative_sf_tile_ub)
            T.copy(
                x_out_ub,
                out[
                    row_offset : row_offset + vector_data_tile_m,
                    col_tile_offset : col_tile_offset + in_sf_block_k,
                ],
            )



    @T.macro
    def apply_relative_sf_tiles_pair_local_exp_tile_row(
        x_sf_exp_i32_row_ub,
        out_sf_exp_row_ub,
        x,
        out,
        row_offset,
        col_offset,
    ):
        input_exp0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        input_exp1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        out_exp_col_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        xsf_group_row_offset_i32_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        xsf_group_row_offset_u32_ub = T.alloc_ub((vector_data_tile_m,), "uint32")
        out_group_row_offset_i32_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        out_group_row_offset_u32_ub = T.alloc_ub((vector_data_tile_m,), "uint32")

        relative_exp0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_exp1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_bits0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_bits1_ub = T.alloc_ub((vector_data_tile_m,), "int32")

        relative_sf0_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf1_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf_tile0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")
        relative_sf_tile1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")

        x_in0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_out0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)

        tile_elem_count = vector_data_tile_m * in_sf_block_k
        T.reinterpretcast(xsf_group_row_offset_u32_ub, xsf_group_row_offset_i32_ub, "uint32_t")
        T.reinterpretcast(out_group_row_offset_u32_ub, out_group_row_offset_i32_ub, "uint32_t")

        for out_sf_k in T.serial(num_out_sf_per_block_k):
            T.tile.arith_progression(
                out_group_row_offset_i32_ub,
                out_sf_k * 4,
                num_out_sf_per_block_k * 4,
                vector_data_tile_m,
            )
            T.tile.gather(
                out_exp_col_ub,
                out_sf_exp_row_ub,
                out_group_row_offset_u32_ub,
                0,
            )
            for reduce_pair in T.serial(num_in_sf_per_out_sf_k // 2):
                reduce_j0 = reduce_pair * 2
                reduce_j1 = reduce_j0 + 1
                sf_k0 = out_sf_k * num_in_sf_per_out_sf_k + reduce_j0
                sf_k1 = sf_k0 + 1

                T.tile.arith_progression(
                    xsf_group_row_offset_i32_ub,
                    sf_k0 * 4,
                    num_in_sf_per_block_k * 4,
                    vector_data_tile_m,
                )
                T.tile.gather(
                    input_exp0_ub,
                    x_sf_exp_i32_row_ub,
                    xsf_group_row_offset_u32_ub,
                    0,
                )
                T.tile.arith_progression(
                    xsf_group_row_offset_i32_ub,
                    sf_k1 * 4,
                    num_in_sf_per_block_k * 4,
                    vector_data_tile_m,
                )
                T.tile.gather(
                    input_exp1_ub,
                    x_sf_exp_i32_row_ub,
                    xsf_group_row_offset_u32_ub,
                    0,
                )

                # vector arithmetic
                T.tile.sub(relative_exp0_ub, input_exp0_ub, out_exp_col_ub)
                T.tile.sub(relative_exp1_ub, input_exp1_ub, out_exp_col_ub)
                T.tile.add(relative_exp0_ub, relative_exp0_ub, 127)
                T.tile.add(relative_exp1_ub, relative_exp1_ub, 127)

                T.tile.bitwise_lshift(relative_bits0_ub, relative_exp0_ub, 23)
                T.tile.bitwise_lshift(relative_bits1_ub, relative_exp1_ub, 23)

                T.reinterpretcast(relative_sf0_ub, relative_bits0_ub, "float")
                T.reinterpretcast(relative_sf1_ub, relative_bits1_ub, "float")

                T.tile.broadcast(relative_sf_tile0_ub, relative_sf0_ub, axis=1)
                T.tile.broadcast(relative_sf_tile1_ub, relative_sf1_ub, axis=1)

                col_tile_offset0 = col_offset + sf_k0 * in_sf_block_k
                col_tile_offset1 = col_offset + sf_k1 * in_sf_block_k
                T.copy(x[row_offset:row_offset + vector_data_tile_m, col_tile_offset0:col_tile_offset0 + in_sf_block_k], x_in0_ub)
                T.copy(x[row_offset:row_offset + vector_data_tile_m, col_tile_offset1:col_tile_offset1 + in_sf_block_k], x_in1_ub)
                T.set_flag("mte2", "v", 1)
                T.wait_flag("mte2", "v", 1)

                T.tile.cast(x_out0_ub, x_in0_ub, mode="CAST_NONE", count=tile_elem_count)
                T.tile.cast(x_out1_ub, x_in1_ub, mode="CAST_NONE", count=tile_elem_count)

                T.tile.mul(x_out0_ub, x_out0_ub, relative_sf_tile0_ub)
                T.tile.mul(x_out1_ub, x_out1_ub, relative_sf_tile1_ub)

                T.set_flag("v", "mte3", 1)
                T.wait_flag("v", "mte3", 1)
                T.copy(x_out0_ub, out[row_offset:row_offset + vector_data_tile_m, col_tile_offset0:col_tile_offset0 + in_sf_block_k])
                T.copy(x_out1_ub, out[row_offset:row_offset + vector_data_tile_m, col_tile_offset1:col_tile_offset1 + in_sf_block_k])

            if num_in_sf_per_out_sf_k % 2 != 0:
                reduce_j = num_in_sf_per_out_sf_k - 1
                sf_k = out_sf_k * num_in_sf_per_out_sf_k + reduce_j

                for row in T.serial(vector_data_tile_m):
                    out_exp = out_sf_exp_row_ub[row, out_sf_k] - 127
                    relative_exp0_ub[row] = x_sf_exp_i32_row_ub[row, sf_k] - out_exp

                T.tile.bitwise_lshift(relative_bits0_ub, relative_exp0_ub, 23)
                T.reinterpretcast(relative_sf0_ub, relative_bits0_ub, "float")
                T.tile.broadcast(relative_sf_tile0_ub, relative_sf0_ub, axis=1)

                col_tile_offset = col_offset + sf_k * in_sf_block_k
                T.copy(x[row_offset:row_offset + vector_data_tile_m, col_tile_offset:col_tile_offset + in_sf_block_k], x_in0_ub)
                T.set_flag("mte2", "v", 1)
                T.wait_flag("mte2", "v", 1)

                T.tile.cast(x_out0_ub, x_in0_ub, mode="CAST_NONE", count=tile_elem_count)
                T.tile.mul(x_out0_ub, x_out0_ub, relative_sf_tile0_ub)

                T.set_flag("v", "mte3", 1)
                T.wait_flag("v", "mte3", 1)
                T.copy(x_out0_ub, out[row_offset:row_offset + vector_data_tile_m, col_tile_offset:col_tile_offset + in_sf_block_k])

    @T.macro
    def apply_relative_sf_tiles_pair_local_exp_tile_fused_max4(
        x_sf_exp_i32_row_ub,
        x,
        out,
        row_offset,
        col_offset,
    ):
        e0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        e1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        e2_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        e3_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        max01_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        max23_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        max_exp_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        out_exp_col_ub = T.alloc_ub((vector_data_tile_m,), "int32")

        xsf_row_offset_i32_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        xsf_row_offset_u32_ub = T.alloc_ub((vector_data_tile_m,), "uint32")

        relative_exp0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_exp1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_exp2_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_exp3_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_bits0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_bits1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_bits2_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_bits3_ub = T.alloc_ub((vector_data_tile_m,), "int32")

        relative_sf0_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf1_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf2_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf3_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf_tile0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")
        relative_sf_tile1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")
        relative_sf_tile2_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")
        relative_sf_tile3_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")

        x_in0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_out0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)

        tile_elem_count = vector_data_tile_m * in_sf_block_k
        T.reinterpretcast(xsf_row_offset_u32_ub, xsf_row_offset_i32_ub, "uint32_t")

        for out_sf_k in T.serial(num_out_sf_per_block_k):
            sf_k_base = out_sf_k * num_in_sf_per_out_sf_k

            T.tile.arith_progression(xsf_row_offset_i32_ub, (sf_k_base + 0) * 4, num_in_sf_per_block_k * 4, vector_data_tile_m)
            T.tile.gather(e0_ub, x_sf_exp_i32_row_ub, xsf_row_offset_u32_ub, 0)
            T.tile.arith_progression(xsf_row_offset_i32_ub, (sf_k_base + 1) * 4, num_in_sf_per_block_k * 4, vector_data_tile_m)
            T.tile.gather(e1_ub, x_sf_exp_i32_row_ub, xsf_row_offset_u32_ub, 0)
            T.tile.arith_progression(xsf_row_offset_i32_ub, (sf_k_base + 2) * 4, num_in_sf_per_block_k * 4, vector_data_tile_m)
            T.tile.gather(e2_ub, x_sf_exp_i32_row_ub, xsf_row_offset_u32_ub, 0)
            T.tile.arith_progression(xsf_row_offset_i32_ub, (sf_k_base + 3) * 4, num_in_sf_per_block_k * 4, vector_data_tile_m)
            T.tile.gather(e3_ub, x_sf_exp_i32_row_ub, xsf_row_offset_u32_ub, 0)

            T.pipe_barrier("all")
            T.tile.max(max01_ub, e0_ub, e1_ub)
            T.tile.max(max23_ub, e2_ub, e3_ub)
            T.tile.max(max_exp_ub, max01_ub, max23_ub)
            T.tile.add(out_exp_col_ub, max_exp_ub, -6)
            T.tile.max(out_exp_col_ub, out_exp_col_ub, 0)

            T.tile.sub(relative_exp0_ub, e0_ub, out_exp_col_ub)
            T.tile.sub(relative_exp1_ub, e1_ub, out_exp_col_ub)
            T.tile.add(relative_exp0_ub, relative_exp0_ub, 127)
            T.tile.add(relative_exp1_ub, relative_exp1_ub, 127)
            T.tile.bitwise_lshift(relative_bits0_ub, relative_exp0_ub, 23)
            T.tile.bitwise_lshift(relative_bits1_ub, relative_exp1_ub, 23)
            T.reinterpretcast(relative_sf0_ub, relative_bits0_ub, "float")
            T.reinterpretcast(relative_sf1_ub, relative_bits1_ub, "float")
            T.tile.broadcast(relative_sf_tile0_ub, relative_sf0_ub, axis=1)
            T.tile.broadcast(relative_sf_tile1_ub, relative_sf1_ub, axis=1)

            col_tile_offset0 = col_offset + (sf_k_base + 0) * in_sf_block_k
            col_tile_offset1 = col_offset + (sf_k_base + 1) * in_sf_block_k
            T.copy(x[row_offset:row_offset + vector_data_tile_m, col_tile_offset0:col_tile_offset0 + in_sf_block_k], x_in0_ub)
            T.copy(x[row_offset:row_offset + vector_data_tile_m, col_tile_offset1:col_tile_offset1 + in_sf_block_k], x_in1_ub)
            T.set_flag("mte2", "v", 1)
            T.wait_flag("mte2", "v", 1)
            T.tile.cast(x_out0_ub, x_in0_ub, mode="CAST_NONE", count=tile_elem_count)
            T.tile.cast(x_out1_ub, x_in1_ub, mode="CAST_NONE", count=tile_elem_count)
            T.tile.mul(x_out0_ub, x_out0_ub, relative_sf_tile0_ub)
            T.tile.mul(x_out1_ub, x_out1_ub, relative_sf_tile1_ub)
            T.set_flag("v", "mte3", 1)
            T.wait_flag("v", "mte3", 1)
            T.copy(x_out0_ub, out[row_offset:row_offset + vector_data_tile_m, col_tile_offset0:col_tile_offset0 + in_sf_block_k])
            T.copy(x_out1_ub, out[row_offset:row_offset + vector_data_tile_m, col_tile_offset1:col_tile_offset1 + in_sf_block_k])

            T.pipe_barrier("all")
            T.tile.sub(relative_exp2_ub, e2_ub, out_exp_col_ub)
            T.tile.sub(relative_exp3_ub, e3_ub, out_exp_col_ub)
            T.tile.add(relative_exp2_ub, relative_exp2_ub, 127)
            T.tile.add(relative_exp3_ub, relative_exp3_ub, 127)
            T.tile.bitwise_lshift(relative_bits2_ub, relative_exp2_ub, 23)
            T.tile.bitwise_lshift(relative_bits3_ub, relative_exp3_ub, 23)
            T.reinterpretcast(relative_sf2_ub, relative_bits2_ub, "float")
            T.reinterpretcast(relative_sf3_ub, relative_bits3_ub, "float")
            T.tile.broadcast(relative_sf_tile2_ub, relative_sf2_ub, axis=1)
            T.tile.broadcast(relative_sf_tile3_ub, relative_sf3_ub, axis=1)

            col_tile_offset0 = col_offset + (sf_k_base + 2) * in_sf_block_k
            col_tile_offset1 = col_offset + (sf_k_base + 3) * in_sf_block_k
            T.copy(x[row_offset:row_offset + vector_data_tile_m, col_tile_offset0:col_tile_offset0 + in_sf_block_k], x_in0_ub)
            T.copy(x[row_offset:row_offset + vector_data_tile_m, col_tile_offset1:col_tile_offset1 + in_sf_block_k], x_in1_ub)
            T.set_flag("mte2", "v", 1)
            T.wait_flag("mte2", "v", 1)
            T.tile.cast(x_out0_ub, x_in0_ub, mode="CAST_NONE", count=tile_elem_count)
            T.tile.cast(x_out1_ub, x_in1_ub, mode="CAST_NONE", count=tile_elem_count)

            T.tile.mul(x_out0_ub, x_out0_ub, relative_sf_tile2_ub)
            T.tile.mul(x_out1_ub, x_out1_ub, relative_sf_tile3_ub)
            T.set_flag("v", "mte3", 1)
            T.wait_flag("v", "mte3", 1)
            T.copy(x_out0_ub, out[row_offset:row_offset + vector_data_tile_m, col_tile_offset0:col_tile_offset0 + in_sf_block_k])
            T.copy(x_out1_ub, out[row_offset:row_offset + vector_data_tile_m, col_tile_offset1:col_tile_offset1 + in_sf_block_k])
            T.pipe_barrier("all")

    @T.macro
    def compute_two_pattern_max(
        x_sf_exp_ub,
        block_idx,
        xsf_row_offset_i32_ub,
        xsf_row_offset_u32_ub,
        e0_ub,
        e1_ub,
        e2_ub,
        e3_ub,
        max01_ub,
        max23_ub,
        out_exp_ub,
    ):
        sf_k_base_bytes = block_idx * num_in_sf_per_out_sf_bytes

        T.tile.arith_progression(xsf_row_offset_i32_ub, 0, compact_sf_row_stride_bytes, vector_data_tile_m)
        T.pipe_barrier("v")
        T.tile.gather(e0_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, sf_k_base_bytes)
        T.tile.gather(e1_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, sf_k_base_bytes + 4)
        T.tile.gather(e2_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, sf_k_base_bytes + 8)
        T.tile.gather(e3_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, sf_k_base_bytes + 12)
        T.pipe_barrier("v")
        T.tile.max(max01_ub, e0_ub, e1_ub)
        T.tile.max(max23_ub, e2_ub, e3_ub)
        T.pipe_barrier("v")
        T.tile.max(out_exp_ub, max01_ub, max23_ub)
        T.pipe_barrier("v")
        T.tile.add(out_exp_ub, out_exp_ub, -6)
        T.pipe_barrier("v")
        T.tile.max(out_exp_ub, out_exp_ub, 0)
        T.pipe_barrier("v")

    @T.macro
    def load_two_pair_input(
        x,
        x_in0_ub,
        x_in1_ub,
        row_offset,
        col_offset,
        input_event,
    ):
        T.wait_flag("v", "mte2", input_event)
        T.copy(
            x[row_offset : row_offset + vector_data_tile_m, col_offset : col_offset + in_sf_block_k],
            x_in0_ub,
        )
        T.copy(
            x[
                row_offset : row_offset + vector_data_tile_m,
                col_offset + in_sf_block_k : col_offset + 2 * in_sf_block_k,
            ],
            x_in1_ub,
        )
        T.set_flag("mte2", "v", input_event)

    @T.macro
    def compute_two_pair_values(
        e0_ub,
        e1_ub,
        out_exp_ub,
        relative_exp0_ub,
        relative_exp1_ub,
        relative_bits0_ub,
        relative_bits1_ub,
        relative_sf0_view_ub,
        relative_sf1_view_ub,
        relative_sf_tile0_ub,
        relative_sf_tile1_ub,
        x_in0_ub,
        x_in1_ub,
        x_out0_ub,
        x_out1_ub,
        input_event,
        output_event,
    ):
        tile_elem_count = vector_data_tile_m * in_sf_block_k
        T.tile.sub(relative_exp0_ub, e0_ub, out_exp_ub)
        T.tile.sub(relative_exp1_ub, e1_ub, out_exp_ub)
        T.pipe_barrier("v")
        T.tile.add(relative_exp0_ub, relative_exp0_ub, 127)
        T.tile.add(relative_exp1_ub, relative_exp1_ub, 127)
        T.pipe_barrier("v")
        T.tile.bitwise_lshift(relative_bits0_ub, relative_exp0_ub, 23)
        T.tile.bitwise_lshift(relative_bits1_ub, relative_exp1_ub, 23)
        T.pipe_barrier("v")
        T.tile.broadcast(relative_sf_tile0_ub, relative_sf0_view_ub, axis=1)
        T.tile.broadcast(relative_sf_tile1_ub, relative_sf1_view_ub, axis=1)

        T.wait_flag("mte2", "v", input_event)
        T.wait_flag("mte3", "v", output_event)
        T.tile.cast(x_out0_ub, x_in0_ub, mode="CAST_NONE", count=tile_elem_count)
        T.tile.cast(x_out1_ub, x_in1_ub, mode="CAST_NONE", count=tile_elem_count)
        T.pipe_barrier("v")
        T.set_flag("v", "mte2", input_event)
        T.tile.mul(x_out0_ub, x_out0_ub, relative_sf_tile0_ub)
        T.tile.mul(x_out1_ub, x_out1_ub, relative_sf_tile1_ub)
        T.set_flag("v", "mte3", output_event)

    @T.macro
    def store_two_pair_output(
        out,
        x_out0_ub,
        x_out1_ub,
        row_offset,
        col_offset,
        output_event,
    ):
        T.wait_flag("v", "mte3", output_event)
        T.copy(
            x_out0_ub,
            out[row_offset : row_offset + vector_data_tile_m, col_offset : col_offset + in_sf_block_k],
        )
        T.copy(
            x_out1_ub,
            out[
                row_offset : row_offset + vector_data_tile_m,
                col_offset + in_sf_block_k : col_offset + 2 * in_sf_block_k,
            ],
        )
        T.set_flag("mte3", "v", output_event)

    @T.macro
    def apply_relative_sf_tiles_compact_pattern_pipeline(
        x_sf,
        x,
        out,
        sf_m,
        sf_k,
        row_offset,
        col_offset,
    ):
        x_sf_load_ub = T.alloc_ub((vector_data_tile_m, compact_sf_row_stride), "float32")
        x_sf_bits_ub = T.alloc_ub((vector_data_tile_m, compact_sf_row_stride), "int32")
        x_sf_exp_ub = T.alloc_ub((vector_data_tile_m, compact_sf_row_stride), "int32")
        T.reinterpretcast(x_sf_bits_ub, x_sf_load_ub, "int")

        # Two input/output slots form a pair ring: slot 0 serves pairs 0/2 and
        # slot 1 serves pairs 1/3.  Each pair stays as two independent 32-col
        # chunks; widening to 64-col stores made MTE3 much longer.
        x_in00_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in01_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in10_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in11_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_out00_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out01_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out10_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out11_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)

        xsf_row_offset_i32_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        xsf_row_offset_u32_ub = T.alloc_ub((vector_data_tile_m,), "uint32")
        e0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        e1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        e2_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        e3_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        max01_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        max23_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        out_exp_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_exp0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_exp1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_bits0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_bits1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_sf0_view_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf1_view_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf_tile0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")
        relative_sf_tile1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")

        T.reinterpretcast(xsf_row_offset_u32_ub, xsf_row_offset_i32_ub, "uint32_t")
        T.reinterpretcast(relative_sf0_view_ub, relative_bits0_ub, "float")
        T.reinterpretcast(relative_sf1_view_ub, relative_bits1_ub, "float")

        T.copy(x_sf[sf_m : sf_m + num_in_sf_per_vector_m, sf_k : sf_k + num_in_sf_per_block_k], x_sf_load_ub)
        T.set_flag("mte2", "v", 4)

        # Seed the ring after the padded SF copy, whose implementation may use
        # event 0 while clearing the 32-byte-aligned UB rows.
        for slot in T.unroll(2):
            T.set_flag("v", "mte2", slot)
            T.set_flag("mte3", "v", slot)

        load_two_pair_input(x, x_in00_ub, x_in01_ub, row_offset, col_offset, 0)
        load_two_pair_input(x, x_in10_ub, x_in11_ub, row_offset, col_offset + 2 * in_sf_block_k, 1)

        T.wait_flag("mte2", "v", 4)
        T.tile.bitwise_rshift(x_sf_exp_ub, x_sf_bits_ub, 23)
        T.pipe_barrier("v")

        compute_two_pattern_max(
            x_sf_exp_ub, 0,
            xsf_row_offset_i32_ub, xsf_row_offset_u32_ub,
            e0_ub, e1_ub, e2_ub, e3_ub, max01_ub, max23_ub, out_exp_ub,
        )

        compute_two_pair_values(
            e0_ub, e1_ub, out_exp_ub,
            relative_exp0_ub, relative_exp1_ub, relative_bits0_ub, relative_bits1_ub,
            relative_sf0_view_ub, relative_sf1_view_ub, relative_sf_tile0_ub, relative_sf_tile1_ub,
            x_in00_ub, x_in01_ub, x_out00_ub, x_out01_ub, 0, 0,
        )

        if num_out_sf_per_block_k == 2:
            load_two_pair_input(x, x_in00_ub, x_in01_ub, row_offset, col_offset + out_sf_block_k, 0)
        store_two_pair_output(out, x_out00_ub, x_out01_ub, row_offset, col_offset, 0)

        compute_two_pair_values(
            e2_ub, e3_ub, out_exp_ub,
            relative_exp0_ub, relative_exp1_ub, relative_bits0_ub, relative_bits1_ub,
            relative_sf0_view_ub, relative_sf1_view_ub, relative_sf_tile0_ub, relative_sf_tile1_ub,
            x_in10_ub, x_in11_ub, x_out10_ub, x_out11_ub, 1, 1,
        )

        if num_out_sf_per_block_k == 2:
            load_two_pair_input(
                x, x_in10_ub, x_in11_ub, row_offset,
                col_offset + out_sf_block_k + 2 * in_sf_block_k, 1,
            )
        store_two_pair_output(out, x_out10_ub, x_out11_ub, row_offset, col_offset + 2 * in_sf_block_k, 1)

        if num_out_sf_per_block_k == 2:
            # Pattern 1 max work overlaps the pair 0/1 MTE3 stores and pair 2/3 prefetches.
            compute_two_pattern_max(
                x_sf_exp_ub, 1,
                xsf_row_offset_i32_ub, xsf_row_offset_u32_ub,
                e0_ub, e1_ub, e2_ub, e3_ub, max01_ub, max23_ub, out_exp_ub,
            )

            compute_two_pair_values(
                e0_ub, e1_ub, out_exp_ub,
                relative_exp0_ub, relative_exp1_ub, relative_bits0_ub, relative_bits1_ub,
                relative_sf0_view_ub, relative_sf1_view_ub, relative_sf_tile0_ub, relative_sf_tile1_ub,
                x_in00_ub, x_in01_ub, x_out00_ub, x_out01_ub, 0, 0,
            )
            store_two_pair_output(out, x_out00_ub, x_out01_ub, row_offset, col_offset + out_sf_block_k, 0)

            compute_two_pair_values(
                e2_ub, e3_ub, out_exp_ub,
                relative_exp0_ub, relative_exp1_ub, relative_bits0_ub, relative_bits1_ub,
                relative_sf0_view_ub, relative_sf1_view_ub, relative_sf_tile0_ub, relative_sf_tile1_ub,
                x_in10_ub, x_in11_ub, x_out10_ub, x_out11_ub, 1, 1,
            )
            store_two_pair_output(
                out, x_out10_ub, x_out11_ub, row_offset,
                col_offset + out_sf_block_k + 2 * in_sf_block_k, 1,
            )
        # Destroy the ring events left by the final slot use. Every seeded or
        # recycled event must be consumed before the vector kernel exits.
        T.wait_flag("v", "mte2", 0)
        T.wait_flag("v", "mte2", 1)
        T.wait_flag("mte3", "v", 0)
        T.wait_flag("mte3", "v", 1)

    @T.macro
    def apply_relative_sf_tiles_compact_pattern_tail64(
        x_sf,
        x,
        out,
        sf_m,
        sf_k,
        row_offset,
        col_offset,
    ):
        tail_x_sf_load_ub = T.alloc_ub((vector_data_tile_m, compact_sf_row_stride), "float32")
        tail_x_sf_bits_ub = T.alloc_ub((vector_data_tile_m, compact_sf_row_stride), "int32")
        tail_x_sf_exp_ub = T.alloc_ub((vector_data_tile_m, compact_sf_row_stride), "int32")
        T.reinterpretcast(tail_x_sf_bits_ub, tail_x_sf_load_ub, "int")

        tail_x_in00_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        tail_x_in01_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        tail_x_out00_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        tail_x_out01_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)

        tail_xsf_row_offset_i32_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        tail_xsf_row_offset_u32_ub = T.alloc_ub((vector_data_tile_m,), "uint32")
        tail_e0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        tail_e1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        tail_e2_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        tail_e3_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        tail_max01_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        tail_max23_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        tail_out_exp_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        tail_relative_exp0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        tail_relative_exp1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        tail_relative_bits0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        tail_relative_bits1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        tail_relative_sf0_view_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        tail_relative_sf1_view_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        tail_relative_sf_tile0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")
        tail_relative_sf_tile1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")

        T.reinterpretcast(tail_xsf_row_offset_u32_ub, tail_xsf_row_offset_i32_ub, "uint32_t")
        T.reinterpretcast(tail_relative_sf0_view_ub, tail_relative_bits0_ub, "float")
        T.reinterpretcast(tail_relative_sf1_view_ub, tail_relative_bits1_ub, "float")

        T.copy(x_sf[sf_m : sf_m + num_in_sf_per_vector_m, sf_k : sf_k + num_in_sf_per_block_k], tail_x_sf_load_ub)
        T.set_flag("mte2", "v", 4)

        T.set_flag("v", "mte2", 0)
        T.set_flag("mte3", "v", 0)
        load_two_pair_input(x, tail_x_in00_ub, tail_x_in01_ub, row_offset, col_offset, 0)

        T.wait_flag("mte2", "v", 4)
        T.tile.bitwise_rshift(tail_x_sf_exp_ub, tail_x_sf_bits_ub, 23)
        T.pipe_barrier("v")

        compute_two_pattern_max(
            tail_x_sf_exp_ub, 0,
            tail_xsf_row_offset_i32_ub, tail_xsf_row_offset_u32_ub,
            tail_e0_ub, tail_e1_ub, tail_e2_ub, tail_e3_ub, tail_max01_ub, tail_max23_ub, tail_out_exp_ub,
        )
        compute_two_pair_values(
            tail_e0_ub, tail_e1_ub, tail_out_exp_ub,
            tail_relative_exp0_ub, tail_relative_exp1_ub, tail_relative_bits0_ub, tail_relative_bits1_ub,
            tail_relative_sf0_view_ub, tail_relative_sf1_view_ub,
            tail_relative_sf_tile0_ub, tail_relative_sf_tile1_ub,
            tail_x_in00_ub, tail_x_in01_ub, tail_x_out00_ub, tail_x_out01_ub, 0, 0,
        )
        store_two_pair_output(out, tail_x_out00_ub, tail_x_out01_ub, row_offset, col_offset, 0)
        T.wait_flag("v", "mte2", 0)
        T.wait_flag("mte3", "v", 0)

    @T.macro
    def apply_relative_sf_tiles_block_pipeline_max4(
        x_sf,
        x,
        out,
        sf_m,
        sf_k,
        row_offset,
        col_offset,
    ):
        tile_elem_count = vector_data_tile_m * in_sf_block_k

        x_sf_load_ub = T.alloc_ub(
            (vector_data_tile_m, num_in_sf_per_block_k),
            "float32",
        )
        x_sf_bits_ub = T.alloc_ub(
            (vector_data_tile_m, num_in_sf_per_block_k),
            "int32",
        )
        x_sf_exp_ub = T.alloc_ub(
            (vector_data_tile_m, num_in_sf_per_block_k),
            "int32",
        )

        # Keep every PTO operand as an independent fixed-shape buffer. The
        # 0624 parser cannot evaluate variable-based ranges into local UB.
        x_in0_slot0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in1_slot0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in0_slot1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in1_slot1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in0_slot2_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in1_slot2_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in0_slot3_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in1_slot3_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)

        x_out00_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out01_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out10_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out11_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out20_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out21_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out30_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out31_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)

        xsf_row_offset_i32_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        xsf_row_offset_u32_ub = T.alloc_ub((vector_data_tile_m,), "uint32")
        e0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        e1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        e2_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        e3_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        max01_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        max23_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        max_exp_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        out_exp_col_ub = T.alloc_ub((vector_data_tile_m,), "int32")

        relative_exp0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_exp1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_bits0_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_bits1_ub = T.alloc_ub((vector_data_tile_m,), "int32")
        relative_sf0_view_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf1_view_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf_tile0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")
        relative_sf_tile1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")

        T.reinterpretcast(x_sf_bits_ub, x_sf_load_ub, "int")
        T.reinterpretcast(xsf_row_offset_u32_ub, xsf_row_offset_i32_ub, "uint32_t")
        T.reinterpretcast(relative_sf0_view_ub, relative_bits0_ub, "float")
        T.reinterpretcast(relative_sf1_view_ub, relative_bits1_ub, "float")

        # Seed all four MTE2 input slots. The first four stages are prefetched
        # before vector work starts, so they do not depend on slot reuse.
        for input_slot in T.unroll(4):
            T.set_flag("v", "mte2", input_slot)

        # Prologue: overlap the full SF load with the first data pair load.
        T.copy(
            x_sf[
                sf_m : sf_m + num_in_sf_per_vector_m,
                sf_k : sf_k + num_in_sf_per_block_k,
            ],
            x_sf_load_ub,
        )
        T.set_flag("mte2", "v", 4)

        # Queue four stages up front. MTE2 continues loading later slots while
        # V decodes SF and starts consuming the earlier slots.
        for preload_stage in T.unroll(num_initial_prefetch_stages):
            preload_slot = preload_stage
            preload_block_idx = preload_stage // 2
            preload_pair_idx = preload_stage % 2
            preload_line0_idx = preload_pair_idx * 2
            preload_line1_idx = preload_line0_idx + 1
            preload_col_base = col_offset + preload_block_idx * out_sf_block_k

            T.wait_flag("v", "mte2", preload_slot)
            if preload_slot == 0:
                T.copy(x[row_offset : row_offset + vector_data_tile_m, preload_col_base + preload_line0_idx * in_sf_block_k : preload_col_base + (preload_line0_idx + 1) * in_sf_block_k], x_in0_slot0_ub)
                T.copy(x[row_offset : row_offset + vector_data_tile_m, preload_col_base + preload_line1_idx * in_sf_block_k : preload_col_base + (preload_line1_idx + 1) * in_sf_block_k], x_in1_slot0_ub)
            elif preload_slot == 1:
                T.copy(x[row_offset : row_offset + vector_data_tile_m, preload_col_base + preload_line0_idx * in_sf_block_k : preload_col_base + (preload_line0_idx + 1) * in_sf_block_k], x_in0_slot1_ub)
                T.copy(x[row_offset : row_offset + vector_data_tile_m, preload_col_base + preload_line1_idx * in_sf_block_k : preload_col_base + (preload_line1_idx + 1) * in_sf_block_k], x_in1_slot1_ub)
            elif preload_slot == 2:
                T.copy(x[row_offset : row_offset + vector_data_tile_m, preload_col_base + preload_line0_idx * in_sf_block_k : preload_col_base + (preload_line0_idx + 1) * in_sf_block_k], x_in0_slot2_ub)
                T.copy(x[row_offset : row_offset + vector_data_tile_m, preload_col_base + preload_line1_idx * in_sf_block_k : preload_col_base + (preload_line1_idx + 1) * in_sf_block_k], x_in1_slot2_ub)
            else:
                T.copy(x[row_offset : row_offset + vector_data_tile_m, preload_col_base + preload_line0_idx * in_sf_block_k : preload_col_base + (preload_line0_idx + 1) * in_sf_block_k], x_in0_slot3_ub)
                T.copy(x[row_offset : row_offset + vector_data_tile_m, preload_col_base + preload_line1_idx * in_sf_block_k : preload_col_base + (preload_line1_idx + 1) * in_sf_block_k], x_in1_slot3_ub)
            T.set_flag("mte2", "v", preload_slot)

        T.wait_flag("mte2", "v", 4)
        T.tile.bitwise_rshift(x_sf_exp_ub, x_sf_bits_ub, 23)
        T.pipe_barrier("v")

        for block_idx in T.unroll(num_out_sf_per_block_k):
            sf_k_base = block_idx * num_in_sf_per_out_sf_k

            T.tile.arith_progression(
                xsf_row_offset_i32_ub,
                (sf_k_base + 0) * 4,
                num_in_sf_per_block_k * 4,
                vector_data_tile_m,
            )
            T.tile.gather(e0_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 0)
            T.tile.arith_progression(
                xsf_row_offset_i32_ub,
                (sf_k_base + 1) * 4,
                num_in_sf_per_block_k * 4,
                vector_data_tile_m,
            )
            T.tile.gather(e1_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 0)
            T.tile.arith_progression(
                xsf_row_offset_i32_ub,
                (sf_k_base + 2) * 4,
                num_in_sf_per_block_k * 4,
                vector_data_tile_m,
            )
            T.tile.gather(e2_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 0)
            T.tile.arith_progression(
                xsf_row_offset_i32_ub,
                (sf_k_base + 3) * 4,
                num_in_sf_per_block_k * 4,
                vector_data_tile_m,
            )
            T.tile.gather(e3_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 0)

            T.pipe_barrier("v")
            T.tile.max(max01_ub, e0_ub, e1_ub)
            T.tile.max(max23_ub, e2_ub, e3_ub)
            T.tile.max(max_exp_ub, max01_ub, max23_ub)
            T.tile.add(out_exp_col_ub, max_exp_ub, -6)
            T.tile.max(out_exp_col_ub, out_exp_col_ub, 0)

            for pair_idx in T.unroll(2):
                stage = block_idx * 2 + pair_idx
                input_slot = stage % 4
                output_slot = stage % 4
                line0_idx = pair_idx * 2
                line1_idx = line0_idx + 1

                # Four-stage look-ahead: only stages beyond the initial four
                # reuse an input slot. Queue the future load before V work so
                # it can overlap the current stage and the previous MTE3.
                if stage + 4 < num_pipeline_pairs:
                    future_stage = stage + 4
                    future_block_idx = future_stage // 2
                    future_pair_idx = future_stage % 2
                    future_line0_idx = future_pair_idx * 2
                    future_line1_idx = future_line0_idx + 1
                    future_col_base = col_offset + future_block_idx * out_sf_block_k

                    T.wait_flag("v", "mte2", input_slot)
                    if input_slot == 0:
                        T.copy(x[row_offset : row_offset + vector_data_tile_m, future_col_base + future_line0_idx * in_sf_block_k : future_col_base + (future_line0_idx + 1) * in_sf_block_k], x_in0_slot0_ub)
                        T.copy(x[row_offset : row_offset + vector_data_tile_m, future_col_base + future_line1_idx * in_sf_block_k : future_col_base + (future_line1_idx + 1) * in_sf_block_k], x_in1_slot0_ub)
                    elif input_slot == 1:
                        T.copy(x[row_offset : row_offset + vector_data_tile_m, future_col_base + future_line0_idx * in_sf_block_k : future_col_base + (future_line0_idx + 1) * in_sf_block_k], x_in0_slot1_ub)
                        T.copy(x[row_offset : row_offset + vector_data_tile_m, future_col_base + future_line1_idx * in_sf_block_k : future_col_base + (future_line1_idx + 1) * in_sf_block_k], x_in1_slot1_ub)
                    elif input_slot == 2:
                        T.copy(x[row_offset : row_offset + vector_data_tile_m, future_col_base + future_line0_idx * in_sf_block_k : future_col_base + (future_line0_idx + 1) * in_sf_block_k], x_in0_slot2_ub)
                        T.copy(x[row_offset : row_offset + vector_data_tile_m, future_col_base + future_line1_idx * in_sf_block_k : future_col_base + (future_line1_idx + 1) * in_sf_block_k], x_in1_slot2_ub)
                    else:
                        T.copy(x[row_offset : row_offset + vector_data_tile_m, future_col_base + future_line0_idx * in_sf_block_k : future_col_base + (future_line0_idx + 1) * in_sf_block_k], x_in0_slot3_ub)
                        T.copy(x[row_offset : row_offset + vector_data_tile_m, future_col_base + future_line1_idx * in_sf_block_k : future_col_base + (future_line1_idx + 1) * in_sf_block_k], x_in1_slot3_ub)
                    T.set_flag("mte2", "v", input_slot)
                if stage >= 4:
                    T.wait_flag("mte3", "v", output_slot)

                if pair_idx == 0:
                    T.tile.sub(relative_exp0_ub, e0_ub, out_exp_col_ub)
                    T.tile.sub(relative_exp1_ub, e1_ub, out_exp_col_ub)
                else:
                    T.tile.sub(relative_exp0_ub, e2_ub, out_exp_col_ub)
                    T.tile.sub(relative_exp1_ub, e3_ub, out_exp_col_ub)

                T.tile.add(relative_exp0_ub, relative_exp0_ub, 127)
                T.tile.add(relative_exp1_ub, relative_exp1_ub, 127)
                T.tile.bitwise_lshift(relative_bits0_ub, relative_exp0_ub, 23)
                T.tile.bitwise_lshift(relative_bits1_ub, relative_exp1_ub, 23)
                T.pipe_barrier("v")
                T.tile.broadcast(relative_sf_tile0_ub, relative_sf0_view_ub, axis=1)
                T.tile.broadcast(relative_sf_tile1_ub, relative_sf1_view_ub, axis=1)

                # Delay the consumer wait until the first input-buffer read so
                # relative-SF work can hide the current MTE2 transfer.
                T.wait_flag("mte2", "v", input_slot)

                if output_slot == 0:
                    if input_slot == 0:
                        T.tile.cast(x_out00_ub, x_in0_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out01_ub, x_in1_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 1:
                        T.tile.cast(x_out00_ub, x_in0_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out01_ub, x_in1_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 2:
                        T.tile.cast(x_out00_ub, x_in0_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out01_ub, x_in1_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                    else:
                        T.tile.cast(x_out00_ub, x_in0_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out01_ub, x_in1_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                    if stage + 4 < num_pipeline_pairs:
                        T.set_flag("v", "mte2", input_slot)
                    T.tile.mul(x_out00_ub, x_out00_ub, relative_sf_tile0_ub)
                    T.tile.mul(x_out01_ub, x_out01_ub, relative_sf_tile1_ub)
                elif output_slot == 1:
                    if input_slot == 0:
                        T.tile.cast(x_out10_ub, x_in0_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out11_ub, x_in1_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 1:
                        T.tile.cast(x_out10_ub, x_in0_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out11_ub, x_in1_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 2:
                        T.tile.cast(x_out10_ub, x_in0_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out11_ub, x_in1_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                    else:
                        T.tile.cast(x_out10_ub, x_in0_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out11_ub, x_in1_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                    if stage + 4 < num_pipeline_pairs:
                        T.set_flag("v", "mte2", input_slot)
                    T.tile.mul(x_out10_ub, x_out10_ub, relative_sf_tile0_ub)
                    T.tile.mul(x_out11_ub, x_out11_ub, relative_sf_tile1_ub)
                elif output_slot == 2:
                    if input_slot == 0:
                        T.tile.cast(x_out20_ub, x_in0_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out21_ub, x_in1_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 1:
                        T.tile.cast(x_out20_ub, x_in0_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out21_ub, x_in1_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 2:
                        T.tile.cast(x_out20_ub, x_in0_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out21_ub, x_in1_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                    else:
                        T.tile.cast(x_out20_ub, x_in0_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out21_ub, x_in1_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                    if stage + 4 < num_pipeline_pairs:
                        T.set_flag("v", "mte2", input_slot)
                    T.tile.mul(x_out20_ub, x_out20_ub, relative_sf_tile0_ub)
                    T.tile.mul(x_out21_ub, x_out21_ub, relative_sf_tile1_ub)
                else:
                    if input_slot == 0:
                        T.tile.cast(x_out30_ub, x_in0_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out31_ub, x_in1_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 1:
                        T.tile.cast(x_out30_ub, x_in0_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out31_ub, x_in1_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 2:
                        T.tile.cast(x_out30_ub, x_in0_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out31_ub, x_in1_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                    else:
                        T.tile.cast(x_out30_ub, x_in0_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out31_ub, x_in1_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                    if stage + 4 < num_pipeline_pairs:
                        T.set_flag("v", "mte2", input_slot)
                    T.tile.mul(x_out30_ub, x_out30_ub, relative_sf_tile0_ub)
                    T.tile.mul(x_out31_ub, x_out31_ub, relative_sf_tile1_ub)

                T.set_flag("v", "mte3", output_slot)
                T.wait_flag("v", "mte3", output_slot)

                current_col_base = col_offset + block_idx * out_sf_block_k
                if output_slot == 0:
                    T.copy(
                        x_out00_ub,
                        out[
                            row_offset : row_offset + vector_data_tile_m,
                            current_col_base + line0_idx * in_sf_block_k :
                            current_col_base + (line0_idx + 1) * in_sf_block_k,
                        ],
                    )
                    T.copy(
                        x_out01_ub,
                        out[
                            row_offset : row_offset + vector_data_tile_m,
                            current_col_base + line1_idx * in_sf_block_k :
                            current_col_base + (line1_idx + 1) * in_sf_block_k,
                        ],
                    )
                elif output_slot == 1:
                    T.copy(
                        x_out10_ub,
                        out[
                            row_offset : row_offset + vector_data_tile_m,
                            current_col_base + line0_idx * in_sf_block_k :
                            current_col_base + (line0_idx + 1) * in_sf_block_k,
                        ],
                    )
                    T.copy(
                        x_out11_ub,
                        out[
                            row_offset : row_offset + vector_data_tile_m,
                            current_col_base + line1_idx * in_sf_block_k :
                            current_col_base + (line1_idx + 1) * in_sf_block_k,
                        ],
                    )
                elif output_slot == 2:
                    T.copy(
                        x_out20_ub,
                        out[
                            row_offset : row_offset + vector_data_tile_m,
                            current_col_base + line0_idx * in_sf_block_k :
                            current_col_base + (line0_idx + 1) * in_sf_block_k,
                        ],
                    )
                    T.copy(
                        x_out21_ub,
                        out[
                            row_offset : row_offset + vector_data_tile_m,
                            current_col_base + line1_idx * in_sf_block_k :
                            current_col_base + (line1_idx + 1) * in_sf_block_k,
                        ],
                    )
                else:
                    T.copy(
                        x_out30_ub,
                        out[
                            row_offset : row_offset + vector_data_tile_m,
                            current_col_base + line0_idx * in_sf_block_k :
                            current_col_base + (line0_idx + 1) * in_sf_block_k,
                        ],
                    )
                    T.copy(
                        x_out31_ub,
                        out[
                            row_offset : row_offset + vector_data_tile_m,
                            current_col_base + line1_idx * in_sf_block_k :
                            current_col_base + (line1_idx + 1) * in_sf_block_k,
                        ],
                    )
                if stage + 4 < num_pipeline_pairs:
                    T.set_flag("mte3", "v", output_slot)

        # No per-slot drain is needed: reuse events are emitted only when a
        # later stage consumes them. Flush the final GM writes locally.
        T.pipe_barrier("mte3")

    @T.macro
    def apply_relative_sf_tiles_local_broadcast(x_sf_load_ub, out_sf_fp32_ub, x, out, row_offset, col_offset):
        input_sf_seed_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        out_sf_seed_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf_rows_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf_tile_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")
        x_in_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_out_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        tile_elem_count = vector_data_tile_m * in_sf_block_k

        for out_sf_k in T.serial(num_out_sf_per_block_k):
            for row in T.serial(vector_data_tile_m):
                local_sf_i = row // in_sf_block_m
                out_sf_m = local_sf_i // num_in_sf_per_out_sf_m
                group = out_sf_m * num_out_sf_per_block_k + out_sf_k
                out_sf_seed_ub[row, 0] = out_sf_fp32_ub[group]
            for reduce_j in T.serial(num_in_sf_per_out_sf_k):
                sf_k = out_sf_k * num_in_sf_per_out_sf_k + reduce_j
                for row in T.serial(vector_data_tile_m):
                    local_sf_i = row // in_sf_block_m
                    if in_config.use_tma_aligned_col_major_sf:
                        input_sf_seed_ub[row, 0] = x_sf_load_ub[sf_k, local_sf_i]
                    else:
                        input_sf_seed_ub[row, 0] = x_sf_load_ub[local_sf_i, sf_k]
                T.tile.div(relative_sf_rows_ub, input_sf_seed_ub, out_sf_seed_ub)
                T.tile.broadcast(relative_sf_tile_ub, relative_sf_rows_ub, axis=1)
                col_tile_offset = col_offset + sf_k * in_sf_block_k
                T.copy(
                    x[
                        row_offset : row_offset + vector_data_tile_m,
                        col_tile_offset : col_tile_offset + in_sf_block_k,
                    ],
                    x_in_ub,
                )
                T.tile.cast(x_out_ub, x_in_ub, mode="CAST_NONE", count=tile_elem_count)
                T.tile.mul(x_out_ub, x_out_ub, relative_sf_tile_ub)
                T.copy(
                    x_out_ub,
                    out[
                        row_offset : row_offset + vector_data_tile_m,
                        col_tile_offset : col_tile_offset + in_sf_block_k,
                    ],
                )


    @T.macro
    def apply_relative_sf_tiles_pair_local_direct(x_sf_load_ub, out_sf_fp32_ub, x, out, row_offset, col_offset):
        input_sf_col0_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        input_sf_col1_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        out_sf_col0_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        out_sf_col1_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf_col0_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf_col1_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf_tile0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")
        relative_sf_tile1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")
        x_in0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_out0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        tile_elem_count = vector_data_tile_m * in_sf_block_k
        for sf_pair in T.serial(num_in_sf_per_block_k // 2):
            sf_k0 = sf_pair * 2
            sf_k1 = sf_k0 + 1
            out_sf_k0 = sf_k0 // num_in_sf_per_out_sf_k
            out_sf_k1 = sf_k1 // num_in_sf_per_out_sf_k
            for row in T.serial(vector_data_tile_m):
                local_sf_i = row // in_sf_block_m
                out_sf_m = local_sf_i // num_in_sf_per_out_sf_m
                group_row_base = out_sf_m * num_out_sf_per_block_k
                group0 = group_row_base + out_sf_k0
                group1 = group_row_base + out_sf_k1
                if in_config.use_tma_aligned_col_major_sf:
                    input_sf_col0_ub[row, 0] = x_sf_load_ub[sf_k0, local_sf_i]
                    input_sf_col1_ub[row, 0] = x_sf_load_ub[sf_k1, local_sf_i]
                else:
                    input_sf_col0_ub[row, 0] = x_sf_load_ub[local_sf_i, sf_k0]
                    input_sf_col1_ub[row, 0] = x_sf_load_ub[local_sf_i, sf_k1]
                out_sf_col0_ub[row, 0] = out_sf_fp32_ub[group0]
                out_sf_col1_ub[row, 0] = out_sf_fp32_ub[group1]
            T.tile.div(relative_sf_col0_ub, input_sf_col0_ub, out_sf_col0_ub)
            T.tile.div(relative_sf_col1_ub, input_sf_col1_ub, out_sf_col1_ub)
            T.tile.broadcast(relative_sf_tile0_ub, relative_sf_col0_ub, axis=1)
            T.tile.broadcast(relative_sf_tile1_ub, relative_sf_col1_ub, axis=1)

            col_tile_offset0 = col_offset + sf_pair * (2 * in_sf_block_k)
            col_tile_offset1 = col_tile_offset0 + in_sf_block_k
            T.copy(
                x[
                    row_offset : row_offset + vector_data_tile_m,
                    col_tile_offset0 : col_tile_offset0 + in_sf_block_k,
                ],
                x_in0_ub,
            )
            T.copy(
                x[
                    row_offset : row_offset + vector_data_tile_m,
                    col_tile_offset1 : col_tile_offset1 + in_sf_block_k,
                ],
                x_in1_ub,
            )
            T.tile.cast(x_out0_ub, x_in0_ub, mode="CAST_NONE", count=tile_elem_count)
            T.tile.cast(x_out1_ub, x_in1_ub, mode="CAST_NONE", count=tile_elem_count)
            T.tile.mul(x_out0_ub, x_out0_ub, relative_sf_tile0_ub)
            T.tile.mul(x_out1_ub, x_out1_ub, relative_sf_tile1_ub)
            T.copy(
                x_out0_ub,
                out[
                    row_offset : row_offset + vector_data_tile_m,
                    col_tile_offset0 : col_tile_offset0 + in_sf_block_k,
                ],
            )
            T.copy(
                x_out1_ub,
                out[
                    row_offset : row_offset + vector_data_tile_m,
                    col_tile_offset1 : col_tile_offset1 + in_sf_block_k,
                ],
            )

    @T.macro
    def apply_relative_sf_tiles_local_direct(x_sf_load_ub, out_sf_fp32_ub, x, out, row_offset, col_offset):
        input_sf_col_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        out_sf_col_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf_col_ub = T.alloc_ub((vector_data_tile_m, 1), "float32")
        relative_sf_tile_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), "float32")
        x_in_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_out_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        tile_elem_count = vector_data_tile_m * in_sf_block_k
        for sf_k in T.serial(num_in_sf_per_block_k):
            out_sf_k = sf_k // num_in_sf_per_out_sf_k
            for row in T.serial(vector_data_tile_m):
                local_sf_i = row // in_sf_block_m
                out_sf_m = local_sf_i // num_in_sf_per_out_sf_m
                group = out_sf_m * num_out_sf_per_block_k + out_sf_k
                if in_config.use_tma_aligned_col_major_sf:
                    input_sf_col_ub[row, 0] = x_sf_load_ub[sf_k, local_sf_i]
                else:
                    input_sf_col_ub[row, 0] = x_sf_load_ub[local_sf_i, sf_k]
                out_sf_col_ub[row, 0] = out_sf_fp32_ub[group]
            T.tile.div(relative_sf_col_ub, input_sf_col_ub, out_sf_col_ub)
            T.tile.broadcast(relative_sf_tile_ub, relative_sf_col_ub, axis=1)
            col_tile_offset = col_offset + sf_k * in_sf_block_k
            T.copy(
                x[
                    row_offset : row_offset + vector_data_tile_m,
                    col_tile_offset : col_tile_offset + in_sf_block_k,
                ],
                x_in_ub,
            )
            T.tile.cast(x_out_ub, x_in_ub, mode="CAST_NONE", count=tile_elem_count)
            T.tile.mul(x_out_ub, x_out_ub, relative_sf_tile_ub)
            T.copy(
                x_out_ub,
                out[
                    row_offset : row_offset + vector_data_tile_m,
                    col_tile_offset : col_tile_offset + in_sf_block_k,
                ],
            )

    @T.prim_func
    def per_block_cast_lossless_kernel(
        x: T.Tensor([num_tokens, hidden], INPUT_DTYPE),
        x_sf: T.Tensor(x_sf_shape, in_config.sf_dtype),
        out: T.Tensor([num_tokens, hidden], OUTPUT_DTYPE),
        out_sf: T.Tensor(out_sf_shape, out_config.sf_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                pid_token = cid // n_num
                pid_hidden = cid % n_num

                row_offset = pid_token * block_m + vid * vector_data_tile_m
                col_offset = pid_hidden * block_k

                sf_row_offset = pid_token * num_in_sf_per_block_m
                sf_col_offset = pid_hidden * num_in_sf_per_block_k
                out_sf_row_offset = pid_token * num_out_sf_per_block_m
                out_sf_col_offset = pid_hidden * num_out_sf_per_block_k

                if use_compact_pattern_pipeline:
                    local_sf_row_offset = sf_row_offset + vid * num_in_sf_per_vector_m
                    if col_offset + block_k <= logical_hidden:
                        apply_relative_sf_tiles_compact_pattern_pipeline(
                            x_sf,
                            x,
                            out,
                            local_sf_row_offset,
                            sf_col_offset,
                            row_offset,
                            col_offset,
                        )
                    elif col_offset < logical_hidden:
                        if logical_hidden - col_offset <= 2 * in_sf_block_k:
                            apply_relative_sf_tiles_compact_pattern_tail64(
                                x_sf,
                                x,
                                out,
                                local_sf_row_offset,
                                sf_col_offset,
                                row_offset,
                                col_offset,
                            )
                        else:
                            apply_relative_sf_tiles_compact_pattern_pipeline(
                                x_sf,
                                x,
                                out,
                                local_sf_row_offset,
                                sf_col_offset,
                                row_offset,
                                col_offset,
                            )
                elif use_local_max4_fast_path:
                    local_sf_row_offset = sf_row_offset + vid * num_in_sf_per_vector_m
                    apply_relative_sf_tiles_block_pipeline_max4(
                        x_sf,
                        x,
                        out,
                        local_sf_row_offset,
                        sf_col_offset,
                        row_offset,
                        col_offset,
                    )
                elif use_vid_local_sf:
                    local_sf_row_offset = sf_row_offset + vid * num_in_sf_per_vector_m
                    local_out_sf_row_offset = (out_sf_row_offset + vid * local_num_out_sf_per_block_m)
                    x_sf_load_ub = T.alloc_ub(x_sf_load_local_shape, "float32")
                    x_sf_exp_grouped_ub = T.alloc_ub((local_num_out_sf_per_block, num_in_sf_per_out_sf_aligned), "float32")
                    x_sf_exp_i32_grouped_ub = T.alloc_ub((local_num_out_sf_per_block, num_in_sf_per_out_sf_aligned), "int32")
                    out_sf_exp_ub = T.alloc_ub((local_num_out_sf_per_block,), "int32")
                    out_sf_fp32_ub = T.alloc_ub((local_num_out_sf_per_block,), "float32")

                    T.tile.fill(x_sf_exp_grouped_ub, 0.0)
                    load_input_sf_exp_local(x_sf_exp_grouped_ub, x_sf_exp_i32_grouped_ub, x_sf_load_ub, x_sf, local_sf_row_offset, sf_col_offset)
                    reduce_output_sf_exp_local(out_sf_exp_ub, x_sf_exp_grouped_ub, out_sf_fp32_ub)

                    out_sf_bits_ub = T.alloc_ub((local_num_out_sf_per_block,), "int32")
                    T.tile.bitwise_lshift(out_sf_bits_ub, out_sf_exp_ub, 23)
                    T.reinterpretcast(out_sf_fp32_ub, out_sf_bits_ub, "float")
                    store_output_sf_local(out_sf, out_sf_fp32_ub, local_out_sf_row_offset, out_sf_col_offset)

                    if (in_sf_block_m == 1 and out_sf_block_m == 1 and num_in_sf_per_out_sf_k >= 2):
                        x_sf_exp_i32_row_ub = T.alloc_ub((vector_data_tile_m, num_in_sf_per_block_k), "int32")
                        out_sf_exp_row_ub = T.alloc_ub((vector_data_tile_m, num_out_sf_per_block_k), "int32")
                        for row in T.serial(vector_data_tile_m):
                            for out_sf_k in T.serial(num_out_sf_per_block_k):
                                group = row * num_out_sf_per_block_k + out_sf_k
                                out_sf_exp_row_ub[row, out_sf_k] = out_sf_exp_ub[group]
                            for sf_k in T.serial(num_in_sf_per_block_k):
                                out_sf_k = sf_k // num_in_sf_per_out_sf_k
                                reduce_j = sf_k - out_sf_k * num_in_sf_per_out_sf_k
                                group = row * num_out_sf_per_block_k + out_sf_k
                                x_sf_exp_i32_row_ub[row, sf_k] = x_sf_exp_i32_grouped_ub[group, reduce_j]
                        apply_relative_sf_tiles_pair_local_exp_tile_row(
                            x_sf_exp_i32_row_ub,
                            out_sf_exp_row_ub,
                            x,
                            out,
                            row_offset,
                            col_offset,
                        )
                    elif in_sf_block_m == 1 and out_sf_block_m == 1:
                        apply_relative_sf_tiles_local_broadcast(x_sf_load_ub, out_sf_fp32_ub, x, out, row_offset, col_offset)
                    elif num_in_sf_per_block_k % 2 == 0:
                        apply_relative_sf_tiles_pair_local_direct(x_sf_load_ub, out_sf_fp32_ub, x, out, row_offset, col_offset)
                    else:
                        apply_relative_sf_tiles_local_direct(x_sf_load_ub, out_sf_fp32_ub, x, out, row_offset, col_offset)
                else:
                    relative_sf_ub = T.alloc_ub((vector_data_tile_m * num_in_sf_per_block_k,), "float32")
                    x_sf_load_ub = T.alloc_ub(x_sf_load_shape, x_sf_load_dtype)
                    x_sf_exp_grouped_ub = T.alloc_ub((num_out_sf_per_block, num_in_sf_per_out_sf_aligned), "float32")
                    out_sf_exp_ub = T.alloc_ub((num_out_sf_per_block,), "int32")
                    out_sf_fp32_ub = T.alloc_ub((num_out_sf_per_block,), "float32")

                    T.tile.fill(x_sf_exp_grouped_ub, 0.0)
                    load_input_sf_exp_block(x_sf_exp_grouped_ub, x_sf_load_ub, x_sf, sf_row_offset, sf_col_offset)
                    reduce_output_sf_exp(out_sf_exp_ub, x_sf_exp_grouped_ub, out_sf_fp32_ub)

                    if vid == 0:
                        if not out_config.use_packed_ue8m0:
                            out_sf_bits_ub = T.alloc_ub((num_out_sf_per_block,), "int32")
                            T.tile.bitwise_lshift(out_sf_bits_ub, out_sf_exp_ub, 23)
                            T.reinterpretcast(out_sf_fp32_ub, out_sf_bits_ub, "float")
                        if not out_config.use_packed_ue8m0 and not out_config.use_tma_aligned_col_major_sf:
                            out_sf_store_ub = T.alloc_ub((num_out_sf_per_block_m, num_out_sf_per_block_k), "float32")
                            for flat in T.serial(num_out_sf_per_block):
                                i = flat // num_out_sf_per_block_k
                                j = flat % num_out_sf_per_block_k
                                out_sf_store_ub[i, j] = out_sf_fp32_ub[flat]
                            T.copy(
                                out_sf_store_ub,
                                out_sf[
                                    out_sf_row_offset:out_sf_row_offset + num_out_sf_per_block_m,
                                    out_sf_col_offset:out_sf_col_offset + num_out_sf_per_block_k,
                                ],
                            )
                        else:
                            for i in T.serial(num_out_sf_per_block_m):
                                for j in T.serial(num_out_sf_per_block_k):
                                    group = i * num_out_sf_per_block_k + j
                                    if out_config.use_packed_ue8m0:
                                        store_sf(out_sf, T.Cast("uint8", out_sf_exp_ub[group]), out_sf_row_offset + i, out_sf_col_offset + j, out_config)
                                    else:
                                        store_sf(out_sf, out_sf_fp32_ub[group], out_sf_row_offset + i, out_sf_col_offset + j, out_config)

                    compute_relative_sf(relative_sf_ub, x_sf_exp_grouped_ub, out_sf_exp_ub, vid * vector_data_tile_m // in_sf_block_m)
                    apply_relative_sf_tiles_transposed(relative_sf_ub, x, out, row_offset, col_offset)
    return per_block_cast_lossless_kernel


def per_block_cast_lossless(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    fmt: str = "fp32",
    x_block_size: tuple[int, int] = DEFAULT_IN_SF_BLOCK,
    out_block_size: tuple[int, int] = DEFAULT_OUT_SF_BLOCK,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    in_use_tma_aligned_col_major_sf: bool | None = None,
    in_round_sf: bool | None = None,
    in_use_packed_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_data, x_sf, in_config = get_cast_input_and_config(
        x,
        x_block_size,
        use_tma_aligned_col_major_sf=in_use_tma_aligned_col_major_sf,
        round_sf=in_round_sf,
        use_packed_ue8m0=in_use_packed_ue8m0,
    )

    if not in_config.with_sf:
        in_config = replace(in_config, with_sf=True, sf_block=x_block_size)

    assert fmt in ("fp32", "float32", "e4m3")

    # NPU path: bf16 + input_sf -> fp32 out + output_sf.
    # The device kernel computes output sf and relative input sf exponents.
    logical_out_fmt = "e4m3" if fmt in ("fp32", "float32") else fmt
    actual_out_fmt = "fp32"
    _ = logical_out_fmt

    assert x_data.dim() == 2 and x_data.is_contiguous()
    assert x_data.device.type == "npu"
    assert x_data.dtype == torch.bfloat16

    num_tokens, hidden = x_data.shape

    out_config = get_cast_output_config(actual_out_fmt, out_block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)

    if num_tokens == 0 or hidden == 0:
        out = torch.empty((num_tokens, hidden), dtype=out_config.torch_dtype, device=x_data.device)
        out_sf = alloc_scaling_factors((num_tokens, hidden), out_config, x_data.device)
        return out, cast_epilogue(out_sf, num_tokens, hidden, out_config)

    layout = _derive_cast_layout(hidden, in_config, out_config)
    block_m = layout["block_m"]
    block_k = layout["block_k"]

    padded_tokens = align_up(num_tokens, block_m)
    padded_hidden = align_up(hidden, block_k)

    if padded_tokens != num_tokens or padded_hidden != hidden:
        x_padded = torch.zeros((padded_tokens, padded_hidden), dtype=x_data.dtype, device=x_data.device)
        x_padded[:num_tokens, :hidden] = x_data
    else:
        x_padded = x_data

    x_sf_shape = get_sf_shape((padded_tokens, padded_hidden), in_config)
    if tuple(x_sf.shape) == x_sf_shape:
        x_sf_padded = x_sf
    else:
        # GPU masks OOB input sf before reduce-max. The NPU wrapper pads with
        # exponent-zero sf so padded values cannot increase an output exponent.
        x_sf_padded = torch.zeros(x_sf_shape, dtype=x_sf.dtype, device=x_data.device)
        copy_shape = tuple(min(src, dst) for src, dst in zip(x_sf.shape, x_sf_shape))
        x_sf_padded[: copy_shape[0], : copy_shape[1]] = x_sf[: copy_shape[0], : copy_shape[1],]

    out = torch.empty((padded_tokens, padded_hidden), dtype=out_config.torch_dtype, device=x_data.device)
    out_sf = alloc_scaling_factors((padded_tokens, padded_hidden), out_config, x_data.device)

    kernel = get_per_block_cast_lossless_kernel(
        hidden=padded_hidden,
        block_m=block_m,
        block_k=block_k,
        in_config=in_config,
        out_config=out_config,
        in_sf_block_m=in_config.sf_block[0],
        in_sf_block_k=in_config.sf_block[1],
        out_sf_block_m=out_config.sf_block[0],
        out_sf_block_k=out_config.sf_block[1],
    )

    print_kernel_source = int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0))
    if print_kernel_source:
        print(kernel.get_kernel_source())

    out, out_sf = kernel(x_padded, x_sf_padded, out, out_sf)

    out = out[:num_tokens, :hidden]
    out_sf = cast_epilogue(out_sf, num_tokens, hidden, out_config)
    return out, out_sf