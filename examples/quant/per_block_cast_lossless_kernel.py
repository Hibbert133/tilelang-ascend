import os
from dataclasses import replace
import torch
import tilelang
import tilelang.language as T
from .my_common import *
tilelang.cache.clear_cache()

<<<<<<< HEAD
=======
def _pad_2d_on_cpu_then_to_device(x: torch.Tensor, target_shape: tuple[int, int], fill_value=0) -> torch.Tensor:
    """Pad a 2D tensor on CPU, then move it back to the original device.

    This avoids launching PyTorch NPU ops such as torch.zeros / slice copy
    before the TileLang kernel when running under msprof op simulator.
    """
    assert x.dim() == 2, f'Only 2D tensors are supported, got shape={tuple(x.shape)}'
    target_device = x.device
    src_m, src_k = x.shape
    dst_m, dst_k = target_shape
    if src_m == dst_m and src_k == dst_k:
        if x.is_contiguous():
            return x
        return x.detach().to('cpu').contiguous().to(target_device)
    x_cpu = x.detach().to('cpu').contiguous()
    out_cpu = torch.full((dst_m, dst_k), fill_value, dtype=x_cpu.dtype, device='cpu')
    copy_m = min(src_m, dst_m)
    copy_k = min(src_k, dst_k)
    out_cpu[:copy_m, :copy_k] = x_cpu[:copy_m, :copy_k]
    return out_cpu.to(target_device).contiguous()

def _crop_2d_on_cpu_then_to_device(x: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    """Crop a 2D tensor on CPU, then move it back to the original device.

    This avoids NPU slicing / TensorMove in Python wrapper epilogue under
    msprof op simulator.
    """
    assert x.dim() == 2, f'Only 2D tensors are supported, got shape={tuple(x.shape)}'
    target_device = x.device
    dst_m, dst_k = target_shape
    if x.shape[0] == dst_m and x.shape[1] == dst_k:
        if x.is_contiguous():
            return x
        return x.detach().to('cpu').contiguous().to(target_device)
    x_cpu = x.detach().to('cpu').contiguous()
    return x_cpu[:dst_m, :dst_k].contiguous().to(target_device)
>>>>>>> 010c7f3 (per_block_cast_lossless_kernel v0705 cleaned_version)
DEFAULT_IN_SF_BLOCK = (1, 32)
DEFAULT_OUT_SF_BLOCK = (1, 128)
VEC_NUM = 2
NUM_ELEMENTS_PER_BLOCK = 8192
NUM_ELEMENTS_PER_VECTOR = 16384
INPUT_DTYPE = 'bfloat16'
OUTPUT_DTYPE = 'float32'
pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False, tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}
pass_configs_v1 = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True, tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}

def _derive_cast_layout(hidden: int, in_config: CastInputConfig, out_config: CastOutputConfig) -> dict[str, int]:
    assert in_config.dtype == INPUT_DTYPE and out_config.dtype == OUTPUT_DTYPE, 'lossless mode only supports bf16 -> fp32 conversion currently'
    assert in_config.with_sf, 'lossless mode requires both input and output scaling factors'
    assert is_power_of_two(in_config.sf_block[1]) and is_power_of_two(out_config.sf_block[1]), 'block_k must be power of 2 for lossless mode'
    assert (
        out_config.sf_block[0] % in_config.sf_block[0] == 0 and
        out_config.sf_block[1] % in_config.sf_block[1] == 0
    ), 'Output block size must be multiple of input block size'
    block_m = max(out_config.sf_block[0], 128)
    block_k = max(out_config.sf_block[1], 512)
    if (
        not in_config.use_packed_ue8m0 and
        (not out_config.use_packed_ue8m0) and
        (not in_config.use_tma_aligned_col_major_sf) and
        (not out_config.use_tma_aligned_col_major_sf) and
        (in_config.sf_block == DEFAULT_IN_SF_BLOCK) and
        (out_config.sf_block == DEFAULT_OUT_SF_BLOCK)
    ):
        block_k_candidates = (256, 512)
        block_k = min(
            (candidate for candidate in block_k_candidates if candidate % in_config.sf_block[1] == 0 and candidate % out_config.sf_block[1] == 0),
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
        'block_m': block_m, 'block_k': block_k, 'num_in_sf_per_block_m': num_in_sf_per_block_m, 'num_in_sf_per_block_k': num_in_sf_per_block_k,
        'num_out_sf_per_block_m': num_out_sf_per_block_m, 'num_out_sf_per_block_k': num_out_sf_per_block_k,
        'num_in_sf_per_out_sf_m': num_in_sf_per_out_sf_m, 'num_in_sf_per_out_sf_k': num_in_sf_per_out_sf_k,
        'num_in_sf_per_out_sf': num_in_sf_per_out_sf,
    }

def _derive_cast_layout_v1(hidden: int, in_config: CastInputConfig, out_config: CastOutputConfig) -> dict[str, int]:
    assert in_config.dtype == INPUT_DTYPE and out_config.dtype == OUTPUT_DTYPE, 'lossless mode only supports bf16 -> fp32 conversion currently'
    assert in_config.with_sf, 'lossless mode requires both input and output scaling factors'
    assert is_power_of_two(in_config.sf_block[1]) and is_power_of_two(out_config.sf_block[1]), 'block_k must be power of 2 for lossless mode'
    assert (
        out_config.sf_block[0] % in_config.sf_block[0] == 0 and
        out_config.sf_block[1] % in_config.sf_block[1] == 0
    ), 'Output block size must be multiple of input block size'
    block_m = max(out_config.sf_block[0], 32)
    block_k = max(out_config.sf_block[1], NUM_ELEMENTS_PER_BLOCK // block_m)
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
        'block_m': block_m, 'block_k': block_k, 'num_in_sf_per_block_m': num_in_sf_per_block_m, 'num_in_sf_per_block_k': num_in_sf_per_block_k,
        'num_out_sf_per_block_m': num_out_sf_per_block_m, 'num_out_sf_per_block_k': num_out_sf_per_block_k,
        'num_in_sf_per_out_sf_m': num_in_sf_per_out_sf_m, 'num_in_sf_per_out_sf_k': num_in_sf_per_out_sf_k,
        'num_in_sf_per_out_sf': num_in_sf_per_out_sf,
    }

def _pack_plain_sf_for_lossless_kernel_v1(
    x_sf_padded: torch.Tensor, padded_tokens: int, padded_hidden: int, block_m: int, block_k: int, in_sf_block: tuple[int, int],
) -> torch.Tensor:
    """Pack plain fp32 input sf into contiguous per-kernel tiles."""
    in_sf_block_m, in_sf_block_k = in_sf_block
    data_tile_m = min(block_m, 32, NUM_ELEMENTS_PER_BLOCK // block_k)
    num_data_tiles_m = block_m // data_tile_m
    num_in_sf_per_block_m = block_m // in_sf_block_m
    num_in_sf_per_block_k = block_k // in_sf_block_k
    num_in_sf_per_data_tile_m = data_tile_m // in_sf_block_m
    m_num = padded_tokens // block_m
    n_num = padded_hidden // block_k
    x_sf_tile = x_sf_padded.view(m_num, num_in_sf_per_block_m, n_num, num_in_sf_per_block_k)
    x_sf_tile = x_sf_tile.view(m_num, num_data_tiles_m, num_in_sf_per_data_tile_m, n_num, num_in_sf_per_block_k)
    x_sf_tile = x_sf_tile.permute(0, 3, 1, 2, 4).contiguous()
    return x_sf_tile.view(m_num, n_num, num_data_tiles_m, num_in_sf_per_data_tile_m * num_in_sf_per_block_k)

def _needs_tile_packed_input_sf_v1(in_config: CastInputConfig, out_config: CastOutputConfig, num_data_tiles_m: int) -> bool:
    """Limit the SF tile-pack workaround to the plain 128x128 output layout."""
    return not in_config.use_packed_ue8m0 and (not in_config.use_tma_aligned_col_major_sf) and (out_config.sf_block == (128, 128)) and (num_data_tiles_m > 1)

def _get_input_sf_load_spec_v1(
    input_sf_is_tile_packed: bool, in_config: CastInputConfig, num_in_sf_per_block_k: int, num_in_sf_per_data_tile_m: int, packed_sf_tile_elems: int,
) -> tuple[tuple[int, ...], str]:
    """Return the UB shape and dtype required by the selected SF layout."""
    if input_sf_is_tile_packed:
        return ((packed_sf_tile_elems,), 'float32')
    if in_config.use_packed_ue8m0:
        assert num_in_sf_per_block_k % 4 == 0
        return ((num_in_sf_per_block_k // 4, num_in_sf_per_data_tile_m * 4), 'uint8')
    if in_config.use_tma_aligned_col_major_sf:
        return ((num_in_sf_per_block_k, num_in_sf_per_data_tile_m), 'float32')
    return ((num_in_sf_per_data_tile_m, num_in_sf_per_block_k), 'float32')

@tilelang.jit(out_idx=[2, 3], pass_configs=pass_configs_v1)
def get_per_block_cast_lossless_kernel_v1(
    hidden: int, block_m: int, block_k: int, in_config: CastInputConfig, out_config: CastOutputConfig, in_sf_block_m: int=DEFAULT_IN_SF_BLOCK[0],
    in_sf_block_k: int=DEFAULT_IN_SF_BLOCK[1], out_sf_block_m: int=DEFAULT_OUT_SF_BLOCK[0], out_sf_block_k: int=DEFAULT_OUT_SF_BLOCK[1],
    input_sf_is_tile_packed: bool=False,
):
    assert block_m > 0 and block_k > 0
    assert block_m % out_sf_block_m == 0
    assert block_k % out_sf_block_k == 0
    assert out_sf_block_m % in_sf_block_m == 0
    assert out_sf_block_k % in_sf_block_k == 0
    num_tokens = T.symbolic('num_tokens')
    m_num = num_tokens // block_m
    n_num = hidden // block_k
    num_in_sf_per_block_m = block_m // in_sf_block_m
    num_in_sf_per_block_k = block_k // in_sf_block_k
    num_out_sf_per_block_m = block_m // out_sf_block_m
    num_out_sf_per_block_k = block_k // out_sf_block_k
    num_in_sf_per_out_sf_m = out_sf_block_m // in_sf_block_m
    num_in_sf_per_out_sf_k = out_sf_block_k // in_sf_block_k
    out_sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    packed_data_tile_m = min(block_m, 32, NUM_ELEMENTS_PER_BLOCK // block_k)
    assert packed_data_tile_m > 0 and block_m % packed_data_tile_m == 0
    assert packed_data_tile_m % in_sf_block_m == 0
    packed_num_data_tiles_m = block_m // packed_data_tile_m
    packed_num_in_sf_per_data_tile_m = packed_data_tile_m // in_sf_block_m
    data_tile_m = packed_data_tile_m if input_sf_is_tile_packed else block_m
    num_data_tiles_m = packed_num_data_tiles_m if input_sf_is_tile_packed else 1
    num_in_sf_per_data_tile_m = packed_num_in_sf_per_data_tile_m if input_sf_is_tile_packed else num_in_sf_per_block_m
    packed_sf_tile_elems = num_in_sf_per_data_tile_m * num_in_sf_per_block_k
    x_sf_shape = (m_num, n_num, num_data_tiles_m, packed_sf_tile_elems) if input_sf_is_tile_packed else get_sf_shape((num_tokens, hidden), in_config)
    x_sf_load_shape, x_sf_load_dtype = _get_input_sf_load_spec_v1(
        input_sf_is_tile_packed, in_config, num_in_sf_per_block_k, num_in_sf_per_data_tile_m, packed_sf_tile_elems,
    )

    @T.macro
    def load_input_sf_block(dst, x_sf, sf_m, sf_k, data_m, pid_token, pid_hidden):
        if input_sf_is_tile_packed:
            T.copy(x_sf[pid_token, pid_hidden, data_m, 0:packed_sf_tile_elems], dst)
        elif in_config.use_packed_ue8m0:
            T.copy(x_sf[sf_k // 4:sf_k // 4 + num_in_sf_per_block_k // 4, sf_m * 4:sf_m * 4 + num_in_sf_per_data_tile_m * 4], dst)
        elif in_config.use_tma_aligned_col_major_sf:
            T.copy(x_sf[sf_k:sf_k + num_in_sf_per_block_k, sf_m:sf_m + num_in_sf_per_data_tile_m], dst)
        else:
            T.copy(x_sf[sf_m:sf_m + num_in_sf_per_data_tile_m, sf_k:sf_k + num_in_sf_per_block_k], dst)

    @T.macro
    def decode_input_sf_exp(dst, src):
        for i in T.serial(num_in_sf_per_data_tile_m):
            for j in T.serial(num_in_sf_per_block_k):
                if in_config.use_packed_ue8m0:
                    dst[i, j] = T.Cast('int32', src[j // 4, i * 4 + j % 4])
                else:
                    sf_value = T.alloc_var('float32', init=0.0)
                    sf_bits = T.alloc_var('int32', init=0)
                    if input_sf_is_tile_packed:
                        sf_value = src[i * num_in_sf_per_block_k + j]
                    elif in_config.use_tma_aligned_col_major_sf:
                        sf_value = src[j, i]
                    else:
                        sf_value = src[i, j]
                    sf_bits = T.reinterpret('int32', sf_value)
                    dst[i, j] = sf_bits >> 23 & 255

    @T.macro
    def store_output_sf(out_sf, out_sf_fp32_ub, out_sf_exp_ub, i, j, sf_m, sf_k):
        if out_config.use_packed_ue8m0:
            out_sf[sf_k // 4, sf_m * 4 + sf_k % 4] = T.Cast('uint8', out_sf_exp_ub[i, j])
        elif out_config.use_tma_aligned_col_major_sf:
            out_sf[sf_k, sf_m] = out_sf_fp32_ub[i, j]
        else:
            out_sf[sf_m, sf_k] = out_sf_fp32_ub[i, j]

    @T.macro
    def load_and_decode_input_sf(x_sf_exp_ub, x_sf_load_ub, x_sf, sf_m, sf_k, data_m, pid_token, pid_hidden):
        load_input_sf_block(x_sf_load_ub, x_sf, sf_m, sf_k, data_m, pid_token, pid_hidden)
        decode_input_sf_exp(x_sf_exp_ub, x_sf_load_ub)

    @T.macro
    def reduce_output_sf_exp(out_sf_exp_ub, x_sf_exp_ub, data_m):
        for i in T.serial(num_in_sf_per_data_tile_m):
            for j in T.serial(num_in_sf_per_block_k):
                out_sf_exp_ub[
                    (data_m * num_in_sf_per_data_tile_m + i) // num_in_sf_per_out_sf_m,
                    j // num_in_sf_per_out_sf_k,
                ] = T.max(
                    out_sf_exp_ub[
                        (data_m * num_in_sf_per_data_tile_m + i) // num_in_sf_per_out_sf_m,
                        j // num_in_sf_per_out_sf_k,
                    ],
                    x_sf_exp_ub[i, j],
                )

    @T.macro
    def store_output_sf_block(out_sf, out_sf_fp32_ub, out_sf_exp_ub, sf_m, sf_k):
        for i in T.serial(num_out_sf_per_block_m):
            for j in T.serial(num_out_sf_per_block_k):
                out_sf_bits = T.alloc_var('int32', init=0)
                out_sf_bits = out_sf_exp_ub[i, j] << 23
                out_sf_fp32_ub[i, j] = T.reinterpret('float32', out_sf_bits)
                store_output_sf(out_sf, out_sf_fp32_ub, out_sf_exp_ub, i, j, sf_m + i, sf_k + j)

    @T.macro
    def update_relative_sf_exp(x_sf_exp_ub, out_sf_exp_ub, data_m):
        for i in T.serial(num_in_sf_per_data_tile_m):
            for j in T.serial(num_in_sf_per_block_k):
                x_sf_exp_ub[i, j] = (
                    x_sf_exp_ub[i, j]
                    - out_sf_exp_ub[
                        (data_m * num_in_sf_per_data_tile_m + i) // num_in_sf_per_out_sf_m,
                        j // num_in_sf_per_out_sf_k,
                    ]
                    + 127
                )

    @T.macro
    def expand_relative_sf(x_relative_sf_ub, x_sf_exp_ub):
        for i in T.serial(data_tile_m):
            for j in T.serial(block_k):
                relative_sf_bits = T.alloc_var('int32', init=0)
                relative_sf_bits = x_sf_exp_ub[i // in_sf_block_m, j // in_sf_block_k] << 23
                x_relative_sf_ub[i, j] = T.reinterpret('float32', relative_sf_bits)

    @T.macro
    def cast_data_tile(x, out, x_in_ub, x_out_ub, x_relative_sf_ub, row_offset, col_offset):
        T.copy(x[row_offset:row_offset + data_tile_m, col_offset:col_offset + block_k], x_in_ub)
        T.tile.cast(x_out_ub, x_in_ub, mode='CAST_NONE', count=data_tile_m * block_k)
        T.tile.mul(x_out_ub, x_out_ub, x_relative_sf_ub)
        T.copy(x_out_ub, out[row_offset:row_offset + data_tile_m, col_offset:col_offset + block_k])

    @T.prim_func
    def per_block_cast_lossless_kernel_v1(
        x: T.Tensor([num_tokens, hidden], INPUT_DTYPE), x_sf: T.Tensor(x_sf_shape, in_config.sf_dtype),
        out: T.Tensor([num_tokens, hidden], OUTPUT_DTYPE), out_sf: T.Tensor(out_sf_shape, out_config.sf_dtype),
    ):
        with T.Kernel(m_num * n_num, threads=1, is_npu=True) as cid:
            with T.Scope('V'):
                pid_token = cid // n_num
                pid_hidden = cid % n_num
                row_offset = pid_token * block_m
                col_offset = pid_hidden * block_k
                x_in_ub = T.alloc_ub((data_tile_m, block_k), INPUT_DTYPE)
                x_out_ub = T.alloc_ub((data_tile_m, block_k), OUTPUT_DTYPE)
                x_relative_sf_ub = T.alloc_ub((data_tile_m, block_k), 'float32')
                x_sf_load_ub = T.alloc_ub(x_sf_load_shape, x_sf_load_dtype)
                x_sf_exp_ub = T.alloc_ub((num_in_sf_per_data_tile_m, num_in_sf_per_block_k), 'int32')
                out_sf_exp_ub = T.alloc_ub((num_out_sf_per_block_m, num_out_sf_per_block_k), 'int32')
                out_sf_fp32_ub = T.alloc_ub((num_out_sf_per_block_m, num_out_sf_per_block_k), 'float32')
                sf_row_offset = pid_token * num_in_sf_per_block_m
                sf_col_offset = pid_hidden * num_in_sf_per_block_k
                out_sf_row_offset = pid_token * num_out_sf_per_block_m
                out_sf_col_offset = pid_hidden * num_out_sf_per_block_k
                for i in T.serial(num_out_sf_per_block_m):
                    for j in T.serial(num_out_sf_per_block_k):
                        out_sf_exp_ub[i, j] = 0
                for data_m in T.serial(num_data_tiles_m):
                    load_and_decode_input_sf(
                        x_sf_exp_ub, x_sf_load_ub, x_sf, sf_row_offset + data_m * num_in_sf_per_data_tile_m, sf_col_offset, data_m, pid_token,
                        pid_hidden,
                    )
                    reduce_output_sf_exp(out_sf_exp_ub, x_sf_exp_ub, data_m)
                for i in T.serial(num_out_sf_per_block_m):
                    for j in T.serial(num_out_sf_per_block_k):
                        out_sf_exp_ub[i, j] = T.max(out_sf_exp_ub[i, j] - 6, 0)
                store_output_sf_block(out_sf, out_sf_fp32_ub, out_sf_exp_ub, out_sf_row_offset, out_sf_col_offset)
                for data_m in T.serial(num_data_tiles_m):
                    if input_sf_is_tile_packed:
                        load_and_decode_input_sf(
                            x_sf_exp_ub, x_sf_load_ub, x_sf, sf_row_offset + data_m * num_in_sf_per_data_tile_m, sf_col_offset, data_m, pid_token,
                            pid_hidden,
                        )
                    update_relative_sf_exp(x_sf_exp_ub, out_sf_exp_ub, data_m)
                    expand_relative_sf(x_relative_sf_ub, x_sf_exp_ub)
                    cast_data_tile(x, out, x_in_ub, x_out_ub, x_relative_sf_ub, row_offset + data_m * data_tile_m, col_offset)
    return per_block_cast_lossless_kernel_v1

@tilelang.jit(out_idx=[2, 3], pass_configs=pass_configs)
def get_per_block_cast_lossless_kernel(
    hidden: int, block_m: int, block_k: int, in_config: CastInputConfig, out_config: CastOutputConfig, logical_hidden: int=-1,
    in_sf_block_m: int=DEFAULT_IN_SF_BLOCK[0], in_sf_block_k: int=DEFAULT_IN_SF_BLOCK[1], out_sf_block_m: int=DEFAULT_OUT_SF_BLOCK[0],
    out_sf_block_k: int=DEFAULT_OUT_SF_BLOCK[1],
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
    num_tokens = T.symbolic('num_tokens')
    m_num = T.ceildiv(num_tokens, block_m)
    n_num = T.ceildiv(hidden, block_k)
    num_in_sf_per_block_m = block_m // in_sf_block_m
    num_in_sf_per_block_k = block_k // in_sf_block_k
    num_out_sf_per_block_k = block_k // out_sf_block_k
    num_in_sf_per_out_sf_m = out_sf_block_m // in_sf_block_m
    num_in_sf_per_out_sf_k = out_sf_block_k // in_sf_block_k
    compact_sf_row_stride = align_up(num_in_sf_per_block_k, 8)
    compact_sf_row_stride_bytes = compact_sf_row_stride * 4
    num_in_sf_per_out_sf_bytes = num_in_sf_per_out_sf_k * 4
    num_pipeline_pairs = num_out_sf_per_block_k * 2
    num_initial_prefetch_stages = min(4, num_pipeline_pairs)
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    out_sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    vector_data_tile_m = block_m // VEC_NUM
    num_in_sf_per_vector_m = vector_data_tile_m // in_sf_block_m
    use_vid_local_sf = not in_config.use_packed_ue8m0 and (not out_config.use_packed_ue8m0)
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

    @T.macro
    def compute_two_pattern_max(
        x_sf_exp_ub, block_idx, xsf_row_offset_i32_ub, xsf_row_offset_u32_ub, e0_ub, e1_ub, e2_ub, e3_ub, max01_ub, max23_ub, out_exp_ub,
    ):
        sf_k_base_bytes = block_idx * num_in_sf_per_out_sf_bytes
        T.tile.arith_progression(xsf_row_offset_i32_ub, 0, compact_sf_row_stride_bytes, vector_data_tile_m)
        T.pipe_barrier('v')
        T.tile.gather(e0_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, sf_k_base_bytes)
        T.tile.gather(e1_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, sf_k_base_bytes + 4)
        T.tile.gather(e2_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, sf_k_base_bytes + 8)
        T.tile.gather(e3_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, sf_k_base_bytes + 12)
        T.pipe_barrier('v')
        T.tile.max(max01_ub, e0_ub, e1_ub)
        T.tile.max(max23_ub, e2_ub, e3_ub)
        T.pipe_barrier('v')
        T.tile.max(out_exp_ub, max01_ub, max23_ub)
        T.pipe_barrier('v')
        T.tile.add(out_exp_ub, out_exp_ub, -6)
        T.pipe_barrier('v')
        T.tile.max(out_exp_ub, out_exp_ub, 0)
        T.pipe_barrier('v')

    @T.macro
    def load_two_pair_input(x, x_in0_ub, x_in1_ub, row_offset, col_offset, input_event):
        T.wait_flag('v', 'mte2', input_event)
        T.copy(x[row_offset:row_offset + vector_data_tile_m, col_offset:col_offset + in_sf_block_k], x_in0_ub)
        T.copy(x[row_offset:row_offset + vector_data_tile_m, col_offset + in_sf_block_k:col_offset + 2 * in_sf_block_k], x_in1_ub)
        T.set_flag('mte2', 'v', input_event)

    @T.macro
    def compute_two_pair_values(
        e0_ub, e1_ub, out_exp_ub, relative_exp0_ub, relative_exp1_ub, relative_bits0_ub, relative_bits1_ub, relative_sf0_view_ub,
        relative_sf1_view_ub, relative_sf_tile0_ub, relative_sf_tile1_ub, x_in0_ub, x_in1_ub, x_out0_ub, x_out1_ub, input_event, output_event,
    ):
        tile_elem_count = vector_data_tile_m * in_sf_block_k
        T.tile.sub(relative_exp0_ub, e0_ub, out_exp_ub)
        T.tile.sub(relative_exp1_ub, e1_ub, out_exp_ub)
        T.pipe_barrier('v')
        T.tile.add(relative_exp0_ub, relative_exp0_ub, 127)
        T.tile.add(relative_exp1_ub, relative_exp1_ub, 127)
        T.pipe_barrier('v')
        T.tile.bitwise_lshift(relative_bits0_ub, relative_exp0_ub, 23)
        T.tile.bitwise_lshift(relative_bits1_ub, relative_exp1_ub, 23)
        T.pipe_barrier('v')
        T.tile.broadcast(relative_sf_tile0_ub, relative_sf0_view_ub, axis=1)
        T.tile.broadcast(relative_sf_tile1_ub, relative_sf1_view_ub, axis=1)
        T.wait_flag('mte2', 'v', input_event)
        T.wait_flag('mte3', 'v', output_event)
        T.tile.cast(x_out0_ub, x_in0_ub, mode='CAST_NONE', count=tile_elem_count)
        T.tile.cast(x_out1_ub, x_in1_ub, mode='CAST_NONE', count=tile_elem_count)
        T.pipe_barrier('v')
        T.set_flag('v', 'mte2', input_event)
        T.tile.mul(x_out0_ub, x_out0_ub, relative_sf_tile0_ub)
        T.tile.mul(x_out1_ub, x_out1_ub, relative_sf_tile1_ub)
        T.set_flag('v', 'mte3', output_event)

    @T.macro
    def store_two_pair_output(out, x_out0_ub, x_out1_ub, row_offset, col_offset, output_event):
        T.wait_flag('v', 'mte3', output_event)
        T.copy(x_out0_ub, out[row_offset:row_offset + vector_data_tile_m, col_offset:col_offset + in_sf_block_k])
        T.copy(x_out1_ub, out[row_offset:row_offset + vector_data_tile_m, col_offset + in_sf_block_k:col_offset + 2 * in_sf_block_k])
        T.set_flag('mte3', 'v', output_event)

    @T.macro
    def apply_relative_sf_tiles_compact_pattern_pipeline(x_sf, x, out, sf_m, sf_k, row_offset, col_offset):
        x_sf_load_ub = T.alloc_ub((vector_data_tile_m, compact_sf_row_stride), 'float32')
        x_sf_bits_ub = T.alloc_ub((vector_data_tile_m, compact_sf_row_stride), 'int32')
        x_sf_exp_ub = T.alloc_ub((vector_data_tile_m, compact_sf_row_stride), 'int32')
        T.reinterpretcast(x_sf_bits_ub, x_sf_load_ub, 'int')
        x_in00_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in01_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in10_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in11_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_out00_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out01_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out10_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out11_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        xsf_row_offset_i32_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        xsf_row_offset_u32_ub = T.alloc_ub((vector_data_tile_m,), 'uint32')
        e0_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        e1_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        e2_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        e3_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        max01_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        max23_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        out_exp_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        relative_exp0_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        relative_exp1_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        relative_bits0_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        relative_bits1_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        relative_sf0_view_ub = T.alloc_ub((vector_data_tile_m, 1), 'float32')
        relative_sf1_view_ub = T.alloc_ub((vector_data_tile_m, 1), 'float32')
        relative_sf_tile0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), 'float32')
        relative_sf_tile1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), 'float32')
        T.reinterpretcast(xsf_row_offset_u32_ub, xsf_row_offset_i32_ub, 'uint32_t')
        T.reinterpretcast(relative_sf0_view_ub, relative_bits0_ub, 'float')
        T.reinterpretcast(relative_sf1_view_ub, relative_bits1_ub, 'float')
        T.copy(x_sf[sf_m:sf_m + num_in_sf_per_vector_m, sf_k:sf_k + num_in_sf_per_block_k], x_sf_load_ub)
        T.set_flag('mte2', 'v', 4)
        for slot in T.unroll(2):
            T.set_flag('v', 'mte2', slot)
            T.set_flag('mte3', 'v', slot)
        load_two_pair_input(x, x_in00_ub, x_in01_ub, row_offset, col_offset, 0)
        load_two_pair_input(x, x_in10_ub, x_in11_ub, row_offset, col_offset + 2 * in_sf_block_k, 1)
        T.wait_flag('mte2', 'v', 4)
        T.tile.bitwise_rshift(x_sf_exp_ub, x_sf_bits_ub, 23)
        T.pipe_barrier('v')
        compute_two_pattern_max(
            x_sf_exp_ub, 0, xsf_row_offset_i32_ub, xsf_row_offset_u32_ub, e0_ub, e1_ub, e2_ub, e3_ub, max01_ub, max23_ub, out_exp_ub,
        )
        compute_two_pair_values(
            e0_ub, e1_ub, out_exp_ub, relative_exp0_ub, relative_exp1_ub, relative_bits0_ub, relative_bits1_ub, relative_sf0_view_ub,
            relative_sf1_view_ub, relative_sf_tile0_ub, relative_sf_tile1_ub, x_in00_ub, x_in01_ub, x_out00_ub, x_out01_ub, 0, 0,
        )
        if num_out_sf_per_block_k == 2:
            load_two_pair_input(x, x_in00_ub, x_in01_ub, row_offset, col_offset + out_sf_block_k, 0)
        store_two_pair_output(out, x_out00_ub, x_out01_ub, row_offset, col_offset, 0)
        compute_two_pair_values(
            e2_ub, e3_ub, out_exp_ub, relative_exp0_ub, relative_exp1_ub, relative_bits0_ub, relative_bits1_ub, relative_sf0_view_ub,
            relative_sf1_view_ub, relative_sf_tile0_ub, relative_sf_tile1_ub, x_in10_ub, x_in11_ub, x_out10_ub, x_out11_ub, 1, 1,
        )
        if num_out_sf_per_block_k == 2:
            load_two_pair_input(x, x_in10_ub, x_in11_ub, row_offset, col_offset + out_sf_block_k + 2 * in_sf_block_k, 1)
        store_two_pair_output(out, x_out10_ub, x_out11_ub, row_offset, col_offset + 2 * in_sf_block_k, 1)
        if num_out_sf_per_block_k == 2:
            compute_two_pattern_max(
                x_sf_exp_ub, 1, xsf_row_offset_i32_ub, xsf_row_offset_u32_ub, e0_ub, e1_ub, e2_ub, e3_ub, max01_ub, max23_ub, out_exp_ub,
            )
            compute_two_pair_values(
                e0_ub, e1_ub, out_exp_ub, relative_exp0_ub, relative_exp1_ub, relative_bits0_ub, relative_bits1_ub, relative_sf0_view_ub,
                relative_sf1_view_ub, relative_sf_tile0_ub, relative_sf_tile1_ub, x_in00_ub, x_in01_ub, x_out00_ub, x_out01_ub, 0, 0,
            )
            store_two_pair_output(out, x_out00_ub, x_out01_ub, row_offset, col_offset + out_sf_block_k, 0)
            compute_two_pair_values(
                e2_ub, e3_ub, out_exp_ub, relative_exp0_ub, relative_exp1_ub, relative_bits0_ub, relative_bits1_ub, relative_sf0_view_ub,
                relative_sf1_view_ub, relative_sf_tile0_ub, relative_sf_tile1_ub, x_in10_ub, x_in11_ub, x_out10_ub, x_out11_ub, 1, 1,
            )
            store_two_pair_output(out, x_out10_ub, x_out11_ub, row_offset, col_offset + out_sf_block_k + 2 * in_sf_block_k, 1)
        T.wait_flag('v', 'mte2', 0)
        T.wait_flag('v', 'mte2', 1)
        T.wait_flag('mte3', 'v', 0)
        T.wait_flag('mte3', 'v', 1)

    @T.macro
    def apply_relative_sf_tiles_compact_pattern_tail64(x_sf, x, out, sf_m, sf_k, row_offset, col_offset):
        tail_x_sf_load_ub = T.alloc_ub((vector_data_tile_m, compact_sf_row_stride), 'float32')
        tail_x_sf_bits_ub = T.alloc_ub((vector_data_tile_m, compact_sf_row_stride), 'int32')
        tail_x_sf_exp_ub = T.alloc_ub((vector_data_tile_m, compact_sf_row_stride), 'int32')
        T.reinterpretcast(tail_x_sf_bits_ub, tail_x_sf_load_ub, 'int')
        tail_x_in00_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        tail_x_in01_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        tail_x_out00_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        tail_x_out01_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        tail_xsf_row_offset_i32_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        tail_xsf_row_offset_u32_ub = T.alloc_ub((vector_data_tile_m,), 'uint32')
        tail_e0_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        tail_e1_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        tail_e2_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        tail_e3_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        tail_max01_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        tail_max23_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        tail_out_exp_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        tail_relative_exp0_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        tail_relative_exp1_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        tail_relative_bits0_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        tail_relative_bits1_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        tail_relative_sf0_view_ub = T.alloc_ub((vector_data_tile_m, 1), 'float32')
        tail_relative_sf1_view_ub = T.alloc_ub((vector_data_tile_m, 1), 'float32')
        tail_relative_sf_tile0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), 'float32')
        tail_relative_sf_tile1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), 'float32')
        T.reinterpretcast(tail_xsf_row_offset_u32_ub, tail_xsf_row_offset_i32_ub, 'uint32_t')
        T.reinterpretcast(tail_relative_sf0_view_ub, tail_relative_bits0_ub, 'float')
        T.reinterpretcast(tail_relative_sf1_view_ub, tail_relative_bits1_ub, 'float')
        T.copy(x_sf[sf_m:sf_m + num_in_sf_per_vector_m, sf_k:sf_k + num_in_sf_per_block_k], tail_x_sf_load_ub)
        T.set_flag('mte2', 'v', 4)
        T.set_flag('v', 'mte2', 0)
        T.set_flag('mte3', 'v', 0)
        load_two_pair_input(x, tail_x_in00_ub, tail_x_in01_ub, row_offset, col_offset, 0)
        T.wait_flag('mte2', 'v', 4)
        T.tile.bitwise_rshift(tail_x_sf_exp_ub, tail_x_sf_bits_ub, 23)
        T.pipe_barrier('v')
        compute_two_pattern_max(
            tail_x_sf_exp_ub, 0, tail_xsf_row_offset_i32_ub, tail_xsf_row_offset_u32_ub, tail_e0_ub, tail_e1_ub, tail_e2_ub, tail_e3_ub,
            tail_max01_ub, tail_max23_ub, tail_out_exp_ub,
        )
        compute_two_pair_values(
            tail_e0_ub, tail_e1_ub, tail_out_exp_ub, tail_relative_exp0_ub, tail_relative_exp1_ub, tail_relative_bits0_ub, tail_relative_bits1_ub,
            tail_relative_sf0_view_ub, tail_relative_sf1_view_ub, tail_relative_sf_tile0_ub, tail_relative_sf_tile1_ub, tail_x_in00_ub,
            tail_x_in01_ub, tail_x_out00_ub, tail_x_out01_ub, 0, 0,
        )
        store_two_pair_output(out, tail_x_out00_ub, tail_x_out01_ub, row_offset, col_offset, 0)
        T.wait_flag('v', 'mte2', 0)
        T.wait_flag('mte3', 'v', 0)

    @T.macro
    def apply_relative_sf_tiles_block_pipeline_max4(x_sf, x, out, sf_m, sf_k, row_offset, col_offset):
        tile_elem_count = vector_data_tile_m * in_sf_block_k
        x_sf_load_ub = T.alloc_ub((vector_data_tile_m, num_in_sf_per_block_k), 'float32')
        x_sf_bits_ub = T.alloc_ub((vector_data_tile_m, num_in_sf_per_block_k), 'int32')
        x_sf_exp_ub = T.alloc_ub((vector_data_tile_m, num_in_sf_per_block_k), 'int32')
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
        xsf_row_offset_i32_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        xsf_row_offset_u32_ub = T.alloc_ub((vector_data_tile_m,), 'uint32')
        e0_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        e1_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        e2_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        e3_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        max01_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        max23_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        max_exp_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        out_exp_col_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        relative_exp0_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        relative_exp1_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        relative_bits0_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        relative_bits1_ub = T.alloc_ub((vector_data_tile_m,), 'int32')
        relative_sf0_view_ub = T.alloc_ub((vector_data_tile_m, 1), 'float32')
        relative_sf1_view_ub = T.alloc_ub((vector_data_tile_m, 1), 'float32')
        relative_sf_tile0_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), 'float32')
        relative_sf_tile1_ub = T.alloc_ub((vector_data_tile_m, in_sf_block_k), 'float32')
        T.reinterpretcast(x_sf_bits_ub, x_sf_load_ub, 'int')
        T.reinterpretcast(xsf_row_offset_u32_ub, xsf_row_offset_i32_ub, 'uint32_t')
        T.reinterpretcast(relative_sf0_view_ub, relative_bits0_ub, 'float')
        T.reinterpretcast(relative_sf1_view_ub, relative_bits1_ub, 'float')
        for input_slot in T.unroll(4):
            T.set_flag('v', 'mte2', input_slot)
        T.copy(x_sf[sf_m:sf_m + num_in_sf_per_vector_m, sf_k:sf_k + num_in_sf_per_block_k], x_sf_load_ub)
        T.set_flag('mte2', 'v', 4)
        for preload_stage in T.unroll(num_initial_prefetch_stages):
            preload_slot = preload_stage
            preload_block_idx = preload_stage // 2
            preload_pair_idx = preload_stage % 2
            preload_line0_idx = preload_pair_idx * 2
            preload_line1_idx = preload_line0_idx + 1
            preload_col_base = col_offset + preload_block_idx * out_sf_block_k
            T.wait_flag('v', 'mte2', preload_slot)
            if preload_slot == 0:
                T.copy(
                    x[
                        row_offset:row_offset + vector_data_tile_m,
                        preload_col_base + preload_line0_idx * in_sf_block_k:preload_col_base + (preload_line0_idx + 1) * in_sf_block_k,
                    ],
                    x_in0_slot0_ub,
                )
                T.copy(
                    x[
                        row_offset:row_offset + vector_data_tile_m,
                        preload_col_base + preload_line1_idx * in_sf_block_k:preload_col_base + (preload_line1_idx + 1) * in_sf_block_k,
                    ],
                    x_in1_slot0_ub,
                )
            elif preload_slot == 1:
                T.copy(
                    x[
                        row_offset:row_offset + vector_data_tile_m,
                        preload_col_base + preload_line0_idx * in_sf_block_k:preload_col_base + (preload_line0_idx + 1) * in_sf_block_k,
                    ],
                    x_in0_slot1_ub,
                )
                T.copy(
                    x[
                        row_offset:row_offset + vector_data_tile_m,
                        preload_col_base + preload_line1_idx * in_sf_block_k:preload_col_base + (preload_line1_idx + 1) * in_sf_block_k,
                    ],
                    x_in1_slot1_ub,
                )
            elif preload_slot == 2:
                T.copy(
                    x[
                        row_offset:row_offset + vector_data_tile_m,
                        preload_col_base + preload_line0_idx * in_sf_block_k:preload_col_base + (preload_line0_idx + 1) * in_sf_block_k,
                    ],
                    x_in0_slot2_ub,
                )
                T.copy(
                    x[
                        row_offset:row_offset + vector_data_tile_m,
                        preload_col_base + preload_line1_idx * in_sf_block_k:preload_col_base + (preload_line1_idx + 1) * in_sf_block_k,
                    ],
                    x_in1_slot2_ub,
                )
            else:
                T.copy(
                    x[
                        row_offset:row_offset + vector_data_tile_m,
                        preload_col_base + preload_line0_idx * in_sf_block_k:preload_col_base + (preload_line0_idx + 1) * in_sf_block_k,
                    ],
                    x_in0_slot3_ub,
                )
                T.copy(
                    x[
                        row_offset:row_offset + vector_data_tile_m,
                        preload_col_base + preload_line1_idx * in_sf_block_k:preload_col_base + (preload_line1_idx + 1) * in_sf_block_k,
                    ],
                    x_in1_slot3_ub,
                )
            T.set_flag('mte2', 'v', preload_slot)
        T.wait_flag('mte2', 'v', 4)
        T.tile.bitwise_rshift(x_sf_exp_ub, x_sf_bits_ub, 23)
        T.pipe_barrier('v')
        for block_idx in T.unroll(num_out_sf_per_block_k):
            sf_k_base = block_idx * num_in_sf_per_out_sf_k
            T.tile.arith_progression(xsf_row_offset_i32_ub, (sf_k_base + 0) * 4, num_in_sf_per_block_k * 4, vector_data_tile_m)
            T.tile.gather(e0_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 0)
            T.tile.arith_progression(xsf_row_offset_i32_ub, (sf_k_base + 1) * 4, num_in_sf_per_block_k * 4, vector_data_tile_m)
            T.tile.gather(e1_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 0)
            T.tile.arith_progression(xsf_row_offset_i32_ub, (sf_k_base + 2) * 4, num_in_sf_per_block_k * 4, vector_data_tile_m)
            T.tile.gather(e2_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 0)
            T.tile.arith_progression(xsf_row_offset_i32_ub, (sf_k_base + 3) * 4, num_in_sf_per_block_k * 4, vector_data_tile_m)
            T.tile.gather(e3_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 0)
            T.pipe_barrier('v')
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
                if stage + 4 < num_pipeline_pairs:
                    future_stage = stage + 4
                    future_block_idx = future_stage // 2
                    future_pair_idx = future_stage % 2
                    future_line0_idx = future_pair_idx * 2
                    future_line1_idx = future_line0_idx + 1
                    future_col_base = col_offset + future_block_idx * out_sf_block_k
                    T.wait_flag('v', 'mte2', input_slot)
                    if input_slot == 0:
                        T.copy(
                            x[
                                row_offset:row_offset + vector_data_tile_m,
                                future_col_base + future_line0_idx * in_sf_block_k:future_col_base + (future_line0_idx + 1) * in_sf_block_k,
                            ],
                            x_in0_slot0_ub,
                        )
                        T.copy(
                            x[
                                row_offset:row_offset + vector_data_tile_m,
                                future_col_base + future_line1_idx * in_sf_block_k:future_col_base + (future_line1_idx + 1) * in_sf_block_k,
                            ],
                            x_in1_slot0_ub,
                        )
                    elif input_slot == 1:
                        T.copy(
                            x[
                                row_offset:row_offset + vector_data_tile_m,
                                future_col_base + future_line0_idx * in_sf_block_k:future_col_base + (future_line0_idx + 1) * in_sf_block_k,
                            ],
                            x_in0_slot1_ub,
                        )
                        T.copy(
                            x[
                                row_offset:row_offset + vector_data_tile_m,
                                future_col_base + future_line1_idx * in_sf_block_k:future_col_base + (future_line1_idx + 1) * in_sf_block_k,
                            ],
                            x_in1_slot1_ub,
                        )
                    elif input_slot == 2:
                        T.copy(
                            x[
                                row_offset:row_offset + vector_data_tile_m,
                                future_col_base + future_line0_idx * in_sf_block_k:future_col_base + (future_line0_idx + 1) * in_sf_block_k,
                            ],
                            x_in0_slot2_ub,
                        )
                        T.copy(
                            x[
                                row_offset:row_offset + vector_data_tile_m,
                                future_col_base + future_line1_idx * in_sf_block_k:future_col_base + (future_line1_idx + 1) * in_sf_block_k,
                            ],
                            x_in1_slot2_ub,
                        )
                    else:
                        T.copy(
                            x[
                                row_offset:row_offset + vector_data_tile_m,
                                future_col_base + future_line0_idx * in_sf_block_k:future_col_base + (future_line0_idx + 1) * in_sf_block_k,
                            ],
                            x_in0_slot3_ub,
                        )
                        T.copy(
                            x[
                                row_offset:row_offset + vector_data_tile_m,
                                future_col_base + future_line1_idx * in_sf_block_k:future_col_base + (future_line1_idx + 1) * in_sf_block_k,
                            ],
                            x_in1_slot3_ub,
                        )
                    T.set_flag('mte2', 'v', input_slot)
                if stage >= 4:
                    T.wait_flag('mte3', 'v', output_slot)
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
                T.pipe_barrier('v')
                T.tile.broadcast(relative_sf_tile0_ub, relative_sf0_view_ub, axis=1)
                T.tile.broadcast(relative_sf_tile1_ub, relative_sf1_view_ub, axis=1)
                T.wait_flag('mte2', 'v', input_slot)
                if output_slot == 0:
                    if input_slot == 0:
                        T.tile.cast(x_out00_ub, x_in0_slot0_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out01_ub, x_in1_slot0_ub, mode='CAST_NONE', count=tile_elem_count)
                    elif input_slot == 1:
                        T.tile.cast(x_out00_ub, x_in0_slot1_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out01_ub, x_in1_slot1_ub, mode='CAST_NONE', count=tile_elem_count)
                    elif input_slot == 2:
                        T.tile.cast(x_out00_ub, x_in0_slot2_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out01_ub, x_in1_slot2_ub, mode='CAST_NONE', count=tile_elem_count)
                    else:
                        T.tile.cast(x_out00_ub, x_in0_slot3_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out01_ub, x_in1_slot3_ub, mode='CAST_NONE', count=tile_elem_count)
                    if stage + 4 < num_pipeline_pairs:
                        T.set_flag('v', 'mte2', input_slot)
                    T.tile.mul(x_out00_ub, x_out00_ub, relative_sf_tile0_ub)
                    T.tile.mul(x_out01_ub, x_out01_ub, relative_sf_tile1_ub)
                elif output_slot == 1:
                    if input_slot == 0:
                        T.tile.cast(x_out10_ub, x_in0_slot0_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out11_ub, x_in1_slot0_ub, mode='CAST_NONE', count=tile_elem_count)
                    elif input_slot == 1:
                        T.tile.cast(x_out10_ub, x_in0_slot1_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out11_ub, x_in1_slot1_ub, mode='CAST_NONE', count=tile_elem_count)
                    elif input_slot == 2:
                        T.tile.cast(x_out10_ub, x_in0_slot2_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out11_ub, x_in1_slot2_ub, mode='CAST_NONE', count=tile_elem_count)
                    else:
                        T.tile.cast(x_out10_ub, x_in0_slot3_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out11_ub, x_in1_slot3_ub, mode='CAST_NONE', count=tile_elem_count)
                    if stage + 4 < num_pipeline_pairs:
                        T.set_flag('v', 'mte2', input_slot)
                    T.tile.mul(x_out10_ub, x_out10_ub, relative_sf_tile0_ub)
                    T.tile.mul(x_out11_ub, x_out11_ub, relative_sf_tile1_ub)
                elif output_slot == 2:
                    if input_slot == 0:
                        T.tile.cast(x_out20_ub, x_in0_slot0_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out21_ub, x_in1_slot0_ub, mode='CAST_NONE', count=tile_elem_count)
                    elif input_slot == 1:
                        T.tile.cast(x_out20_ub, x_in0_slot1_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out21_ub, x_in1_slot1_ub, mode='CAST_NONE', count=tile_elem_count)
                    elif input_slot == 2:
                        T.tile.cast(x_out20_ub, x_in0_slot2_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out21_ub, x_in1_slot2_ub, mode='CAST_NONE', count=tile_elem_count)
                    else:
                        T.tile.cast(x_out20_ub, x_in0_slot3_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out21_ub, x_in1_slot3_ub, mode='CAST_NONE', count=tile_elem_count)
                    if stage + 4 < num_pipeline_pairs:
                        T.set_flag('v', 'mte2', input_slot)
                    T.tile.mul(x_out20_ub, x_out20_ub, relative_sf_tile0_ub)
                    T.tile.mul(x_out21_ub, x_out21_ub, relative_sf_tile1_ub)
                else:
                    if input_slot == 0:
                        T.tile.cast(x_out30_ub, x_in0_slot0_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out31_ub, x_in1_slot0_ub, mode='CAST_NONE', count=tile_elem_count)
                    elif input_slot == 1:
                        T.tile.cast(x_out30_ub, x_in0_slot1_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out31_ub, x_in1_slot1_ub, mode='CAST_NONE', count=tile_elem_count)
                    elif input_slot == 2:
                        T.tile.cast(x_out30_ub, x_in0_slot2_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out31_ub, x_in1_slot2_ub, mode='CAST_NONE', count=tile_elem_count)
                    else:
                        T.tile.cast(x_out30_ub, x_in0_slot3_ub, mode='CAST_NONE', count=tile_elem_count)
                        T.tile.cast(x_out31_ub, x_in1_slot3_ub, mode='CAST_NONE', count=tile_elem_count)
                    if stage + 4 < num_pipeline_pairs:
                        T.set_flag('v', 'mte2', input_slot)
                    T.tile.mul(x_out30_ub, x_out30_ub, relative_sf_tile0_ub)
                    T.tile.mul(x_out31_ub, x_out31_ub, relative_sf_tile1_ub)
                T.set_flag('v', 'mte3', output_slot)
                T.wait_flag('v', 'mte3', output_slot)
                current_col_base = col_offset + block_idx * out_sf_block_k
                if output_slot == 0:
                    T.copy(
                        x_out00_ub,
                        out[
                            row_offset:row_offset + vector_data_tile_m,
                            current_col_base + line0_idx * in_sf_block_k:current_col_base + (line0_idx + 1) * in_sf_block_k,
                        ],
                    )
                    T.copy(
                        x_out01_ub,
                        out[
                            row_offset:row_offset + vector_data_tile_m,
                            current_col_base + line1_idx * in_sf_block_k:current_col_base + (line1_idx + 1) * in_sf_block_k,
                        ],
                    )
                elif output_slot == 1:
                    T.copy(
                        x_out10_ub,
                        out[
                            row_offset:row_offset + vector_data_tile_m,
                            current_col_base + line0_idx * in_sf_block_k:current_col_base + (line0_idx + 1) * in_sf_block_k,
                        ],
                    )
                    T.copy(
                        x_out11_ub,
                        out[
                            row_offset:row_offset + vector_data_tile_m,
                            current_col_base + line1_idx * in_sf_block_k:current_col_base + (line1_idx + 1) * in_sf_block_k,
                        ],
                    )
                elif output_slot == 2:
                    T.copy(
                        x_out20_ub,
                        out[
                            row_offset:row_offset + vector_data_tile_m,
                            current_col_base + line0_idx * in_sf_block_k:current_col_base + (line0_idx + 1) * in_sf_block_k,
                        ],
                    )
                    T.copy(
                        x_out21_ub,
                        out[
                            row_offset:row_offset + vector_data_tile_m,
                            current_col_base + line1_idx * in_sf_block_k:current_col_base + (line1_idx + 1) * in_sf_block_k,
                        ],
                    )
                else:
                    T.copy(
                        x_out30_ub,
                        out[
                            row_offset:row_offset + vector_data_tile_m,
                            current_col_base + line0_idx * in_sf_block_k:current_col_base + (line0_idx + 1) * in_sf_block_k,
                        ],
                    )
                    T.copy(
                        x_out31_ub,
                        out[
                            row_offset:row_offset + vector_data_tile_m,
                            current_col_base + line1_idx * in_sf_block_k:current_col_base + (line1_idx + 1) * in_sf_block_k,
                        ],
                    )
                if stage + 4 < num_pipeline_pairs:
                    T.set_flag('mte3', 'v', output_slot)
        T.pipe_barrier('mte3')

    @T.prim_func
    def per_block_cast_lossless_kernel(
        x: T.Tensor([num_tokens, hidden], INPUT_DTYPE), x_sf: T.Tensor(x_sf_shape, in_config.sf_dtype),
        out: T.Tensor([num_tokens, hidden], OUTPUT_DTYPE), out_sf: T.Tensor(out_sf_shape, out_config.sf_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            with T.Scope('V'):
                pid_token = cid // n_num
                pid_hidden = cid % n_num
                row_offset = pid_token * block_m + vid * vector_data_tile_m
                col_offset = pid_hidden * block_k
                sf_row_offset = pid_token * num_in_sf_per_block_m
                sf_col_offset = pid_hidden * num_in_sf_per_block_k
                if use_compact_pattern_pipeline:
                    local_sf_row_offset = sf_row_offset + vid * num_in_sf_per_vector_m
                    if col_offset + block_k <= logical_hidden:
                        apply_relative_sf_tiles_compact_pattern_pipeline(x_sf, x, out, local_sf_row_offset, sf_col_offset, row_offset, col_offset)
                    elif col_offset < logical_hidden:
                        if logical_hidden - col_offset <= 2 * in_sf_block_k:
                            apply_relative_sf_tiles_compact_pattern_tail64(x_sf, x, out, local_sf_row_offset, sf_col_offset, row_offset, col_offset)
                        else:
                            apply_relative_sf_tiles_compact_pattern_pipeline(x_sf, x, out, local_sf_row_offset, sf_col_offset, row_offset, col_offset)
                elif use_local_max4_fast_path:
                    local_sf_row_offset = sf_row_offset + vid * num_in_sf_per_vector_m
                    apply_relative_sf_tiles_block_pipeline_max4(x_sf, x, out, local_sf_row_offset, sf_col_offset, row_offset, col_offset)
    return per_block_cast_lossless_kernel

def per_block_cast_lossless(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor], fmt: str='fp32', x_block_size: tuple[int, int]=DEFAULT_IN_SF_BLOCK,
    out_block_size: tuple[int, int]=DEFAULT_OUT_SF_BLOCK, use_tma_aligned_col_major_sf: bool=False, round_sf: bool=False,
    use_packed_ue8m0: bool=False, in_use_tma_aligned_col_major_sf: bool | None=None, in_round_sf: bool | None=None,
    in_use_packed_ue8m0: bool | None=None,
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
    assert fmt in ('fp32', 'float32', 'e4m3')
    logical_out_fmt = 'e4m3' if fmt in ('fp32', 'float32') else fmt
    actual_out_fmt = 'fp32'
    _ = logical_out_fmt
    assert x_data.dim() == 2 and x_data.is_contiguous()
    assert x_data.device.type == 'npu'
    assert x_data.dtype == torch.bfloat16
    num_tokens, hidden = x_data.shape
    out_config = get_cast_output_config(actual_out_fmt, out_block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)
    if num_tokens == 0 or hidden == 0:
        out = torch.empty((num_tokens, hidden), dtype=out_config.torch_dtype, device=x_data.device)
        out_sf = alloc_scaling_factors((num_tokens, hidden), out_config, x_data.device)
        return (out, cast_epilogue(out_sf, num_tokens, hidden, out_config))
    use_v1_kernel = (
        in_config.use_packed_ue8m0
        or out_config.use_packed_ue8m0
        or in_config.use_tma_aligned_col_major_sf
        or out_config.use_tma_aligned_col_major_sf
    )
    layout = _derive_cast_layout_v1(hidden, in_config, out_config) if use_v1_kernel else _derive_cast_layout(hidden, in_config, out_config)
    block_m = layout['block_m']
    block_k = layout['block_k']
    padded_tokens = align_up(num_tokens, block_m)
    padded_hidden = align_up(hidden, block_k)
<<<<<<< HEAD

    if padded_tokens != num_tokens or padded_hidden != hidden:
        x_padded = torch.zeros((padded_tokens, padded_hidden), dtype=x_data.dtype, device=x_data.device)
        x_padded[:num_tokens, :hidden] = x_data
    else:
        x_padded = x_data

=======
    x_padded = _pad_2d_on_cpu_then_to_device(x_data, (padded_tokens, padded_hidden), fill_value=0)
>>>>>>> 010c7f3 (per_block_cast_lossless_kernel v0705 cleaned_version)
    x_sf_shape = get_sf_shape((padded_tokens, padded_hidden), in_config)
    x_sf_padded = _pad_2d_on_cpu_then_to_device(x_sf, x_sf_shape, fill_value=0)
    out = torch.empty((padded_tokens, padded_hidden), dtype=out_config.torch_dtype, device=x_data.device)
    out_sf = alloc_scaling_factors((padded_tokens, padded_hidden), out_config, x_data.device)
    if use_v1_kernel:
        data_tile_m = min(block_m, 32, NUM_ELEMENTS_PER_BLOCK // block_k)
        num_data_tiles_m = block_m // data_tile_m
        input_sf_is_tile_packed = _needs_tile_packed_input_sf_v1(in_config, out_config, num_data_tiles_m)
        if input_sf_is_tile_packed:
            x_sf_for_kernel = _pack_plain_sf_for_lossless_kernel_v1(
                x_sf_padded.contiguous(), padded_tokens, padded_hidden, block_m, block_k, in_config.sf_block,
            )
        else:
            x_sf_for_kernel = x_sf_padded
        kernel = get_per_block_cast_lossless_kernel_v1(
            hidden=padded_hidden, block_m=block_m, block_k=block_k, in_config=in_config, out_config=out_config, in_sf_block_m=in_config.sf_block[0],
            in_sf_block_k=in_config.sf_block[1], out_sf_block_m=out_config.sf_block[0], out_sf_block_k=out_config.sf_block[1],
            input_sf_is_tile_packed=input_sf_is_tile_packed,
        )
        kernel_input_sf = x_sf_for_kernel
    else:
        kernel = get_per_block_cast_lossless_kernel(
            hidden=padded_hidden, block_m=block_m, block_k=block_k, in_config=in_config, out_config=out_config, logical_hidden=hidden,
            in_sf_block_m=in_config.sf_block[0], in_sf_block_k=in_config.sf_block[1], out_sf_block_m=out_config.sf_block[0],
            out_sf_block_k=out_config.sf_block[1],
        )
        kernel_input_sf = x_sf_padded
    print_kernel_source = int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0))
    if print_kernel_source:
        print(kernel.get_kernel_source())
    out, out_sf = kernel(x_padded, kernel_input_sf, out, out_sf)
<<<<<<< HEAD

    out = out[:num_tokens, :hidden]
=======
    out = _crop_2d_on_cpu_then_to_device(out, (num_tokens, hidden))
>>>>>>> 010c7f3 (per_block_cast_lossless_kernel v0705 cleaned_version)
    out_sf = cast_epilogue(out_sf, num_tokens, hidden, out_config)
    return (out, out_sf)