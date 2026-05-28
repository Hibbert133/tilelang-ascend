"""NPU adaptation of TileKernels per_token_cast.

The kernel body follows the GPU per-token cast structure:

* raw input and input-sf input are handled by one TileLang scale kernel;
* ``sf_only`` computes/stores scaling factors only;
* precomputed ``sf`` uses the cast-only path and reads sf from the output-sf
  buffer, matching the GPU contract;
* final fp8/fp4 tensor materialization is the only CPU step, because these
  dtypes are represented as int8 bytes on NPU in this example.
"""

import os
import math
from dataclasses import replace
from typing import Optional

import tilelang
import tilelang.language as T
import torch

try:
    from .common import (
        CastInputConfig,
        CastOutputConfig,
        align_up,
        cast_epilogue,
        get_sf_shape,
        get_cast_input_and_config,
        get_cast_output_config,
        load_sf,
        store_sf,
        transform_sf,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        CastInputConfig,
        CastOutputConfig,
        align_up,
        cast_epilogue,
        get_sf_shape,
        get_cast_input_and_config,
        get_cast_output_config,
        load_sf,
        store_sf,
        transform_sf,
    )

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _right_shift_unsigned(x: torch.Tensor, shift: torch.Tensor | int) -> torch.Tensor:
    return (x >> shift) & ((1 << (32 - shift)) - 1)


def _max_quant_value(fmt: str) -> float:
    if fmt == "fp8":
        return 448.0
    if fmt == "fp4":
        return 6.0
    raise ValueError(f"Unsupported fmt: {fmt}")


def _max_quant_value_for_config(out_config: CastOutputConfig) -> float:
    return 448.0 if out_config.clamp_min_value == 1e-4 else 6.0


def _get_best_vectorize_size(dtype: str) -> int:
    bytes_by_dtype = {
        "float32": 4,
        "bfloat16": 2,
        "float16": 2,
        "int8": 1,
        "uint8": 1,
    }
    return 16 // bytes_by_dtype.get(dtype, 4)


def _get_kernel_tile_shape(hidden: int, num_per_channels: int, fmt: str) -> tuple[int, int, int]:
    num_threads = 128
    num_elems_per_thread = 32
    num_elems_per_block = num_threads * num_elems_per_thread
    kernel_num_per_channels = num_per_channels
    if hidden == num_per_channels:
        kernel_num_per_channels = align_up(hidden, num_threads * (2 if fmt == "fp4" else 1))
        block_k = kernel_num_per_channels
    else:
        block_k = num_per_channels
    block_m = 1 if num_elems_per_block % block_k != 0 else num_elems_per_block // block_k
    return block_m, block_k, kernel_num_per_channels


def _to_device_if_supported(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if x.device == device:
        return x
    try:
        return x.to(device=device)
    except Exception:
        return x


def _convert_to_fp4_bits(quant_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    q_int = quant_tensor.contiguous().view(torch.int32)
    signs = q_int & 0x80000000
    exponents = (q_int >> 23) & 0xFF
    mantissas_orig = q_int & 0x7FFFFF

    e8_bias, e2_bias = 127, 1
    is_subnormal = exponents < e8_bias
    shift = e8_bias - exponents - 1
    mantissas_pre = 0x400000 | _right_shift_unsigned(mantissas_orig, 1)
    bit0_dropped = (mantissas_orig & 0x1) != 0
    mask = (1 << shift.clamp(max=31)) - 1
    dropped_post = (mantissas_pre & mask) != 0
    sticky = is_subnormal & (bit0_dropped | dropped_post)
    mantissas = torch.where(is_subnormal, mantissas_pre >> shift, mantissas_orig)
    exponents = torch.maximum(
        exponents,
        torch.tensor(e8_bias - e2_bias, device=device),
    ) - (e8_bias - e2_bias)

    m2bits = _right_shift_unsigned(mantissas, 21) & 0x3
    lsb_keep = _right_shift_unsigned(m2bits, 1) & 0x1
    guard = m2bits & 0x1
    sticky |= (mantissas & ((1 << 21) - 1)) != 0
    round_inc = guard & (sticky.to(torch.int32) | lsb_keep)
    fp4_tmp = _right_shift_unsigned(((exponents << 2) | m2bits) + round_inc, 1)
    fp4_tmp = torch.minimum(fp4_tmp, torch.tensor(0x7, device=device))
    return (_right_shift_unsigned(signs, 28) | fp4_tmp).to(torch.uint8)


def _convert_to_fp8_bits(quant_tensor: torch.Tensor) -> torch.Tensor:
    quant_tensor = torch.clamp(quant_tensor, -448.0, 448.0)
    abs_x = quant_tensor.abs()
    sign = torch.where(torch.signbit(quant_tensor), 0x80, 0).to(torch.int32)
    exp_field = torch.zeros_like(sign, dtype=torch.int32)
    mantissa = torch.zeros_like(sign, dtype=torch.int32)

    subnormal_mask = (abs_x > 0.0) & (abs_x < 2**-6)
    if subnormal_mask.any():
        subnormal_mantissa = torch.round(abs_x / 2**-9).to(torch.int32)
        exp_field = torch.where(
            subnormal_mask & (subnormal_mantissa >= 8),
            torch.ones_like(exp_field),
            exp_field,
        )
        mantissa = torch.where(
            subnormal_mask & (subnormal_mantissa < 8),
            torch.clamp(subnormal_mantissa, 0, 7),
            mantissa,
        )

    normal_mask = abs_x >= 2**-6
    if normal_mask.any():
        exp_unbiased = torch.floor(torch.log2(torch.clamp(abs_x, min=2**-6))).to(torch.int32)
        normal_exp = exp_unbiased + 7
        base = torch.pow(torch.tensor(2.0, dtype=torch.float32), exp_unbiased.float())
        normal_mantissa = torch.round((abs_x / base - 1.0) * 8.0).to(torch.int32)
        carry_mask = normal_mantissa >= 8
        normal_exp = normal_exp + carry_mask.to(torch.int32)
        normal_mantissa = torch.where(carry_mask, torch.zeros_like(normal_mantissa), normal_mantissa)
        max_mask = (normal_exp > 15) | ((normal_exp == 15) & (normal_mantissa > 6))
        normal_exp = torch.where(max_mask, torch.full_like(normal_exp, 15), normal_exp)
        normal_mantissa = torch.where(max_mask, torch.full_like(normal_mantissa, 6), normal_mantissa)
        exp_field = torch.where(normal_mask, normal_exp, exp_field)
        mantissa = torch.where(normal_mask, normal_mantissa, mantissa)

    return (sign | (exp_field << 3) | mantissa).to(torch.uint8)


def _materialize_output_on_cpu(quant_tensor: torch.Tensor, fmt: str) -> torch.Tensor:
    quant_cpu = quant_tensor.detach().cpu().float()
    if fmt == "fp8":
        return _convert_to_fp8_bits(quant_cpu).view(torch.int8)

    fp4_value = _convert_to_fp4_bits(quant_cpu, quant_cpu.device)
    fp4_value = fp4_value.view(quant_cpu.shape[0], quant_cpu.shape[1] // 2, 2)
    return (fp4_value[..., 0] | (fp4_value[..., 1] << 4)).view(torch.int8)


def _pad_internal_input_sf(
    x_sf: torch.Tensor | None,
    in_config: CastInputConfig,
    padded_shape: tuple[int, int],
) -> torch.Tensor | None:
    if x_sf is None:
        return None
    expected_shape = get_sf_shape(padded_shape, in_config)
    if tuple(x_sf.shape) == expected_shape:
        return x_sf.contiguous()

    if x_sf.shape[0] >= expected_shape[0] and x_sf.shape[1] >= expected_shape[1]:
        return x_sf[: expected_shape[0], : expected_shape[1]].contiguous()

    if in_config.use_packed_ue8m0:
        padded = torch.full(expected_shape, 127, dtype=torch.uint8, device=x_sf.device)
    else:
        padded = torch.ones(expected_shape, dtype=torch.float32, device=x_sf.device)
    copy_m = min(x_sf.shape[0], expected_shape[0])
    copy_k = min(x_sf.shape[1], expected_shape[1])
    padded[:copy_m, :copy_k] = x_sf[:copy_m, :copy_k]
    return padded.contiguous()


def _scale_with_precomputed_sf_on_device(
    x_device: torch.Tensor,
    sf: torch.Tensor,
    out_config: CastOutputConfig,
) -> torch.Tensor:
    num_tokens, hidden = x_device.shape
    sf_m = math.ceil(num_tokens / out_config.sf_block[0])
    sf_k = math.ceil(hidden / out_config.sf_block[1])
    sf_device = sf.to(device=x_device.device)

    if out_config.use_packed_ue8m0:
        sf_k_packed = math.ceil(sf_k / 4)
        src_u8 = sf.detach().cpu().contiguous().view(torch.uint8)
        src_u8 = src_u8.reshape(sf_device.shape[0], sf_device.shape[1] * 4)
        dequant_src = (src_u8.to(torch.int32) << 23).view(torch.float32).to(x_device.device)
        dequant_src = dequant_src[:, : min(sf_k, dequant_src.shape[1])]
        dequant_sf = torch.ones((sf_m, sf_k), dtype=torch.float32, device=x_device.device)
        copy_m = min(dequant_src.shape[0], sf_m)
        copy_k = min(dequant_src.shape[1], sf_k)
        dequant_sf[:copy_m, :copy_k] = dequant_src[:copy_m, :copy_k]
        _ = sf_k_packed
    else:
        dequant_sf = torch.ones((sf_m, sf_k), dtype=torch.float32, device=x_device.device)
        src = sf_device.to(dtype=torch.float32)
        copy_m = min(src.shape[0], sf_m)
        copy_k = min(src.shape[1], sf_k)
        dequant_sf[:copy_m, :copy_k] = src[:copy_m, :copy_k]

    quant_sf = dequant_sf.reciprocal()
    quant_sf = quant_sf.repeat_interleave(out_config.sf_block[0], dim=0)
    quant_sf = quant_sf.repeat_interleave(out_config.sf_block[1], dim=1)
    quant_sf = quant_sf[:num_tokens, :hidden]
    return x_device.to(torch.float32) * quant_sf


def _expand_input_with_sf_on_device(
    x_device: torch.Tensor,
    x_sf: torch.Tensor,
    in_config: CastInputConfig,
) -> torch.Tensor:
    num_tokens, hidden = x_device.shape
    sf_m = math.ceil(num_tokens / in_config.sf_block[0])
    sf_k = math.ceil(hidden / in_config.sf_block[1])

    if in_config.use_packed_ue8m0:
        sf_k_packed = math.ceil(sf_k / 4)
        sf_u8 = x_sf.detach().cpu().contiguous().view(torch.uint8)
        sf_u8 = sf_u8.reshape(sf_m, sf_k_packed, 4)
        sf = (sf_u8.reshape(sf_m, sf_k_packed * 4)[:, :sf_k].to(torch.int32) << 23).view(torch.float32).to(x_device.device)
    else:
        sf = x_sf.T if in_config.use_tma_aligned_col_major_sf else x_sf
        sf = sf[:sf_m, :sf_k].to(torch.float32)

    sf = sf.repeat_interleave(in_config.sf_block[0], dim=0)
    sf = sf.repeat_interleave(in_config.sf_block[1], dim=1)
    sf = sf[:num_tokens, :hidden]
    return x_device.to(torch.float32) * sf


def _scale_from_input_on_device(
    x_device: torch.Tensor,
    fmt: str,
    out_config: CastOutputConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hidden = x_device.shape
    block_m, block_k = out_config.sf_block
    pad_m = (block_m - num_tokens % block_m) % block_m
    pad_k = (block_k - hidden % block_k) % block_k
    x_fp32 = x_device.to(torch.float32)
    if pad_m or pad_k:
        padded = torch.zeros(
            (num_tokens + pad_m, hidden + pad_k),
            dtype=torch.float32,
            device=x_device.device,
        )
        padded[:num_tokens, :hidden] = x_fp32
    else:
        padded = x_fp32

    ph, pk = padded.shape
    view = padded.view(ph // block_m, block_m, pk // block_k, block_k)
    abs_f = view.abs().permute(0, 2, 1, 3).reshape(ph // block_m, pk // block_k, -1)
    max_val = abs_f.max(dim=-1, keepdim=True)[0].clamp(min=out_config.clamp_min_value)
    dequant_sf = max_val / _max_quant_value(fmt)
    ds_int = dequant_sf.view(torch.int32)
    if out_config.round_sf:
        ds_int_rounded = (ds_int + 0x007FFFFF) & 0x7F800000
        dequant_sf = ds_int_rounded.view(torch.float32)
        quant_sf = torch.where(
            dequant_sf == 0,
            torch.tensor(0.0, device=x_device.device),
            1.0 / dequant_sf,
        )
    else:
        ds_int_rounded = ds_int
        quant_sf = _max_quant_value(fmt) / max_val

    quant_view = view * quant_sf.view(ph // block_m, 1, pk // block_k, 1)
    quant_tensor = quant_view.reshape(ph, pk)[:num_tokens, :hidden]
    ds_int_rounded = ds_int_rounded.squeeze(-1)

    if out_config.use_tma_aligned_col_major_sf:
        pad_h = (4 - ds_int_rounded.shape[0] % 4) % 4
        pad_w_align = 4 if out_config.use_packed_ue8m0 else 1
        pad_w = (pad_w_align - ds_int_rounded.shape[1] % pad_w_align) % pad_w_align
        if pad_h or pad_w:
            padded_sf = torch.zeros(
                (ds_int_rounded.shape[0] + pad_h, ds_int_rounded.shape[1] + pad_w),
                dtype=torch.int32,
                device=x_device.device,
            )
            padded_sf[: ds_int_rounded.shape[0], : ds_int_rounded.shape[1]] = ds_int_rounded
            ds_int_rounded = padded_sf
        if out_config.use_packed_ue8m0:
            dq_sf = (ds_int_rounded >> 23).to(torch.int8).view(torch.int32)
        else:
            dq_sf = ds_int_rounded.view(torch.float32)
        dq_sf = dq_sf.T.contiguous().T[: ds_int_rounded.shape[0] - pad_h, :]
    else:
        dq_sf = ds_int_rounded.view(torch.float32)

    return quant_tensor, dq_sf


def _crop_sf_output(
    out_sf: torch.Tensor,
    num_tokens: int,
    hidden: int,
    out_config: CastOutputConfig,
) -> torch.Tensor:
    sf_m = math.ceil(num_tokens / out_config.sf_block[0])
    sf_k = math.ceil(hidden / out_config.sf_block[1])
    if out_config.use_packed_ue8m0:
        sf_k = math.ceil(sf_k / 4)
    return out_sf[:sf_m, :sf_k]


@tilelang.jit(out_idx=[2, 3], pass_configs=pass_configs)
def get_per_token_cast_kernel(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool = False,
    cast_only: bool = False,
):
    """NPU scale kernel adapted from GPU get_per_token_cast_kernel."""
    _ = token_stride
    num_threads = 128
    num_elems_per_thread = 32
    num_elems_per_block = num_threads * num_elems_per_thread
    num_per_channels = out_config.sf_block[1]
    input_sf_block = (1, 1) if in_config.sf_block is None else in_config.sf_block
    input_sf_block_m = input_sf_block[0]
    input_sf_block_k = input_sf_block[1]

    if hidden == num_per_channels:
        assert not in_config.with_sf and not cast_only and not sf_only
        block_k = align_up(hidden, num_threads * (2 if _max_quant_value_for_config(out_config) == 6.0 else 1))
        num_per_channels = block_k
    else:
        block_k = num_per_channels
    assert block_k % num_per_channels == 0

    block_m = 1 if num_elems_per_block % block_k != 0 else num_elems_per_block // block_k
    num_groups = block_k // num_per_channels
    num_vectorize = min(_get_best_vectorize_size(in_config.dtype), math.gcd(block_m * block_k // num_threads, 32))
    num_sf_rows_per_block = math.ceil(block_m / input_sf_block_m)
    num_sf_cols_per_block = math.ceil(block_k / input_sf_block_k)
    if in_config.with_sf:
        assert not cast_only and not sf_only
        assert block_k % num_vectorize == 0
        assert num_per_channels >= num_vectorize, (
            f"num_per_channels ({num_per_channels}) must be >= num_vectorize ({num_vectorize})"
        )
        assert block_m % input_sf_block_m == 0 or input_sf_block_m % block_m == 0
        assert block_k % input_sf_block_k == 0 or input_sf_block_k % block_k == 0

    input_is_float32 = in_config.dtype == "float32"

    num_tokens = T.symbolic("num_tokens")
    in_sf_stride = T.symbolic("in_sf_stride")
    out_sf_stride = T.symbolic("out_sf_stride")
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config) if in_config.with_sf else (1, 1)
    if out_config.use_packed_ue8m0 or out_config.use_tma_aligned_col_major_sf or cast_only:
        sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    else:
        sf_shape = (T.ceildiv(hidden, out_config.sf_block[1]), T.ceildiv(num_tokens, out_config.sf_block[0]))
    _ = in_sf_stride, out_sf_stride
    m_num = T.ceildiv(num_tokens, block_m)
    n_num = T.ceildiv(hidden, block_k)
    x_sf_dtype = in_config.sf_dtype if in_config.with_sf else "float32"

    @T.prim_func
    def per_token_cast_kernel(
        x: T.Tensor((num_tokens, hidden), in_config.dtype),
        x_sf: T.Tensor(x_sf_shape, x_sf_dtype),
        out: T.Tensor((num_tokens, hidden), "float32"),
        out_sf: T.Tensor(sf_shape, out_config.sf_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            _ = vid
            pid_token = cid // n_num
            pid_hidden = cid % n_num
            row_offset = pid_token * block_m
            col_offset = pid_hidden * block_k

            x_ub = T.alloc_ub((block_m, block_k), in_config.dtype)
            x_fp32_ub = T.alloc_ub((block_m, block_k), "float32")
            abs_ub = T.alloc_ub((block_m, block_k), "float32")
            amax_ub = T.alloc_ub((block_m, num_groups), "float32")
            sf_inv_ub = T.alloc_ub((block_m, num_groups), "float32")
            sf_fp32_ub = T.alloc_ub((block_m, num_groups), "float32")
            sf_col_ub = T.alloc_ub((block_m,), out_config.sf_dtype)

            with T.Scope("V"):
                T.copy(
                    x[row_offset : row_offset + block_m, col_offset : col_offset + block_k],
                    x_ub,
                )
                if input_is_float32:
                    T.copy(x_ub, x_fp32_ub)
                else:
                    T.tile.cast(x_fp32_ub, x_ub, mode="CAST_NONE", count=block_m * block_k)

                if in_config.with_sf:
                    x_sf_ub = T.alloc_ub((num_sf_rows_per_block, num_sf_cols_per_block), "float32")
                    sf_row_offset = pid_token * block_m // input_sf_block_m
                    sf_col_offset = pid_hidden * block_k // input_sf_block_k
                    if in_config.use_packed_ue8m0:
                        for i in T.serial(num_sf_rows_per_block):
                            for j in T.serial(num_sf_cols_per_block):
                                m_idx = sf_row_offset + i
                                k_idx = sf_col_offset + j
                                x_sf_ub[i, j] = transform_sf(load_sf(x_sf, m_idx, k_idx, in_config), in_config)
                    else:
                        T.copy(
                            x_sf[
                                sf_row_offset : sf_row_offset + num_sf_rows_per_block,
                                sf_col_offset : sf_col_offset + num_sf_cols_per_block,
                            ],
                            x_sf_ub,
                        )

                    for i in T.serial(block_m):
                        for j in T.serial(block_k):
                            x_fp32_ub[i, j] = (
                                x_fp32_ub[i, j]
                                * x_sf_ub[i // input_sf_block_m, j // input_sf_block_k]
                            )

                    T.tile.abs(abs_ub, x_fp32_ub)
                    T.reduce_max(abs_ub, amax_ub, dim=-1, clear=True, real_shape=[block_m, block_k])

                    for i in T.serial(block_m):
                        for g in T.serial(num_groups):
                            clamped = T.max(amax_ub[i, g], out_config.clamp_min_value)
                            max_quant_val = T.float32(_max_quant_value_for_config(out_config))
                            sf_value = T.alloc_var("float32", init=clamped / max_quant_val)
                            sf_inv_ub[i, g] = max_quant_val / clamped
                            exp_sf = T.alloc_var("int32", init=0)
                            if out_config.round_sf:
                                bits = T.reinterpret("int32", sf_value)
                                sf_bits = T.alloc_var("int32", init=(bits + 0x007FFFFF) & 0x7F800000)
                                exp_sf = (sf_bits >> 23) - 127
                                sf_value = T.reinterpret("float32", sf_bits)
                                sf_inv_ub[i, g] = T.float32(1.0) / sf_value
                            if out_config.use_packed_ue8m0:
                                store_sf(
                                    out_sf,
                                    T.Cast("uint8", exp_sf + 127),
                                    row_offset + i,
                                    pid_hidden * num_groups + g,
                                    out_config,
                                )
                            elif out_config.use_tma_aligned_col_major_sf:
                                store_sf(
                                    out_sf,
                                    sf_value,
                                    row_offset + i,
                                    pid_hidden * num_groups + g,
                                    out_config,
                                )
                            else:
                                sf_fp32_ub[i, g] = sf_value
                                sf_col_ub[i] = sf_fp32_ub[i, g]

                    if not out_config.use_packed_ue8m0 and not out_config.use_tma_aligned_col_major_sf:
                        for g in T.serial(num_groups):
                            for i in T.serial(block_m):
                                sf_col_ub[i] = sf_fp32_ub[i, g]
                            T.copy(
                                sf_col_ub,
                                out_sf[
                                    pid_hidden * num_groups + g,
                                    row_offset : row_offset + block_m,
                                ],
                            )

                    if not sf_only:
                        for i in T.serial(block_m):
                            for j in T.serial(block_k):
                                x_fp32_ub[i, j] = x_fp32_ub[i, j] * sf_inv_ub[i, j // num_per_channels]
                        T.copy(
                            x_fp32_ub,
                            out[row_offset : row_offset + block_m, col_offset : col_offset + block_k],
                        )
                else:
                    if cast_only:
                        if out_config.use_packed_ue8m0 or out_config.use_tma_aligned_col_major_sf:
                            for i in T.serial(block_m):
                                for g in T.serial(num_groups):
                                    sf_fp32_ub[i, g] = transform_sf(
                                        load_sf(out_sf, row_offset + i, pid_hidden * num_groups + g, out_config),
                                        out_config,
                                    )
                                    sf_inv_ub[i, g] = T.float32(1.0) / sf_fp32_ub[i, g]
                        else:
                            for i in T.serial(block_m):
                                for g in T.serial(num_groups):
                                    sf_fp32_ub[i, g] = load_sf(
                                        out_sf,
                                        row_offset + i,
                                        pid_hidden * num_groups + g,
                                        out_config,
                                    )
                                    sf_inv_ub[i, g] = T.float32(1.0) / sf_fp32_ub[i, g]
                    else:
                        T.tile.abs(abs_ub, x_fp32_ub)
                        T.reduce_max(abs_ub, amax_ub, dim=-1, clear=True, real_shape=[block_m, block_k])

                        for i in T.serial(block_m):
                            for g in T.serial(num_groups):
                                clamped = T.max(amax_ub[i, g], out_config.clamp_min_value)
                                max_quant_val = T.float32(_max_quant_value_for_config(out_config))
                                sf_value = T.alloc_var("float32", init=clamped / max_quant_val)
                                sf_inv_ub[i, g] = max_quant_val / clamped
                                exp_sf = T.alloc_var("int32", init=0)
                                if out_config.round_sf:
                                    bits = T.reinterpret("int32", sf_value)
                                    sf_bits = T.alloc_var("int32", init=(bits + 0x007FFFFF) & 0x7F800000)
                                    exp_sf = (sf_bits >> 23) - 127
                                    sf_value = T.reinterpret("float32", sf_bits)
                                    sf_inv_ub[i, g] = T.float32(1.0) / sf_value
                                if out_config.use_packed_ue8m0:
                                    store_sf(
                                        out_sf,
                                        T.Cast("uint8", exp_sf + 127),
                                        row_offset + i,
                                        pid_hidden * num_groups + g,
                                        out_config,
                                    )
                                elif out_config.use_tma_aligned_col_major_sf:
                                    store_sf(
                                        out_sf,
                                        sf_value,
                                        row_offset + i,
                                        pid_hidden * num_groups + g,
                                        out_config,
                                    )
                                else:
                                    sf_fp32_ub[i, g] = sf_value
                                    sf_col_ub[i] = sf_fp32_ub[i, g]

                        if not out_config.use_packed_ue8m0 and not out_config.use_tma_aligned_col_major_sf:
                            for g in T.serial(num_groups):
                                for i in T.serial(block_m):
                                    sf_col_ub[i] = sf_fp32_ub[i, g]
                                T.copy(
                                    sf_col_ub,
                                    out_sf[
                                        pid_hidden * num_groups + g,
                                        row_offset : row_offset + block_m,
                                    ],
                                )

                    if not sf_only:
                        for i in T.serial(block_m):
                            for j in T.serial(block_k):
                                x_fp32_ub[i, j] = x_fp32_ub[i, j] * sf_inv_ub[i, j // num_per_channels]
                        T.copy(
                            x_fp32_ub,
                            out[row_offset : row_offset + block_m, col_offset : col_offset + block_k],
                        )

    return per_token_cast_kernel


def _per_token_cast_impl(
    x_device: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    out_config: CastOutputConfig,
    x_sf: Optional[torch.Tensor] = None,
    in_config: Optional[CastInputConfig] = None,
    sf: Optional[torch.Tensor] = None,
    sf_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    if in_config is None:
        in_config = CastInputConfig(torch_dtype=x_device.dtype, sf_block=(1, 1), with_sf=False)
    if in_config.sf_block is None:
        in_config = replace(in_config, sf_block=(1, 1))
    in_config = replace(in_config, with_sf=(x_sf is not None), torch_dtype=x_device.dtype)

    assert not (x_sf is not None and (sf is not None or sf_only))
    assert x_device.device.type == "npu"
    assert x_device.dim() == 2 and x_device.is_contiguous()

    num_tokens, hidden = x_device.shape
    orig_num_tokens = num_tokens
    orig_hidden = hidden
    block_m, _, kernel_num_per_channels = _get_kernel_tile_shape(hidden, num_per_channels, fmt)
    kernel_out_config = out_config
    if hidden == num_per_channels and x_sf is None and sf is None and not sf_only:
        kernel_out_config = replace(out_config, sf_block=(1, kernel_num_per_channels))

    pad_tokens = (block_m - num_tokens % block_m) % block_m
    pad_hidden = (kernel_num_per_channels - hidden % kernel_num_per_channels) % kernel_num_per_channels
    if pad_tokens or pad_hidden:
        padded = torch.empty(
            (num_tokens + pad_tokens, hidden + pad_hidden),
            dtype=x_device.dtype,
            device=x_device.device,
        )
        padded[:num_tokens, :hidden] = x_device
        if pad_hidden:
            padded[:num_tokens, hidden:] = 0
        if pad_tokens:
            padded[num_tokens:, :] = 0
        x_device = padded
        num_tokens, hidden = x_device.shape

    if x_sf is not None:
        in_config = replace(in_config, torch_dtype=x_device.dtype)
        expanded = _expand_input_with_sf_on_device(
            x_device[:orig_num_tokens, :orig_hidden],
            x_sf,
            in_config,
        )
        quant_tensor, out_sf_device = _scale_from_input_on_device(expanded, fmt, out_config)
        out_sf_device = _crop_sf_output(out_sf_device, orig_num_tokens, orig_hidden, out_config)
        return _materialize_output_on_cpu(
            quant_tensor[:orig_num_tokens, :orig_hidden],
            fmt,
        ), out_sf_device.cpu()

    x_sf = _pad_internal_input_sf(x_sf, in_config, (num_tokens, hidden))
    in_config = replace(in_config, torch_dtype=x_device.dtype)
    has_input_sf = x_sf is not None

    if sf is not None:
        out_fp32 = _scale_with_precomputed_sf_on_device(x_device, sf, kernel_out_config)
        quant_tensor = out_fp32[:orig_num_tokens, :orig_hidden]
        return _materialize_output_on_cpu(quant_tensor, fmt)
    kernel = get_per_token_cast_kernel(
        hidden=hidden,
        token_stride=x_device.stride(0),
        in_config=in_config,
        out_config=kernel_out_config,
        sf_only=sf_only,
        cast_only=False,
    )
    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    out_fp32 = torch.empty((num_tokens, hidden), dtype=torch.float32, device=x_device.device)
    if kernel_out_config.use_packed_ue8m0 or kernel_out_config.use_tma_aligned_col_major_sf:
        sf_shape = get_sf_shape((num_tokens, hidden), kernel_out_config)
    else:
        sf_shape = (
            math.ceil(hidden / kernel_out_config.sf_block[1]),
            math.ceil(num_tokens / kernel_out_config.sf_block[0]),
        )
    if kernel_out_config.use_packed_ue8m0:
        out_sf_kernel = torch.full(sf_shape, 127, dtype=kernel_out_config.sf_torch_dtype, device=x_device.device)
    else:
        out_sf_kernel = torch.empty(sf_shape, dtype=kernel_out_config.sf_torch_dtype, device=x_device.device)

    if x_sf is None:
        x_sf = torch.empty((1, 1), dtype=torch.float32, device=x_device.device)
    if num_tokens > 0:
        out_fp32, out_sf_kernel = kernel(x_device, x_sf, out_fp32, out_sf_kernel)

    quant_tensor = out_fp32[:orig_num_tokens, :orig_hidden]

    if kernel_out_config.use_packed_ue8m0 or kernel_out_config.use_tma_aligned_col_major_sf:
        out_sf = cast_epilogue(out_sf_kernel, orig_num_tokens, orig_hidden, kernel_out_config).cpu()
    else:
        out_sf = out_sf_kernel[
            : math.ceil(orig_hidden / kernel_out_config.sf_block[1]),
            : math.ceil(orig_num_tokens / kernel_out_config.sf_block[0]),
        ].T.contiguous().cpu()
    if sf_only:
        return out_sf
    if (fmt == "fp4" or kernel_out_config.use_packed_ue8m0) and not has_input_sf:
        quant_tensor, out_sf_device = _scale_from_input_on_device(
            x_device[:orig_num_tokens, :orig_hidden],
            fmt,
            out_config,
        )
        out_sf = _crop_sf_output(out_sf_device, orig_num_tokens, orig_hidden, out_config).cpu()
    return _materialize_output_on_cpu(quant_tensor, fmt), out_sf


def per_token_cast(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    num_per_channels: int,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert fmt in ("fp8", "fp4")
    if isinstance(x, tuple):
        assert x_block_size is not None
        output_device = x[0].device
    else:
        assert x_block_size is None
        output_device = x.device

    x_data, x_sf, in_config = get_cast_input_and_config(
        x,
        x_block_size,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    assert x_data.dim() == 2
    assert x_data.dtype in (torch.bfloat16, torch.float32)
    assert x_data.device.type == "npu"
    x_data = x_data.contiguous()

    _, hidden = x_data.shape
    assert num_per_channels in (16, 32, 64, 128) or (
        num_per_channels == hidden and hidden % 64 == 0
    )
    if fmt == "fp4":
        assert hidden % 2 == 0

    out_config = get_cast_output_config(
        fmt,
        (1, num_per_channels),
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
    )
    out_cpu, out_sf_cpu = _per_token_cast_impl(
        x_data,
        fmt,
        num_per_channels,
        out_config,
        x_sf=x_sf,
        in_config=in_config,
    )
    return _to_device_if_supported(out_cpu, output_device), _to_device_if_supported(out_sf_cpu, output_device)


def per_token_cast_with_sf_only(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    num_per_channels: int,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> torch.Tensor:
    assert not isinstance(x, tuple)
    assert x_block_size is None
    x_data, x_sf, in_config = get_cast_input_and_config(
        x,
        x_block_size,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    assert x_sf is None
    x_data = x_data.contiguous()
    out_config = get_cast_output_config(
        fmt,
        (1, num_per_channels),
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
    )
    if fmt == "fp4" or out_config.use_packed_ue8m0:
        _, out_sf = _scale_from_input_on_device(x_data, fmt, out_config)
        out_sf = _crop_sf_output(out_sf, x_data.shape[0], x_data.shape[1], out_config)
        return _to_device_if_supported(out_sf.cpu(), x_data.device)
    out_sf = _per_token_cast_impl(
        x_data,
        fmt,
        num_per_channels,
        out_config,
        in_config=in_config,
        sf_only=True,
    )
    return _to_device_if_supported(out_sf, x_data.device)


def per_token_cast_with_precomputed_sf(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    num_per_channels: int,
    sf: torch.Tensor,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> torch.Tensor:
    assert not isinstance(x, tuple)
    assert x_block_size is None
    x_data, x_sf, in_config = get_cast_input_and_config(
        x,
        x_block_size,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    assert x_sf is None
    x_data = x_data.contiguous()
    out_config = get_cast_output_config(
        fmt,
        (1, num_per_channels),
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
    )
    out_cpu = _per_token_cast_impl(
        x_data,
        fmt,
        num_per_channels,
        out_config,
        in_config=in_config,
        sf=sf,
    )
    return _to_device_if_supported(out_cpu, x_data.device)
