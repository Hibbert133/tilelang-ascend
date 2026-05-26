"""NPU per_token_cast_to_e5m6: bf16 → e5m6 (packed uint32 output).

Aligns GPU ``per_token_cast_to_e5m6_kernel`` logic:
  1. Load (block_m_per_vec, block_k) fragment via 2D grid (m_num × n_num) + vector cores.
  2. Per-tile absmax → amax per token sub-block.
  3. sf / sf_inv computation (amax / 65024).
  4. Scale: out = x * sf_inv.
  5. Host-side e5m6 packing (8 fp32 → 3 uint32).
"""

import os
from dataclasses import replace

import tilelang
import tilelang.language as T
import torch

try:
    from .common import *
except ImportError:
    from common import *  # type: ignore[no-redef]

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
}



def get_sf_and_inv_e5m6(amax: T.float32, out_config: CastOutputConfig):
    """Aligns GPU get_sf_and_inv_e5m6: sf = amax / 65024, optionally round to power of 2.

    NPU: T.reinterpret uses string dtype (e.g. "uint32", "float32"), verified in act_quant.py.
    """
    clamped_amax = T.max(amax, out_config.clamp_min_value)

    max_value = T.float32(65024.0)
    sf = T.alloc_var("float32", init=0.0)
    sf = clamped_amax / max_value
    if not out_config.round_sf:
        return sf, max_value / clamped_amax

    # Round into 2's power
    bits = T.reinterpret("uint32", sf)
    # amax >= 1e-4 ensures sign bit = 0 and bits != 0 (no denorm/zero).
    # `(bits - 1) >> 23 + 1` gives ceil(log2).
    exp_sf = ((bits - 1) >> 23) + 1 - 127
    sf_inv = T.reinterpret("float32", (127 - exp_sf) << 23)
    if out_config.use_packed_ue8m0:
        return (exp_sf + 127) & 0xFF, sf_inv
    else:
        return T.reinterpret("float32", (127 + exp_sf) << 23), sf_inv



@T.macro
def float_to_e5m6(x, out):
    """Pack 8 fp32 values into 3 uint32 (12-bit e5m6).

    Aligns GPU float_to_e5m6 macro.
    NPU: T.tile.cast with CAST_TRUNC replaces GPU __float2half_rz.
    """
    half_u16 = T.alloc_ub((8,), "uint16")
    half_u32 = T.alloc_ub((8,), "uint32")
    remain_u32 = T.alloc_ub((8,), "uint32")
    x_f16 = T.alloc_ub((8,), "float16")

    # NPU: T.tile.cast for fp32→fp16 (GPU: T.call_extern __float2half_rz)
    T.tile.cast(x_f16, x, mode="CAST_TRUNC", count=8)

    kCutBits = T.uint32(0x1FFFF)
    kThreshold = T.uint32(0x10000)
    one_u32 = T.uint32(1)

    for i in T.unroll(8):
        half_u16[i] = T.reinterpret("uint16", x_f16[i])
        value_u32 = T.reinterpret("uint32", x[i])
        remain_u32[i] = value_u32 & kCutBits
        half_u16[i] = half_u16[i] >> 4

    T.tile.cast(half_u32, half_u16, mode="CAST_NONE", count=8)
    for i in T.unroll(8):
        cond = ((half_u32[i] & one_u32) + remain_u32[i] > kThreshold)
        half_u32[i] = T.if_then_else(cond, half_u32[i] + one_u32, half_u32[i])

    out[0] = (
        (half_u32[0] << 20) |
        (half_u32[1] << 8) |
        (half_u32[2] >> 4)
    )

    out[1] = (
        (half_u32[2] << 28) |
        (half_u32[3] << 16) |
        (half_u32[4] << 4) |
        (half_u32[5] >> 8)
    )

    out[2] = (
        (half_u32[5] << 24) |
        (half_u32[6] << 12) |
        half_u32[7]
    )


def _right_shift_unsigned(x: torch.Tensor | int, shift: torch.Tensor | int):
    return (x >> shift) & ((1 << (32 - shift)) - 1)


def _float32_to_fp16_rtz_bits(x: torch.Tensor) -> torch.Tensor:
    x_bits = x.contiguous().view(torch.int32)
    sign = (x_bits >> 16) & 0x8000
    exp = (x_bits >> 23) & 0xFF
    mant = x_bits & 0x7FFFFF

    normal = (exp >= 113) & (exp <= 142)
    subnormal = (exp >= 103) & (exp <= 112)
    overflow = (exp > 142) & (exp < 255)
    underflow = exp < 103
    is_nan = exp == 255

    exp_f16 = (exp - 112).to(torch.int32)
    mant_f16 = (mant >> 13).to(torch.int32)
    shift = (113 - exp).to(torch.int32)
    mant_sub = _right_shift_unsigned(0x800000 | mant, shift + 13)

    result = sign.to(torch.int32)
    result = torch.where(normal, result | (exp_f16 << 10) | mant_f16, result)
    result = torch.where(subnormal, result | mant_sub, result)
    result = torch.where(overflow | (is_nan & (mant == 0)), result | 0x7C00, result)
    result = torch.where(is_nan & (mant != 0), result | 0x7FFF, result)
    result = torch.where(underflow, sign.to(torch.int32), result)
    return result.to(torch.uint16)


def _cast_to_e5m6_host(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.dtype == torch.float32
    num_tokens, hidden = x.shape
    assert hidden % 8 == 0

    x_bits = x.contiguous().view(torch.int32)
    fp16_bits = _float32_to_fp16_rtz_bits(x)
    remain_bits = x_bits & 0x1FFFF
    e5m6_bits = _right_shift_unsigned(fp16_bits.to(torch.int32), 4)
    lsb = e5m6_bits & 1
    cond = (lsb.to(torch.int64) + remain_bits.to(torch.int64)) > 0x10000
    e5m6_bits = (e5m6_bits + cond.to(torch.int32)) & 0xFFF

    e5m6 = e5m6_bits.to(torch.int64).view(num_tokens, hidden // 8, 8)
    h = [e5m6[..., i] for i in range(8)]
    w0 = (h[0] << 20) | (h[1] << 8) | (h[2] >> 4)
    w1 = (h[2] << 28) | (h[3] << 16) | (h[4] << 4) | (h[5] >> 8)
    w2 = (h[5] << 24) | (h[6] << 12) | h[7]
    return torch.stack([w0, w1, w2], dim=-1).to(torch.uint32).view(
        num_tokens, hidden // 8 * 3
    ).view(torch.uint8)


@tilelang.jit(out_idx=[1, 2], pass_configs=pass_configs)
def get_per_token_cast_to_e5m6_kernel(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
):
    """Generate NPU per-token cast to e5m6 kernel.

    Aligns GPU get_per_token_cast_to_e5m6_kernel tiling and structure:
      GPU: 2D grid (pid_token, pid_hidden) with num_threads=128.
      NPU: 2D grid (m_num * n_num) with vector cores (vid).
    """
    num_per_channels = out_config.sf_block[1]
    assert hidden == num_per_channels
    assert not in_config.with_sf

    block_m = 16
    # block_k must divide hidden evenly to avoid out-of-bounds access on partial tiles.
    if hidden <= 256:
        block_k = hidden
    else:
        block_k = 256
        while hidden % block_k != 0:
            block_k //= 2
    VEC_NUM = 2
    block_m_per_vec = block_m // VEC_NUM
    num_groups = 1
    num_tokens = T.symbolic("num_tokens")
    m_num = T.ceildiv(num_tokens, block_m)
    n_num = T.ceildiv(hidden, block_k)
    round_sf = bool(out_config.round_sf)
    use_packed_ue8m0 = bool(out_config.use_packed_ue8m0)

    @T.prim_func
    def per_token_cast_to_e5m6_kernel(
        x: T.Tensor([num_tokens, hidden], in_config.dtype),
        out: T.Tensor([num_tokens, hidden], "float32"),
        out_sf: T.Tensor([num_tokens], out_config.sf_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            pid_token = cid // n_num
            pid_hidden = cid % n_num
            row_offset = pid_token * block_m + vid * block_m_per_vec
            col_offset = pid_hidden * block_k

            scale = block_m_per_vec * block_k

            x_ub = T.alloc_ub((block_m_per_vec, block_k), in_config.dtype)
            x_fp32_ub = T.alloc_ub((block_m_per_vec, block_k), "float32")
            abs_ub = T.alloc_ub((block_m_per_vec, block_k), "float32")
            amax_ub = T.alloc_ub((block_m_per_vec, 1), "float32")
            # part_amax_ub = T.alloc_ub((block_m_per_vec,), "float32")
            sf_inv_ub = T.alloc_ub((block_m_per_vec,1), "float32")
            sf_fp32_ub = T.alloc_ub((block_m_per_vec,1), "float32")
            sf_out_ub = T.alloc_ub((block_m_per_vec,), out_config.sf_dtype)
            

            with T.Scope("V"):
                # tmp_ub = T.alloc_ub((block_m_per_vec,), "float32")
                local_max_ub = T.alloc_ub((block_m_per_vec, 1), "float32")
                T.tile.fill(amax_ub, 0.0)

                for ph in T.serial(n_num):
                    col_off = ph * block_k

                    # load tile
                    T.copy(x[row_offset : row_offset + block_m_per_vec,col_off : col_off + block_k], x_ub)

                    T.tile.cast(x_fp32_ub, x_ub, mode="CAST_NONE", count=scale)
                    T.tile.abs(abs_ub, x_fp32_ub)
                   
                    # ===== key change =====
                    T.reduce_max(abs_ub, local_max_ub, dim=-1, clear=True, real_shape=[block_m_per_vec, block_k])
                    # T.dump_tensor(local_max_ub,111,64)
                    # accumulate across ph
                    for i in T.serial(block_m_per_vec):
                        amax_ub[i,0] = T.max(amax_ub[i,0], local_max_ub[i,0])
                # ########        
                # tmp_out = T.alloc_ub((block_m_per_vec,), "float32")
                # for i in T.serial(block_m_per_vec):
                #     tmp_out[i] = amax_ub[i,0]
                # T.dump_tensor(tmp_out, 111, block_m_per_vec)

                # Compute sf / sf_inv from global amax
                sf_debug_ub = T.alloc_ub((block_m_per_vec,), "float32")
                for i in T.serial(block_m_per_vec):
                    clamped = T.max(amax_ub[i,0], out_config.clamp_min_value)
                    sf_val = clamped / T.float32(65024.0)
                    sf_debug_ub[i] = sf_val
                    sf_inv_ub[i,0] = T.float32(65024.0) / clamped
                # T.dump_tensor(sf_debug_ub, 111, block_m_per_vec)
                    if round_sf:
                        sf_val = T.min(sf_val, T.float32(1.0))
                        bits = T.reinterpret("int32", sf_val)
                        exp_sf = ((bits - 1) >> 23) + 1 - 127
                        if use_packed_ue8m0:
                            sf_inv_bits = T.alloc_var("int32", init=(127 - exp_sf) << 23)
                            sf_inv_ub[i,0] = T.reinterpret("float32", sf_inv_bits)
                            sf_fp32_ub[i,0] = T.cast((exp_sf + 127) & 0xFF, "float32")
                        else:
                            sf_bits = T.alloc_var("int32", init=(127 + exp_sf) << 23)
                            sf_fp32_ub[i,0] = T.reinterpret("float32", sf_bits)
                            sf_inv_ub[i,0] = T.float32(1.0) / sf_fp32_ub[i,0]
                    else:
                        sf_fp32_ub[i,0] = sf_val
                    # T.dump_tensor(sf_fp32_ub, 111, block_m_per_vec)

                if pid_hidden == 0:
                    for i in T.serial(block_m_per_vec):
                        sf_out_ub[i] = sf_fp32_ub[i,0]
                    #   T.dump_tensor(sf_out_ub, 111, block_m_per_vec)
                    T.copy(sf_out_ub, out_sf[row_offset : row_offset + block_m_per_vec])

                # out_sf_flat = T.alloc_ub((block_m_per_vec,), "float32")

                # for i in T.serial(block_m_per_vec):
                #     out_sf_flat[i] = sf_fp32_ub[i, 0]

                # T.copy(
                #     out_sf_flat,
                #     out_sf[row_offset : row_offset + block_m_per_vec, 0]
                # )

                # # Pass 2: scale my tile with global sf_inv
                T.copy(x[row_offset : row_offset + block_m_per_vec, col_offset : col_offset + block_k], x_ub)
                T.tile.cast(x_fp32_ub, x_ub, mode="CAST_NONE", count=scale)
                for i in T.serial(block_m_per_vec):
                    for j in T.serial(block_k):
                        x_fp32_ub[i, j] = x_fp32_ub[i, j] * sf_inv_ub[i,0]
                T.copy(x_fp32_ub, out[row_offset : row_offset + block_m_per_vec, col_offset : col_offset + block_k])

    return per_token_cast_to_e5m6_kernel


def per_token_cast_to_e5m6(
    x: torch.Tensor,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> tuple:
    """Cast a matrix to E5M6 (12-bit truncated half-precision) with per-token scaling factors.

    Aligns GPU per_token_cast_to_e5m6. E5M6 packs each value into 12 bits
    (1-bit sign, 5-bit exponent, 6-bit mantissa) and stores 8 values as
    3 uint32 words (96 bits). The output is returned as uint8.

    Args:
        x: Input 2D tensor of shape (num_tokens, hidden), BF16.
        num_per_channels: Number of channels in each scaling block. Must equal ``hidden``.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.

    Returns:
        A tuple ``(out, out_sf)`` where ``out`` is a uint8 tensor of shape
        ``(num_tokens, hidden // 8 * 3)`` and ``out_sf`` is the sf-factor tensor.
    """
    assert x.dim() == 2 and x.is_contiguous()
    assert x.dtype == torch.bfloat16
    assert x.device.type == "npu"
    num_tokens, hidden = x.shape
    assert num_per_channels == hidden
    assert hidden % 8 == 0

    x_data, _, in_config = get_cast_input_and_config(x, None)
    in_config = replace(in_config, with_sf=False)
    orig_num_tokens = num_tokens
    if num_tokens % 16 != 0:
        padded_num_tokens = ((num_tokens + 15) // 16) * 16
        x_padded = torch.empty(
            (padded_num_tokens, hidden),
            dtype=x_data.dtype,
            device=x_data.device,
        )
        x_padded[:num_tokens, :] = x_data
        x_padded[num_tokens:, :] = 0
        x_data = x_padded
        num_tokens = padded_num_tokens

    out_config = get_cast_output_config(
        "e5m6", (1, num_per_channels),
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        custom_clamp_min_value=1e-4,
    )

    # Get kernel implementation (aligns GPU get_per_token_cast_to_e5m6_kernel call)
    kernel = get_per_token_cast_to_e5m6_kernel(
        hidden=hidden,
        token_stride=x_data.stride(0),
        in_config=in_config,
        out_config=out_config,
    )

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    out_fp32 = torch.empty((num_tokens, hidden), dtype=torch.float32, device=x.device)
    out_sf = torch.empty((num_tokens,), dtype=out_config.sf_torch_dtype, device=x.device)
    if num_tokens > 0:
        out_fp32, out_sf = kernel(x_data, out_fp32, out_sf)
    out_sf = out_sf.reshape(num_tokens, 1)
    out_sf = out_sf[:orig_num_tokens, :]
    if round_sf:
        x_cpu = x[:orig_num_tokens, :].cpu().float()
        amax = x_cpu.abs().amax(dim=-1, keepdim=True).clamp(min=out_config.clamp_min_value)
        dequant_sf = amax / torch.tensor(65024.0, dtype=torch.float32)
        dequant_sf = dequant_sf.clamp(max=1.0)
        dequant_sf_int = dequant_sf.view(torch.int32)
        exp_sf = ((dequant_sf_int - 1) >> 23) + 1 - 127
        sf_inv_bits = ((127 - exp_sf).clamp(min=0) << 23).to(torch.int32)
        sf_inv = sf_inv_bits.view(torch.float32)
        x_scaled = x_cpu * sf_inv
        out_packed = _cast_to_e5m6_host(x_scaled)
    else:
        out_packed = _cast_to_e5m6_host(out_fp32[:orig_num_tokens, :].cpu())
    return out_packed.to(x.device), out_sf
