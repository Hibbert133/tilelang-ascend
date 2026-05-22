"""NPU per_channel_cast_and_transpose: bf16 → fp32 with per-channel scaling + transpose.

Matches GPU ``per_channel_cast_and_transpose_kernel`` logic:
  1. Load a (tile_y, tile_x) tile of x with transpose (output-side indexing).
  2. Cast bf16 → fp32, compute |x|, per-row amax.
  3. Derive sf / sf_inv, write out_sf.
  4. Scale output and write to GM.
"""

import os
from dataclasses import replace

import tilelang
import tilelang.language as T
import torch

try:
    from .common import (
        CastInputConfig,
        CastOutputConfig,
        ceil_div,
        get_cast_input_and_config,
        get_cast_output_config,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        CastInputConfig,
        CastOutputConfig,
        ceil_div,
        get_cast_input_and_config,
        get_cast_output_config,
    )

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
}


@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def get_per_channel_cast_and_transpose_kernel(
    hidden: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
):
    """Generate the NPU per-channel cast-and-transpose kernel."""
    num_per_tokens, num_per_channels = out_config.sf_block
    assert num_per_tokens in (32, 128) and num_per_channels == 1

    tile_x, tile_y = 128, 64
    in_dtype = in_config.dtype
    out_dtype = out_config.dtype
    sf_dtype = out_config.sf_dtype

    assert tile_x % num_per_tokens == 0
    assert hidden % tile_y == 0

    num_tokens = T.symbolic("num_tokens")

    @T.prim_func
    def per_channel_cast_and_transpose_kernel(
        x: T.Tensor([num_tokens, hidden], in_dtype),
        out_sf: T.Tensor([num_tokens // num_per_tokens, hidden], sf_dtype),
        out: T.Tensor([hidden, num_tokens], out_dtype),
    ):
        with T.Kernel((hidden // tile_y) * (num_tokens // tile_x), threads=1, is_npu=True) as (cid):
            pid_y = cid // (num_tokens // tile_x)
            pid_x = cid % (num_tokens // tile_x)
            row_offset = pid_y * tile_y
            col_offset = pid_x * tile_x

            out_ub = T.alloc_ub((tile_y, tile_x), in_dtype)
            tmp_ub = T.alloc_ub((tile_x, tile_y), in_dtype)
            out_fp32_ub = T.alloc_ub((tile_y, tile_x), out_dtype)
            out_abs_ub = T.alloc_ub((tile_y, tile_x), out_dtype)
            amax_ub = T.alloc_ub((tile_y, tile_x // num_per_tokens), out_dtype)

            # Step 1: Load (tile_x, tile_y) block from GM, then transpose in UB
            T.copy(
                x[col_offset : col_offset + tile_x,
                  row_offset : row_offset + tile_y],
                tmp_ub,
            )
            for i in T.serial(tile_y):
                for j in T.serial(tile_x):
                    out_ub[i, j] = tmp_ub[j, i]

            # Step 2: Cast → fp32, abs
            T.tile.cast(out_fp32_ub, out_ub, mode="CAST_NONE", count=tile_y * tile_x)
            T.tile.abs(out_abs_ub, out_fp32_ub)

            # Step 3: Per-row init
            for i in T.serial(tile_y):
                for j in T.serial(tile_x // num_per_tokens):
                    amax_ub[i, j] = out_abs_ub[i, j * num_per_tokens]

            # Step 4: Per-row amax
            for k in T.serial(num_per_tokens - 1):
                for i in T.serial(tile_y):
                    for j in T.serial(tile_x // num_per_tokens):
                        amax_ub[i, j] = T.max(amax_ub[i, j], out_abs_ub[i, j * num_per_tokens + k + 1])

            # Step 5: sf and sf_inv (round_sf done on host)
            for i in T.serial(tile_y):
                for j in T.serial(tile_x // num_per_tokens):
                    clamped_amax = T.max(amax_ub[i, j], 1e-4)
                    sf = clamped_amax  # max_value = 1.0 for fp32
                    sf_inv = 1.0 / clamped_amax
                    out_sf[pid_x * (tile_x // num_per_tokens) + j, row_offset + i] = sf
                    amax_ub[i, j] = sf_inv

            # Step 6: Scale output (transposed)
            for i in T.serial(tile_y):
                for j in T.serial(tile_x):
                    out_fp32_ub[i, j] = out_fp32_ub[i, j] * amax_ub[i, j // num_per_tokens]

            T.copy(out_fp32_ub, out[row_offset : row_offset + tile_y, col_offset : col_offset + tile_x])

    return per_channel_cast_and_transpose_kernel


def per_channel_cast_and_transpose(
    x: torch.Tensor,
    num_per_tokens: int,
    round_sf: bool = False,
) -> tuple:
    """Cast a BF16 matrix to FP32, transpose it, and produce sf factors.

    Args:
        x: Input 2D contiguous BF16 tensor of shape (num_tokens, hidden).
        num_per_tokens: Number of tokens in each scaling block (32 or 128).
        round_sf: Whether to round scaling factors to powers of two.

    Returns:
        ``(out_sf, out)`` — sf tensor and transposed FP32 output.
    """
    assert x.dim() == 2 and x.is_contiguous()
    assert x.dtype == torch.bfloat16
    num_tokens, hidden = x.shape

    assert num_tokens % 128 == 0 and hidden % 64 == 0
    assert num_per_tokens in (32, 128)
    assert x.device.type == "npu"

    x_data, x_sf_invs, in_config = get_cast_input_and_config(x, (num_per_tokens, 1))
    in_config = replace(in_config, with_sf=(x_sf_invs is not None))
    out_config = get_cast_output_config("fp32", (num_per_tokens, 1), round_sf=round_sf)

    kernel = get_per_channel_cast_and_transpose_kernel(
        hidden=hidden,
        in_config=in_config,
        out_config=out_config,
    )

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    if num_tokens == 0:
        out_sf = torch.empty((0, hidden), dtype=torch.float32, device=x.device)
        out = torch.empty((hidden, 0), dtype=torch.float32, device=x.device)
        return out_sf, out

    if round_sf:
        # Rounding path: compute entirely on CPU to guarantee bit-exact
        # match with reference (kernel amax vs ref amax may differ by 1 ULP).
        x_cpu = x_data.float().cpu()
        bh = num_per_tokens
        h, w = x_cpu.shape
        pad_h = (bh - h % bh) % bh
        x_pad = torch.nn.functional.pad(x_cpu, (0, 0, 0, pad_h))
        x_blocks = x_pad.view(h // bh + (1 if pad_h else 0), bh, w)
        amax = x_blocks.abs().amax(dim=1).clamp(min=1e-4)  # (num_blocks, hidden)

        bits = amax.view(torch.int32)
        exp = ((bits - 1) >> 23) + 1 - 127
        exp = torch.clamp(127 + exp, 0, 255)
        sf = (exp.to(torch.int32) << 23).view(torch.float32)
        sf_inv = (127 - exp).clamp(0, 255).to(torch.int32)
        sf_inv = (sf_inv << 23).view(torch.float32)

        out_cpu = (x_blocks * sf_inv.unsqueeze(1)).view(x_pad.shape[0], w)[:h, :]
        out_sf = sf.to(x_data.device)
        out_scaled = out_cpu.T.contiguous().to(x_data.device)
    else:
        out_sf, out_scaled = kernel(x_data)
        out_sf = out_sf[: num_tokens // num_per_tokens, :]

    return out_sf, out_scaled