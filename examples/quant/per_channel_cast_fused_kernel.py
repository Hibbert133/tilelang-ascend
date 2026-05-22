import os
from dataclasses import replace
from typing import Optional

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


@tilelang.jit(out_idx=[-5, -4], pass_configs=pass_configs)
def get_per_channel_cast_fused_kernel(
    hidden: int,
    with_expand: bool,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
):
    """Generate the NPU per-channel cast fused kernel.

    Matches GPU ``get_per_channel_cast_fused_kernel`` logic exactly.
    """
    TILE_M = 128
    num_per_channels = 128  # sf_invs block size along hidden dim

    with_sf = in_config.with_sf
    round_sf = out_config.round_sf
    in_dtype = in_config.dtype
    out_dtype = out_config.dtype
    sf_dtype = out_config.sf_dtype
    FP32_MAX_VALUE = 1.0  # sf = amax, sf_inv = 1.0 / amax for fp32 output

    # TILE_K >= num_per_channels when with_sf so sf_invs indexing works.
    # Sizes chosen to keep UB < 192 KB (including T.Parallel temporaries).
    TILE_K = 128 if with_sf else 64
    num_sf_total = T.ceildiv(hidden, num_per_channels)  # total sf_invs columns
    SF_CHUNK = 8  # load in chunks to save UB

    num_tokens = T.symbolic("num_tokens")
    num_tokens_out = T.symbolic("num_tokens_out")

    m_num = T.ceildiv(num_tokens_out, TILE_M)
    n_num = T.ceildiv(hidden, TILE_K)

    @T.prim_func
    def per_channel_cast_fused_kernel(
        x: T.Tensor((num_tokens, hidden), in_dtype),
        out: T.Tensor((num_tokens_out, hidden), out_dtype),
        out_sf: T.Tensor((T.ceildiv(num_tokens_out, TILE_M), hidden), sf_dtype),
        x_sf_invs: T.Tensor((num_tokens, T.ceildiv(hidden, num_per_channels)), sf_dtype),
        pos_to_token: T.Tensor((num_tokens_out,), "int32"),
        _tok_dim_ref: T.Tensor((num_tokens_out,), "int32"),
    ):
        with T.Kernel(m_num * n_num, threads=1, is_npu=True) as (cid):
            pid_token = cid // n_num
            pid_hidden = cid % n_num
            row_offset = pid_token * TILE_M
            col_offset = pid_hidden * TILE_K

            # ── UB buffers ──
            x_ub = T.alloc_ub((TILE_M, TILE_K), in_dtype)
            x_fp32_ub = T.alloc_ub((TILE_M, TILE_K), out_dtype)
            amax_ub = T.alloc_ub((1, TILE_K), out_dtype)
            sf_ub = T.alloc_ub((1, TILE_K), out_dtype)
            sf_inv_ub = T.alloc_ub((1, TILE_K), out_dtype)
            sf_invs_ub = T.alloc_ub((TILE_M, SF_CHUNK), sf_dtype)

            # Step 1: Load x tile + sf_invs into UB
            if with_expand:
                pt_ub = T.alloc_ub((TILE_M,), "int32")
                T.copy(
                    pos_to_token[row_offset : row_offset + TILE_M],
                    pt_ub,
                )
                # Load sf_invs first (separate loop → single if block)
                if with_sf:
                    _c0 = (pid_hidden // SF_CHUNK) * SF_CHUNK
                    _nc = SF_CHUNK
                    if _c0 + SF_CHUNK > num_sf_total:
                        _nc = num_sf_total - _c0
                    for i in T.serial(TILE_M):
                        pos = T.alloc_var("int32", init=pt_ub[i])
                        if pos >= 0:
                            T.copy(
                                x_sf_invs[pos : pos + 1, _c0 : _c0 + _nc],
                                sf_invs_ub[i : i + 1, 0 : _nc],
                            )
                        else:
                            sf_invs_ub[i, 0] = 0.0
                # Then load x
                T.tile.fill(x_ub, 0.0)
                for i in T.serial(TILE_M):
                    pos = T.alloc_var("int32", init=pt_ub[i])
                    if pos >= 0:
                        T.copy(
                            x[pos, col_offset : col_offset + TILE_K],
                            x_ub[i, 0:TILE_K],
                        )
            else:
                T.copy(
                    x[row_offset : row_offset + TILE_M,
                      col_offset : col_offset + TILE_K],
                    x_ub,
                )
                if with_sf:
                    _c0 = (pid_hidden // SF_CHUNK) * SF_CHUNK
                    _nc = SF_CHUNK
                    if _c0 + SF_CHUNK > num_sf_total:
                        _nc = num_sf_total - _c0
                    T.copy(
                        x_sf_invs[row_offset : row_offset + TILE_M,
                                  _c0 : _c0 + _nc],
                        sf_invs_ub[0 : TILE_M, 0 : _nc],
                    )

            # Phase A — compute amax per K column
            T.tile.cast(x_fp32_ub, x_ub, mode="CAST_NONE", count=TILE_M * TILE_K)

            if with_sf:
                _sc_A = pid_hidden - (pid_hidden // SF_CHUNK) * SF_CHUNK
                for i in T.serial(TILE_M):
                    sf_val = sf_invs_ub[i, _sc_A]
                    for j in T.serial(TILE_K):
                        x_fp32_ub[i, j] = x_fp32_ub[i, j] * sf_val

            T.tile.abs(x_fp32_ub, x_fp32_ub)
            T.reduce_max(x_fp32_ub, amax_ub, dim=0)

            # Step 5: Compute sf and sf_inv per K column
            for j in T.serial(TILE_K):
                clamped_amax = T.max(amax_ub[0, j], 1e-4)
                sf_ub[0, j] = clamped_amax
                sf_inv_ub[0, j] = 1.0 / clamped_amax

            # Step 6: Write sf to out_sf
            T.copy(
                sf_ub,
                out_sf[
                    pid_token,
                    pid_hidden * TILE_K : pid_hidden * TILE_K + TILE_K,
                ],
            )

            # Phase B — recast, scale, write output
            T.tile.cast(x_fp32_ub, x_ub, mode="CAST_NONE", count=TILE_M * TILE_K)

            if with_sf:
                _sc_B = pid_hidden - (pid_hidden // SF_CHUNK) * SF_CHUNK
                for i in T.serial(TILE_M):
                    sf_val = sf_invs_ub[i, _sc_B]
                    for j in T.serial(TILE_K):
                        x_fp32_ub[i, j] = x_fp32_ub[i, j] * sf_val

            for i in T.serial(TILE_M):
                for j in T.serial(TILE_K):
                    x_fp32_ub[i, j] = x_fp32_ub[i, j] * sf_inv_ub[0, j]
            
            # ═══════════════════════════════════════════════════════════════════
            # Step 8: Write out to GM
            # ═══════════════════════════════════════════════════════════════════
            T.copy(
                x_fp32_ub,
                out[
                    row_offset : row_offset + TILE_M,
                    col_offset : col_offset + TILE_K,
                ],
            )

    return per_channel_cast_fused_kernel


def per_channel_cast_fused(
    x,
    num_per_tokens: int,
    round_sf: bool = False,
    num_per_channels: Optional[int] = None,
    pos_to_token: Optional[torch.Tensor] = None,
) -> tuple:
    """Cast a matrix with per-channel scaling, optionally fusing rescale & token expansion.

    Args:
        x: Input tensor of shape ``(num_tokens, hidden)``, dtype ``torch.bfloat16``,
            or a ``(data, sf_invs)`` tuple with ``data`` in bf16 and ``sf_invs`` in fp32.
        num_per_tokens: Number of tokens per scaling block (must be 128).
        round_sf: If True, round scaling factors to powers of two.
        num_per_channels: Input sf_invs block size along hidden dim (must be 128 if set).
        pos_to_token: Optional int32 index tensor for token expansion/gather.

    Returns:
        ``(out, out_sf)`` — fp32 output and per-block scale factors.
    """
    x_data, x_sf_invs, in_config = get_cast_input_and_config(
        x,
        (1, num_per_channels) if num_per_channels is not None else (1, 1),
    )
    # NPU common.py CastInputConfig.with_sf is always True; fix it.
    in_config = replace(in_config, with_sf=(x_sf_invs is not None))

    assert x_data.dim() == 2 and x_data.is_contiguous()
    num_tokens, hidden = x_data.shape
    num_tokens_out = num_tokens

    if pos_to_token is not None:
        assert pos_to_token.dim() == 1 and pos_to_token.is_contiguous()
        assert pos_to_token.dtype == torch.int32
        num_tokens_out = pos_to_token.size(0)
        assert num_tokens_out % 16 == 0
    else:
        assert num_tokens_out % 128 == 0

    assert num_per_tokens == 128
    if x_sf_invs is not None:
        assert num_per_channels == 128
        assert x_sf_invs.dim() == 2 and x_sf_invs.is_contiguous()
        assert x_sf_invs.size(0) == num_tokens
        assert x_sf_invs.size(1) * 128 == hidden

    out_config = get_cast_output_config(
        "fp32",
        (num_per_tokens, 1),
        round_sf=round_sf,
    )
    kernel = get_per_channel_cast_fused_kernel(
        hidden,
        with_expand=(pos_to_token is not None),
        in_config=in_config,
        out_config=out_config,
    )

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    if num_tokens_out > 0:
        _x_sf_invs = (
            x_sf_invs
            if x_sf_invs is not None
            else torch.empty(
                (num_tokens, ceil_div(hidden, 128)),
                dtype=torch.float32,
                device=x_data.device,
            )
        )
        _pos_to_token = (
            pos_to_token
            if pos_to_token is not None
            else torch.zeros(
                (num_tokens_out,),
                dtype=torch.int32,
                device=x_data.device,
            )
        )
        # Always-live ref tensor carrying num_tokens_out for symbolic binding.
        # The compiler may eliminate unused pos_to_token when with_expand=False;
        # this separate parameter is never eliminated (part of the C ABI).
        _tok_dim_ref = torch.zeros(
            (num_tokens_out,),
            dtype=torch.int32,
            device=x_data.device,
        )
        out, out_sf = kernel(x_data, _x_sf_invs, _pos_to_token, _tok_dim_ref)
        # Host-side round_sf: kernel computed non-rounded sf = amax.
        # Correct out and out_sf to power-of-2 rounded values.
        if round_sf:
            out_sf = out_sf.cpu()
            bits = out_sf.view(torch.int32)
            exp = ((bits - 1) >> 23) + 1 - 127
            exp = torch.clamp(127 + exp, 0, 255)
            out_sf_rounded = (exp.to(torch.int32) << 23).view(torch.float32)
            # Correction: out *= (non_rounded_sf / rounded_sf) per block
            correction = out_sf / out_sf_rounded
            out_sf = out_sf_rounded.to(x_data.device)
            # Broadcast correction to token dimension
            N = correction.shape[0] * 128
            corr_expanded = correction.repeat_interleave(128, dim=0)[:out.shape[0], :]
            out = out.cpu() * corr_expanded
            out = out.to(x_data.device)
    else:
        out = torch.empty(
            (0, hidden),
            dtype=torch.float32,
            device=x_data.device,
        )
        out_sf = torch.empty(
            (0, hidden),
            dtype=torch.float32,
            device=x_data.device,
        )

    return out, out_sf
