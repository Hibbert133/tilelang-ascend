import os
from dataclasses import replace

import torch
import tilelang
import tilelang.language as T

try:
    from .common import *
except ImportError:
    from common import *


tilelang.cache.clear_cache()

DEFAULT_IN_SF_BLOCK = (1, 32)
DEFAULT_OUT_SF_BLOCK = (1, 128)
NUM_ELEMENTS_PER_BLOCK = 8192
INPUT_DTYPE = "bfloat16"
OUTPUT_DTYPE = "float32"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _derive_cast_layout(
    hidden: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
) -> dict[str, int]:
    assert in_config.dtype == INPUT_DTYPE and out_config.dtype == OUTPUT_DTYPE, (
        "lossless mode only supports bf16 -> fp32 conversion currently"
    )
    assert in_config.with_sf, "lossless mode requires both input and output scaling factors"
    assert is_power_of_two(in_config.sf_block[1]) and is_power_of_two(out_config.sf_block[1]), (
        "block_k must be power of 2 for lossless mode"
    )
    assert (
        out_config.sf_block[0] % in_config.sf_block[0] == 0
        and out_config.sf_block[1] % in_config.sf_block[1] == 0
    ), "Output block size must be multiple of input block size"

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


def _compute_input_sf_exp_internal(
    x_sf_internal: torch.Tensor,
    shape: tuple[int, int],
    in_config: CastInputConfig,
) -> torch.Tensor:
    """Convert internal-layout sf to padded logical int32 biased exponents.

    This matches GPU logic:

        input_exp = (reinterpret(sf, int32) >> 23) & 0xFF

    The output layout is always logical row-major [sf_m, sf_k].
    """
    num_sf_m = ceil_div(shape[0], in_config.sf_block[0])
    num_sf_k = ceil_div(shape[1], in_config.sf_block[1])
    sf_cpu = x_sf_internal.detach().cpu().contiguous()

    if in_config.use_packed_ue8m0:
        packed_sf_k = sf_cpu.shape[0]
        packed_sf_m = sf_cpu.shape[1] // 4
        source_exp = (
            sf_cpu.reshape(packed_sf_k, packed_sf_m, 4)
            .permute(1, 0, 2)
            .reshape(packed_sf_m, packed_sf_k * 4)
        )
    else:
        if in_config.use_tma_aligned_col_major_sf:
            sf_cpu = sf_cpu.T
        sf_bits = sf_cpu.to(torch.float32).contiguous().view(torch.int32)
        source_exp = ((sf_bits >> 23) & 0xFF).to(torch.int32)

    exp_cpu = torch.zeros((num_sf_m, num_sf_k), dtype=torch.int32, device="cpu")
    copy_m = min(source_exp.shape[0], num_sf_m)
    copy_k = min(source_exp.shape[1], num_sf_k)
    exp_cpu[:copy_m, :copy_k] = source_exp[:copy_m, :copy_k]
    return exp_cpu.to(device=x_sf_internal.device)


def _compute_lossless_sf_internal(
    x_sf_exp_internal: torch.Tensor,
    shape: tuple[int, int],
    logical_shape: tuple[int, int],
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dense relative sf and output sf using GPU lossless rules."""
    input_exp = x_sf_exp_internal.detach().cpu()
    input_exp = input_exp.contiguous()

    in_per_out_m = out_config.sf_block[0] // in_config.sf_block[0]
    in_per_out_k = out_config.sf_block[1] // in_config.sf_block[1]
    out_sf_m = ceil_div(shape[0], out_config.sf_block[0])
    out_sf_k = ceil_div(shape[1], out_config.sf_block[1])
    out_exp = torch.empty((out_sf_m, out_sf_k), dtype=torch.int32, device="cpu")

    for out_m in range(out_sf_m):
        for out_k in range(out_sf_k):
            src = input_exp[
                out_m * in_per_out_m : (out_m + 1) * in_per_out_m,
                out_k * in_per_out_k : (out_k + 1) * in_per_out_k,
            ]
            out_exp[out_m, out_k] = max(int(src.max().item()) - 6, 0)

    expanded_out_exp = (
        out_exp.repeat_interleave(in_per_out_m, dim=0)
        .repeat_interleave(in_per_out_k, dim=1)
    )
    relative_exp = input_exp - expanded_out_exp + 127
    relative_sf = (relative_exp << 23).contiguous().view(torch.float32)
    relative_sf = (
        relative_sf.repeat_interleave(in_config.sf_block[0], dim=0)
        .repeat_interleave(in_config.sf_block[1], dim=1)
    )
    relative_sf = relative_sf[: shape[0], : shape[1]]

    if out_config.use_packed_ue8m0:
        assert out_config.use_tma_aligned_col_major_sf, (
            "packed UE8M0 scaling factors require TMA-aligned col-major layout"
        )
        logical_out_sf_m = ceil_div(logical_shape[0], out_config.sf_block[0])
        logical_out_sf_k = ceil_div(logical_shape[1], out_config.sf_block[1])
        packed_out_exp = out_exp[:logical_out_sf_m, :logical_out_sf_k]
        padded_out_sf_m = align_up(logical_out_sf_m, 4)
        padded_out_sf_k = align_up(logical_out_sf_k, 4)
        out_sf_padded = torch.zeros(
            (padded_out_sf_m, padded_out_sf_k),
            dtype=torch.uint8,
            device="cpu",
        )
        out_sf_padded[
            :logical_out_sf_m,
            :logical_out_sf_k,
        ] = packed_out_exp.to(torch.uint8)
        out_sf = (
            out_sf_padded.reshape(padded_out_sf_m, padded_out_sf_k // 4, 4)
            .permute(1, 0, 2)
            .reshape(padded_out_sf_k // 4, padded_out_sf_m * 4)
        )
    else:
        out_sf = (out_exp << 23).contiguous().view(torch.float32)
    if out_config.use_tma_aligned_col_major_sf and not out_config.use_packed_ue8m0:
        out_sf = out_sf.T
    return (
        relative_sf.contiguous().to(device=x_sf_exp_internal.device),
        out_sf.contiguous().to(device=x_sf_exp_internal.device),
    )


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def get_per_block_cast_lossless_kernel(
    hidden: int,
    block_m: int,
    block_k: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    in_sf_block_m: int = DEFAULT_IN_SF_BLOCK[0],
    in_sf_block_k: int = DEFAULT_IN_SF_BLOCK[1],
    out_sf_block_m: int = DEFAULT_OUT_SF_BLOCK[0],
    out_sf_block_k: int = DEFAULT_OUT_SF_BLOCK[1],
):
    assert block_m > 0 and block_k > 0
    assert block_m % out_sf_block_m == 0
    assert block_k % out_sf_block_k == 0
    assert out_sf_block_m % in_sf_block_m == 0
    assert out_sf_block_k % in_sf_block_k == 0

    num_tokens = T.symbolic("num_tokens")

    m_num = T.ceildiv(num_tokens, block_m)
    n_num = T.ceildiv(hidden, block_k)

    @T.prim_func
    def per_block_cast_lossless_kernel(
        x: T.Tensor([num_tokens, hidden], INPUT_DTYPE),
        x_relative_sf: T.Tensor([num_tokens, hidden], "float32"),
        out: T.Tensor([num_tokens, hidden], OUTPUT_DTYPE),
    ):
        with T.Kernel(m_num * n_num, threads=1, is_npu=True) as cid:
            pid_token = cid // n_num
            pid_hidden = cid % n_num

            row_offset = pid_token * block_m
            col_offset = pid_hidden * block_k

            x_in_ub = T.alloc_ub((block_m, block_k), INPUT_DTYPE)
            x_out_ub = T.alloc_ub((block_m, block_k), OUTPUT_DTYPE)
            x_relative_sf_ub = T.alloc_ub((block_m, block_k), "float32")

            with T.Scope("V"):
                T.copy(
                    x[
                        row_offset : row_offset + block_m,
                        col_offset : col_offset + block_k,
                    ],
                    x_in_ub,
                )

                T.copy(
                    x_relative_sf[
                        row_offset : row_offset + block_m,
                        col_offset : col_offset + block_k,
                    ],
                    x_relative_sf_ub,
                )

                T.tile.cast(
                    x_out_ub,
                    x_in_ub,
                    mode="CAST_NONE",
                    count=block_m * block_k,
                )

                T.tile.mul(x_out_ub, x_out_ub, x_relative_sf_ub)

                T.copy(
                    x_out_ub,
                    out[
                        row_offset : row_offset + block_m,
                        col_offset : col_offset + block_k,
                    ],
                )

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

    # Current NPU path:
    # bf16 + input_sf -> fp32 out + fp32 out_sf.
    logical_out_fmt = "e4m3" if fmt in ("fp32", "float32") else fmt
    actual_out_fmt = "fp32"
    _ = logical_out_fmt

    assert x_data.dim() == 2 and x_data.is_contiguous()
    assert x_data.device.type == "npu"
    assert x_data.dtype == torch.bfloat16

    num_tokens, hidden = x_data.shape

    out_config = get_cast_output_config(
        actual_out_fmt,
        out_block_size,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
    )

    if num_tokens == 0 or hidden == 0:
        out = torch.empty(
            (num_tokens, hidden),
            dtype=out_config.torch_dtype,
            device=x_data.device,
        )
        out_sf = alloc_scaling_factors((num_tokens, hidden), out_config, x_data.device)
        return out, cast_epilogue(out_sf, num_tokens, hidden, out_config)

    layout = _derive_cast_layout(hidden, in_config, out_config)
    block_m = layout["block_m"]
    block_k = layout["block_k"]

    padded_tokens = align_up(num_tokens, block_m)
    padded_hidden = align_up(hidden, block_k)

    if padded_tokens != num_tokens or padded_hidden != hidden:
        x_padded = torch.zeros(
            (padded_tokens, padded_hidden),
            dtype=x_data.dtype,
            device=x_data.device,
        )
        x_padded[:num_tokens, :hidden] = x_data
    else:
        x_padded = x_data

    # Precompute input and relative sf biased exponents on host side. The NPU
    # kernel only applies the relative scale to produce raw fp32 output.
    x_sf_exp_padded = _compute_input_sf_exp_internal(
        x_sf,
        (padded_tokens, padded_hidden),
        in_config,
    )
    x_relative_sf_padded, out_sf = _compute_lossless_sf_internal(
        x_sf_exp_padded,
        (padded_tokens, padded_hidden),
        (num_tokens, hidden),
        in_config,
        out_config,
    )

    out = torch.empty(
        (padded_tokens, padded_hidden),
        dtype=out_config.torch_dtype,
        device=x_data.device,
    )

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

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    out = kernel(x_padded, x_relative_sf_padded, out)

    out = out[:num_tokens, :hidden]
    out_sf = cast_epilogue(out_sf, num_tokens, hidden, out_config)
    return out, out_sf
