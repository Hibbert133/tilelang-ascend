import os
from dataclasses import replace

import torch
import tilelang
import tilelang.language as T

try:
    from .my_common import *
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


def _pack_plain_sf_for_lossless_kernel(
    x_sf_padded: torch.Tensor,
    padded_tokens: int,
    padded_hidden: int,
    block_m: int,
    block_k: int,
    in_sf_block: tuple[int, int],
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

    x_sf_tile = x_sf_padded.view(
        m_num,
        num_in_sf_per_block_m,
        n_num,
        num_in_sf_per_block_k,
    )
    x_sf_tile = x_sf_tile.view(
        m_num,
        num_data_tiles_m,
        num_in_sf_per_data_tile_m,
        n_num,
        num_in_sf_per_block_k,
    )
    x_sf_tile = x_sf_tile.permute(0, 3, 1, 2, 4).contiguous()
    return x_sf_tile.view(
        m_num,
        n_num,
        num_data_tiles_m,
        num_in_sf_per_data_tile_m * num_in_sf_per_block_k,
    )


def _needs_tile_packed_input_sf(
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    num_data_tiles_m: int,
) -> bool:
    """Limit the SF tile-pack workaround to the plain 128x128 output layout."""
    return (
        not in_config.use_packed_ue8m0
        and not in_config.use_tma_aligned_col_major_sf
        and out_config.sf_block == (128, 128)
        and num_data_tiles_m > 1
    )


def _get_input_sf_load_spec(
    input_sf_is_tile_packed: bool,
    in_config: CastInputConfig,
    num_in_sf_per_block_k: int,
    num_in_sf_per_data_tile_m: int,
    packed_sf_tile_elems: int,
) -> tuple[tuple[int, ...], str]:
    """Return the UB shape and dtype required by the selected SF layout."""
    if input_sf_is_tile_packed:
        return (packed_sf_tile_elems,), "float32"
    if in_config.use_packed_ue8m0:
        assert num_in_sf_per_block_k % 4 == 0
        return (
            num_in_sf_per_block_k // 4,
            num_in_sf_per_data_tile_m * 4,
        ), "uint8"
    if in_config.use_tma_aligned_col_major_sf:
        return (
            num_in_sf_per_block_k,
            num_in_sf_per_data_tile_m,
        ), "float32"
    return (
        num_in_sf_per_data_tile_m,
        num_in_sf_per_block_k,
    ), "float32"


@tilelang.jit(out_idx=[2, 3], pass_configs=pass_configs)
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
    input_sf_is_tile_packed: bool = False,
):
    assert block_m > 0 and block_k > 0
    assert block_m % out_sf_block_m == 0
    assert block_k % out_sf_block_k == 0
    assert out_sf_block_m % in_sf_block_m == 0
    assert out_sf_block_k % in_sf_block_k == 0

    num_tokens = T.symbolic("num_tokens")

    # The Python wrapper pads both dimensions before launching this kernel.
    # Keep the launch grid exact so safe-memory legalization can prove that
    # the SF accesses are in bounds without injecting scalar if_then_else.
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
    num_in_sf_per_data_tile_m = (
        packed_num_in_sf_per_data_tile_m
        if input_sf_is_tile_packed
        else num_in_sf_per_block_m
    )
    packed_sf_tile_elems = num_in_sf_per_data_tile_m * num_in_sf_per_block_k
    x_sf_shape = (
        (m_num, n_num, num_data_tiles_m, packed_sf_tile_elems)
        if input_sf_is_tile_packed
        else get_sf_shape((num_tokens, hidden), in_config)
    )
    x_sf_load_shape, x_sf_load_dtype = _get_input_sf_load_spec(
        input_sf_is_tile_packed,
        in_config,
        num_in_sf_per_block_k,
        num_in_sf_per_data_tile_m,
        packed_sf_tile_elems,
    )

    @T.macro
    def load_input_sf_block(dst, x_sf, sf_m, sf_k, data_m, pid_token, pid_hidden):
        if input_sf_is_tile_packed:
            T.copy(
                x_sf[
                    pid_token,
                    pid_hidden,
                    data_m,
                    0:packed_sf_tile_elems,
                ],
                dst,
            )
        elif in_config.use_packed_ue8m0:
            T.copy(
                x_sf[
                    sf_k // 4 : sf_k // 4 + num_in_sf_per_block_k // 4,
                    sf_m * 4 : sf_m * 4 + num_in_sf_per_data_tile_m * 4,
                ],
                dst,
            )
        elif in_config.use_tma_aligned_col_major_sf:
            T.copy(
                x_sf[
                    sf_k : sf_k + num_in_sf_per_block_k,
                    sf_m : sf_m + num_in_sf_per_data_tile_m,
                ],
                dst,
            )
        else:
            T.copy(
                x_sf[
                    sf_m : sf_m + num_in_sf_per_data_tile_m,
                    sf_k : sf_k + num_in_sf_per_block_k,
                ],
                dst,
            )

    @T.macro
    def decode_input_sf_exp(dst, src):
        for i in T.serial(num_in_sf_per_data_tile_m):
            for j in T.serial(num_in_sf_per_block_k):
                if in_config.use_packed_ue8m0:
                    dst[i, j] = T.Cast("int32", src[j // 4, i * 4 + j % 4])
                else:
                    sf_value = T.alloc_var("float32", init=0.0)
                    sf_bits = T.alloc_var("int32", init=0)
                    if input_sf_is_tile_packed:
                        sf_value = src[i * num_in_sf_per_block_k + j]
                    elif in_config.use_tma_aligned_col_major_sf:
                        sf_value = src[j, i]
                    else:
                        sf_value = src[i, j]
                    sf_bits = T.reinterpret("int32", sf_value)
                    dst[i, j] = (sf_bits >> 23) & 0xFF

    @T.macro
    def store_output_sf(out_sf, out_sf_fp32_ub, out_sf_exp_ub, i, j, sf_m, sf_k):
        if out_config.use_packed_ue8m0:
            out_sf[sf_k // 4, sf_m * 4 + sf_k % 4] = T.Cast(
                "uint8",
                out_sf_exp_ub[i, j],
            )
        elif out_config.use_tma_aligned_col_major_sf:
            out_sf[sf_k, sf_m] = out_sf_fp32_ub[i, j]
        else:
            out_sf[sf_m, sf_k] = out_sf_fp32_ub[i, j]

    @T.macro
    def load_and_decode_input_sf(
        x_sf_exp_ub,
        x_sf_load_ub,
        x_sf,
        sf_m,
        sf_k,
        data_m,
        pid_token,
        pid_hidden,
    ):
        load_input_sf_block(
            x_sf_load_ub,
            x_sf,
            sf_m,
            sf_k,
            data_m,
            pid_token,
            pid_hidden,
        )
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
                        (data_m * num_in_sf_per_data_tile_m + i)
                        // num_in_sf_per_out_sf_m,
                        j // num_in_sf_per_out_sf_k,
                    ],
                    x_sf_exp_ub[i, j],
                )

    @T.macro
    def store_output_sf_block(out_sf, out_sf_fp32_ub, out_sf_exp_ub, sf_m, sf_k):
        for i in T.serial(num_out_sf_per_block_m):
            for j in T.serial(num_out_sf_per_block_k):
                out_sf_bits = T.alloc_var("int32", init=0)
                out_sf_bits = out_sf_exp_ub[i, j] << 23
                out_sf_fp32_ub[i, j] = T.reinterpret("float32", out_sf_bits)
                store_output_sf(
                    out_sf,
                    out_sf_fp32_ub,
                    out_sf_exp_ub,
                    i,
                    j,
                    sf_m + i,
                    sf_k + j,
                )

    @T.macro
    def update_relative_sf_exp(x_sf_exp_ub, out_sf_exp_ub, data_m):
        for i in T.serial(num_in_sf_per_data_tile_m):
            for j in T.serial(num_in_sf_per_block_k):
                x_sf_exp_ub[i, j] = (
                    x_sf_exp_ub[i, j]
                    - out_sf_exp_ub[
                        (data_m * num_in_sf_per_data_tile_m + i)
                        // num_in_sf_per_out_sf_m,
                        j // num_in_sf_per_out_sf_k,
                    ]
                    + 127
                )

    @T.macro
    def expand_relative_sf(x_relative_sf_ub, x_sf_exp_ub):
        for i in T.serial(data_tile_m):
            for j in T.serial(block_k):
                relative_sf_bits = T.alloc_var("int32", init=0)
                relative_sf_bits = x_sf_exp_ub[i // in_sf_block_m, j // in_sf_block_k] << 23
                x_relative_sf_ub[i, j] = T.reinterpret("float32", relative_sf_bits)

    @T.macro
    def cast_data_tile(x, out, x_in_ub, x_out_ub, x_relative_sf_ub, row_offset, col_offset):
        T.copy(
            x[
                row_offset : row_offset + data_tile_m,
                col_offset : col_offset + block_k,
            ],
            x_in_ub,
        )
        T.tile.cast(
            x_out_ub,
            x_in_ub,
            mode="CAST_NONE",
            count=data_tile_m * block_k,
        )
        T.tile.mul(x_out_ub, x_out_ub, x_relative_sf_ub)
        T.copy(
            x_out_ub,
            out[
                row_offset : row_offset + data_tile_m,
                col_offset : col_offset + block_k,
            ],
        )

    @T.prim_func
    def per_block_cast_lossless_kernel(
        x: T.Tensor([num_tokens, hidden], INPUT_DTYPE),
        x_sf: T.Tensor(x_sf_shape, in_config.sf_dtype),
        out: T.Tensor([num_tokens, hidden], OUTPUT_DTYPE),
        out_sf: T.Tensor(out_sf_shape, out_config.sf_dtype),
    ):
        with T.Kernel(m_num * n_num, threads=1, is_npu=True) as cid:
            with T.Scope("V"):
                pid_token = cid // n_num
                pid_hidden = cid % n_num

                row_offset = pid_token * block_m
                col_offset = pid_hidden * block_k

                x_in_ub = T.alloc_ub((data_tile_m, block_k), INPUT_DTYPE)
                x_out_ub = T.alloc_ub((data_tile_m, block_k), OUTPUT_DTYPE)
                x_relative_sf_ub = T.alloc_ub((data_tile_m, block_k), "float32")
                x_sf_load_ub = T.alloc_ub(x_sf_load_shape, x_sf_load_dtype)
                x_sf_exp_ub = T.alloc_ub(
                    (num_in_sf_per_data_tile_m, num_in_sf_per_block_k),
                    "int32",
                )
                out_sf_exp_ub = T.alloc_ub(
                    (num_out_sf_per_block_m, num_out_sf_per_block_k),
                    "int32",
                )
                out_sf_fp32_ub = T.alloc_ub(
                    (num_out_sf_per_block_m, num_out_sf_per_block_k),
                    "float32",
                )

                sf_row_offset = pid_token * num_in_sf_per_block_m
                sf_col_offset = pid_hidden * num_in_sf_per_block_k
                out_sf_row_offset = pid_token * num_out_sf_per_block_m
                out_sf_col_offset = pid_hidden * num_out_sf_per_block_k

                for i in T.serial(num_out_sf_per_block_m):
                    for j in T.serial(num_out_sf_per_block_k):
                        out_sf_exp_ub[i, j] = 0

                for data_m in T.serial(num_data_tiles_m):
                    load_and_decode_input_sf(
                        x_sf_exp_ub,
                        x_sf_load_ub,
                        x_sf,
                        sf_row_offset + data_m * num_in_sf_per_data_tile_m,
                        sf_col_offset,
                        data_m,
                        pid_token,
                        pid_hidden,
                    )
                    reduce_output_sf_exp(out_sf_exp_ub, x_sf_exp_ub, data_m)

                for i in T.serial(num_out_sf_per_block_m):
                    for j in T.serial(num_out_sf_per_block_k):
                        out_sf_exp_ub[i, j] = T.max(out_sf_exp_ub[i, j] - 6, 0)

                store_output_sf_block(
                    out_sf,
                    out_sf_fp32_ub,
                    out_sf_exp_ub,
                    out_sf_row_offset,
                    out_sf_col_offset,
                )

                for data_m in T.serial(num_data_tiles_m):
                    if input_sf_is_tile_packed:
                        load_and_decode_input_sf(
                            x_sf_exp_ub,
                            x_sf_load_ub,
                            x_sf,
                            sf_row_offset + data_m * num_in_sf_per_data_tile_m,
                            sf_col_offset,
                            data_m,
                            pid_token,
                            pid_hidden,
                        )
                    update_relative_sf_exp(x_sf_exp_ub, out_sf_exp_ub, data_m)
                    expand_relative_sf(x_relative_sf_ub, x_sf_exp_ub)
                    cast_data_tile(
                        x,
                        out,
                        x_in_ub,
                        x_out_ub,
                        x_relative_sf_ub,
                        row_offset + data_m * data_tile_m,
                        col_offset,
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

    # NPU path: bf16 + input_sf -> fp32 out + output_sf.
    # The device kernel computes output sf and relative input sf exponents.
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

    x_sf_shape = get_sf_shape((padded_tokens, padded_hidden), in_config)
    if tuple(x_sf.shape) == x_sf_shape:
        x_sf_padded = x_sf
    else:
        # GPU masks OOB input sf before reduce-max. The NPU wrapper pads with
        # exponent-zero sf so padded values cannot increase an output exponent.
        x_sf_padded = torch.zeros(
            x_sf_shape,
            dtype=x_sf.dtype,
            device=x_data.device,
        )
        copy_shape = tuple(
            min(src, dst)
            for src, dst in zip(x_sf.shape, x_sf_shape)
        )
        x_sf_padded[: copy_shape[0], : copy_shape[1]] = x_sf[
            : copy_shape[0],
            : copy_shape[1],
        ]

    data_tile_m = min(block_m, 32, NUM_ELEMENTS_PER_BLOCK // block_k)
    num_data_tiles_m = block_m // data_tile_m
    input_sf_is_tile_packed = _needs_tile_packed_input_sf(
        in_config,
        out_config,
        num_data_tiles_m,
    )
    if input_sf_is_tile_packed:
        x_sf_for_kernel = _pack_plain_sf_for_lossless_kernel(
            x_sf_padded.contiguous(),
            padded_tokens,
            padded_hidden,
            block_m,
            block_k,
            in_config.sf_block,
        )
    else:
        x_sf_for_kernel = x_sf_padded

    out = torch.empty(
        (padded_tokens, padded_hidden),
        dtype=out_config.torch_dtype,
        device=x_data.device,
    )
    out_sf = alloc_scaling_factors(
        (padded_tokens, padded_hidden),
        out_config,
        x_data.device,
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
        input_sf_is_tile_packed=input_sf_is_tile_packed,
    )

    print_kernel_source = int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0))
    if print_kernel_source:
        print(kernel.get_kernel_source())

    out, out_sf = kernel(x_padded, x_sf_for_kernel, out, out_sf)

    out = out[:num_tokens, :hidden]
    out_sf = cast_epilogue(out_sf, num_tokens, hidden, out_config)
    return out, out_sf