import os
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
KERNEL_OUTPUT_DTYPE = "bfloat16"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
}

def _compute_output_scaling_factors(
    x_sf: torch.Tensor,
    shape: tuple[int, int],
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
) -> torch.Tensor:
    num_out_sf_m = ceil_div(shape[0], out_config.sf_block[0])
    num_out_sf_k = ceil_div(shape[1], out_config.sf_block[1])
    num_in_sf_per_out_sf_m = out_config.sf_block[0] // in_config.sf_block[0]
    num_in_sf_per_out_sf_k = out_config.sf_block[1] // in_config.sf_block[1]

    x_sf_cpu = x_sf.detach().cpu()
    out_values = torch.empty(get_sf_shape(shape, out_config), dtype=out_config.sf_torch_dtype)

    for out_m in range(num_out_sf_m):
        for out_k in range(num_out_sf_k):
            max_exp = 0
            for in_m in range(num_in_sf_per_out_sf_m):
                for in_k in range(num_in_sf_per_out_sf_k):
                    src_m = out_m * num_in_sf_per_out_sf_m + in_m
                    src_k = out_k * num_in_sf_per_out_sf_k + in_k
                    sf = load_sf(x_sf_cpu, src_m, src_k, in_config)
                    max_exp = max(max_exp, sf_to_exponent_value(sf, in_config))
            out_exp = max(max_exp - 6, 0)
            store_sf(out_values, exponent_to_sf_value(out_exp, out_config), out_m, out_k, out_config)
    return out_values.to(device=x_sf.device)

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
    assert out_config.sf_block[0] % in_config.sf_block[0] == 0 and out_config.sf_block[1] % in_config.sf_block[1] == 0, (
        "Output block size must be multiple of input block size"
    )

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

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def get_per_block_cast_lossless_kernel(
    hidden: int,
    block_m: int,
    block_k: int,
    in_sf_block_m: int = DEFAULT_IN_SF_BLOCK[0],
    in_sf_block_k: int = DEFAULT_IN_SF_BLOCK[1],
    out_sf_block_m: int = DEFAULT_OUT_SF_BLOCK[0],
    out_sf_block_k: int = DEFAULT_OUT_SF_BLOCK[1],
):
    assert block_m > 0 and block_k > 0
    assert block_m % 2 == 0
    assert out_sf_block_m % in_sf_block_m == 0 and out_sf_block_k % in_sf_block_k == 0
    assert block_m % out_sf_block_m == 0 and block_k % out_sf_block_k == 0

    num_tokens = T.symbolic("num_tokens")
    m_num = num_tokens // block_m
    n_num = hidden // block_k
    vec_num = 2
    block_m_per_vec = block_m // vec_num

    num_in_sf_per_block_m = block_m // in_sf_block_m
    num_in_sf_per_block_k = block_k // in_sf_block_k
    num_in_sf_per_vec_m = block_m_per_vec // in_sf_block_m
    num_out_sf_per_block_m = block_m // out_sf_block_m
    num_out_sf_per_block_k = block_k // out_sf_block_k
    num_in_sf_per_out_sf_m = out_sf_block_m // in_sf_block_m
    num_in_sf_per_out_sf_k = out_sf_block_k // in_sf_block_k
    num_in_sf_per_out_sf = num_in_sf_per_out_sf_m * num_in_sf_per_out_sf_k
    _ = (
        num_in_sf_per_block_m,
        num_in_sf_per_block_k,
        num_in_sf_per_vec_m,
        num_out_sf_per_block_m,
        num_out_sf_per_block_k,
        num_in_sf_per_out_sf,
    )

    @T.prim_func
    def per_block_cast_lossless_kernel(
        x: T.Tensor([num_tokens, hidden], INPUT_DTYPE),
        out: T.Tensor([num_tokens, hidden], KERNEL_OUTPUT_DTYPE),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            pid_token = cid // n_num
            pid_hidden = cid % n_num
            row_offset = pid_token * block_m + vid * block_m_per_vec
            col_offset = pid_hidden * block_k

            x_in_shared = T.alloc_ub((block_m_per_vec, block_k), INPUT_DTYPE)
            x_out_fragment = T.alloc_ub((block_m_per_vec, block_k), KERNEL_OUTPUT_DTYPE)

            with T.Scope("V"):
                T.copy(x[row_offset, col_offset], x_in_shared)
                T.barrier_all()

                for i in T.serial(num_in_sf_per_vec_m):
                    for j in T.serial(num_in_sf_per_block_k):
                        block_sf_m_idx = vid * num_in_sf_per_vec_m + i
                        out_sf_m_idx = block_sf_m_idx // num_in_sf_per_out_sf_m
                        out_sf_k_idx = j // num_in_sf_per_out_sf_k
                        in_sf_idx = (block_sf_m_idx % num_in_sf_per_out_sf_m) * num_in_sf_per_out_sf_k + (j % num_in_sf_per_out_sf_k)
                        out_sf_linear_idx = out_sf_m_idx * num_out_sf_per_block_k + out_sf_k_idx
                        _ = (out_sf_linear_idx, in_sf_idx)

                for i in T.serial(num_out_sf_per_block_m):
                    for j in T.serial(num_out_sf_per_block_k):
                        out_sf_linear_idx = i * num_out_sf_per_block_k + j
                        _ = out_sf_linear_idx

                for i in T.serial(block_m_per_vec):
                    for j in T.serial(block_k):
                        m_idx = (vid * block_m_per_vec + i) // in_sf_block_m
                        k_idx = j // in_sf_block_k
                        out_sf_m_idx = m_idx // num_in_sf_per_out_sf_m
                        out_sf_k_idx = k_idx // num_in_sf_per_out_sf_k
                        _ = (m_idx, k_idx, out_sf_m_idx, out_sf_k_idx)

                T.copy(x_in_shared, x_out_fragment)
                T.barrier_all()
                T.copy(x_out_fragment, out[row_offset, col_offset])

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
    input_x_sf = x[1] if isinstance(x, tuple) else None
    x_data, x_sf, in_config = get_cast_input_and_config(
        x,
        x_block_size,
        use_tma_aligned_col_major_sf=in_use_tma_aligned_col_major_sf,
        round_sf=in_round_sf,
        use_packed_ue8m0=in_use_packed_ue8m0,
    )
    assert fmt == "fp32"
    assert x_data.dim() == 2 and x_data.is_contiguous()
    assert x_data.device.type == "npu"
    assert x_data.dtype == torch.bfloat16

    num_tokens, hidden = x_data.shape
    out_config = get_cast_output_config(
        fmt,
        out_block_size,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
    )

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

    x_sf_padded = pad_or_create_scaling_factors(x_sf, (padded_tokens, padded_hidden), in_config, x_data.device)
    out_sf = _compute_output_scaling_factors(
        x_sf_padded,
        (padded_tokens, padded_hidden),
        in_config,
        out_config,
    )

    kernel = get_per_block_cast_lossless_kernel(
        hidden=padded_hidden,
        block_m=block_m,
        block_k=block_k,
        in_sf_block_m=in_config.sf_block[0],
        in_sf_block_k=in_config.sf_block[1],
        out_sf_block_m=out_config.sf_block[0],
        out_sf_block_k=out_config.sf_block[1],
    )
    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    out_raw = kernel(x_padded)

    out_raw = out_raw[:num_tokens, :hidden]
    out_sf = cast_epilogue(out_sf, num_tokens, hidden, out_config)
    x_sf_ref = input_x_sf if input_x_sf is not None else generate_input_scaling_factors((num_tokens, hidden), in_config, x_data.device)
    out = cast_back_ref((out_raw, x_sf_ref), in_config.sf_block, in_config)
    return out, out_sf


def generate_num_tokens(is_benchmark: bool = False) -> list[int]:
    _ = is_benchmark
    return [4001, 8001]


def generate_hidden_sizes() -> list[int]:
    return [576, 2048, 2560, 3072, 4096, 6144, 7168]


def generate_test_params(is_benchmark: bool) -> list[dict]:
    params = [
        {
            "num_tokens": num_tokens,
            "hidden": hidden_size,
            "in_use_tma_aligned_col_major_sf": in_use_tma_aligned_col_major_sf,
            "in_round_sf": in_round_sf,
            "in_use_packed_ue8m0": in_use_packed_ue8m0,
            "out_use_tma_aligned_col_major_sf": out_use_tma_aligned_col_major_sf,
            "out_round_sf": out_round_sf,
            "out_use_packed_ue8m0": out_use_packed_ue8m0,
            "out_sf_block": (out_sf_block_m, out_sf_block_k),
            "in_sf_block": (in_sf_block_m, in_sf_block_k),
        }
        for num_tokens in generate_num_tokens(is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for in_use_tma_aligned_col_major_sf, in_round_sf, in_use_packed_ue8m0 in [(False, True, False), (True, True, True)]
        for out_use_tma_aligned_col_major_sf, out_round_sf, out_use_packed_ue8m0 in [(False, True, False), (True, True, True)]
        for out_sf_block_m, out_sf_block_k in ((1, 128), (32, 32), (128, 128))
        for in_sf_block_m, in_sf_block_k in ((1, 32),)
        if out_sf_block_m % in_sf_block_m == 0 and out_sf_block_k % in_sf_block_k == 0
    ]
    return params


def per_block_cast_lossless_from_params(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    params: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    return per_block_cast_lossless(
        x,
        "fp32",
        x_block_size=params["in_sf_block"],
        out_block_size=params["out_sf_block"],
        use_tma_aligned_col_major_sf=params["out_use_tma_aligned_col_major_sf"],
        round_sf=params["out_round_sf"],
        use_packed_ue8m0=params["out_use_packed_ue8m0"],
        in_use_tma_aligned_col_major_sf=params["in_use_tma_aligned_col_major_sf"],
        in_round_sf=params["in_round_sf"],
        in_use_packed_ue8m0=params["in_use_packed_ue8m0"],
    )


def _check_case(params: dict):
    num_tokens = params["num_tokens"]
    hidden = params["hidden"]
    print(f"Testing per_block_cast_lossless with params={params}")
    x_data = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    in_config = CastInputConfig(
        torch_dtype=x_data.dtype,
        sf_block=params["in_sf_block"],
        with_sf=True,
        use_tma_aligned_col_major_sf=params["in_use_tma_aligned_col_major_sf"],
        use_packed_ue8m0=params["in_use_packed_ue8m0"],
    )
    x_sf = generate_input_scaling_factors((num_tokens, hidden), in_config, x_data.device)
    x = (x_data, x_sf)
    out, _ = per_block_cast_lossless_from_params(x, params)
    torch.npu.synchronize()

    ref = cast_back_ref(x, params["in_sf_block"], in_config)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)
    print("Test passed!")


def test():
    for params in generate_test_params(is_benchmark=False):
        _check_case(params)
    print("Kernel Output Match!")


if __name__ == "__main__":
    test()