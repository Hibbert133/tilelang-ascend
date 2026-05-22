"""Tests for NPU per_block_cast_lossless_kernel."""

import torch

try:
    from .common import (
        CastInputConfig,
        CastOutputConfig,
        align_up,
        alloc_scaling_factors,
        cast_back_ref,
        cast_epilogue,
        ceil_div,
        generate_input_scaling_factors,
        get_cast_input_and_config,
        get_cast_output_config,
        get_sf_shape,
        pad_or_create_scaling_factors,
    )
    from .per_block_cast_lossless_kernel import (
        _compute_output_scaling_factors,
        _derive_cast_layout,
        per_block_cast_lossless,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        CastInputConfig,
        CastOutputConfig,
        align_up,
        alloc_scaling_factors,
        cast_back_ref,
        cast_epilogue,
        ceil_div,
        generate_input_scaling_factors,
        get_cast_input_and_config,
        get_cast_output_config,
        get_sf_shape,
        pad_or_create_scaling_factors,
    )
    from per_block_cast_lossless_kernel import (  # type: ignore[no-redef]
        _compute_output_scaling_factors,
        _derive_cast_layout,
        per_block_cast_lossless,
    )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

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


def per_block_cast_lossless_from_params(x, params: dict):
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
    x_sf = generate_input_scaling_factors(
        (num_tokens, hidden),
        CastInputConfig(torch_dtype=x_data.dtype, sf_block=params["in_sf_block"],
                        with_sf=True,
                        use_tma_aligned_col_major_sf=params["in_use_tma_aligned_col_major_sf"],
                        use_packed_ue8m0=params["in_use_packed_ue8m0"]),
        x_data.device,
    )
    x = (x_data, x_sf)
    # Use same config pipeline as host wrapper (auto-detects layout from sf tensor)
    x_data2, x_sf2, in_config = get_cast_input_and_config(
        x, params["in_sf_block"],
        use_tma_aligned_col_major_sf=params["in_use_tma_aligned_col_major_sf"],
        use_packed_ue8m0=params["in_use_packed_ue8m0"],
    )
    out, out_sf = per_block_cast_lossless_from_params(x, params)
    torch.npu.synchronize()

    # Reference out
    ref = cast_back_ref(x, params["in_sf_block"], in_config)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)

    # Reference out_sf
    out_config = get_cast_output_config(
        "fp32", params["out_sf_block"],
        use_tma_aligned_col_major_sf=params["out_use_tma_aligned_col_major_sf"],
        round_sf=params["out_round_sf"],
        use_packed_ue8m0=params["out_use_packed_ue8m0"],
    )
    layout = _derive_cast_layout(hidden, in_config, out_config)
    block_m = layout["block_m"]
    block_k = layout["block_k"]
    padded_tokens = align_up(num_tokens, block_m)
    padded_hidden = align_up(hidden, block_k)
    x_sf_padded = pad_or_create_scaling_factors(x_sf2, (padded_tokens, padded_hidden), in_config, x_data.device)
    ref_sf = _compute_output_scaling_factors(x_sf_padded, (padded_tokens, padded_hidden), in_config, out_config)
    ref_sf = cast_epilogue(ref_sf, num_tokens, hidden, out_config)
    torch.testing.assert_close(out_sf.cpu(), ref_sf.cpu(), rtol=1e-2, atol=1e-2)
    print(f"  PASS")


def test():
    for params in generate_test_params(is_benchmark=False):
        _check_case(params)
    print("Kernel Output Match!")


if __name__ == "__main__":
    test()
