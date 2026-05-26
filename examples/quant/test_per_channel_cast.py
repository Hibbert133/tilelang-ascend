"""Tests for NPU per_channel_cast_kernel with fp32 output.

Run on NPU hardware:
    cd examples/quant
    python test_per_channel_cast.py
"""

import os

import torch

try:
    from .common import get_cast_output_config
    from .per_channel_cast_kernel import per_channel_cast
except ImportError:
    from common import get_cast_output_config  # type: ignore[no-redef]
    from per_channel_cast_kernel import per_channel_cast  # type: ignore[no-redef]


def generate_hidden_sizes() -> list[int]:
    return [h for h in [576, 2048, 2560, 3072, 4096, 6144, 7168] if h % 64 == 0]


def generate_num_tokens() -> list[int]:
    if os.getenv("TK_FULL_TEST") in ["1", "true", "True"]:
        return [128, 4096, 8064]
    return [128, 4096]


def generate_test_params() -> list[dict]:
    return [
        {
            "num_per_tokens": 128,
            "num_tokens": num_tokens,
            "hidden": hidden,
            "round_sf": round_sf,
            "dtype": torch.bfloat16,
        }
        for num_tokens in generate_num_tokens()
        for hidden in generate_hidden_sizes()
        for round_sf in (False, True)
    ]


def _round_sf_to_power_of_two(sf: torch.Tensor) -> torch.Tensor:
    bits = sf.view(torch.int32)
    exp = ((bits - 1) >> 23) + 1 - 127
    exp = torch.clamp(127 + exp, 0, 255)
    return (exp.to(torch.int32) << 23).view(torch.float32)


def per_channel_cast_ref(
    x: torch.Tensor,
    num_per_tokens: int,
    round_sf: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.device.type == "cpu"
    assert x.ndim == 2
    num_tokens, hidden = x.shape
    out_config = get_cast_output_config("fp32", (num_per_tokens, 1), round_sf=round_sf)

    x_fp32 = x.float()
    num_blocks = (num_tokens + num_per_tokens - 1) // num_per_tokens
    pad_tokens = num_blocks * num_per_tokens - num_tokens
    if pad_tokens:
        x_work = torch.nn.functional.pad(x_fp32, (0, 0, 0, pad_tokens))
    else:
        x_work = x_fp32

    x_blocks = x_work.view(num_blocks, num_per_tokens, hidden)
    out_sf = x_blocks.abs().amax(dim=1).clamp(min=out_config.clamp_min_value)
    if round_sf:
        out_sf = _round_sf_to_power_of_two(out_sf)

    sf_expanded = out_sf.repeat_interleave(num_per_tokens, dim=0)[:num_tokens, :]
    out = x_fp32 / sf_expanded
    return out, out_sf


def _check_case(params: dict):
    num_tokens = params["num_tokens"]
    hidden = params["hidden"]
    round_sf = params["round_sf"]
    num_per_tokens = params["num_per_tokens"]

    print(
        f"Testing per_channel_cast: tokens={num_tokens}, hidden={hidden}, "
        f"round_sf={round_sf}"
    )
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")

    out, out_sf = per_channel_cast(
        x,
        "fp32",
        num_per_tokens=num_per_tokens,
        round_sf=round_sf,
    )
    torch.npu.synchronize()

    ref_out, ref_sf = per_channel_cast_ref(
        x.cpu(),
        num_per_tokens=num_per_tokens,
        round_sf=round_sf,
    )

    torch.testing.assert_close(out.cpu(), ref_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_sf.cpu(), ref_sf, rtol=1e-2, atol=1e-2)
    print("  PASS")


def test():
    for params in generate_test_params():
        _check_case(params)
    print("Kernel Output Match!")


if __name__ == "__main__":
    test()
