"""Tests for NPU per_token_cast_to_e5m6_kernel (bf16→e5m6).

Reference ``cast_to_e5m6`` is the exact GPU PyTorch reference from
``tile_kernels.torch.cast_to_e5m6``.
"""

import os

import torch

try:
    from .per_token_cast_to_e5m6_kernel import per_token_cast_to_e5m6
except ImportError:
    from per_token_cast_to_e5m6_kernel import (  # type: ignore[no-redef]
        per_token_cast_to_e5m6,
    )


# ---------------------------------------------------------------------------
# GPU reference e5m6 packing (host-side, using exact RTZ float32→fp16)
# ---------------------------------------------------------------------------

def _right_shift_unsigned(x, shift):
    return (x >> shift) & ((1 << (32 - shift)) - 1)


def _float32_to_fp16_rtz_bits(x: torch.Tensor) -> torch.Tensor:
    """float32 → fp16 round-toward-zero bit conversion (GPU __float2half_rz)."""
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


def _cast_to_e5m6(x: torch.Tensor) -> torch.Tensor:
    """GPU float_to_e5m6: pack 8 fp32 into 3 uint32 (12-bit e5m6)."""
    assert x.ndim == 2 and x.dtype in (torch.float32, torch.bfloat16)
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float32)
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
    packed = torch.stack([w0, w1, w2], dim=-1).to(torch.uint32)
    return packed.view(num_tokens, hidden // 8 * 3).view(torch.uint8)


# ---------------------------------------------------------------------------
# Reference (exact GPU tile_kernels.torch.cast_to_e5m6)
# ---------------------------------------------------------------------------

def cast_to_e5m6(
    x: torch.Tensor,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> tuple:
    """GPU PyTorch reference for per-token cast to e5m6."""
    assert x.ndim == 2
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float32)

    num_tokens, hidden = x.shape
    assert hidden % num_per_channels == 0
    assert hidden % 8 == 0
    num_groups = hidden // num_per_channels
    clamp_min = 1e-4
    max_value = torch.tensor(65024.0, dtype=torch.float32)

    x_view = x.view(num_tokens, num_groups, num_per_channels)
    amax = x_view.abs().amax(dim=-1).clamp(min=clamp_min)
    dequant_sf = amax / max_value
    dequant_sf_int = dequant_sf.view(torch.int32)

    if round_sf:
        exp_sf = ((dequant_sf_int - 1) >> 23) + 1 - 127
        sf_inv_bits = (127 - exp_sf).clamp(min=0) << 23
        sf_inv = sf_inv_bits.view(torch.float32)
        sf_inv = torch.where(dequant_sf_int == 0, torch.zeros_like(sf_inv), sf_inv)
        sf_out_bits = (exp_sf + 127) << 23
        sf_out = sf_out_bits.view(torch.float32)
    else:
        sf_inv = torch.where(
            dequant_sf_int == 0,
            torch.zeros_like(amax),
            max_value / amax,
        )
        sf_out = dequant_sf

    sf_inv_expanded = sf_inv.unsqueeze(-1).expand(num_tokens, num_groups, num_per_channels)
    sf_inv_expanded = sf_inv_expanded.reshape(num_tokens, hidden)
    x_scaled = x * sf_inv_expanded
    packed = _cast_to_e5m6(x_scaled)
    _ = use_tma_aligned_col_major_sf, use_packed_ue8m0  # NPU: sf always float32 row-major

    return packed, sf_out


# ---------------------------------------------------------------------------
# Test helpers (aligned with GPU test_per_token_cast_to_e5m6.py)
# ---------------------------------------------------------------------------

_E5M6_SPECIAL_VALUES = (
    2 ** -20,              # min subnormal
    2 ** -14 * 63 / 64,    # max subnormal
    2 ** -14,              # min normal
)


def generate_e5m6_inputs(num_tokens: int, hidden: int, dtype: torch.dtype):
    """Yield (x, is_special) pairs: random then e5m6 special values."""
    yield torch.randn((num_tokens, hidden), dtype=dtype, device="npu"), False
    for value in _E5M6_SPECIAL_VALUES:
        x = torch.full((num_tokens, hidden), value, dtype=dtype, device="npu")
        x[:, -1] = 65024.0  # max e5m6 value
        yield x, True


def generate_hidden_sizes() -> list[int]:
    return [h for h in [576, 2048, 2560, 3072, 4096, 6144, 7168] if h % 8 == 0]


def generate_test_params() -> list[dict]:
    return [
        {"num_tokens": nt, "hidden": h, "round_sf": rs,
         "use_tma_aligned_col_major_sf": tma, "use_packed_ue8m0": ue8}
        for nt in [128, 4001]
        for h in generate_hidden_sizes()
        for rs in (False, True)
        for tma, ue8 in [(False, False)]  # NPU: sf always float32 row-major
    ]


def _check_case(params: dict):
    num_tokens = params["num_tokens"]
    hidden = params["hidden"]
    round_sf = params["round_sf"]
    tma = params["use_tma_aligned_col_major_sf"]
    ue8 = params["use_packed_ue8m0"]

    for x, is_special in generate_e5m6_inputs(num_tokens, hidden, torch.bfloat16):
        tag = "special" if is_special else "random"
        print(f"Testing per_token_cast_to_e5m6: tokens={num_tokens}, hidden={hidden}, "
              f"round_sf={round_sf}, tma={tma}, ue8={ue8} [{tag}]")

        out_packed, out_sf = per_token_cast_to_e5m6(x, num_per_channels=hidden, round_sf=round_sf)
        torch.npu.synchronize()

        ref_packed, ref_sf = cast_to_e5m6(x.cpu(), hidden, tma, round_sf, ue8)
        torch.testing.assert_close(out_sf.cpu(), ref_sf, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(out_packed.int().cpu(), ref_packed.int().cpu(), rtol=1e-3, atol=1e-3)
        print("  PASS")


def test():
    for params in generate_test_params():
        _check_case(params)
    print("Kernel Output Match!")


if __name__ == "__main__":
    test()
