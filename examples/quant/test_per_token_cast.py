"""Tests for NPU per_token_cast wrapper with int8-carried fp8/fp4 output.

Run on NPU hardware:
    cd examples/quant
    python test_per_token_cast.py
"""

import os
import math

import torch
import torch.nn.functional as F

try:
    from .common import CastInputConfig, transform_sf_for_ref
    from .per_token_cast_kernel import (
        per_token_cast,
        per_token_cast_with_precomputed_sf,
        per_token_cast_with_sf_only,
    )
except ImportError:
    from common import CastInputConfig, transform_sf_for_ref  # type: ignore[no-redef]
    from per_token_cast_kernel import (  # type: ignore[no-redef]
        per_token_cast,
        per_token_cast_with_precomputed_sf,
        per_token_cast_with_sf_only,
    )


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def right_shift_unsigned(x: torch.Tensor, shift: torch.Tensor | int) -> torch.Tensor:
    return (x >> shift) & ((1 << (32 - shift)) - 1)


def get_min_clamp_val(fmt: str) -> float:
    return 1e-4 if fmt == "fp8" else 6.0 * 2 ** (-126)


def get_max_quant_val(fmt: str) -> float:
    return 448.0 if fmt == "fp8" else 6.0


def convert_to_fp4_bits(quant_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    q_int = quant_tensor.contiguous().view(torch.int32)
    signs = q_int & 0x80000000
    exponents = (q_int >> 23) & 0xFF
    mantissas_orig = q_int & 0x7FFFFF

    e8_bias, e2_bias = 127, 1
    is_subnormal = exponents < e8_bias
    shift = e8_bias - exponents - 1
    mantissas_pre = 0x400000 | right_shift_unsigned(mantissas_orig, 1)
    bit0_dropped = (mantissas_orig & 0x1) != 0
    mask = (1 << shift.clamp(max=31)) - 1
    dropped_post = (mantissas_pre & mask) != 0
    sticky = is_subnormal & (bit0_dropped | dropped_post)
    mantissas = torch.where(is_subnormal, mantissas_pre >> shift, mantissas_orig)
    exponents = torch.maximum(
        exponents,
        torch.tensor(e8_bias - e2_bias, device=device),
    ) - (e8_bias - e2_bias)

    m2bits = right_shift_unsigned(mantissas, 21) & 0x3
    lsb_keep = right_shift_unsigned(m2bits, 1) & 0x1
    guard = m2bits & 0x1
    sticky |= (mantissas & ((1 << 21) - 1)) != 0
    round_inc = guard & (sticky.to(torch.int32) | lsb_keep)
    fp4_tmp = right_shift_unsigned(((exponents << 2) | m2bits) + round_inc, 1)
    fp4_tmp = torch.minimum(fp4_tmp, torch.tensor(0x7, device=device))
    return (right_shift_unsigned(signs, 28) | fp4_tmp).to(torch.uint8)


def convert_to_fp8_bits(quant_tensor: torch.Tensor) -> torch.Tensor:
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


def unpack_fp8(encoded: torch.Tensor) -> torch.Tensor:
    u = encoded.to(torch.uint8).to(torch.int32)
    sign = torch.where((u & 0x80) != 0, -1.0, 1.0)
    exp_field = (u >> 3) & 0x0F
    mantissa = u & 0x07
    subnormal = mantissa.to(torch.float32) * (2.0 ** -9)
    normal = (1.0 + mantissa.to(torch.float32) / 8.0) * torch.pow(
        torch.tensor(2.0, dtype=torch.float32),
        exp_field.to(torch.float32) - 7.0,
    )
    return torch.where(exp_field == 0, subnormal, normal) * sign


def unpack_fp4(encoded: torch.Tensor) -> torch.Tensor:
    u = encoded.to(torch.uint8)
    low = u & 0x0F
    high = (u >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).reshape(*u.shape[:-1], u.shape[-1] * 2)
    v = unpacked.to(torch.int32)
    sign = torch.where((v & 0x08) != 0, -1.0, 1.0)
    exp_field = (v >> 1) & 0x03
    mantissa = v & 0x01
    subnormal = mantissa.to(torch.float32) * 0.5
    normal = (1.0 + mantissa.to(torch.float32) * 0.5) * torch.pow(
        torch.tensor(2.0, dtype=torch.float32),
        exp_field.to(torch.float32) - 1.0,
    )
    return torch.where(exp_field == 0, subnormal, normal) * sign


def transform_sf(
    sf: torch.Tensor,
    data_shape: tuple[int, int],
    block_size: tuple[int, int],
    use_packed_ue8m0: bool,
) -> torch.Tensor:
    if not use_packed_ue8m0:
        return sf.to(torch.float32)

    num_sf_m = ceil_div(data_shape[0], block_size[0])
    num_sf_k = ceil_div(data_shape[1], block_size[1])
    num_sf_k_packed = ceil_div(num_sf_k, 4)
    sf_u8 = sf.detach().cpu().contiguous().view(torch.uint8)
    sf_u8 = sf_u8.reshape(num_sf_m, num_sf_k_packed, 4)
    sf_u8 = sf_u8.reshape(num_sf_m, num_sf_k_packed * 4)[:, :num_sf_k]
    return (sf_u8.to(torch.int32) << 23).view(torch.float32).to(sf.device)


def make_input_scaling_factors(
    shape: tuple[int, int],
    block_size: tuple[int, int],
    use_tma_aligned_col_major_sf: bool,
    use_packed_ue8m0: bool,
    device: torch.device,
) -> torch.Tensor:
    num_sf_m = ceil_div(shape[0], block_size[0])
    num_sf_k = ceil_div(shape[1], block_size[1])
    if use_packed_ue8m0:
        num_sf_k_packed = ceil_div(num_sf_k, 4)
        sf_u8 = torch.randint(125, 130, (num_sf_m, num_sf_k_packed * 4), dtype=torch.uint8)
        sf_u8[:, num_sf_k:] = 127
        sf = sf_u8.view(torch.int32).to(device)
        return sf.T if use_tma_aligned_col_major_sf else sf

    sf_exp = torch.randint(-2, 3, (num_sf_m, num_sf_k), dtype=torch.int32, device=device)
    sf = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=device), sf_exp.to(torch.float32))
    return sf.T if use_tma_aligned_col_major_sf else sf


def expand_input_with_sf_ref(
    x: tuple[torch.Tensor, torch.Tensor],
    block_size: tuple[int, int],
    use_tma_aligned_col_major_sf: bool,
    use_packed_ue8m0: bool,
) -> torch.Tensor:
    x_data, x_sf = x
    config = CastInputConfig(
        torch_dtype=x_data.dtype,
        sf_block=block_size,
        with_sf=True,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    if use_tma_aligned_col_major_sf:
        x_sf = x_sf.T
        if use_packed_ue8m0:
            x_sf = x_sf.contiguous().view(torch.uint8)
    x_sf = transform_sf_for_ref(x_sf, config, tuple(x_data.shape), already_internal_layout=False)
    x_sf = x_sf.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=1)
    x_sf = x_sf[: x_data.shape[0], : x_data.shape[1]]
    return x_data.to(torch.float32) * x_sf.to(device=x_data.device, dtype=torch.float32)


def cast_back_quantized(
    x: tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    block_size: tuple[int, int],
    use_packed_ue8m0: bool,
) -> torch.Tensor:
    input_tensor, input_sf = x
    encoded = input_tensor.detach().cpu().contiguous().view(torch.uint8)
    input_tensor = unpack_fp4(encoded) if fmt == "fp4" else unpack_fp8(encoded)
    input_tensor = input_tensor.to(input_sf.device)
    input_sf = transform_sf(input_sf, input_tensor.shape, block_size, use_packed_ue8m0)
    input_sf = input_sf.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=1)
    input_sf = input_sf[: input_tensor.shape[0], : input_tensor.shape[1]]
    return input_tensor * input_sf


def clear_unused_sf(sf: torch.Tensor, hidden: int, num_per_channels: int) -> torch.Tensor:
    num_channel_blocks = ceil_div(hidden, num_per_channels)
    aligned_num_channel_blocks = align(num_channel_blocks, 4)
    sf_flattened = sf.contiguous().flatten().view(torch.uint8).view(-1, aligned_num_channel_blocks)
    sf_flattened[:, num_channel_blocks:] = 0
    return sf_flattened


def check_bias(x: torch.Tensor, ref_x: torch.Tensor) -> None:
    count = x.numel()
    if count == 0:
        return
    less_count = (x < ref_x).sum()
    equal_count = (x == ref_x).sum()
    less_ratio = (less_count + equal_count / 2) / count
    allowed_diff_ratio = 10 / math.sqrt(count)
    assert abs(less_ratio - 0.5) < allowed_diff_ratio, (
        f"Less than ratio not close to 0.5 (size = {count}): {less_ratio=:.4f}"
    )


def format_sf(
    ds_int_rounded: torch.Tensor,
    use_tma_aligned_col_major_sf: bool,
    use_packed_ue8m0: bool,
) -> torch.Tensor:
    if use_tma_aligned_col_major_sf:
        pad_h = align(ds_int_rounded.shape[0], 4) - ds_int_rounded.shape[0]
        pad_w = align(ds_int_rounded.shape[1], 4 if use_packed_ue8m0 else 1) - ds_int_rounded.shape[1]
        ds_int_rounded = F.pad(ds_int_rounded, (0, pad_w, 0, pad_h))
        if use_packed_ue8m0:
            dq_sf = (ds_int_rounded >> 23).to(torch.int8).view(torch.int32)
        else:
            dq_sf = ds_int_rounded.view(torch.float32)
        return dq_sf.T.contiguous().T[: ds_int_rounded.shape[0] - pad_h, :]

    return ds_int_rounded.view(torch.float32)


def cast_ref(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    block_size: tuple[int, int],
    sf: torch.Tensor | None = None,
    x_block_size: tuple[int, int] | None = None,
    round_sf: bool = False,
    use_tma_aligned_col_major_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if isinstance(x, tuple):
        assert x_block_size is not None
        x = expand_input_with_sf_ref(
            x,
            x_block_size,
            use_tma_aligned_col_major_sf,
            use_packed_ue8m0,
        )
    else:
        assert x_block_size is None
    assert x.dtype in (torch.bfloat16, torch.float32)

    assert x.ndim == 2
    h, w = x.shape
    bh, bw = block_size
    is_fp4 = fmt == "fp4"
    device = x.device
    max_quant_val = get_max_quant_val(fmt)

    if h == 0:
        out_w = w // 2 if is_fp4 else w
        out = torch.empty((0, out_w), dtype=torch.int8, device=device)
        if sf is not None:
            return out
        sf_h = 0
        sf_w = ceil_div(w, bw)
        if use_packed_ue8m0:
            dq_sf = torch.empty((sf_h, ceil_div(sf_w, 4)), dtype=torch.int32, device=device)
        else:
            dq_sf = torch.empty((sf_h, sf_w), dtype=torch.float32, device=device)
        return out, dq_sf

    if is_fp4:
        assert w % 2 == 0

    pad_h = (bh - h % bh) % bh
    pad_w = (bw - w % bw) % bw
    padded_src = F.pad(x.to(torch.float32), (0, pad_w, 0, pad_h))
    ph, pw = padded_src.shape
    valid_mask = torch.zeros((ph, pw), dtype=torch.bool, device=device)
    valid_mask[:h, :w] = True

    if sf is None:
        reshaped_for_max = padded_src.view(ph // bh, bh, pw // bw, bw).permute(0, 2, 1, 3).reshape(ph // bh, pw // bw, -1)
        reshaped_mask = valid_mask.view(ph // bh, bh, pw // bw, bw).permute(0, 2, 1, 3).reshape(ph // bh, pw // bw, -1)
        abs_f = torch.where(
            reshaped_mask,
            reshaped_for_max.abs(),
            torch.tensor(-1.0, device=device, dtype=torch.float32),
        )
        max_val = abs_f.max(dim=-1, keepdim=True)[0].clamp(min=get_min_clamp_val(fmt))
        max_quant_val_expanded = max_val.new_full(max_val.shape, max_quant_val, dtype=torch.float32)
        dequant_sf = max_val / max_quant_val_expanded
        ds_int = dequant_sf.view(torch.int32)
        if round_sf:
            ds_int_rounded = (ds_int + 0x007FFFFF) & 0x7F800000
            dequant_sf_rounded = ds_int_rounded.view(torch.float32)
            quant_sf = torch.where(
                dequant_sf_rounded == 0,
                torch.tensor(0.0, device=device),
                1.0 / dequant_sf_rounded,
            )
        else:
            ds_int_rounded = ds_int
            quant_sf = torch.where(
                ds_int_rounded == 0,
                torch.tensor(0.0, device=device),
                max_quant_val_expanded / max_val,
            )
    else:
        expected_sf_shape = (ph // bh, pw // bw)
        sf_config = CastInputConfig(
            torch_dtype=torch.float32,
            sf_block=block_size,
            with_sf=True,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            use_packed_ue8m0=use_packed_ue8m0,
        )
        sf = transform_sf_for_ref(sf, sf_config, (h, w), already_internal_layout=False)
        assert tuple(sf.shape) == expected_sf_shape, (tuple(sf.shape), expected_sf_shape)
        quant_sf = sf.to(device=device, dtype=torch.float32).reciprocal().unsqueeze(-1)

    padded_src_view = padded_src.view(ph // bh, bh, pw // bw, bw)
    quant_sf_view = quant_sf.view(ph // bh, 1, pw // bw, 1)
    quant_tensor = (padded_src_view * quant_sf_view).reshape(ph, pw)[:h, :w]

    if fmt == "fp8":
        out = convert_to_fp8_bits(quant_tensor.detach().cpu()).view(torch.int8).to(device)
    else:
        fp4_value = convert_to_fp4_bits(quant_tensor.detach().cpu(), torch.device("cpu"))
        fp4_value = fp4_value.view(h, w // 2, 2)
        out = (fp4_value[..., 0] | (fp4_value[..., 1] << 4)).view(torch.int8).to(device)

    if sf is not None:
        return out

    dq_sf = format_sf(ds_int_rounded.squeeze(-1).detach().cpu(), use_tma_aligned_col_major_sf, use_packed_ue8m0)
    return out, dq_sf.to(device)


def generate_num_tokens() -> list[int]:
    base = [4001, 8001]
    return ([0] + base) if os.getenv("TK_FULL_TEST") in ("1", "true", "True") else base


def generate_hidden_sizes() -> list[int]:
    return [576, 2048, 2560, 3072, 4096, 6144, 7168]


def generate_test_params() -> list[dict]:
    return [
        {
            "num_tokens": num_tokens,
            "hidden": hidden,
            "in_dtype": in_dtype,
            "input_with_sf": input_with_sf,
            "in_fmt": "with_sf" if input_with_sf else None,
            "fmt": fmt,
            "num_per_channels": num_per_channels,
            "x_block_size": x_block_size,
            "use_tma_aligned_col_major_sf": use_tma_aligned_col_major_sf,
            "round_sf": round_sf,
            "use_packed_ue8m0": use_packed_ue8m0,
        }
        for num_tokens in generate_num_tokens()
        for hidden in generate_hidden_sizes()
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in (
            (False, True, False),
            (True, True, True),
        )
        for input_with_sf in (False, True)
        for in_dtype in (torch.float32, torch.bfloat16)
        for num_per_channels in ((32, 128) if input_with_sf else (32, 64, 128, hidden))
        for x_block_size in (((128, 128), (32, 32)) if input_with_sf else (None,))
        for fmt in ("fp8", "fp4")
    ]


def assert_bit_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_cpu = actual.cpu().contiguous()
    expected_cpu = expected.cpu().contiguous()
    assert actual_cpu.shape == expected_cpu.shape, (actual_cpu.shape, expected_cpu.shape)
    assert actual_cpu.dtype == expected_cpu.dtype, (actual_cpu.dtype, expected_cpu.dtype)
    torch.testing.assert_close(
        actual_cpu.view(torch.uint8),
        expected_cpu.view(torch.uint8),
        rtol=1e-3,
        atol=1e-3,
    )


def _check_case(params: dict):
    num_tokens = params["num_tokens"]
    hidden = params["hidden"]
    in_dtype = params["in_dtype"]
    input_with_sf = params["input_with_sf"]
    in_fmt = params["in_fmt"]
    fmt = params["fmt"]
    num_per_channels = params["num_per_channels"]
    x_block_size = params.get("x_block_size")
    use_tma_aligned_col_major_sf = params["use_tma_aligned_col_major_sf"]
    round_sf = params["round_sf"]
    use_packed_ue8m0 = params["use_packed_ue8m0"]
    assert (in_fmt is not None) == input_with_sf
    assert (x_block_size is not None) == input_with_sf
    print(
        f"Testing per_token_cast: tokens={num_tokens}, hidden={hidden}, "
        f"in_dtype={in_dtype}, in_fmt={in_fmt}, fmt={fmt}, "
        f"block={num_per_channels}, x_block={x_block_size}, "
        f"tma={use_tma_aligned_col_major_sf}, "
        f"round_sf={round_sf}, ue8={use_packed_ue8m0}"
    )

    x_data = torch.randn((num_tokens, hidden), dtype=in_dtype, device="npu")
    if input_with_sf:
        x_sf = make_input_scaling_factors(
            (num_tokens, hidden),
            x_block_size,
            use_tma_aligned_col_major_sf,
            use_packed_ue8m0,
            x_data.device,
        )
        x = (x_data, x_sf)
        original_x = expand_input_with_sf_ref(
            x,
            x_block_size,
            use_tma_aligned_col_major_sf,
            use_packed_ue8m0,
        )
    else:
        x = x_data
        original_x = x_data

    out, out_sf = per_token_cast(
        x,
        fmt,
        num_per_channels=num_per_channels,
        x_block_size=x_block_size,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    torch.npu.synchronize()

    ref_out, ref_sf = cast_ref(
        x,
        fmt,
        (1, num_per_channels),
        x_block_size=x_block_size,
        round_sf=round_sf,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    assert_bit_equal(out, ref_out)
    if use_packed_ue8m0:
        assert_bit_equal(
            clear_unused_sf(out_sf, hidden, num_per_channels),
            clear_unused_sf(ref_sf, hidden, num_per_channels),
        )
    else:
        assert_bit_equal(out_sf, ref_sf)

    out_back = cast_back_quantized(
        (out, out_sf),
        fmt,
        (1, num_per_channels),
        use_packed_ue8m0,
    )
    check_bias(out_back, original_x)

    if not input_with_sf and fmt == "fp8" and not use_packed_ue8m0 and num_tokens > 0:
        x_non_contiguous = torch.randn((num_tokens, hidden * 2), dtype=in_dtype, device="npu")[:, :hidden]
        x_non_contiguous.copy_(original_x)
        non_contiguous_out, non_contiguous_sf = per_token_cast(
            x_non_contiguous,
            fmt,
            num_per_channels=num_per_channels,
            x_block_size=x_block_size,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            round_sf=round_sf,
            use_packed_ue8m0=use_packed_ue8m0,
        )
        assert_bit_equal(non_contiguous_out, out)
        assert_bit_equal(non_contiguous_sf, out_sf)

    if not input_with_sf and num_per_channels != hidden:
        if not use_tma_aligned_col_major_sf:
            out_with_sf = per_token_cast_with_precomputed_sf(
                x,
                fmt,
                num_per_channels=num_per_channels,
                sf=out_sf,
                x_block_size=x_block_size,
                use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
                round_sf=round_sf,
                use_packed_ue8m0=use_packed_ue8m0,
            )
            ref_with_sf = cast_ref(
                x,
                fmt,
                (1, num_per_channels),
                sf=out_sf,
                x_block_size=x_block_size,
                round_sf=round_sf,
                use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
                use_packed_ue8m0=use_packed_ue8m0,
            )
            assert_bit_equal(out_with_sf, ref_with_sf)

        sf_only = per_token_cast_with_sf_only(
            x,
            fmt,
            num_per_channels=num_per_channels,
            x_block_size=x_block_size,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            round_sf=round_sf,
            use_packed_ue8m0=use_packed_ue8m0,
        )
        if use_packed_ue8m0:
            assert_bit_equal(
                clear_unused_sf(sf_only, hidden, num_per_channels),
                clear_unused_sf(ref_sf, hidden, num_per_channels),
            )
        else:
            assert_bit_equal(sf_only, ref_sf)

    print("  PASS")


def test():
    for params in generate_test_params():
        _check_case(params)
    print("Kernel Output Match!")


if __name__ == "__main__":
    test()
