"""Tests for NPU per_block_cast_lossless_kernel.

Aligned with GPU correctness logic:

GPU-style raw-output correctness:
  1. Prepare fp32 input.
  2. Cast fp32 -> bf16_data + input_sf.
  3. Run NPU per_block_cast_lossless.
  4. Compare kernel raw output against a raw-output reference:

       kernel_out
           ==
       input_bf16 * input_sf / expected_output_sf

The test does not assert output scaling-factor equality.

This version includes detailed debug printing on mismatch:
  - max diff location
  - actual/ref dequant ratio
  - input sf, actual out sf, expected out sf
  - expected/actual relative scale
  - biased exponent fields
"""

import os
import math

import pytest
import torch
import torch.nn.functional as F

try:
    from .common import (
        CastInputConfig,
        get_cast_input_and_config,
        get_cast_output_config,
    )
    from .per_block_cast_lossless_kernel import per_block_cast_lossless
except ImportError:
    from common import (  # type: ignore[no-redef]
        CastInputConfig,
        get_cast_input_and_config,
        get_cast_output_config,
    )
    from per_block_cast_lossless_kernel import per_block_cast_lossless  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WRAPPER_FMT = "fp32"

INPUT_MAX_QUANT_VAL = 6.0
INPUT_MIN_CLAMP_VAL = 6.0 * 2.0 ** (-126)

RTOL = float(os.getenv("RTOL", "1e-3"))
ATOL = float(os.getenv("ATOL", "1e-3"))

DEBUG_ON_FAILURE = int(os.getenv("DEBUG_ON_FAILURE", "1"))
DEBUG_ALWAYS = int(os.getenv("DEBUG_ALWAYS", "0"))


# ---------------------------------------------------------------------------
# Param generation
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
            "in_round_sf": True,
            "in_use_packed_ue8m0": False,
            "out_use_tma_aligned_col_major_sf": out_use_tma_aligned_col_major_sf,
            "out_round_sf": True,
            "out_use_packed_ue8m0": False,
            "out_sf_block": (out_sf_block_m, out_sf_block_k),
            "in_sf_block": (1, 32),
        }
        for num_tokens in generate_num_tokens(is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for in_use_tma_aligned_col_major_sf in (False, True)
        for out_use_tma_aligned_col_major_sf in (False, True)
        for out_sf_block_m, out_sf_block_k in ((1, 128), (32, 32), (128, 128))
        if out_sf_block_m % 1 == 0 and out_sf_block_k % 32 == 0
    ]
    return params


def make_param_id(params: dict) -> str:
    return (
        f"num_tokens={params['num_tokens']}-"
        f"hidden={params['hidden']}-"
        f"in_sf_block={params['in_sf_block']}-"
        f"out_sf_block={params['out_sf_block']}-"
        f"in_tma={params['in_use_tma_aligned_col_major_sf']}-"
        f"out_tma={params['out_use_tma_aligned_col_major_sf']}-"
        f"in_packed={params['in_use_packed_ue8m0']}-"
        f"out_packed={params['out_use_packed_ue8m0']}-"
        f"in_round={params['in_round_sf']}-"
        f"out_round={params['out_round_sf']}"
    )


def per_block_cast_lossless_from_params(
    x: tuple[torch.Tensor, torch.Tensor],
    params: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    return per_block_cast_lossless(
        x,
        WRAPPER_FMT,
        x_block_size=params["in_sf_block"],
        out_block_size=params["out_sf_block"],
        use_tma_aligned_col_major_sf=params["out_use_tma_aligned_col_major_sf"],
        round_sf=params["out_round_sf"],
        use_packed_ue8m0=params["out_use_packed_ue8m0"],
        in_use_tma_aligned_col_major_sf=params["in_use_tma_aligned_col_major_sf"],
        in_round_sf=params["in_round_sf"],
        in_use_packed_ue8m0=params["in_use_packed_ue8m0"],
    )


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _format_sf_for_config(
    sf_logical: torch.Tensor,
    config: CastInputConfig,
    device: torch.device,
) -> torch.Tensor:
    """Format logical row-major fp32 sf into kernel/internal layout."""
    assert not config.use_packed_ue8m0, (
        "packed UE8M0 is not supported by current NPU test"
    )

    if config.use_tma_aligned_col_major_sf:
        return sf_logical.T.contiguous().to(device=device)

    return sf_logical.contiguous().to(device=device)


def _logical_sf_from_external(
    sf_external: torch.Tensor,
    config: CastInputConfig,
) -> torch.Tensor:
    """Return logical row-major fp32 sf from an external-layout tensor."""
    assert not config.use_packed_ue8m0, (
        "packed UE8M0 is not supported by current NPU test"
    )

    if config.use_tma_aligned_col_major_sf:
        return sf_external.detach().to(torch.float32).T.contiguous()

    return sf_external.detach().to(torch.float32).contiguous()


# ---------------------------------------------------------------------------
# GPU-style cast / cast_back helpers adapted for bf16+sf input
# ---------------------------------------------------------------------------

def cast(
    x: torch.Tensor,
    block_size: tuple[int, int],
    round_sf: bool = False,
    use_tma_aligned_col_major_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-style cast adapted for this NPU lossless test.

    Input:
      fp32 x

    Output:
      bf16 quantized data, fp32 scaling factors in requested layout.

    Flow:
      padding -> valid mask -> block amax -> dequant_sf -> quantize data -> format sf
    """
    assert x.ndim == 2
    assert x.dtype == torch.float32
    assert not use_packed_ue8m0, (
        "packed UE8M0 is not supported by current NPU test"
    )

    h, w = x.shape
    bh, bw = block_size
    device = x.device

    x_fp32 = x.to(torch.float32)

    pad_h = (bh - h % bh) % bh
    pad_w = (bw - w % bw) % bw

    padded_src = F.pad(x_fp32, (0, pad_w, 0, pad_h))
    valid_mask = F.pad(
        torch.ones_like(x_fp32, dtype=torch.bool),
        (0, pad_w, 0, pad_h),
    )

    ph, pw = padded_src.shape

    reshaped_for_max = (
        padded_src.view(ph // bh, bh, pw // bw, bw)
        .permute(0, 2, 1, 3)
        .reshape(ph // bh, pw // bw, -1)
    )
    reshaped_mask = (
        valid_mask.view(ph // bh, bh, pw // bw, bw)
        .permute(0, 2, 1, 3)
        .reshape(ph // bh, pw // bw, -1)
    )

    abs_f = torch.abs(reshaped_for_max)
    abs_f = torch.where(
        reshaped_mask,
        abs_f,
        torch.tensor(-1.0, device=device, dtype=abs_f.dtype),
    )

    max_val, _ = abs_f.max(dim=-1, keepdim=True)
    max_val = torch.clamp(max_val, min=INPUT_MIN_CLAMP_VAL)

    dequant_sf = max_val / INPUT_MAX_QUANT_VAL
    dequant_sf_bits = dequant_sf.contiguous().view(torch.int32)

    if round_sf:
        dequant_sf_bits = (dequant_sf_bits + 0x007FFFFF) & 0x7F800000
        dequant_sf = dequant_sf_bits.view(torch.float32)
    else:
        dequant_sf = dequant_sf_bits.view(torch.float32)

    quant_sf = torch.full_like(dequant_sf, 1.0, dtype=torch.float32) / dequant_sf

    padded_src_view = padded_src.view(ph // bh, bh, pw // bw, bw)
    quant_sf_view = quant_sf.view(ph // bh, 1, pw // bw, 1)

    quant_tensor = (padded_src_view * quant_sf_view).reshape(ph, pw)
    quant_tensor = quant_tensor[:h, :w]

    x_bf16 = quant_tensor.to(torch.bfloat16).contiguous()

    sf_logical = dequant_sf_bits.squeeze(-1).view(torch.float32).contiguous()

    in_config = CastInputConfig(
        torch_dtype=torch.bfloat16,
        sf_block=block_size,
        with_sf=True,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    x_sf = _format_sf_for_config(sf_logical, in_config, device)
    return x_bf16, x_sf


def cast_back_input(
    x: tuple[torch.Tensor, torch.Tensor],
    block_size: tuple[int, int],
    config: CastInputConfig,
    logical_shape: tuple[int, int],
) -> torch.Tensor:
    """GPU-style cast_back for input bf16 data + input sf."""
    x_data, x_sf = x
    h, w = logical_shape

    sf_logical = x_sf.detach().to(torch.float32).contiguous()
    sf_expanded = (
        sf_logical.repeat_interleave(block_size[0], dim=0)
        .repeat_interleave(block_size[1], dim=1)
    )
    sf_expanded = sf_expanded[:h, :w].to(device=x_data.device)

    return x_data.to(torch.float32) * sf_expanded


# ---------------------------------------------------------------------------
# Reference / debug sf computation
# ---------------------------------------------------------------------------

def _biased_exp_from_sf(sf: torch.Tensor) -> torch.Tensor:
    """Return biased fp32 exponent field: (bits >> 23) & 0xFF."""
    sf_cpu = sf.detach().cpu().to(torch.float32).contiguous()
    bits = sf_cpu.view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.int32)


def _sf_from_biased_exp(exp: torch.Tensor) -> torch.Tensor:
    """Return fp32 sf from biased exponent field."""
    exp_cpu = exp.detach().cpu().to(torch.int32).contiguous()
    bits = exp_cpu << 23
    return bits.view(torch.float32)


def compute_expected_out_sf_logical(
    x_sf: torch.Tensor,
    in_config: CastInputConfig,
    out_config,
    logical_shape: tuple[int, int],
) -> torch.Tensor:
    """Compute expected logical row-major out_sf using GPU biased exponent logic.

    GPU lossless:
      out_exp = max(input_exp) - 6, saturated at 0
      out_sf = reinterpret(out_exp << 23, fp32)
    """
    h, w = logical_shape
    in_bm, in_bk = in_config.sf_block
    out_bm, out_bk = out_config.sf_block

    in_sf_logical = _logical_sf_from_external(x_sf, in_config).detach().cpu().float()
    in_exp = _biased_exp_from_sf(in_sf_logical)

    out_sf_m = math.ceil(h / out_bm)
    out_sf_k = math.ceil(w / out_bk)

    in_per_out_m = out_bm // in_bm
    in_per_out_k = out_bk // in_bk

    out_exp = torch.zeros((out_sf_m, out_sf_k), dtype=torch.int32)

    for om in range(out_sf_m):
        for ok in range(out_sf_k):
            src = in_exp[
                om * in_per_out_m : (om + 1) * in_per_out_m,
                ok * in_per_out_k : (ok + 1) * in_per_out_k,
            ]
            max_exp = int(src.max().item())
            out_exp[om, ok] = max(max_exp - 6, 0)

    return _sf_from_biased_exp(out_exp)


def _safe_ratio(a: float, b: float) -> float:
    if b == 0.0:
        return float("inf") if a != 0.0 else 1.0
    return a / b


def _safe_log2_ratio(a: float, b: float) -> float:
    ratio = abs(_safe_ratio(a, b))
    if ratio <= 0 or math.isinf(ratio) or math.isnan(ratio):
        return float("nan")
    return math.log2(ratio)


def _biased_exp_scalar(v: float) -> int:
    t = torch.tensor([v], dtype=torch.float32)
    return int(((t.view(torch.int32) >> 23) & 0xFF).item())


def _print_failure_debug(
    *,
    params: dict,
    x_casted: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    out_sf: torch.Tensor,
    out_dequant: torch.Tensor,
    ref_dequant: torch.Tensor,
) -> None:
    x_data, x_sf = x_casted

    _, _, in_config = get_cast_input_and_config(
        x_casted,
        params["in_sf_block"],
        use_tma_aligned_col_major_sf=params["in_use_tma_aligned_col_major_sf"],
        round_sf=params["in_round_sf"],
        use_packed_ue8m0=params["in_use_packed_ue8m0"],
    )

    out_config = get_cast_output_config(
        WRAPPER_FMT,
        params["out_sf_block"],
        use_tma_aligned_col_major_sf=params["out_use_tma_aligned_col_major_sf"],
        round_sf=params["out_round_sf"],
        use_packed_ue8m0=params["out_use_packed_ue8m0"],
    )

    actual_cpu = out_dequant.detach().cpu().float()
    expected_cpu = ref_dequant.detach().cpu().float()
    diff = (actual_cpu - expected_cpu).abs()

    flat_idx = int(diff.argmax().item())
    m_t, k_t = torch.unravel_index(torch.tensor(flat_idx), diff.shape)
    m, k = int(m_t.item()), int(k_t.item())

    x_cpu = x_data.detach().cpu().float()
    out_cpu = out.detach().cpu().float()

    in_sf_logical = _logical_sf_from_external(x_sf, in_config).detach().cpu().float()
    out_sf_logical = out_sf.detach().cpu().float().contiguous()

    expected_out_sf_logical = compute_expected_out_sf_logical(
        x_sf,
        in_config,
        out_config,
        tuple(x_data.shape),
    )

    in_bm, in_bk = params["in_sf_block"]
    out_bm, out_bk = params["out_sf_block"]

    in_sf_m = m // in_bm
    in_sf_k = k // in_bk
    out_sf_m = m // out_bm
    out_sf_k = k // out_bk

    x_val = float(x_cpu[m, k].item())
    out_val = float(out_cpu[m, k].item())
    actual_val = float(actual_cpu[m, k].item())
    expected_val = float(expected_cpu[m, k].item())

    input_sf_val = float(in_sf_logical[in_sf_m, in_sf_k].item())
    actual_out_sf_val = float(out_sf_logical[out_sf_m, out_sf_k].item())
    expected_out_sf_val = float(expected_out_sf_logical[out_sf_m, out_sf_k].item())

    expected_rel_sf = _safe_ratio(input_sf_val, expected_out_sf_val)
    actual_rel_sf = _safe_ratio(out_val, x_val)
    rel_sf_from_actual_outsf = _safe_ratio(input_sf_val, actual_out_sf_val)

    print("\n" + "=" * 100)
    print("DEBUG FAILURE: per_block_cast_lossless")
    print("=" * 100)
    print("params:", params)
    print("max_abs_idx:", (m, k))
    print("max_abs_diff:", float(diff[m, k].item()))
    print("actual/ref ratio:", _safe_ratio(actual_val, expected_val))
    print("log2(abs(actual/ref)):", _safe_log2_ratio(actual_val, expected_val))
    print("-" * 100)

    print("x_bf16 value:", x_val)
    print("kernel out value:", out_val)
    print("actual raw out:", actual_val)
    print("expected raw out_ref:", expected_val)

    print("-" * 100)
    print("input sf idx:", (in_sf_m, in_sf_k))
    print("input sf value:", input_sf_val)
    print("input sf biased exp:", _biased_exp_scalar(input_sf_val))

    print("out sf idx:", (out_sf_m, out_sf_k))
    print("actual out_sf value:", actual_out_sf_val)
    print("actual out_sf biased exp:", _biased_exp_scalar(actual_out_sf_val))
    print("expected out_sf value:", expected_out_sf_val)
    print("expected out_sf biased exp:", _biased_exp_scalar(expected_out_sf_val))
    print("actual_out_sf / expected_out_sf:", _safe_ratio(actual_out_sf_val, expected_out_sf_val))
    print("log2(abs(actual_out_sf / expected_out_sf)):", _safe_log2_ratio(actual_out_sf_val, expected_out_sf_val))

    print("-" * 100)
    print("expected relative_sf = input_sf / expected_out_sf:", expected_rel_sf)
    print("actual relative_sf approx = out / x_bf16:", actual_rel_sf)
    print("input_sf / actual_out_sf:", rel_sf_from_actual_outsf)
    print("actual_rel / expected_rel:", _safe_ratio(actual_rel_sf, expected_rel_sf))
    print("log2(abs(actual_rel / expected_rel)):", _safe_log2_ratio(actual_rel_sf, expected_rel_sf))

    print("-" * 100)
    in_per_out_m = out_bm // in_bm
    in_per_out_k = out_bk // in_bk
    group_m0 = out_sf_m * in_per_out_m
    group_m1 = (out_sf_m + 1) * in_per_out_m
    group_k0 = out_sf_k * in_per_out_k
    group_k1 = (out_sf_k + 1) * in_per_out_k
    group_sf = in_sf_logical[group_m0:group_m1, group_k0:group_k1]
    group_exp = _biased_exp_from_sf(group_sf)

    print("input sf group range m:", (group_m0, group_m1), "k:", (group_k0, group_k1))
    print("input sf group values:")
    print(group_sf)
    print("input sf group biased exps:")
    print(group_exp)
    print("expected max input exp:", int(group_exp.max().item()))
    print("expected out exp = max - 6:", max(int(group_exp.max().item()) - 6, 0))
    print("=" * 100 + "\n")


# ---------------------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------------------

def generate_test_data(
    params: dict,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Generate fp32 source and its bf16+sf casted input."""
    num_tokens = params["num_tokens"]
    hidden = params["hidden"]

    x_fp32 = torch.randn((num_tokens, hidden), dtype=torch.float32, device="npu")

    x_casted = cast(
        x_fp32,
        params["in_sf_block"],
        round_sf=params["in_round_sf"],
        use_tma_aligned_col_major_sf=params["in_use_tma_aligned_col_major_sf"],
        use_packed_ue8m0=params["in_use_packed_ue8m0"],
    )

    return x_fp32, x_casted


def compute_reference_out(
    x: tuple[torch.Tensor, torch.Tensor],
    params: dict,
) -> torch.Tensor:
    x_data, x_sf = x
    x_data2, x_sf2, in_config = get_cast_input_and_config(
        x,
        params["in_sf_block"],
        use_tma_aligned_col_major_sf=params["in_use_tma_aligned_col_major_sf"],
        round_sf=params["in_round_sf"],
        use_packed_ue8m0=params["in_use_packed_ue8m0"],
    )
    out_config = get_cast_output_config(
        WRAPPER_FMT,
        params["out_sf_block"],
        use_tma_aligned_col_major_sf=params["out_use_tma_aligned_col_major_sf"],
        round_sf=params["out_round_sf"],
        use_packed_ue8m0=params["out_use_packed_ue8m0"],
    )
    input_dequant = cast_back_input(
        (x_data2, x_sf2),
        in_config.sf_block,
        in_config,
        tuple(x_data.shape),
    )
    expected_out_sf = compute_expected_out_sf_logical(
        x_sf,
        in_config,
        out_config,
        tuple(x_data.shape),
    ).to(device=x_data.device)
    expected_out_sf = (
        expected_out_sf.repeat_interleave(out_config.sf_block[0], dim=0)
        .repeat_interleave(out_config.sf_block[1], dim=1)
    )
    expected_out_sf = expected_out_sf[: x_data.shape[0], : x_data.shape[1]]
    return input_dequant / expected_out_sf


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_tensor_match(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()

    assert actual_cpu.shape == expected_cpu.shape, (
        f"{name} shape mismatch: "
        f"actual={tuple(actual_cpu.shape)}, expected={tuple(expected_cpu.shape)}"
    )

    try:
        torch.testing.assert_close(actual_cpu, expected_cpu, rtol=RTOL, atol=ATOL)
    except AssertionError as exc:
        diff = (actual_cpu.to(torch.float32) - expected_cpu.to(torch.float32)).abs()

        max_abs = diff.max().item() if diff.numel() else 0.0
        max_idx = None
        if diff.numel():
            flat_idx = int(diff.argmax().item())
            max_idx = tuple(torch.unravel_index(torch.tensor(flat_idx), diff.shape))

        raise AssertionError(
            f"{name} mismatch\n"
            f"max_abs_diff={max_abs}\n"
            f"max_abs_idx={max_idx}\n"
            f"{exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

def _check_case(params: dict) -> None:
    print(f"Testing per_block_cast_lossless with params={params}")

    _, x_casted = generate_test_data(params)

    out, out_sf = per_block_cast_lossless_from_params(x_casted, params)
    torch.npu.synchronize()

    out_ref = compute_reference_out(x_casted, params)

    if DEBUG_ALWAYS:
        _print_failure_debug(
            params=params,
            x_casted=x_casted,
            out=out,
            out_sf=out_sf,
            out_dequant=out,
            ref_dequant=out_ref,
        )

    try:
        assert_tensor_match(
            out,
            out_ref,
            "out_vs_out_ref",
        )
    except AssertionError:
        if DEBUG_ON_FAILURE:
            _print_failure_debug(
                params=params,
                x_casted=x_casted,
                out=out,
                out_sf=out_sf,
                out_dequant=out,
                ref_dequant=out_ref,
            )
        raise

    print("  PASS")


@pytest.mark.parametrize(
    "params",
    generate_test_params(is_benchmark=False),
    ids=make_param_id,
)
def test_per_block_cast_lossless(params: dict) -> None:
    _check_case(params)


def run_all_cases() -> None:
    for params in generate_test_params(is_benchmark=False):
        _check_case(params)
    print("Kernel Output Match!")


if __name__ == "__main__":
    run_all_cases()
