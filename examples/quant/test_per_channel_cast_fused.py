"""Tests and reference for NPU per_channel_cast_fused_kernel.

Reference follows the GPU ``per_channel_cast_fused`` logic (token expansion,
sf_invs propagation, per-block scaling).  Generator helpers are replicated
from ``tile_kernels.testing.generator`` to avoid depending on ``tile_kernels``.

Run on NPU hardware:
    cd examples/quant
    python test_per_channel_cast_fused.py
"""

import itertools
import os
from typing import Iterable, Optional, Union

import torch

try:
    from .common import ceil_div
    from .per_channel_cast_fused_kernel import per_channel_cast_fused
except ImportError:
    from common import ceil_div  # type: ignore[no-redef]
    from per_channel_cast_fused_kernel import per_channel_cast_fused  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Generator helpers (replicated from tile_kernels.testing.generator)
# ---------------------------------------------------------------------------


def generate_hidden_sizes(align: int = 64) -> list[int]:
    base_list = [576, 2048, 2560, 3072, 4096, 6144, 7168]
    return [h for h in base_list if h % align == 0]


def generate_moe_params(is_benchmark: bool = False) -> Iterable[dict]:
    do_full_test = os.getenv("TK_FULL_TEST") in ["1", "true", "True"]
    extra_num_topk_list = (1, 7) if do_full_test else ()
    extra_num_experts_list = (288, 384) if do_full_test else ()
    extra_num_ep_ranks_list = (1, 72, 256) if do_full_test else ()

    if do_full_test and not is_benchmark:
        yield {"num_send_tokens": 0, "num_topk": 1, "num_experts": 1, "num_ep_ranks": 1}

    for num_tokens in (4001,):
        for num_topk in (2, 6, 8, 9) + extra_num_topk_list:
            for num_experts in (72, 256) + extra_num_experts_list:
                for num_ep_ranks in (8, 64) + extra_num_ep_ranks_list:
                    if num_experts % num_ep_ranks == 0:
                        yield {
                            "num_send_tokens": num_tokens,
                            "num_topk": num_topk,
                            "num_experts": num_experts // num_ep_ranks,
                            "num_ep_ranks": num_ep_ranks,
                        }


def generate_topk_idx(params: dict) -> torch.Tensor:
    num_send_tokens = params["num_send_tokens"]
    num_experts = params["num_experts"]
    num_topk = params["num_topk"]
    num_ep_ranks = params["num_ep_ranks"]
    device = "npu"

    if num_send_tokens == 0:
        return torch.empty((0, num_topk), dtype=torch.int64, device=device)
    scores = torch.rand(
        (num_send_tokens * num_ep_ranks, num_experts * num_ep_ranks),
        dtype=torch.bfloat16,
        device=device,
    )
    _, topk_idx = torch.topk(scores, k=num_topk, dim=-1, sorted=False)
    mask = topk_idx >= num_experts
    topk_idx[mask] = -1
    mask = mask.all(dim=1)
    topk_idx = topk_idx[~mask]
    return topk_idx


# ---------------------------------------------------------------------------
# Reference — mirrors GPU per_channel_cast_fused, adapted for fp32 output
# ---------------------------------------------------------------------------


def _round_sf_to_power_of_two_ref(sf: torch.Tensor) -> torch.Tensor:
    """Round sf up to the nearest power of two (GPU get_sf_and_inv round_sf path)."""
    bits = sf.view(torch.int32)
    exp = ((bits - 1) >> 23) + 1 - 127
    exp = torch.clamp(127 + exp, 0, 255)
    return (exp.to(torch.int32) << 23).view(torch.float32)


def _round_sf_inv_to_power_of_two_ref(sf: torch.Tensor) -> torch.Tensor:
    """sf_inv = 2^-ceil(log2(sf))."""
    bits = sf.view(torch.int32)
    exp = ((bits - 1) >> 23) + 1 - 127
    exp_inv = torch.clamp(127 - exp, 0, 255)
    return (exp_inv.to(torch.int32) << 23).view(torch.float32)


def per_channel_cast_fused_ref(
    x: Union[torch.Tensor, tuple],
    num_per_tokens: int,
    num_per_channels: Optional[int],
    round_sf: bool,
    pos_to_token: Optional[torch.Tensor],
    max_value: float = 448.0,
) -> tuple:
    """PyTorch reference for per_channel_cast_fused.

    Args:
        max_value: Target format max representable value.
                   fp32 output -> 1.0 (normalized to [-1, 1]).
    """
    is_fused_cast_back = isinstance(x, tuple)
    num_per_channels = num_per_channels if is_fused_cast_back else None

    # ── Token expansion ──
    if pos_to_token is not None:
        x_data = x[0] if is_fused_cast_back else x
        x_gathered = x_data[pos_to_token.clamp(min=0)]
        valid_mask = (pos_to_token >= 0).unsqueeze(1)
        x_gathered = torch.where(
            valid_mask,
            x_gathered.to(torch.float32),
            torch.zeros_like(x_gathered, dtype=torch.float32),
        ).to(x_data.dtype)
        if is_fused_cast_back:
            x_sf = x[1]
            x_sf_gathered = x_sf[pos_to_token.clamp(min=0)]
            x_sf_gathered = torch.where(
                valid_mask, x_sf_gathered, torch.zeros_like(x_sf_gathered)
            )
            x = (x_gathered, x_sf_gathered)
        else:
            x = x_gathered

    # ── Unpack ──
    if is_fused_cast_back:
        x_data, x_sf_invs = x
        x_fp32 = x_data.float()
        x_sf_invs = x_sf_invs.float()
        assert num_per_channels is not None
    else:
        x_data = x
        x_fp32 = x_data.float()
        x_sf_invs = None
        num_per_channels = None

    num_tokens, hidden = x_fp32.shape

    out = torch.zeros(num_tokens, hidden, dtype=torch.float32)
    out_sf = torch.zeros(
        ceil_div(num_tokens, num_per_tokens), hidden, dtype=torch.float32
    )

    num_blocks_m = ceil_div(num_tokens, num_per_tokens)

    for bm in range(num_blocks_m):
        r0 = bm * num_per_tokens
        r1 = min(r0 + num_per_tokens, num_tokens)

        block = x_fp32[r0:r1, :].clone()

        if x_sf_invs is not None:
            for c0 in range(0, hidden, num_per_channels):
                c1 = min(c0 + num_per_channels, hidden)
                k_idx = c0 // num_per_channels
                sf_col = x_sf_invs[r0:r1, k_idx].unsqueeze(1)
                block[:, c0:c1] = block[:, c0:c1] * sf_col

        amax = block.abs().amax(dim=0, keepdim=False).clamp(min=1e-4)

        if round_sf:
            sf_val = amax / max_value
            sf = _round_sf_to_power_of_two_ref(sf_val)
            sf_inv = _round_sf_inv_to_power_of_two_ref(sf_val)
        else:
            sf = amax / max_value
            sf_inv = max_value / amax

        out_sf[bm, :] = sf

        if x_sf_invs is not None:
            for c0 in range(0, hidden, num_per_channels):
                c1 = min(c0 + num_per_channels, hidden)
                k_idx = c0 // num_per_channels
                sf_col = x_sf_invs[r0:r1, k_idx].unsqueeze(1)
                out[r0:r1, c0:c1] = (
                    x_fp32[r0:r1, c0:c1] * sf_col * sf_inv[c0:c1]
                )
        else:
            out[r0:r1, :] = x_fp32[r0:r1, :] * sf_inv

    return out, out_sf


# ---------------------------------------------------------------------------
# Test data generation (aligned with GPU test_per_channel_cast_fused.py)
# ---------------------------------------------------------------------------


def _generate_random_pos_to_token(
    num_tokens: int, num_topk: int, topk_idx: torch.Tensor, device: str
) -> torch.Tensor:
    """Generate a random pos_to_token tensor matching the shape GPU MOE expansion
    would produce.  Includes -1 entries to simulate padding.
    """
    # Approximate the expanded size: each token appears up to num_topk times,
    # padded to 128-alignment.
    valid_count = (topk_idx >= 0).sum().item()
    padded = ((valid_count + 127) // 128) * 128
    if padded == 0:
        padded = 128
    pt = torch.randint(0, num_tokens, (padded,), dtype=torch.int32, device=device)
    pt[valid_count:] = -1
    return pt


def generate_test_data(params: dict):
    num_send_tokens = params["num_send_tokens"]
    num_topk = params["num_topk"]
    num_experts = params["num_experts"]
    hidden = params["hidden"]
    num_per_tokens = params["num_per_tokens"]
    num_per_channels = params["num_per_channels"]
    is_fused_cast_back = params["is_fused_cast_back"]
    round_sf = params["round_sf"]
    device = "npu"

    pos_to_token = None
    if num_topk > 0:
        topk_idx = generate_topk_idx(params)
        num_tokens = topk_idx.shape[0]
        pos_to_token = _generate_random_pos_to_token(num_tokens, num_topk, topk_idx, device)
        # Random expanded x of the shape pos_to_token would produce
        x = torch.randn((pos_to_token.shape[0], hidden), dtype=torch.bfloat16, device=device)
    else:
        num_tokens = num_send_tokens
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device)

    if is_fused_cast_back:
        sf_shape = (x.shape[0], ceil_div(hidden, num_per_channels))
        x_sf_invs_val = (
            torch.rand(sf_shape, dtype=torch.float32, device=device) * 0.5 + 0.5
        )
        x = (x, x_sf_invs_val)

    def func():
        return per_channel_cast_fused(
            x,
            num_per_tokens=num_per_tokens,
            round_sf=round_sf,
            num_per_channels=num_per_channels if is_fused_cast_back else None,
            pos_to_token=pos_to_token,
        )

    def func_ref():
        ch = num_per_channels if is_fused_cast_back else None
        # Move to CPU — NPU doesn't support .view(torch.int32) bit ops in ref.
        x_cpu = (
            (x[0].cpu(), x[1].cpu()) if isinstance(x, tuple) else x.cpu()
        )
        pt_cpu = pos_to_token.cpu() if pos_to_token is not None else None
        return per_channel_cast_fused_ref(
            x_cpu, num_per_tokens, ch, round_sf, pt_cpu, max_value=1.0
        )

    return func, func_ref


def generate_test_params(is_benchmark: bool = False) -> list[dict]:
    params = [
        {
            **moe,
            "hidden": hidden_size,
            "num_per_tokens": num_per_tokens,
            "num_per_channels": num_per_channels,
            "is_fused_cast_back": is_fused_cast_back,
            "round_sf": round_sf,
        }
        for moe in itertools.chain(
            iter([{"num_send_tokens": 4096, "num_topk": 0, "num_experts": 0, "num_ep_ranks": 0}]),
            generate_moe_params(is_benchmark),
        )
        for hidden_size in generate_hidden_sizes(128)
        for num_per_tokens, num_per_channels in [(128, 128)]
        for is_fused_cast_back in (False, True)
        for round_sf in (False, True)
    ]
    if is_benchmark:
        params = [p for p in params if p["num_topk"] in (0, 6)]
    return params

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _check_case(params: dict):
    """Single test case: run kernel vs reference, compare with assert_close."""
    func, func_ref = generate_test_data(params)

    out, out_sf = func()
    torch.npu.synchronize()

    ref_out, ref_sf = func_ref()
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), ref_out.cpu(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_sf.cpu(), ref_sf.cpu(), rtol=1e-2, atol=1e-2)


def test():
    """Run all test cases."""
    for params in generate_test_params(is_benchmark=False):
        print(f"Testing per_channel_cast_fused: {params}")
        _check_case(params)
        print(f"  PASS")
    print("Kernel Output Match!")


if __name__ == "__main__":
    test()
