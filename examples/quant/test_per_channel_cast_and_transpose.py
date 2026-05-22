"""Tests for NPU per_channel_cast_and_transpose_kernel (bf16 → fp32)."""

import torch

try:
    from .per_channel_cast_and_transpose_kernel import per_channel_cast_and_transpose
except ImportError:
    from per_channel_cast_and_transpose_kernel import per_channel_cast_and_transpose


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def _cast_ref(
    x: torch.Tensor,
    block_size: tuple[int, int],
    round_sf: bool = False,
) -> tuple:
    """Replicate ``tile_kernels.torch.cast`` logic for bf16→fp32.

    Args:
        x: (num_tokens, hidden) bf16 tensor.
        block_size: (bh, bw) block shape.  bw=1 for per-channel.
        round_sf: round sf to power of 2.

    Returns:
        (out, sf) — fp32 output (num_tokens, hidden) and sf (num_blocks, hidden).
    """
    assert x.ndim == 2
    x_fp32 = x.float()
    h, w = x_fp32.shape
    bh, bw = block_size
    assert bw == 1
    max_val = 1.0  # fp32 output
    min_clamp = 1e-4

    # 1) Padding to block boundaries
    pad_h = (bh - h % bh) % bh
    pad_w = (bw - w % bw) % bw
    padded = torch.nn.functional.pad(x_fp32, (0, pad_w, 0, pad_h))
    valid = torch.nn.functional.pad(
        torch.ones_like(x_fp32, dtype=torch.bool), (0, pad_w, 0, pad_h)
    )
    ph, pw = padded.shape

    # 2) Block-wise max: (H/bh, bh, W/bw, bw) → (H/bh, W/bw, bh*bw)
    reshaped = (
        padded.view(ph // bh, bh, pw // bw, bw)
        .permute(0, 2, 1, 3)
        .reshape(ph // bh, pw // bw, -1)
    )
    rmask = (
        valid.view(ph // bh, bh, pw // bw, bw)
        .permute(0, 2, 1, 3)
        .reshape(ph // bh, pw // bw, -1)
    )
    abs_f = torch.abs(reshaped)
    abs_f = torch.where(rmask, abs_f, torch.tensor(-1.0, device=abs_f.device, dtype=abs_f.dtype))
    amax, _ = abs_f.max(dim=-1, keepdim=True)  # (H/bh, W/bw, 1)
    amax = amax.clamp(min=min_clamp).squeeze(-1)  # (num_blocks, hidden)

    # 3) Compute sf / sf_inv
    if round_sf:
        bits = amax.view(torch.int32)
        exp = ((bits - 1) >> 23) + 1 - 127
        exp = torch.clamp(127 + exp, 0, 255)
        sf = (exp.to(torch.int32) << 23).view(torch.float32)
        sf_inv = (127 - exp).clamp(0, 255).to(torch.int32)
        sf_inv = (sf_inv << 23).view(torch.float32)
    else:
        sf = amax / max_val
        sf_inv = max_val / amax

    # 4) Scale: broadcast sf_inv per block
    sf_inv_view = sf_inv.view(ph // bh, 1, pw // bw, 1)
    padded_view = padded.view(ph // bh, bh, pw // bw, bw)
    out_padded = (padded_view * sf_inv_view).reshape(ph, pw)

    # 5) Crop back
    out = out_padded[:h, :w]

    return out, sf


def per_channel_cast_and_transpose_reference(
    x: torch.Tensor,
    num_per_tokens: int,
    round_sf: bool = False,
) -> tuple:
    """GPU-test reference: ``cast(x, block_size=(num_per_tokens, 1)).T``."""
    out, sf = _cast_ref(x, (num_per_tokens, 1), round_sf)
    return sf, out.T.contiguous()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def align_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def generate_num_tokens(alignment: int = 1) -> list[int]:
    return [align_up(nt, alignment) for nt in [4001, 8001]]


def generate_hidden_sizes() -> list[int]:
    return [576, 2048, 2560, 3072, 4096, 6144, 7168]


def generate_test_params() -> list[dict]:
    return [
        {"num_tokens": nt, "hidden": h, "round_sf": rs, "num_per_tokens": npt}
        for nt in generate_num_tokens(128)
        for h in generate_hidden_sizes()
        for rs in (True, False)
        for npt in (32, 128)
    ]


def _check_case(params: dict):
    print(f"Testing per_channel_cast_and_transpose: {params}")
    npt = params["num_per_tokens"]
    rs = params["round_sf"]
    x = torch.randn((params["num_tokens"], params["hidden"]), dtype=torch.bfloat16, device="npu")

    out_sf, out = per_channel_cast_and_transpose(x, npt, rs)
    torch.npu.synchronize()

    ref_sf, ref_out = per_channel_cast_and_transpose_reference(x.cpu(), npt, rs)
    torch.testing.assert_close(out_sf.cpu(), ref_sf, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out.cpu(), ref_out, rtol=1e-2, atol=1e-2)
    print("  PASS")


def test():
    for params in generate_test_params():
        _check_case(params)
    print("Kernel Output Match!")


if __name__ == "__main__":
    test()
