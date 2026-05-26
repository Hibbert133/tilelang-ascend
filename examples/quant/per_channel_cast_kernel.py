"""NPU per_channel_cast wrapper with fp32 output.

This is the per-channel specialization of ``per_channel_cast_fused`` without
token expansion or input scaling factors:
  scale block = (num_per_tokens, 1)
  out_sf = per-block per-column scale
  out = x / out_sf
"""

import torch

try:
    from .common import get_cast_input_and_config, get_cast_output_config
    from .per_channel_cast_fused_kernel import per_channel_cast_fused
except ImportError:
    from common import get_cast_input_and_config, get_cast_output_config  # type: ignore[no-redef]
    from per_channel_cast_fused_kernel import per_channel_cast_fused  # type: ignore[no-redef]


def per_channel_cast(
    x: torch.Tensor,
    fmt: str = "fp32",
    num_per_tokens: int = 128,
    round_sf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast a matrix to fp32 with per-channel scaling factors.

    Args:
        x: 2D contiguous BF16 tensor on NPU.
        fmt: Output format, currently ``"fp32"`` or ``"float32"``.
        num_per_tokens: Number of tokens in each per-column scale block.
        round_sf: Whether to round scales up to powers of two.
    """
    assert fmt in ("fp32", "float32")
    assert x.dim() == 2 and x.is_contiguous()
    assert x.dtype == torch.bfloat16
    assert x.device.type == "npu"
    assert num_per_tokens == 128

    x_data, x_sf_invs, in_config = get_cast_input_and_config(x, (num_per_tokens, 1))
    assert x_sf_invs is None
    _ = in_config

    num_tokens, hidden = x_data.shape
    assert num_tokens % 128 == 0 and hidden % 64 == 0
    _ = get_cast_output_config("fp32", (num_per_tokens, 1), round_sf=round_sf)

    return per_channel_cast_fused(
        x_data,
        num_per_tokens=num_per_tokens,
        round_sf=round_sf,
        num_per_channels=None,
        pos_to_token=None,
    )
