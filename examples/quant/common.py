from dataclasses import dataclass, replace
from typing import Optional

import torch

__all__ = [
    "BaseCastConfig",
    "CastInputConfig",
    "CastOutputConfig",
    "ceil_div",
    "align_up",
    "is_power_of_two",
    "scale_fill_value",
    "get_sf_shape",
    "alloc_scaling_factors",
    "pad_or_create_scaling_factors",
    "cast_epilogue",
    "get_cast_input_and_config",
    "get_cast_output_config",
    "load_sf",
    "store_sf",
    "sf_to_exponent_value",
    "exponent_to_sf_value",
    "transform_sf_for_ref",
    "cast_back_ref",
    "generate_input_scaling_factors",
]


@dataclass(frozen=True)
class BaseCastConfig:
    torch_dtype: torch.dtype = torch.float32
    sf_block: tuple[int, int] = (1, 1)
    use_tma_aligned_col_major_sf: bool = False
    use_packed_ue8m0: bool = False

    @property
    def dtype(self) -> str:
        return str(self.torch_dtype).replace("torch.", "")

    @property
    def sf_torch_dtype(self) -> torch.dtype:
        return torch.uint8 if self.use_packed_ue8m0 else torch.float32

    @property
    def sf_dtype(self) -> str:
        return str(self.sf_torch_dtype).replace("torch.", "")


@dataclass(frozen=True)
class CastInputConfig(BaseCastConfig):
    torch_dtype: torch.dtype = torch.bfloat16
    with_sf: bool = True


@dataclass(frozen=True)
class CastOutputConfig(BaseCastConfig):
    torch_dtype: torch.dtype = torch.float32
    round_sf: bool = False
    custom_clamp_min_value: Optional[float] = None

    @property
    def clamp_min_value(self) -> float:
        if self.custom_clamp_min_value is not None:
            return self.custom_clamp_min_value
        if self.dtype == "float32":
            return torch.finfo(torch.float32).tiny
        if self.dtype == "bfloat16":
            return torch.finfo(torch.bfloat16).tiny
        raise ValueError(f"Unsupported dtype {self.dtype}")


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align_up(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def scale_fill_value(config: BaseCastConfig) -> int | float:
    return 127 if config.use_packed_ue8m0 else 1.0


def get_sf_shape(shape: tuple[int, int], config: BaseCastConfig) -> tuple[int, int]:
    num_block_m = ceil_div(shape[0], config.sf_block[0])
    num_block_k = ceil_div(shape[1], config.sf_block[1])
    if config.use_packed_ue8m0:
        num_block_m *= 4
        num_block_k = ceil_div(num_block_k, 4)
    return (num_block_k, num_block_m) if config.use_tma_aligned_col_major_sf else (num_block_m, num_block_k)


def alloc_scaling_factors(
    shape: tuple[int, int],
    out_config: BaseCastConfig,
    device: torch.device,
) -> torch.Tensor:
    if out_config.use_packed_ue8m0:
        assert out_config.use_tma_aligned_col_major_sf, (
            "packed UE8M0 scaling factors require TMA-aligned col-major layout"
        )
    sf_shape = get_sf_shape(shape, out_config)
    aligned_sf_shape = sf_shape[1]
    if out_config.use_tma_aligned_col_major_sf:
        aligned_sf_shape = align_up(sf_shape[1], 16 if out_config.use_packed_ue8m0 else 4)

    scaling_factor = torch.empty(
        size=(sf_shape[0], aligned_sf_shape),
        dtype=out_config.sf_torch_dtype,
        device=device,
    )
    if out_config.use_tma_aligned_col_major_sf:
        scaling_factor = scaling_factor[:, : sf_shape[1]]
    return scaling_factor


def pad_or_create_scaling_factors(
    x_sf: torch.Tensor | None,
    shape: tuple[int, int],
    config: BaseCastConfig,
    device: torch.device,
) -> torch.Tensor:
    sf_shape = get_sf_shape(shape, config)
    fill_value = scale_fill_value(config)
    if x_sf is None:
        return torch.full(sf_shape, fill_value, dtype=config.sf_torch_dtype, device=device)
    if tuple(x_sf.shape) == sf_shape:
        return x_sf

    padded = torch.full(sf_shape, fill_value, dtype=x_sf.dtype, device=x_sf.device)
    copy_shape = tuple(min(src, dst) for src, dst in zip(x_sf.shape, sf_shape))
    padded[: copy_shape[0], : copy_shape[1]] = x_sf[: copy_shape[0], : copy_shape[1]]
    return padded


def cast_epilogue(
    out_sf: torch.Tensor,
    num_tokens: int,
    hidden: int,
    config: BaseCastConfig,
) -> torch.Tensor:
    if config.use_packed_ue8m0:
        if num_tokens == 0:
            out_sf = torch.empty(
                (out_sf.shape[0], out_sf.shape[1] // 4),
                dtype=torch.int32,
                device=out_sf.device,
            )
        else:
            out_sf = out_sf.view(dtype=torch.int32)
    out_sf = out_sf.T if config.use_tma_aligned_col_major_sf else out_sf
    return out_sf[
        : ceil_div(num_tokens, config.sf_block[0]),
        : ceil_div(hidden, config.sf_block[1]),
    ]


def get_cast_input_and_config(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    sf_block: tuple[int, int],
    use_tma_aligned_col_major_sf: bool | None = None,
    round_sf: bool | None = None,
    use_packed_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, CastInputConfig]:
    _ = round_sf
    if isinstance(x, tuple):
        x_data, x_sf = x
        config = CastInputConfig(torch_dtype=x_data.dtype, sf_block=sf_block, with_sf=True)
        assert x_data.dtype == torch.bfloat16
        assert isinstance(x_sf, torch.Tensor)
        if use_tma_aligned_col_major_sf is None:
            use_tma_aligned_col_major_sf = x_sf.stride(0) == 1
        if use_packed_ue8m0 is None:
            use_packed_ue8m0 = x_sf.dtype == torch.int32
        config = replace(
            config,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            use_packed_ue8m0=use_packed_ue8m0,
        )
        if config.use_tma_aligned_col_major_sf:
            x_sf = x_sf.T
            if config.use_packed_ue8m0:
                assert x_sf.dtype == torch.int32
                x_sf = x_sf.view(torch.uint8)
        else:
            assert x_sf.stride(1) == 1
            assert x_sf.dtype == torch.float32
        return x_data, x_sf, config

    assert x.dtype == torch.bfloat16
    return x, None, CastInputConfig(
        torch_dtype=x.dtype,
        sf_block=sf_block,
        with_sf=True,
        use_tma_aligned_col_major_sf=bool(use_tma_aligned_col_major_sf),
        use_packed_ue8m0=bool(use_packed_ue8m0),
    )


def get_cast_output_config(
    fmt: str,
    sf_block: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    custom_clamp_min_value: Optional[float] = None,
) -> CastOutputConfig:
    assert fmt in ("fp32", "float32", "e5m6")
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "e5m6": torch.uint32,  # kernel outputs packed e5m6 as uint32
    }
    return CastOutputConfig(
        torch_dtype=mapping[fmt],
        sf_block=sf_block,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        custom_clamp_min_value=custom_clamp_min_value,
    )


def load_sf(tensor: torch.Tensor, m_idx: int, k_idx: int, config: BaseCastConfig):
    if config.use_packed_ue8m0:
        return tensor[k_idx // 4, m_idx * 4 + k_idx % 4]
    if config.use_tma_aligned_col_major_sf:
        return tensor[k_idx, m_idx]
    return tensor[m_idx, k_idx]


def store_sf(tensor: torch.Tensor, sf, m_idx: int, k_idx: int, config: BaseCastConfig) -> None:
    if config.use_packed_ue8m0:
        tensor[k_idx // 4, m_idx * 4 + k_idx % 4] = sf
    elif config.use_tma_aligned_col_major_sf:
        tensor[k_idx, m_idx] = sf
    else:
        tensor[m_idx, k_idx] = sf


def sf_to_exponent_value(sf, config: BaseCastConfig) -> int:
    if config.use_packed_ue8m0:
        return int(sf.item())
    value = float(sf.item())
    if value <= 0.0:
        return 0
    return int(torch.tensor(value, dtype=torch.float32).view(torch.int32).item() >> 23)


def exponent_to_sf_value(exp: int, config: BaseCastConfig):
    if config.use_packed_ue8m0:
        return exp & 0xFF
    return torch.tensor(exp << 23, dtype=torch.int32).view(torch.float32).item()


def transform_sf_for_ref(
    sf: torch.Tensor,
    config: BaseCastConfig,
    data_shape: tuple[int, int],
    already_internal_layout: bool = False,
) -> torch.Tensor:
    num_sf_m = ceil_div(data_shape[0], config.sf_block[0])
    num_sf_k = ceil_div(data_shape[1], config.sf_block[1])
    if config.use_packed_ue8m0:
        num_sf_k_packed = ceil_div(num_sf_k, 4)
        sf_uint8 = sf.detach().cpu().contiguous().view(torch.uint8)
        if already_internal_layout:
            sf_uint8 = sf_uint8[:num_sf_k_packed, : num_sf_m * 4]
            sf_uint8 = sf_uint8.reshape(num_sf_k_packed, num_sf_m, 4).permute(1, 0, 2)
        else:
            sf_uint8 = sf_uint8.reshape(num_sf_m, num_sf_k_packed, 4)
        sf_uint8 = sf_uint8.reshape(num_sf_m, num_sf_k_packed * 4)[:, :num_sf_k]
        sf_cpu = (sf_uint8.to(torch.int32) << 23).view(torch.float32)
        return sf_cpu.to(device=sf.device)
    if config.use_tma_aligned_col_major_sf and already_internal_layout:
        sf = sf.T
    return sf[:num_sf_m, :num_sf_k].to(torch.float32)


def cast_back_ref(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    block_size: tuple[int, int],
    in_config: CastInputConfig | None = None,
    already_internal_layout: bool = False,
) -> torch.Tensor:
    if not isinstance(x, tuple):
        return x.to(torch.float32)

    x_data, x_sf = x
    if in_config is None:
        in_config = CastInputConfig(torch_dtype=x_data.dtype, sf_block=block_size, with_sf=True)
    x_sf = transform_sf_for_ref(x_sf, in_config, x_data.shape, already_internal_layout)
    x_sf = x_sf.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=1)
    x_sf = x_sf[: x_data.shape[0], : x_data.shape[1]]
    return x_data.to(torch.float32) * x_sf


def generate_input_scaling_factors(
    shape: tuple[int, int],
    config: CastInputConfig,
    device: torch.device,
) -> torch.Tensor:
    sf_shape = get_sf_shape(shape, config)
    if config.use_packed_ue8m0:
        sf_uint8 = torch.full(sf_shape, 127, dtype=torch.uint8, device=device)
        sf = sf_uint8.view(torch.int32)
    else:
        sf = torch.ones(sf_shape, dtype=torch.float32, device=device)
    return sf.T if config.use_tma_aligned_col_major_sf else sf

