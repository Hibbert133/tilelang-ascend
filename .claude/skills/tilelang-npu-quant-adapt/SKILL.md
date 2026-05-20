---
name: tilelang-npu-quant-adapt
description: "TileLang-Ascend NPU quant/cast 算子适配工作流。用于把 TileKernels/GPU quant 算子、scale-factor 布局、cast/dequant/cast_back 逻辑迁移到 Ascend NPU，尤其是 BF16 输入、FP32 输出、per-block scaling、packed UE8M0、TMA-aligned col-major scale factor、无运行环境纯代码开发。触发：用户提到 quant、cast、per_block_cast_lossless、scale factor、UE8M0、TMA layout、BF16→FP32、对齐 GPU 测试或 TileKernels quant 实现时，应优先使用本 skill。"
---

# TileLang-Ascend NPU Quant/Cast 算子适配

本 skill 用于把 TileKernels/GPU quant 算子适配到 TileLang-Ascend NPU。重点不是逐行照搬 GPU kernel，而是对齐 GPU 的接口、测试参数、scale-factor 语义和 correctness 逻辑，同时按 Ascend 当前稳定能力重写 kernel。

典型任务：

- 适配 `examples/quant/*cast*kernel.py`
- 对齐 TileKernels 中 `tile_kernels/quant/*.py`
- 对齐 `tile_kernels/torch/cast.py` 的 `cast_back` / dequant 语义
- 处理 `BaseCastConfig`、`CastInputConfig`、`CastOutputConfig`
- 处理 per-block scale factor、packed UE8M0、TMA-aligned col-major scale-factor layout
- 在没有 NPU/CANN/torch-npu 运行环境时做纯代码开发和静态核对

---

## 1. 先确认适配范围

开始写代码前，明确本次 NPU 支持范围。不要默认 GPU 的 dtype 和完整量化路径都可用。

### 1.1 本阶段常见安全目标

对于 `per_block_cast_lossless` 这类算子，当前推荐最小稳定目标：

| 项 | 推荐约束 |
|---|---|
| 输入 tensor dtype | `torch.bfloat16` |
| kernel 输入 dtype | `"bfloat16"` |
| Python API 输出 dtype | `torch.float32` |
| `fmt` | 只支持 `"fp32"` / `"float32"` |
| kernel 内部输出 raw dtype | 优先 `"bfloat16"`，必要时 wrapper 再 cast_back 到 fp32 |
| scale-factor 复杂逻辑 | 优先 Python 侧处理，先保证可运行 |

不要在 Ascend NPU 代码中直接承诺支持 GPU 的：

- `T.float4_e2m1fn`
- `T.float8_e4m3fn`
- `torch.float8_e4m3fn` kernel 路径
- FP4 E2M1 packing/unpacking device kernel
- CUDA-specific pass configs
- GPU 的 `threads=128/256`
- 三维 `T.Kernel`

### 1.2 用户要求优先级

当用户说“对齐 GPU 实现”时，优先对齐这些层面：

1. Python wrapper 入参名称和默认值
2. 测试参数生成方式
3. config/dataclass 结构
4. scale-factor shape/layout 语义
5. correctness reference，例如 GPU 的 `torch.cast_back(...)`
6. block 推导和 scale block 聚合公式

不要把“对齐 GPU”理解成把 GPU kernel API、dtype、线程模型逐行搬到 NPU。

---

## 2. 必读参考文件

根据任务位置读取相关文件。常见参考路径如下：

| 文件 | 用途 |
|---|---|
| `Tilekernel/TileKernels/tile_kernels/quant/common.py` | GPU quant config、SF shape、alloc、epilogue、load/store 语义 |
| `Tilekernel/TileKernels/tile_kernels/torch/cast.py` | PyTorch reference，尤其是 `cast_back` correctness 语义 |
| `Tilekernel/TileKernels/tile_kernels/quant/per_block_cast_lossless_kernel.py` | GPU kernel 的 block 推导、SF 聚合逻辑 |
| `Tilekernel/TileKernels/tests/quant/test_per_block_cast_lossless.py` | 测试入参矩阵、数据集生成、正确性比较方式 |
| `tilelang-ascend/docs/TileLang-Ascend Programming Guide.md` | Ascend dtype、T.Kernel、T.copy、UB、Scope 等语法 |
| `.claude/skills/npu-adpat-skills.md` | GPU→NPU 通用适配规则 |
| `.claude/skills/tilelang-custom-skill/tilelang-api-best-practices/SKILL.md` | Ascend TileLang API 用法 |

如果 GPU reference 和本项目 examples/API 文档冲突，以 Ascend 当前 examples 和 API 文档为准。

---

## 3. NPU dtype 和 API 规则

### 3.1 kernel 内 dtype 用字符串

Ascend kernel 里不要用 GPU 风格 `T.dtype` 或 torch dtype 对象。使用字符串：

```python
INPUT_DTYPE = "bfloat16"
OUTPUT_DTYPE = "float32"
KERNEL_OUTPUT_DTYPE = "bfloat16"
```

Tensor annotation：

```python
x: T.Tensor([num_tokens, hidden], INPUT_DTYPE)
out: T.Tensor([num_tokens, hidden], KERNEL_OUTPUT_DTYPE)
```

UB allocation：

```python
x_ub = T.alloc_ub((block_m_per_vec, block_k), INPUT_DTYPE)
out_ub = T.alloc_ub((block_m_per_vec, block_k), KERNEL_OUTPUT_DTYPE)
```

### 3.2 动态维度用项目当前兼容写法

本项目 NPU 适配常用：

```python
num_tokens = T.symbolic("num_tokens")
```

不要从 GPU 直接照搬不兼容的 dynamic/strided API。若文档和现有示例不一致，优先保持当前文件附近 examples 已验证的写法。

### 3.3 避免高风险 device-side 路径

这些路径在 NPU quant 适配中容易导致 segfault、编译失败或精度问题，除非已有现成通过示例，否则先不要使用：

- device-side `uint32` bit reinterpret / shift
- `T.reinterpret` 实现 BF16 bit expand 到 FP32
- packed UE8M0 在 kernel 中读写 `uint8/int32` 复杂布局
- 三维 reduce 或复杂 dynamic reduce
- `device_assert`
- `T.Parallel` 内嵌 `T.serial`
- 直接复用 CUDA pass config
- `T.StridedTensor`
- `T.alloc_fragment` 用于 NPU expert-mode UB 计算

---

## 4. 推荐代码组织

### 4.1 使用 quant common 文件

仿照 GPU 的：

```python
from tile_kernels.quant.common import *
```

NPU examples 中建议写成：

```python
try:
    from .common import *
except ImportError:
    from common import *
```

这样既支持包导入，也支持直接运行 example 文件。

### 4.2 `examples/quant/common.py` 应包含的内容

把跨 quant/cast 算子复用的逻辑放入 `common.py`：

```python
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
```

建议放入 common 的函数类别：

| 类别 | 内容 |
|---|---|
| config | `BaseCastConfig`、`CastInputConfig`、`CastOutputConfig` |
| shape/layout | `ceil_div`、`align_up`、`is_power_of_two`、`get_sf_shape` |
| scale-factor allocation | `scale_fill_value`、`alloc_scaling_factors`、`pad_or_create_scaling_factors`、`cast_epilogue` |
| scale-factor access | `load_sf`、`store_sf` |
| exponent conversion | `sf_to_exponent_value`、`exponent_to_sf_value` |
| reference/test | `transform_sf_for_ref`、`cast_back_ref`、`generate_input_scaling_factors` |

保留在具体算子文件中的内容：

- 该算子的 layout 推导，如 `_derive_cast_layout`
- 该算子的 output SF 聚合，如 `_compute_output_scaling_factors`
- JIT kernel factory
- public wrapper
- test parameter generator

如果某个 helper 已被第二个 quant 算子复用，再考虑从算子文件上移到 common。

---

## 5. Config 和 scale-factor layout 语义

### 5.1 Base config

NPU 侧 config 的 `dtype` 返回字符串，不能返回 `T.dtype`：

```python
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
```

### 5.2 Output config 限制 fmt

当前 BF16→FP32 适配中：

```python
def get_cast_output_config(fmt: str, ...):
    assert fmt in ("fp32", "float32")
    return CastOutputConfig(torch_dtype=torch.float32, ...)
```

不要保留 GPU 的：

```python
assert fmt in ("e5m6", "e4m3", "e2m1")
```

除非你已经实现并验证了对应 NPU dtype 路径。

### 5.3 `get_sf_shape` 对齐 GPU

```python
def get_sf_shape(shape: tuple[int, int], config: BaseCastConfig) -> tuple[int, int]:
    num_block_m = ceil_div(shape[0], config.sf_block[0])
    num_block_k = ceil_div(shape[1], config.sf_block[1])
    if config.use_packed_ue8m0:
        num_block_m *= 4
        num_block_k = ceil_div(num_block_k, 4)
    return (num_block_k, num_block_m) if config.use_tma_aligned_col_major_sf else (num_block_m, num_block_k)
```

这个 shape 是物理存储 shape，不是逻辑 `[num_sf_m, num_sf_k]`。后续 `cast_back_ref` 需要还原逻辑 shape。

### 5.4 packed UE8M0 与 TMA

GPU 参考中 packed UE8M0 通常和 TMA col-major 配套出现。NPU 侧如果支持 packed，优先要求：

```python
if out_config.use_packed_ue8m0:
    assert out_config.use_tma_aligned_col_major_sf
```

原因：packed UE8M0 的物理布局把 4 个 exponent byte 打进 `int32` view，TMA layout 又会对外呈现转置后的 layout。混用时很容易 shape 正确但语义错。

---

## 6. `cast_back_ref` 是 correctness 的核心

Quant 算子测试通常不是比较 raw tensor：

```python
out == x.to(torch.float32)
```

而是比较 dequant/cast-back 后的值：

```python
cast_back(output_quant, output_sf, out_block) == cast_back(input_quant, input_sf, in_block)
```

GPU reference 常见模式：

```python
x_fp8_fp32_ref = tile_kernels.torch.cast_back(x_fp4, "fp32", in_block)
x_fp8_fp32 = tile_kernels.torch.cast_back(x_fp8, "fp32", out_block)
```

NPU BF16/FP32 适配可用：

```python
out = cast_back_ref((out_raw, x_sf_ref), in_config.sf_block, in_config)
ref = cast_back_ref((x_data, x_sf), in_config.sf_block, in_config)
torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)
```

`cast_back_ref` 要处理：

1. 普通 float32 SF
2. TMA external layout
3. packed UE8M0 `int32` external layout
4. wrapper 内部已转置/已 view 成 `uint8` 的 internal layout

推荐签名：

```python
def cast_back_ref(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    block_size: tuple[int, int],
    in_config: CastInputConfig | None = None,
    already_internal_layout: bool = False,
) -> torch.Tensor:
    ...
```

默认 `already_internal_layout=False`，即像 GPU `torch.cast_back` 一样处理用户传入的外部 SF。

---

## 7. per-block lossless 的推荐实现策略

### 7.1 对齐 GPU 的 block 推导

```python
NUM_ELEMENTS_PER_BLOCK = 8192
block_m = max(out_config.sf_block[0], 32)
block_k = max(out_config.sf_block[1], NUM_ELEMENTS_PER_BLOCK // block_m)
assert block_m % out_config.sf_block[0] == 0
assert block_k % out_config.sf_block[1] == 0
```

并计算：

```python
num_in_sf_per_block_m = block_m // in_config.sf_block[0]
num_in_sf_per_block_k = block_k // in_config.sf_block[1]
num_out_sf_per_block_m = block_m // out_config.sf_block[0]
num_out_sf_per_block_k = block_k // out_config.sf_block[1]
num_in_sf_per_out_sf_m = out_config.sf_block[0] // in_config.sf_block[0]
num_in_sf_per_out_sf_k = out_config.sf_block[1] // in_config.sf_block[1]
num_in_sf_per_out_sf = num_in_sf_per_out_sf_m * num_in_sf_per_out_sf_k
```

### 7.2 非整除 shape 用 host padding + crop

GPU tests 往往包含 `4001`、`8001` 等非整除 token 数。NPU `T.copy` tail mask 支持不明确时，不要让 kernel 处理越界 tail。

推荐：

```python
padded_tokens = align_up(num_tokens, block_m)
padded_hidden = align_up(hidden, block_k)

if padded_tokens != num_tokens or padded_hidden != hidden:
    x_padded = torch.zeros((padded_tokens, padded_hidden), dtype=x_data.dtype, device=x_data.device)
    x_padded[:num_tokens, :hidden] = x_data
else:
    x_padded = x_data
```

kernel 只处理整 tile。kernel 后：

```python
out_raw = out_raw[:num_tokens, :hidden]
out_sf = cast_epilogue(out_sf, num_tokens, hidden, out_config)
```

### 7.3 Kernel 先保持稳定

如果 BF16→FP32 device-side cast 不稳定，采用 raw BF16 kernel 输出：

```python
KERNEL_OUTPUT_DTYPE = "bfloat16"
```

kernel 内：

```python
T.copy(x[row_offset, col_offset], x_in_shared)
T.barrier_all()
T.copy(x_in_shared, x_out_fragment)
T.barrier_all()
T.copy(x_out_fragment, out[row_offset, col_offset])
```

wrapper 再：

```python
out = cast_back_ref((out_raw, x_sf_ref), in_config.sf_block, in_config)
```

这样对齐 GPU correctness 语义，同时规避 NPU device cast 的不稳定路径。

### 7.4 output SF 聚合先放 Python 侧

对齐 GPU lossless 的核心聚合：

```python
for each output scale block:
    max_exp = max(input scale exponents covered by this output block)
    out_exp = max(max_exp - 6, 0)
```

Python 侧实现时，把 NPU tensor 先搬 CPU，避免 NPU 上 bit op 触发 ACL/CANN 编译问题：

```python
x_sf_cpu = x_sf.detach().cpu()
out_values = torch.empty(get_sf_shape(shape, out_config), dtype=out_config.sf_torch_dtype)
```

再用 `load_sf` / `store_sf` 和 exponent helper 聚合，最后：

```python
return out_values.to(device=x_sf.device)
```

---

## 8. 测试参数对齐 GPU

如果用户要求对齐 GPU test，保留相同参数矩阵，但实际 dtype/format 受 NPU 当前支持范围限制。

推荐生成：

```python
def generate_num_tokens(is_benchmark: bool = False) -> list[int]:
    _ = is_benchmark
    return [4001, 8001]


def generate_hidden_sizes() -> list[int]:
    return [576, 2048, 2560, 3072, 4096, 6144, 7168]


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
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
```

Tolerance 根据用户要求和 BF16/NPU 实际路径可用：

```python
torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)
```

---

## 9. 已知错误与修复策略

### 9.1 Segmentation fault

常见触发：

- 把复杂 `x_sf/out_sf` tensor 传入 kernel 并做 packed layout 写入
- device-side `uint32`/`uint8` UB 操作
- 三维 reduce 或复杂 nested loop
- `device_assert`

处理：

- kernel 退回稳定 BF16 block copy
- SF 聚合移到 Python 侧
- 先不在 kernel 内处理 packed UE8M0/TMA 写入

### 9.2 ACL/CANN bit-op 编译错误

典型现象：

```text
SetPrecisionMode ... AclSetCompileopt ... error code 500001
Cannot find global function cce.product_init
```

或 NPU tensor 上：

```python
(sf.view(torch.int32) >> 23)
(exp.to(torch.int32) << 23).view(torch.float32)
```

处理：

- `.detach().cpu()` 后做 scalar/CPU bit 逻辑
- 避免 NPU tensor 上直接位运算

### 9.3 `T.Parallel` / `T.serial` 嵌套错误

典型错误：

```text
InternalError: Check failed: (op->kind == ForKind::kParallel) is false
```

处理：

- problematic 路径改为 `T.serial`
- 不在 `T.Parallel` 内嵌 `T.serial`
- 先保持简单静态循环

### 9.4 BF16→FP32 device cast 精度不对

可能尝试过：

- `T.tile.cast`
- scalar loop `T.float32(...)`
- `T.cast(..., "float32")`
- copy-on-cast
- reinterpret bit expansion

若仍 mismatch，采用：

- kernel 输出 BF16 raw
- wrapper `cast_back_ref` 生成 FP32
- reference 也用 `cast_back_ref`

### 9.5 packed/TMA SF shape mismatch

典型错误：

```text
The size of tensor a (4001) must match the size of tensor b (5) at non-singleton dimension 0
```

处理：

- 不要直接把 packed physical shape 当 logical `[num_sf_m, num_sf_k]`
- 用 `data_shape` 和 `sf_block` 计算逻辑 SF shape
- packed 时用 `num_sf_k_packed = ceil_div(num_sf_k, 4)`
- 最终展开为 `[num_sf_m, num_sf_k]` 后再 repeat_interleave

---

## 10. 无运行环境时的工作方式

如果用户明确说“没有运行环境”“纯代码开发”“不要运行”，遵守以下规则：

- 不运行 kernel
- 不运行 pytest
- 不运行 example
- 不运行 py_compile，除非用户后来明确允许
- 不运行格式化命令
- 可以使用 Read/Grep/Glob/Edit/Write 做静态开发
- 结论中明确说明“未运行验证，只做静态检查”

静态核对重点：

1. wrapper 签名是否对齐 GPU
2. dtype 是否都是 NPU 字符串
3. `fmt` 是否限制在当前支持范围
4. block 推导是否和 GPU 一致
5. padding/crop 是否覆盖非整除 shape
6. `get_sf_shape` / `cast_epilogue` 是否符合 packed/TMA layout
7. `cast_back_ref` 是否使用外部 SF 布局作为默认输入
8. `common.py` 的 `__all__` 是否覆盖星号导入需要的符号
9. 是否有旧函数名残留，例如 `_pad_or_create_scaling_factors`
10. 是否意外引入 GPU-only API

---

## 11. 代码修改优先级

处理用户反馈时，按这个顺序迭代：

1. **先修可运行性**：segfault、编译失败、明显 API 不支持优先。
2. **再修 correctness**：检查 reference 是否对齐 GPU cast_back，而不是简单 `.to(float32)`。
3. **再对齐 GPU 结构**：补齐 config、入参、test params、common 抽象。
4. **最后考虑下沉逻辑到 kernel**：只有在用户环境已跑通稳定后，再逐步把 Python-side SF 聚合迁入 kernel。

不要为了“代码量看起来像 GPU”牺牲 NPU 可运行性。

---

## 12. 适配 checklist

提交或回复前逐项检查：

- [ ] 是否读取了 GPU quant common / torch cast reference / GPU test
- [ ] 是否明确当前 NPU 支持 dtype 和 fmt
- [ ] kernel dtype 是否使用字符串
- [ ] 是否避免 `T.float4_e2m1fn` / `T.float8_e4m3fn`
- [ ] 是否避免 CUDA-only pass configs
- [ ] wrapper 入参是否对齐 GPU
- [ ] test params 是否对齐 GPU 数据集
- [ ] block 推导是否对齐 GPU
- [ ] 非整除 shape 是否 host padding + crop
- [ ] scale-factor shape 是否对齐 `get_sf_shape`
- [ ] packed UE8M0 是否正确处理 `int32` view `uint8`
- [ ] TMA layout 是否区分外部 layout 和内部 layout
- [ ] reference 是否用 `cast_back_ref`
- [ ] 是否避免 NPU tensor 上危险 bit op
- [ ] 是否避免复杂 device-side SF reduce/write
- [ ] `common.py` 是否有 `__all__`
- [ ] 星号导入是否保留 relative/direct fallback
- [ ] 无运行环境时是否未运行命令并如实说明

---

## 13. 推荐完成报告格式

简短说明改动，避免声称未验证内容已通过。

```markdown
已完成 NPU quant/cast 适配整理：

- 对齐 GPU wrapper 入参和测试参数生成。
- 将通用 config、SF shape/layout、cast_back_ref 等移动到 `examples/quant/common.py`。
- kernel 内保留稳定 BF16 block copy，FP32 correctness 通过 wrapper 侧 `cast_back_ref` 对齐 GPU `torch.cast_back` 语义。
- packed UE8M0/TMA scale factor 默认按外部布局处理，并保留内部布局兼容参数。

未运行 NPU/CANN 测试；本次只做纯代码静态修改。
```

---

## 14. 核心经验

TileLang-Ascend NPU quant 适配的关键是：

> 对齐 GPU 的接口、测试矩阵、scale-factor 语义和 correctness reference；kernel 实现则按 Ascend 当前稳定 API 重写。

Quant/cast 算子的正确性往往来自：

```text
raw tensor + scale factor + cast_back/dequant
```

而不是 raw tensor 本身。

因此遇到精度 mismatch 时，优先检查 reference 是否应使用 `cast_back_ref`，再检查 scale-factor layout，而不是立刻改 kernel cast 指令。
