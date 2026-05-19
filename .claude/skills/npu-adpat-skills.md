# NPU Adapter

为TileKernels项目中的kernel文件进行Tilelang NPU适配，将GPU版本的代码转换为NPU兼容版本。

## 触发条件

当用户需要将TileKernels项目中的测试文件或kernel文件从GPU版本适配到NPU版本时使用此技能。

## 技能描述

本技能专门用于TileKernels项目的NPU适配工作，能够：
1. 分析测试文件的依赖关系，确定需要适配的kernel文件
2. 应用12个NPU适配规则对kernel文件和测试文件进行修改
3. 提供运行环境配置建议

## NPU适配规则

### 规则 1: 注释不支持的 GPU 配置项
 - **问题**: NPU 不支持某些 GPU 特定的编译选项，会导致 `AttributeError`。
 - **修改**: 将 `@tilelang.jit` 装饰器的 `pass_configs` 中不支持的配置项注释掉，只保留空字典 `{}`。
 - **不支持的配置项**:
     - `TL_DISABLE_WARP_SPECIALIZED`
     - `TL_DISABLE_THREAD_STORAGE_SYNC`
     - `TL_DISABLE_OUT_OF_BOUND_WARNING`
     - `TL_DISABLE_WGMMA`
     - `TL_PTXAS_REGISTER_USAGE_LEVEL`
     - `TL_DISABLE_VECTORIZE_256`

### 规则 2: dtype 类型适配（`torch.dtype` + 字符串转换）
 - **问题**: NPU 环境的 `tilelang.language` 没有 `dtype` 属性，且底层 TVM API（如 `T.Tensor`、`T.alloc_ub`）不接受 `torch.dtype` 对象，必须使用字符串形式的 dtype（如 `"float32"`）。
 - **修改**:
     - 函数参数中的 `T.dtype` 全部替换为 `torch.dtype`，默认值从 `T.float32` 改为 `torch.float32`。
     - 在 kernel 函数体内部，通过 `dtype_str = str(dtype).replace('torch.', '')` 将 `torch.dtype` 转换为字符串。
     - 所有出现 dtype 的地方（`T.Tensor` 类型注解、`T.alloc_ub`、`T.alloc_buffer` 等）均使用 `dtype_str`。

### 规则3: 移除T.Ref声明
 - **问题**: NPU环境不支持 `T.Ref` 声明
 - **修改**: 删除所有的 `T.Ref` 声明

### 规则4: 替换T.LocalBuffer为T.Buffer
 - **问题**: NPU环境使用 `T.Buffer` 替代 `T.LocalBuffer`
 - **修改**: 将所有 `T.LocalBuffer` 调用替换为 `T.Buffer`


### 规则 5: 动态形状变量声明（`T.dynamic` → `T.symbolic`）
 - **问题**: NPU 环境的 `tilelang.language` 没有 `dynamic` 属性，必须使用 `T.symbolic` 声明动态维度
 - **修改**: 将所有 `T.dynamic('var_name')` 替换为 `T.symbolic('var_name')`

 - **示例**:
    ```python
    # 原代码
    num_tokens = T.dynamic('num_tokens')

    # NPU 适配后
    num_tokens = T.symbolic('num_tokens')
    ```

### 规则6: 替换T.StridedTensor为T.Tensor
 - **问题**: NPU环境不支持 `T.StridedTensor`
 - **修改**: 将所有 `T.StridedTensor` 类型声明替换为 `T.Tensor`

### 规则7: 替换T.clear为T.tile.fill
 - **问题**: NPU环境使用 `T.tile.fill` 替代 `T.clear`
 - **修改**: 将所有 `T.clear` 调用替换为 `T.tile.fill(buffer, 0.0)`

### 规则8: 修复T.reduce_sum接口
 - **问题**: NPU环境的 `T.reduce_sum` 需要显式指定 `dim` 参数
 - *修改**: 为 `T.reduce_sum` 调用添加 `dim` 参数

### 规则9: 替换T.fill为T.tile.fill
 - **问题**: NPU环境使用 `T.tile.fill` 替代 `T.fill`
 - **修改**: 将所有 `T.fill` 调用替换为 `T.tile.fill(buffer, 0.0)`

### 规则10: 测试文件NPU适配
 - **问题**: 测试文件需要将默认设备从 'cuda' 改为 'npu'
 - **修改**: 将测试数据生成函数的默认设备参数从 `'cuda'` 改为 `'npu'`，例如 `device: str = 'npu'`

### 规则 11: 替换T.alloc_fragment为T.alloc_ub 片内内存分配
 - **问题**: NPU 环境不支持 GPU 的 `T.alloc_fragment`，需使用 NPU 的统一缓冲区分配器 `T.alloc_ub`。
 - **修改**:
   - 将所有 `T.alloc_fragment(shape, dtype)` 替换为 `T.alloc_ub(shape, dtype_str)`，其中 `dtype_str` 为规则 2 转换后的字符串 dtype。
   - 将所有参与计算的数据（包括原本在全局内存中的张量）先通过 T.copy 复制到 UB 中，之后再参与算术运算。
   - 禁止在 T.Parallel 循环内直接使用全局内存张量的下标（如 mhc_scale[0]），必须使用 UB 中的副本。
 - **示例**:
    ```python
    # 原代码
    frag = T.alloc_fragment((block_size, dim), dtype)
    # NPU 适配后
    dtype_str = str(dtype).replace('torch.', '')
    frag = T.alloc_ub((block_size, dim), dtype_str)

    input_ub = T.alloc_ub((block_size, dim), dtype_str)
    base_ub  = T.alloc_ub((mhc_mult3,), dtype_str)
    T.copy(input_mixes[pid * block_size, 0], input_ub)
    T.copy(mhc_base, base_ub)
    # 之后全部使用 input_ub, base_ub 进行计算
    ```

### 规则 12: 移除T.annotate_layout布局优化
 - **问题**: `T.annotate_layout` 是 GPU 优化布局，NPU 不需要此优化。
 - **修改**: 将前向、反向两个 kernel 中的 `T.annotate_layout(...)` 代码块全部删除（或注释）。


### 规则 13: NPU 环境适配（避免 torch.cuda，获取设备核心数）
 - **问题**: 原代码通过 torch.cuda.get_device_properties 获取 SM 数量，NPU 环境无 CUDA。
 - **修改**: 将前向、反向两个 kernel 中的 `T.annotate_layout(...)` 代码块全部删除（或注释）。
   - 实现一个 NPU 兼容的核心数获取函数，优先使用 torch_npu，降级为环境变量或默认值。
   - 在需要 num_sms 的地方（如反向 kernel 的 partial grad 分配）使用该函数。
   - **示例**:
      ```python
      def _get_npu_num_sms(default=80):
          try:
              import torch_npu
              return torch_npu.npu.get_device_properties(0).multi_processor_count
          except Exception:
              return int(os.environ.get("ASCEND_NUM_SMS", default))
      ```


## 工作流程

1. **依赖关系分析**
   - 查看测试文件导入的函数
   - 追踪函数的实现位置
   - 确定需要适配的kernel文件和测试文件

2. **应用NPU适配规则**
   - 对kernel文件应用规则1-9、规则11和规则12
   - 对测试文件应用规则10（将默认设备从 'cuda' 改为 'npu'）
   - 根据实际文件内容选择需要的规则
   - 在修改处添加注释说明是NPU适配

3. **运行环境配置**
   - 提供PYTHONPATH配置建议
   - 避免影响其他项目的方法

## 使用示例

### 通用适配场景
用户说："请为TileKernels项目进行NPU适配"
或
用户说："适配这个kernel文件到NPU环境"

### 特定测试文件适配场景
用户说："请为 test_pre_split_mixes.py 进行NPU适配"
或
用户说："这个测试文件在NPU环境下运行失败，需要适配"

## 注意事项

1. **精准适配**: 只根据依赖关系分析确定需要适配的文件，不要过度适配
2. **规则选择**: 不是每个文件都需要应用所有12个规则，根据实际内容选择
   - kernel文件应用规则1-9、规则11和规则12
   - 测试文件应用规则10（将默认设备从 'cuda' 改为 'npu'）
3. **注释标记**: 在修改处添加 `# NPU适配` 注释，便于追踪
4. **环境配置**: 提供PYTHONPATH配置建议，确保使用修改后的源代码

## 技能限制

- 专门针对TileKernels项目的NPU适配工作
- 不适用于其他项目的NPU适配
- 需要用户提供具体的测试文件或kernel文件路径

## 输出格式

1. 依赖关系分析结果
2. 需要适配的文件列表
3. 每个文件的适配详情（应用的规则和修改内容）
4. 运行环境配置建议