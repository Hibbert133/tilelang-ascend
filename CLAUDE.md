# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

TileLang-Ascend is a TileLang variant for Huawei Ascend NPUs. It provides a Python DSL (`@tilelang.jit`, `tilelang.language as T`) on top of TVM, lowers TileLang/TIR through Python and C++ passes, generates Ascend C or PTO code, and runs kernels through the CANN/NPU runtime.

Primary compilation flow:

```text
Python DSL (@tilelang.jit)
  -> tilelang/engine/lower.py
  -> tilelang/engine/phase.py + tilelang/transform/ + src/transform/
  -> src/target/codegen_ascend_pto.cc or src/target/codegen_ascend.cc
  -> src/tl_templates/ + runtime module
  -> CANN toolchain / NPU execution
```

This branch focuses on the Ascend C + PTO route. `tilelang/engine/lower.py` dispatches `target.model == "pto"` to `target.build.tilelang_ascend_pto` and `target.model == "ascendc"`/`auto` to `target.build.tilelang_ascend`.

## Repository skills

Project-specific Claude skills live under `.claude/skills/`. Before doing specialized TileLang-Ascend work, check the relevant skill `SKILL.md` instead of relying on memory:

- `.claude/skills/tilelang-custom-skill/tilelang-api-best-practices/` for TileLang Ascend API usage while writing or debugging kernels.
- `.claude/skills/tilelang-op-design/` when designing a new operator or generating a `design.md`.
- `.claude/skills/tilelang-op-generate/` when implementing an operator from a design.
- `.claude/skills/tilelang-pass-analyzer/` and `.claude/skills/tilelang-pass-workflow-analyzer/` for pass functionality, ordering, and dependency questions.
- `.claude/skills/tilelang-pass-design/` when designing a new compiler pass.
- `.claude/skills/tilelang-perf-optimization/` for TileLang-Ascend performance tuning.
- `.claude/skills/npu-adpat-skills.md` for adapting TileKernels GPU kernels/tests to NPU-compatible TileLang.

## Environment and build commands

TileLang-Ascend supports Linux/WSL-style build environments. The Ascend scripts require Python >= 3.10 and a working Ascend stack, with CANN sourced and `ASCEND_HOME_PATH` set.

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# Full source build for Ascend.
bash install_ascend.sh

# Optional modes.
bash install_ascend.sh --enable-llvm
bash install_ascend.sh --enable-shmem
bash install_ascend.sh --enable-incremental

# Build and install an Ascend wheel.
./build_wheel_ascend.sh
./build_wheel_ascend.sh --enable-llvm
pip install dist/tilelang-*.whl

# Load local runtime/library paths after source build.
source set_env.sh
```

`install_ascend.sh` installs build/runtime requirements, initializes submodules, configures CMake with `USE_ASCEND ON`, and builds under `build/`. `build_wheel_ascend.sh` exports `USE_ASCEND=true`, refreshes requirements/submodules, cleans prior build artifacts, and runs `python setup.py bdist_wheel`.

## Tests

```bash
# Run all Python tests used by CI.
export TILELANG_CLEAR_CACHE=1
python -m pytest testing/python

# Run one test file.
python -m pytest -q testing/python/language/test_tilelang_ascend_language_parallel.py

# Run one test node.
python -m pytest -q testing/python/language/test_tilelang_ascend_language_parallel.py::test_name

# Run example tests as CI does.
cd examples && python -m pytest '**/test*.py'

# Run a single example manually.
python examples/gemm/example_gemm.py
```

Many tests/examples require CANN, torch-npu, built TileLang libraries, and NPU hardware/runtime. If any layer is unavailable, state exactly what was not validated.

## Formatting and linting

```bash
pip install -r requirements-lint.txt

# Format/lint changed Python and C/C++ files relative to the base branch.
bash format.sh

# Format/lint specific files.
bash format.sh --files path/to/file.py path/to/file.cc

# Format/lint all eligible files.
bash format.sh --all

# PR-style checks for changed files.
ruff format --check <changed-python-files>
ruff check <changed-python-files>
clang-format --dry-run --Werror <changed-cpp-files>
```

`format.sh` expects the versions pinned in `requirements-lint.txt` for yapf, ruff, codespell, and clang-format when available. `.github/workflows/ci_ascend.yml` checks changed Python files with Ruff and changed C/C++ files with clang-format.

## High-level architecture

### Python frontend

- `tilelang/jit/` wraps TIR functions as callable kernels. `tilelang/jit/kernel.py` compiles a `PrimFunc`, creates a `cython`, `ctypes`, or `dlpack` adapter, handles output/workspace indices, and exposes profiling helpers.
- `tilelang/engine/` orchestrates lowering. `lower.py` converts a `PrimFunc` to an `IRModule`, applies lower/legalize and target optimization phases, then invokes backend codegen.
- `tilelang/language/` defines the DSL surface used in kernels. Key Ascend files are `ascend.py` for synchronization, barriers, scopes, and Ascend intrinsics; `ascend_tile.py` for tile-level vector/copy/math operations; plus `allocate.py`, `copy.py`, `gemm.py`, `parallel.py`, `pipeline.py`, and reduction modules.
- `tilelang/transform/pass_config.py` defines pass config keys, including Ascend automation switches: `TL_ASCEND_AUTO_SYNC`, `TL_ASCEND_MEMORY_PLANNING`, `TL_ASCEND_AUTO_CV_COMBINE`, and `TL_ASCEND_AUTO_CV_SYNC`.
- `tilelang/carver/`, `autotuner/`, `profiler/`, and `cache/` support scheduling/resource analysis, tuning, profiling, and compiled-kernel reuse.

### C++ backend

- `src/transform/` contains the main IR transformations: parallel-to-vector lowering, sync insertion, memory planning, CV combine/sync, storage rewrite, workspace reduction, buffer shape collection, and PTO buffer-shape saving.
- `src/target/` contains code generators and runtime modules. For this branch, `codegen_ascend_pto.cc` and `rt_mod_ascend_pto.cc` are the PTO path; `codegen_ascend.cc` and `rt_mod_ascend.cc` are the Ascend C path.
- `src/op/` registers/lowers TileLang operations such as copy, GEMM, elementwise, reduce, logical, parallel, and Ascend-specific ops.
- `src/layout/` implements layout/swizzle utilities used by codegen and optimization.
- `src/tl_templates/` holds backend C/C++ template headers for generated code. Ascend/PTO templates live under `src/tl_templates/ascend/` and `src/tl_templates/pto/`.

### Examples and tests

- `examples/` contains operator examples and benchmark-style scripts for GEMM, attention, normalization, activation, convolution, dispatch/combine, torch integration, ACLGraph, and other kernels.
- `testing/python/language/` contains focused language/API tests for Ascend features. Add coverage here when extending `tilelang/language/` or backend lowering.

## Ascend programming model notes

- Ascend kernels use a strict memory hierarchy: GM -> L1/UB -> L0A/L0B -> L0C. Use `T.copy` for movement and primitives such as `T.gemm`, `T.mma`, `T.Parallel`, and `T.tile.*` for compute.
- Developer mode relies on automatic compiler behavior (`T.Parallel`, automatic scope separation/sync/memory planning via pass configs). Expert mode uses explicit `T.Scope("C")` / `T.Scope("V")`, explicit `T.alloc_L1/ub/L0A/L0B/L0C`, and manual synchronization (`T.set_flag`, `T.wait_flag`, `T.set_cross_flag`, `T.wait_cross_flag`, barriers).
- Before writing or changing kernels, start from the closest existing example in `examples/`, check `.claude/skills/tilelang-custom-skill/tilelang-api-best-practices/`, and verify API signatures in `tilelang/language/ascend.py`, `tilelang/language/ascend_tile.py`, and `testing/python/language/`.
- Repository guidance marks `tilelang/language/pto.py` as deprecated; prefer current Ascend APIs unless maintaining existing PTO-specific code.

## When changing language primitives or passes

Language/API changes usually require coordinated Python and C++ updates:

1. Python DSL definition in `tilelang/language/`.
2. Python pass config or lowering hooks in `tilelang/transform/` or `tilelang/engine/` if applicable.
3. C++ op registration/lowering in `src/op/` or transform passes in `src/transform/`.
4. Backend codegen in `src/target/codegen_ascend_pto.cc` or `src/target/codegen_ascend.cc` when emitted code changes.
5. Template/runtime support under `src/tl_templates/` or `src/target/rt_mod_*` when generated code depends on helper definitions.
6. Focused tests under `testing/python/language/` and, when useful, an executable example under a dedicated `examples/<operator>/` directory.

For new operator examples, repository guidance expects a dedicated `examples/<operator_name>/` directory rather than new top-level `.py` files directly under `examples/` or mixing unrelated operators into existing example directories.

## Debugging generated kernels

- Use `T.printf` and `T.dump_tensor` in kernels for device-side debugging.
- Inspect generated Ascend C/PTO code under build/cache outputs when lowering or codegen behavior is unclear.
- Recent history added `TL_PTO_DEBUG` for PTO tensor-dump debug printing; grep current code before relying on exact behavior.
