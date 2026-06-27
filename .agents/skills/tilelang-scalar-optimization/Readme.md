# TileLang Scalar Optimization Skill

## Overview

`tilelang-scalar-optimization` is a TileLang-Ascend optimization skill for reducing scalar-heavy hot paths in Ascend vector-core kernels.

It is intended for kernels that are already functionally correct, but profiling or generated Ascend C shows excessive scalar work, such as:

- `T.serial` loops that copy one row, column, or scalar at a time.
- Per-row scalar `T.alloc_var`, `if`, `T.max`, `T.min`, or reduce chains.
- Serial gather/index remap from UB before vector computation.
- Manual column expansion before `cast`, `mul`, or `store`.
- Fixed-layout or fixed-shape cases that can be specialized with a safe fast path.

The main goal is to convert regular runtime scalar work into `T.tile` operations, such as:

- `T.tile.gather`
- `T.tile.broadcast`
- `T.tile.max`
- `T.tile.add`
- `T.tile.sub`
- `T.tile.mul`
- `T.tile.bitwise_rshift`
- `T.tile.bitwise_lshift`

This skill should be used together with historical successful commits and precision-passing evidence. Do not treat dirty worktree experiments, failed precision attempts, or segfaulting versions as reusable optimization patterns.

---

## Directory Layout

```text
.agents/skills/tilelang-scalar-optimization/
├── SKILL.md
├── README.md
├── agents/
│   └── openai.yaml
└── references/
    ├── scalar-patterns.md
    └── per-block-cast-evidence.md
