# CLAUDE.md

本项目以 `spec.org` 为规范标准，目标是实验“特征集合 T”和“状态集合 S”之间的映射、相似性诱导与复合规则学习。

## 当前结构

```text
src/
├── lib.rs
├── feature.rs
├── feature/
│   └── space.rs      # Feature, Neighborhood, SimilaritySpace
├── state.rs
├── state/
│   ├── link.rs       # Type1Link, Type2Link
│   └── space.rs      # State, StateSpace
├── mapping.rs
├── mapping/
│   ├── compose.rs    # CompositionExample, CompositionLearner
│   └── output.rs     # OutputMapping, StateOutputDistribution
└── probability.rs    # AmplitudeDistribution, Born normalization, selection
examples/
├── composition_transfer.rs
└── fixed_composition_transfer.rs
tests/
└── structure_tests.rs
```

模块不使用新的 `mod.rs`。使用 `feature.rs + feature/...`、`state.rs + state/...`、`mapping.rs + mapping/...` 的布局。

## 设计原则

- `T` 是特征集合，承载外部输入、输出及可观察特征。
- `T` 上的相似空间由 `Neighborhood` 构成。
- `Neighborhood` 只保存成员；置信度保存在 `Feature` 对邻域的隶属关系上。
- `S` 是状态集合，保持干净，不写人工语义 tag。
- `State` 通过 `support_features` 关联到 `T`，状态相似性由支持特征在 `T` 上的相似性诱导。
- `S -> S` 中元素到元素的连线是 1 型连线 `Type1Link`。
- `S -> S` 中状态子集到状态子集的映射是 2 型连线 `Type2Link`。
- `Type2Link` 内部由 `Type1Link` 组成，每条 1 型连线带一个系数。
- 复合规则学习放在 `mapping` 模块中，第一版只实现最小可验证的单目标迁移。
- 三类映射统一使用实数振幅相干叠加、Born 归一化和 argmax 选择函数。

## 实验入口

训练、模拟和实验入口放在 `examples/` 中，不使用 `src/main.rs`。

当前 example：

```bash
cargo run --example composition_transfer
cargo run --example fixed_composition_transfer
```

## 检查命令

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

## 代码规范

- 使用 Rust 2021。
- 不使用 `println!` / `eprintln!`；实验输出使用 `log` + `env_logger`。
- 公共 API 写 `///` 文档注释。
- 优先使用 newtype ID，例如 `FeatureId`、`NeighborhoodId`、`StateId`、`LinkId`。
- 不在核心状态结构里写语义 tag。
- 先保证结构清晰和可测试，再引入训练机制。
