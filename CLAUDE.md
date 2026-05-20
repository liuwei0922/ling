# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ling is a Rust toy model exploring language and thinking through amplitude propagation and conceptual operator learning. The core idea: instead of learning direct input→output mappings, learn **operators** (transformations in a conceptual space) and how commands superpose them coherently (quantum-style).

## Build & Test Commands

```bash
cargo build                  # Build library
cargo build --example toy_directions  # Build toy example
cargo run --example toy_directions    # Run toy experiment
cargo test                   # Run all tests
cargo test --test core_tests # Run integration tests only
cargo doc --open             # Build and open docs
```

## Project Structure

```
src/
├── lib.rs              # Public API: re-exports core, concept, engine, toy
├── core/               # Math foundation — amplitude states, operators, activation
│   ├── amplitude.rs    # AmplitudeState, Operator, OperatorSuperposition
│   └── activation.rs   # Activation trait + implementations (BornRule, SigmoidGate, MagnitudeSoftmax)
├── concept/            # Dynamic concept space with pruning
│   └── space.rs        # ConceptSpace, Label, PruningConfig (frequency/overlap/entropy pruning)
├── engine/             # Training — gradient-free optimizers for small param spaces
│   └── trainer.rs      # Trainer (finite-difference), EvolutionaryOptimizer
└── toy/                # 4-direction toy experiment
    ├── data.rs         # Dataset: commands with train/test split, vocab encoding
    └── agent.rs        # ToyAgent: amplitude-propagation model, LossEvaluator
examples/
└── toy_directions.rs   # Runnable experiment script
tests/
└── core_tests.rs       # Core math tests
```

## Current Toy Results (Baseline)

The 4-direction experiment shows the model can generalize seen concepts to novel phrasings, but only when the underlying concept (e.g., direction "北") appears frequently across diverse patterns. Concepts with sparse coverage fail to transfer.

- **Train accuracy**: 98.08% (fits training data)
- **Test accuracy**: 52.08% (vs random 25%)
- "北"/"南" generalize to unseen commands (e.g., "看向北" → 100%); "东"/"西" mostly don't
- Confirms the premise: concept learning requires **frequency + diversity** — mimicking how humans need repeated exposure to foundational concepts before building up hierarchy

## Architecture Principles

- **core/**: Pure math, no training logic. `AmplitudeState` is a complex vector; `Operator` is a complex matrix; `OperatorSuperposition` combines them coherently (interference). Activation traits convert amplitudes to probabilities.
- **concept/**: Dynamic label sets. Labels are over-generated then pruned by frequency, overlap, and entropy.
- **engine/**: Optimizers work on flat `Vec<f64>` parameter slices. Toy models expose `flatten_params()`/`restore_params()` for this.
- **toy/**: Concrete experiments. `ToyAgent` uses feature vectors → operator coefficients (tanh-activated) → coherent superposition → Born rule output.

## Key Design Decisions

- Operators store flat as `Vec<f64>` for easy optimization access
- LossEvaluator uses RefCell to avoid cloning on every loss evaluation
- Activation trait allows switching measurement regime (Born/sigmoid/softmax)
- All activation types derive Clone for use in models
