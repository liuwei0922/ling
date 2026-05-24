//! Mapping-composition primitives.

pub mod compose;
pub mod output;

pub use compose::{CompositionExample, CompositionLearner};
pub use output::{FeatureWeight, OutputMapping, StateOutputDistribution};
