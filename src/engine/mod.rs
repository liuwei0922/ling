/// Training and optimization engine.
///
/// Provides optimizers for the amplitude-based models. The parameter
/// spaces in the toy are small enough that gradient-free methods
/// (finite-difference, evolutionary strategies) are viable.
pub mod trainer;

pub use trainer::{Trainer, EvolutionaryOptimizer};
