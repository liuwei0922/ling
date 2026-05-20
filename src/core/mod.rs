/// Mathematical core: amplitude states, operators, and activation functions.
///
/// This module defines the fundamental types for the amplitude-propagation
/// framework. Everything in `core` is pure math — no training logic,
/// no data generation, just the representation of quantum-like states
/// and their transformations.
pub mod amplitude;
pub mod activation;

pub use amplitude::{AmplitudeState, Operator, OperatorSuperposition};
pub use activation::{Activation, BornRule, SigmoidGate, MagnitudeSoftmax};
