/// Toy experiments for the amplitude-propagation framework.
///
/// toy::data generates the 4-direction command dataset with train/test split.
/// toy::agent implements the amplitude-based model that learns conceptual operators.
pub mod data;
pub mod agent;

pub use data::{generate_dataset, build_vocab, generate_examples, Direction, Action};
pub use agent::{ToyAgent, LossEvaluator};
