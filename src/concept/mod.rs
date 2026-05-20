/// Concept space: dynamic label set management and pruning.
///
/// In the theoretical framework:
/// - S is a labeled set
/// - scr(F) is the function space on S modulo an equivalence relation
/// - The concept space manages this label structure and the evolution
///   of scr(F) through stacking
///
/// For the toy, this manages candidate concept labels with pruning
/// heuristics (frequency, overlap, entropy).
pub mod space;

pub use space::{ConceptSpace, Label, PruningConfig};
