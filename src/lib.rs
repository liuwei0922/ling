/// Ling: experiments for modeling language-like composition through mappings
/// between a feature set `T` and a state set `S`.
///
/// # Modules
///
/// - **feature**: Feature elements and the similarity space over `T`
/// - **state**: State elements and type-2 links inside `S`
/// - **mapping**: Composition records and transfer rules
/// - **probability**: Amplitude accumulation, Born normalization, and selection
pub mod feature;
pub mod mapping;
pub mod probability;
pub mod state;
