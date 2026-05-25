/// Identifier for an element in the feature set `T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FeatureId(pub usize);

/// A feature element in `T`.
///
/// Features carry no similarity information; the similarity space lives on
/// the state set `S`. A feature is simply an external input/output identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Feature {
    /// Stable feature identifier.
    pub id: FeatureId,
}

impl Feature {
    /// Create a feature from its identifier.
    pub fn new(id: FeatureId) -> Self {
        Self { id }
    }
}
