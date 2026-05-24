use crate::feature::{FeatureId, SimilaritySpace};

/// Identifier for an element in the state set `S`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateId(pub usize);

/// A state element in `S`.
///
/// A state carries no hand-written semantic tag. Similarity between states is
/// induced from the feature elements that support them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    /// Stable state identifier.
    pub id: StateId,
    /// Feature elements that support this state.
    pub support_features: Vec<FeatureId>,
}

impl State {
    /// Create a state from its supporting features.
    pub fn new(id: StateId, support_features: Vec<FeatureId>) -> Self {
        Self {
            id,
            support_features,
        }
    }
}

/// State space containing state elements and type-2 links.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StateSpace {
    /// All known state elements.
    pub states: Vec<State>,
    /// All known type-2 links.
    pub links: Vec<super::link::Type2Link>,
}

impl StateSpace {
    /// Create an empty state space.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a state.
    pub fn add_state(&mut self, state: State) {
        self.states.push(state);
    }

    /// Add a type-2 link.
    pub fn add_link(&mut self, link: super::link::Type2Link) {
        self.links.push(link);
    }

    /// Return a state by identifier.
    pub fn state(&self, id: StateId) -> Option<&State> {
        self.states.iter().find(|state| state.id == id)
    }

    /// Compute state similarity from supporting feature similarity.
    ///
    /// The first implementation uses max-pair similarity, which is conservative
    /// in structure and easy to inspect in toy experiments.
    pub fn state_similarity(
        &self,
        left: StateId,
        right: StateId,
        similarity_space: &SimilaritySpace,
    ) -> f64 {
        let Some(left) = self.state(left) else {
            return 0.0;
        };
        let Some(right) = self.state(right) else {
            return 0.0;
        };

        left.support_features
            .iter()
            .flat_map(|left_feature| {
                right.support_features.iter().map(move |right_feature| {
                    similarity_space.feature_similarity(*left_feature, *right_feature)
                })
            })
            .fold(0.0, f64::max)
    }
}
