use rand::Rng;

use super::space::StateId;
use crate::probability::{AmplitudeDistribution, Probability, SelectionMode};

/// Identifier for a type-2 link in the state set `S`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LinkId(pub usize);

/// A type-1 link from one state element to another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Type1Link {
    /// Source state.
    pub source: StateId,
    /// Target state.
    pub target: StateId,
    /// Coefficient carried by this link.
    pub coefficient: f64,
}

/// A type-2 link from a subset of `S` to another subset of `S`.
#[derive(Debug, Clone, PartialEq)]
pub struct Type2Link {
    /// Stable link identifier.
    pub id: LinkId,
    /// Source subset.
    pub source: Vec<StateId>,
    /// Target subset.
    pub target: Vec<StateId>,
    /// Type-1 links between source and target states.
    pub type1_links: Vec<Type1Link>,
}

impl Type2Link {
    /// Create a type-2 link and generate all source-target type-1 links.
    pub fn complete(
        id: LinkId,
        source: Vec<StateId>,
        target: Vec<StateId>,
        coefficient: f64,
    ) -> Self {
        let type1_links = source
            .iter()
            .flat_map(|source| {
                target.iter().map(move |target| Type1Link {
                    source: *source,
                    target: *target,
                    coefficient,
                })
            })
            .collect();

        Self {
            id,
            source,
            target,
            type1_links,
        }
    }

    /// Create an identity/self-loop link for a single state.
    pub fn identity(id: LinkId, state: StateId) -> Self {
        Self::complete(id, vec![state], vec![state], 1.0)
    }

    /// Activate type-1 links whose source is currently active.
    pub fn activate(&self, active_sources: &[StateId]) -> Vec<Probability<StateId>> {
        let mut amplitudes = AmplitudeDistribution::new();
        for link in &self.type1_links {
            if active_sources.contains(&link.source) {
                amplitudes.add(link.target, link.coefficient);
            }
        }
        amplitudes.probabilities()
    }

    /// Select a target state from the active paths using the given selection mode.
    pub fn select_target(
        &self,
        active_sources: &[StateId],
        mode: SelectionMode,
        rng: &mut impl Rng,
    ) -> Option<StateId> {
        let mut amplitudes = AmplitudeDistribution::new();
        for link in &self.type1_links {
            if active_sources.contains(&link.source) {
                amplitudes.add(link.target, link.coefficient);
            }
        }
        amplitudes.select(mode, Some(&self.target), rng)
    }
}
