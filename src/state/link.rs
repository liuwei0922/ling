use rand::Rng;

use super::space::{NeighborhoodId, StateId};
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
    /// Source neighborhood (the subset of `S` this link activates from).
    pub source: NeighborhoodId,
    /// Target subset.
    pub target: Vec<StateId>,
    /// Type-1 links between source and target states.
    pub type1_links: Vec<Type1Link>,
    /// Target neighborhoods generated from this link.
    pub target_neighborhoods: Vec<NeighborhoodId>,
}

impl Type2Link {
    /// Create a type-2 link and generate all source-target type-1 links.
    ///
    /// `source_members` is the member list of the source neighborhood, used to
    /// generate the complete bipartite Type1 links. `target_neighborhoods` is
    /// initialized empty and filled by `StateSpace::add_link`.
    pub fn complete(
        id: LinkId,
        source: NeighborhoodId,
        source_members: &[StateId],
        target: Vec<StateId>,
        coefficient: f64,
    ) -> Self {
        let type1_links = source_members
            .iter()
            .flat_map(|&src| {
                target.iter().map(move |&tgt| Type1Link {
                    source: src,
                    target: tgt,
                    coefficient,
                })
            })
            .collect();

        Self {
            id,
            source,
            target,
            type1_links,
            target_neighborhoods: Vec::new(),
        }
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
