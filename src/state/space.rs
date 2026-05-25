use std::collections::BTreeMap;

use super::link::LinkId;
use crate::feature::FeatureId;

/// Identifier for an element in the state set `S`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateId(pub usize);

/// Identifier for a neighborhood in the similarity space over `S`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NeighborhoodId(pub usize);

/// Reference from a state to one of its neighborhoods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighborhoodRef {
    /// The referenced neighborhood.
    pub id: NeighborhoodId,
}

impl From<NeighborhoodId> for NeighborhoodRef {
    fn from(id: NeighborhoodId) -> Self {
        Self { id }
    }
}

/// A state element in `S`.
///
/// A state carries no hand-written semantic tag. It stores the neighborhoods
/// it belongs to and a confidence for each membership relation. Similarity
/// between states is computed directly from neighborhood overlap.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    /// Stable state identifier.
    pub id: StateId,
    /// Feature elements that support this state.
    pub support_features: Vec<FeatureId>,
    /// Neighborhood memberships for this state.
    pub neighborhoods: Vec<NeighborhoodRef>,
    /// Confidence values aligned with `neighborhoods`.
    pub confidences: Vec<f64>,
}

impl State {
    /// Create a state from its supporting features and neighborhood memberships.
    pub fn new(
        id: StateId,
        support_features: Vec<FeatureId>,
        neighborhoods: Vec<NeighborhoodRef>,
        confidences: Vec<f64>,
    ) -> Self {
        assert_eq!(
            neighborhoods.len(),
            confidences.len(),
            "neighborhoods and confidences must have the same length"
        );
        Self {
            id,
            support_features,
            neighborhoods,
            confidences,
        }
    }

    fn membership_map(&self) -> BTreeMap<NeighborhoodId, f64> {
        self.neighborhoods
            .iter()
            .zip(&self.confidences)
            .filter(|(_, confidence)| **confidence > 0.0)
            .map(|(neighborhood, confidence)| (neighborhood.id, *confidence))
            .collect()
    }
}

/// A neighborhood in the similarity space over `S`.
///
/// The neighborhood records its members and the links that use it as source.
/// Membership confidence is stored on each `State`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Neighborhood {
    /// Stable neighborhood identifier.
    pub id: NeighborhoodId,
    /// States that belong to this neighborhood.
    pub members: Vec<StateId>,
    /// Type-2 links that have this neighborhood as their source.
    pub source_links: Vec<LinkId>,
}

impl Neighborhood {
    /// Create a neighborhood from its members.
    pub fn new(id: NeighborhoodId, members: Vec<StateId>) -> Self {
        Self {
            id,
            members,
            source_links: Vec::new(),
        }
    }
}

/// State space containing state elements, type-2 links, and the similarity
/// space (neighborhoods over `S`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StateSpace {
    /// All known state elements.
    pub states: Vec<State>,
    /// All known type-2 links.
    pub links: Vec<super::link::Type2Link>,
    /// All known neighborhoods in the similarity space over `S`.
    pub neighborhoods: Vec<Neighborhood>,
    /// Index from (source_neighborhood, target_state) to generated target neighborhood.
    target_neighborhood_index: BTreeMap<(NeighborhoodId, StateId), NeighborhoodId>,
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

    /// Add a type-2 link and automatically build target neighborhoods.
    ///
    /// For each target state, a Born-normalized confidence is computed from the
    /// coherent amplitude sum of all Type1 links pointing to that target. When
    /// multiple links share the same (source_neighborhood, target_state) pair,
    /// their confidences are averaged.
    pub fn add_link(&mut self, mut link: super::link::Type2Link) {
        // Compute Born amplitudes per target.
        let mut target_amp: BTreeMap<StateId, f64> = BTreeMap::new();
        for tl in &link.type1_links {
            *target_amp.entry(tl.target).or_insert(0.0) += tl.coefficient;
        }
        let total_sq: f64 = target_amp.values().map(|a| a * a).sum();

        // Look up source neighborhood members.
        let source_members: Vec<StateId> = self
            .neighborhoods
            .iter()
            .find(|n| n.id == link.source)
            .map(|n| n.members.clone())
            .unwrap_or_default();

        if total_sq > 0.0 {
            for (&target, &amp) in &target_amp {
                let p = amp * amp / total_sq;
                if p <= 0.0 {
                    continue;
                }

                let key = (link.source, target);
                if let Some(&existing_nid) = self.target_neighborhood_index.get(&key) {
                    // Merge: take the mean of all contributing links.
                    let existing = self
                        .neighborhoods
                        .iter()
                        .find(|n| n.id == existing_nid)
                        .expect("target_neighborhood_index refers to non-existent neighborhood");
                    let n = existing.source_links.len() as f64;
                    let old_conf = self
                        .states
                        .iter()
                        .find(|s| s.id == existing.members[0])
                        .and_then(|s| {
                            s.neighborhoods
                                .iter()
                                .position(|r| r.id == existing_nid)
                                .map(|i| s.confidences[i])
                        })
                        .unwrap_or(p);
                    let new_conf = (old_conf * n + p) / (n + 1.0);

                    // Update all member states' confidences.
                    for state in &mut self.states {
                        if let Some(pos) = state
                            .neighborhoods
                            .iter()
                            .position(|r| r.id == existing_nid)
                        {
                            state.confidences[pos] = new_conf;
                        }
                    }

                    // Update neighborhood's source_links.
                    if let Some(nh) = self.neighborhoods.iter_mut().find(|n| n.id == existing_nid) {
                        nh.source_links.push(link.id);
                    }

                    link.target_neighborhoods.push(existing_nid);
                } else {
                    // Create a new target neighborhood.
                    let nid = NeighborhoodId(self.neighborhoods.len());
                    let nref = NeighborhoodRef::from(nid);

                    for state in &mut self.states {
                        if source_members.contains(&state.id) {
                            state.neighborhoods.push(nref);
                            state.confidences.push(p);
                        }
                    }

                    self.neighborhoods.push(Neighborhood {
                        id: nid,
                        members: source_members.clone(),
                        source_links: vec![link.id],
                    });
                    self.target_neighborhood_index.insert(key, nid);
                    link.target_neighborhoods.push(nid);
                }
            }
        }

        // Record this link on the source neighborhood.
        if let Some(src_nh) = self.neighborhoods.iter_mut().find(|n| n.id == link.source) {
            src_nh.source_links.push(link.id);
        }

        self.links.push(link);
    }

    /// Add a neighborhood to the similarity space.
    pub fn add_neighborhood(&mut self, neighborhood: Neighborhood) {
        self.neighborhoods.push(neighborhood);
    }

    /// Return a state by identifier.
    pub fn state(&self, id: StateId) -> Option<&State> {
        self.states.iter().find(|state| state.id == id)
    }

    /// Return a neighborhood by identifier.
    pub fn neighborhood(&self, id: NeighborhoodId) -> Option<&Neighborhood> {
        self.neighborhoods.iter().find(|n| n.id == id)
    }

    /// Compute state similarity from direct neighborhood overlap.
    ///
    /// This is a weighted Jaccard score over the neighborhoods that two states
    /// belong to:
    ///
    /// `sum(shared min(conf_a, conf_b)) / sum(union max(conf_a, conf_b))`.
    pub fn state_similarity(&self, left: StateId, right: StateId) -> f64 {
        let Some(left) = self.state(left) else {
            return 0.0;
        };
        let Some(right) = self.state(right) else {
            return 0.0;
        };

        let left_memberships = left.membership_map();
        let right_memberships = right.membership_map();
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (neighborhood, left_confidence) in &left_memberships {
            let right_confidence = right_memberships.get(neighborhood).copied().unwrap_or(0.0);
            numerator += left_confidence.min(right_confidence);
            denominator += left_confidence.max(right_confidence);
        }

        for (neighborhood, right_confidence) in &right_memberships {
            if !left_memberships.contains_key(neighborhood) {
                denominator += right_confidence;
            }
        }

        if denominator > 0.0 {
            numerator / denominator
        } else if left.id == right.id {
            1.0
        } else {
            0.0
        }
    }
}
