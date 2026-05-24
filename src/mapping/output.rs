use crate::feature::FeatureId;
use crate::probability::AmplitudeDistribution;
use crate::state::{StateId, Type2Link};

/// A feature amplitude or probability in an output distribution over `T`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureWeight {
    /// Output feature.
    pub feature: FeatureId,
    /// Coefficient for this feature.
    pub weight: f64,
}

/// Output distribution from one state element in `S` back to features in `T`.
#[derive(Debug, Clone, PartialEq)]
pub struct StateOutputDistribution {
    /// Source state.
    pub state: StateId,
    /// Candidate output features and their weights.
    pub features: Vec<FeatureWeight>,
}

impl StateOutputDistribution {
    /// Create a state-to-feature output distribution.
    pub fn new(state: StateId, features: Vec<FeatureWeight>) -> Self {
        Self { state, features }
    }
}

/// Mapping from `S` back to observable features in `T`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OutputMapping {
    /// Output distributions for basic states.
    pub state_outputs: Vec<StateOutputDistribution>,
}

impl OutputMapping {
    /// Create an empty output mapping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an output distribution for one state.
    pub fn add_state_output(&mut self, output: StateOutputDistribution) {
        self.state_outputs.push(output);
    }

    /// Return the output distribution for one state.
    pub fn state_output(&self, state: StateId) -> Option<&StateOutputDistribution> {
        self.state_outputs
            .iter()
            .find(|output| output.state == state)
    }

    /// Compute the feature probability distribution emitted by a type-2 link.
    ///
    /// Type-1 link coefficients and state-output coefficients are treated as
    /// real amplitudes. Amplitudes are summed first, then squared and normalized.
    pub fn link_output_distribution(&self, link: &Type2Link) -> Vec<FeatureWeight> {
        let mut amplitudes = AmplitudeDistribution::new();

        for type1_link in &link.type1_links {
            let Some(target_output) = self.state_output(type1_link.target) else {
                continue;
            };
            for feature in &target_output.features {
                amplitudes.add(feature.feature, type1_link.coefficient * feature.weight);
            }
        }

        amplitudes
            .probabilities()
            .into_iter()
            .map(|entry| FeatureWeight {
                feature: entry.item,
                weight: entry.probability,
            })
            .collect()
    }

    /// Choose the highest-weight output feature, optionally restricted to a
    /// requested output feature subset.
    pub fn select_feature(
        &self,
        distribution: &[FeatureWeight],
        allowed_features: Option<&[FeatureId]>,
    ) -> Option<FeatureId> {
        let mut amplitudes = AmplitudeDistribution::new();
        for feature in distribution {
            amplitudes.add(feature.feature, feature.weight.sqrt());
        }
        amplitudes.select_argmax(allowed_features)
    }
}
