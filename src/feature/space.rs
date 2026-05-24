use std::collections::BTreeMap;

/// Identifier for an element in the feature set `T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FeatureId(pub usize);

/// Identifier for a neighborhood in the similarity space over `T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NeighborhoodId(pub usize);

/// Reference from a feature to one of its neighborhoods.
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

/// A feature element in `T`.
///
/// A feature stores the neighborhoods it belongs to and a confidence for each
/// membership relation. The confidence belongs to the relation
/// `feature ∈ neighborhood`, not to the neighborhood itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Feature {
    /// Stable feature identifier.
    pub id: FeatureId,
    /// Neighborhood memberships for this feature.
    pub neighborhoods: Vec<NeighborhoodRef>,
    /// Confidence values aligned with `neighborhoods`.
    pub confidences: Vec<f64>,
}

impl Feature {
    /// Create a feature with explicit neighborhood membership confidences.
    pub fn new(id: FeatureId, neighborhoods: Vec<NeighborhoodRef>, confidences: Vec<f64>) -> Self {
        assert_eq!(
            neighborhoods.len(),
            confidences.len(),
            "neighborhoods and confidences must have the same length"
        );
        Self {
            id,
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

/// A neighborhood in the similarity space over `T`.
///
/// The neighborhood only records its members. Membership confidence is stored
/// on each `Feature`, because confidence describes a specific membership
/// relation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Neighborhood {
    /// Stable neighborhood identifier.
    pub id: NeighborhoodId,
    /// Features that belong to this neighborhood.
    pub members: Vec<FeatureId>,
}

impl Neighborhood {
    /// Create a neighborhood from its members.
    pub fn new(id: NeighborhoodId, members: Vec<FeatureId>) -> Self {
        Self { id, members }
    }
}

/// Similarity space over feature elements.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SimilaritySpace {
    /// All known feature elements.
    pub features: Vec<Feature>,
    /// All known neighborhoods.
    pub neighborhoods: Vec<Neighborhood>,
}

impl SimilaritySpace {
    /// Create an empty similarity space.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a feature to the space.
    pub fn add_feature(&mut self, feature: Feature) {
        self.features.push(feature);
    }

    /// Add a neighborhood to the space.
    pub fn add_neighborhood(&mut self, neighborhood: Neighborhood) {
        self.neighborhoods.push(neighborhood);
    }

    /// Return a feature by identifier.
    pub fn feature(&self, id: FeatureId) -> Option<&Feature> {
        self.features.iter().find(|feature| feature.id == id)
    }

    /// Compute weighted neighborhood-overlap similarity between two features.
    ///
    /// This is a weighted Jaccard score:
    ///
    /// `sum(shared min(conf_a, conf_b)) / sum(union max(conf_a, conf_b))`.
    pub fn feature_similarity(&self, left: FeatureId, right: FeatureId) -> f64 {
        let Some(left) = self.feature(left) else {
            return 0.0;
        };
        let Some(right) = self.feature(right) else {
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
