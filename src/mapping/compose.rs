use crate::state::{LinkId, StateId, StateSpace, Type2Link};

/// A training observation for composition: `operator + argument -> result`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompositionExample {
    /// Operator-like state, such as the state induced by "turn".
    pub operator: StateId,
    /// Argument state, such as the state induced by a direction symbol.
    pub argument: StateId,
    /// The observed result link.
    pub result_link: LinkId,
}

/// Learns and applies minimal composition transfer rules.
#[derive(Debug, Clone, PartialEq)]
pub struct CompositionLearner {
    /// Observed composition examples.
    pub examples: Vec<CompositionExample>,
    /// Required state similarity for transferring an observed composition.
    pub similarity_threshold: f64,
}

impl CompositionLearner {
    /// Create a learner with a similarity threshold.
    pub fn new(similarity_threshold: f64) -> Self {
        Self {
            examples: Vec::new(),
            similarity_threshold,
        }
    }

    /// Add an observed composition example.
    pub fn add_example(&mut self, example: CompositionExample) {
        self.examples.push(example);
    }

    /// Generate a candidate transferred link for `operator + argument`.
    ///
    /// The first implementation supports single-target links. If a stored
    /// example has a target similar enough to the requested argument, that
    /// target is replaced by the requested argument.
    pub fn transfer_single_target(
        &self,
        operator: StateId,
        argument: StateId,
        new_link_id: LinkId,
        state_space: &StateSpace,
    ) -> Option<Type2Link> {
        self.examples.iter().find_map(|example| {
            if example.operator != operator {
                return None;
            }

            let result_link = state_space
                .links
                .iter()
                .find(|link| link.id == example.result_link)?;

            if result_link.target.len() != 1 {
                return None;
            }

            let similarity = state_space.state_similarity(example.argument, argument);
            if similarity < self.similarity_threshold {
                return None;
            }

            let source_nh = state_space.neighborhood(result_link.source)?;

            Some(Type2Link::complete(
                new_link_id,
                result_link.source,
                &source_nh.members,
                vec![argument],
                1.0,
            ))
        })
    }
}
