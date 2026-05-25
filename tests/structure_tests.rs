use rand::rngs::StdRng;
use rand::SeedableRng;

use ling::feature::FeatureId;
use ling::mapping::{
    CompositionExample, CompositionLearner, FeatureWeight, OutputMapping, StateOutputDistribution,
};
use ling::probability::{AmplitudeDistribution, SelectionMode};
use ling::state::{
    LinkId, Neighborhood, NeighborhoodId, NeighborhoodRef, State, StateId, StateSpace, Type1Link,
    Type2Link,
};

#[test]
fn state_similarity_comes_from_shared_neighborhoods() {
    let direction = NeighborhoodId(0);
    let other = NeighborhoodId(1);

    let mut state_space = StateSpace::new();
    state_space.add_neighborhood(Neighborhood::new(
        direction,
        vec![StateId(0), StateId(1), StateId(2), StateId(3)],
    ));
    state_space.add_neighborhood(Neighborhood::new(other, vec![StateId(4)]));
    state_space.add_state(State::new(
        StateId(0),
        vec![],
        vec![NeighborhoodRef::from(direction)],
        vec![1.0],
    ));
    state_space.add_state(State::new(
        StateId(1),
        vec![],
        vec![NeighborhoodRef::from(direction)],
        vec![0.8],
    ));
    state_space.add_state(State::new(
        StateId(4),
        vec![],
        vec![NeighborhoodRef::from(other)],
        vec![1.0],
    ));

    assert!(state_space.state_similarity(StateId(0), StateId(1)) > 0.7);
    assert_eq!(state_space.state_similarity(StateId(0), StateId(4)), 0.0);
}

#[test]
fn state_similarity_is_direct_neighborhood_overlap() {
    let direction = NeighborhoodId(0);
    let mut state_space = StateSpace::new();
    state_space.add_neighborhood(Neighborhood::new(direction, vec![StateId(0), StateId(1)]));
    state_space.add_state(State::new(
        StateId(0),
        vec![],
        vec![NeighborhoodRef::from(direction)],
        vec![1.0],
    ));
    state_space.add_state(State::new(
        StateId(1),
        vec![],
        vec![NeighborhoodRef::from(direction)],
        vec![1.0],
    ));

    assert_eq!(state_space.state_similarity(StateId(0), StateId(1)), 1.0);
}

#[test]
fn type2_link_generates_complete_type1_links() {
    let link = Type2Link::complete(
        LinkId(0),
        vec![StateId(0), StateId(1), StateId(2)],
        vec![StateId(3), StateId(4)],
        0.5,
    );

    assert_eq!(link.type1_links.len(), 6);
    assert!(link
        .type1_links
        .iter()
        .any(|link| link.source == StateId(0) && link.target == StateId(3)));
    assert!(link.type1_links.iter().all(|link| link.coefficient == 0.5));
}

#[test]
fn amplitude_distribution_uses_born_rule_and_argmax_selection() {
    let mut distribution = AmplitudeDistribution::new();
    distribution.add(StateId(0), 1.0);
    distribution.add(StateId(1), -2.0);

    let probabilities = distribution.probabilities();

    assert_eq!(probabilities.len(), 2);
    assert_eq!(distribution.select_argmax(None), Some(StateId(1)));
}

#[test]
fn type2_link_activation_filters_by_active_source() {
    let link = Type2Link::complete(
        LinkId(0),
        vec![StateId(0), StateId(1)],
        vec![StateId(2), StateId(3)],
        1.0,
    );

    let probabilities = link.activate(&[StateId(0)]);
    let targets: Vec<StateId> = probabilities.iter().map(|entry| entry.item).collect();

    assert_eq!(targets, vec![StateId(2), StateId(3)]);
    let mut rng = StdRng::seed_from_u64(42);
    assert_eq!(
        link.select_target(&[StateId(0)], SelectionMode::Argmax, &mut rng),
        Some(StateId(2))
    );
}

#[test]
fn composition_transfer_uses_state_similarity() {
    let direction = NeighborhoodId(0);
    let mut state_space = StateSpace::new();
    state_space.add_neighborhood(Neighborhood::new(
        direction,
        vec![StateId(0), StateId(1), StateId(2)],
    ));

    let turn = StateId(10);
    let a = StateId(0);
    let c = StateId(2);
    state_space.add_state(State::new(
        a,
        vec![FeatureId(0)],
        vec![NeighborhoodRef::from(direction)],
        vec![1.0],
    ));
    state_space.add_state(State::new(
        StateId(1),
        vec![FeatureId(1)],
        vec![NeighborhoodRef::from(direction)],
        vec![1.0],
    ));
    state_space.add_state(State::new(
        c,
        vec![FeatureId(2)],
        vec![NeighborhoodRef::from(direction)],
        vec![1.0],
    ));
    state_space.add_state(State::new(turn, vec![], vec![], vec![]));
    state_space.add_link(Type2Link::complete(
        LinkId(0),
        vec![a, StateId(1), c],
        vec![a],
        1.0,
    ));

    let mut learner = CompositionLearner::new(0.9);
    learner.add_example(CompositionExample {
        operator: turn,
        argument: a,
        result_link: LinkId(0),
    });

    let generated = learner
        .transfer_single_target(turn, c, LinkId(1), &state_space)
        .expect("similar state should transfer");

    assert_eq!(generated.target, vec![c]);
    assert_eq!(generated.source, vec![a, StateId(1), c]);
}

#[test]
fn type2_link_output_is_projected_back_to_features() {
    let source = vec![StateId(0), StateId(1)];
    let target = vec![StateId(2)];
    let link = Type2Link::complete(LinkId(0), source, target, 1.0);

    let mut output_mapping = OutputMapping::new();
    output_mapping.add_state_output(StateOutputDistribution::new(
        StateId(2),
        vec![
            FeatureWeight {
                feature: FeatureId(20),
                weight: 0.25,
            },
            FeatureWeight {
                feature: FeatureId(21),
                weight: 0.75,
            },
        ],
    ));

    let distribution = output_mapping.link_output_distribution(&link);
    let mut rng = StdRng::seed_from_u64(42);
    let selected = output_mapping.select_feature(
        &distribution,
        SelectionMode::Argmax,
        Some(&[FeatureId(20), FeatureId(21)]),
        &mut rng,
    );

    assert_eq!(selected, Some(FeatureId(21)));
}

#[test]
fn output_projection_uses_coherent_amplitude_sum() {
    let link = Type2Link {
        id: LinkId(0),
        source: vec![StateId(0), StateId(1)],
        target: vec![StateId(2), StateId(3)],
        type1_links: vec![
            Type1Link {
                source: StateId(0),
                target: StateId(2),
                coefficient: 1.0,
            },
            Type1Link {
                source: StateId(1),
                target: StateId(3),
                coefficient: -1.0,
            },
        ],
    };

    let mut output_mapping = OutputMapping::new();
    output_mapping.add_state_output(StateOutputDistribution::new(
        StateId(2),
        vec![FeatureWeight {
            feature: FeatureId(20),
            weight: 1.0,
        }],
    ));
    output_mapping.add_state_output(StateOutputDistribution::new(
        StateId(3),
        vec![FeatureWeight {
            feature: FeatureId(20),
            weight: 1.0,
        }],
    ));

    let distribution = output_mapping.link_output_distribution(&link);

    assert!(distribution.is_empty());
}
