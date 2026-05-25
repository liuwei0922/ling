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
    let source_members = vec![StateId(0), StateId(1), StateId(2)];
    let link = Type2Link::complete(
        LinkId(0),
        NeighborhoodId(0),
        &source_members,
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
    let source_members = vec![StateId(0), StateId(1)];
    let link = Type2Link::complete(
        LinkId(0),
        NeighborhoodId(0),
        &source_members,
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

    let source_members = vec![a, StateId(1), c];
    state_space.add_link(Type2Link::complete(
        LinkId(0),
        direction,
        &source_members,
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
    assert_eq!(generated.source, direction);
}

#[test]
fn type2_link_output_is_projected_back_to_features() {
    let source_members = vec![StateId(0), StateId(1)];
    let link = Type2Link::complete(
        LinkId(0),
        NeighborhoodId(0),
        &source_members,
        vec![StateId(2)],
        1.0,
    );

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
        source: NeighborhoodId(0),
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
        target_neighborhoods: vec![],
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

#[test]
fn add_link_creates_target_neighborhood_with_born_confidence() {
    let src_nh = NeighborhoodId(0);
    let mut space = StateSpace::new();
    space.add_neighborhood(Neighborhood::new(
        src_nh,
        vec![StateId(0), StateId(1), StateId(2)],
    ));

    let a = StateId(0);
    let b = StateId(1);
    let c = StateId(2);
    let e = StateId(10);
    let d = StateId(11);
    for id in [a, b, c] {
        space.add_state(State::new(id, vec![], vec![], vec![]));
    }

    // {a,b,c} -> {e,d}, all coefficients 1.0
    // amp(e)=3, amp(d)=3, Born: P(e)=0.5, P(d)=0.5
    let source_members = vec![a, b, c];
    space.add_link(Type2Link::complete(
        LinkId(0),
        src_nh,
        &source_members,
        vec![e, d],
        1.0,
    ));

    // src_nh + 2 target neighborhoods
    assert_eq!(space.neighborhoods.len(), 3);

    let target_nhs: Vec<_> = space
        .neighborhoods
        .iter()
        .filter(|n| n.id != src_nh)
        .collect();
    assert_eq!(target_nhs.len(), 2);
    for nh in &target_nhs {
        assert_eq!(nh.members, vec![a, b, c]);
        assert_eq!(nh.source_links, vec![LinkId(0)]);
    }

    // Each source state gets confidence 0.5 in each target neighborhood.
    for id in [a, b, c] {
        let state = space.state(id).unwrap();
        assert_eq!(state.confidences, vec![0.5, 0.5]);
    }

    // The link records its target neighborhoods.
    let link = &space.links[0];
    assert_eq!(link.target_neighborhoods.len(), 2);
}

#[test]
fn merge_averages_confidences_for_same_source_and_target() {
    let src_nh = NeighborhoodId(0);
    let mut space = StateSpace::new();
    space.add_neighborhood(Neighborhood::new(src_nh, vec![StateId(0), StateId(1)]));

    let a = StateId(0);
    let b = StateId(1);
    let e = StateId(10);
    for id in [a, b] {
        space.add_state(State::new(id, vec![], vec![], vec![]));
    }

    // Link 1: {a,b} -> {e}, P(e)=1.0
    let members = vec![a, b];
    space.add_link(Type2Link::complete(
        LinkId(0),
        src_nh,
        &members,
        vec![e],
        1.0,
    ));

    // Link 2: {a,b} -> {e}, same (src, target) pair, P(e)=1.0
    // Born: amp(e)=2, total_sq=4, P=1.0. Merge: mean = (1.0+1.0)/2 = 1.0
    space.add_link(Type2Link::complete(
        LinkId(1),
        src_nh,
        &members,
        vec![e],
        1.0,
    ));

    // Only 1 target neighborhood (merged), plus src_nh = 2 total
    assert_eq!(space.neighborhoods.len(), 2);

    // Both links reference the same target neighborhood
    assert_eq!(
        space.links[0].target_neighborhoods,
        space.links[1].target_neighborhoods
    );

    let target_nh = space.neighborhoods.iter().find(|n| n.id != src_nh).unwrap();
    assert_eq!(target_nh.source_links, vec![LinkId(0), LinkId(1)]);

    // Confidence should be mean of 1.0 and 1.0 = 1.0
    for id in [a, b] {
        let state = space.state(id).unwrap();
        assert_eq!(state.confidences.len(), 1);
        assert!((state.confidences[0] - 1.0).abs() < 1e-10);
    }
}

#[test]
fn different_source_neighborhoods_are_not_merged() {
    let nh0 = NeighborhoodId(0);
    let nh1 = NeighborhoodId(1);
    let mut space = StateSpace::new();
    space.add_neighborhood(Neighborhood::new(nh0, vec![StateId(0), StateId(1)]));
    space.add_neighborhood(Neighborhood::new(
        nh1,
        vec![StateId(0), StateId(1), StateId(2)],
    ));

    let a = StateId(0);
    let b = StateId(1);
    let c = StateId(2);
    let e = StateId(10);
    for id in [a, b, c] {
        space.add_state(State::new(id, vec![], vec![], vec![]));
    }

    // Link from nh0: {a,b} -> {e}, P(e)=1.0
    space.add_link(Type2Link::complete(LinkId(0), nh0, &[a, b], vec![e], 1.0));

    // Link from nh1: {a,b,c} -> {e}, P(e)=1.0
    // Different source neighborhood → separate target neighborhood
    space.add_link(Type2Link::complete(
        LinkId(1),
        nh1,
        &[a, b, c],
        vec![e],
        1.0,
    ));

    // nh0 + nh1 + 2 separate target neighborhoods = 4
    assert_eq!(space.neighborhoods.len(), 4);

    // The two target neighborhoods have different members
    let target_nhs: Vec<_> = space
        .neighborhoods
        .iter()
        .filter(|n| n.id != nh0 && n.id != nh1)
        .collect();
    assert_eq!(target_nhs.len(), 2);

    let members_sets: Vec<Vec<StateId>> = target_nhs.iter().map(|n| n.members.clone()).collect();
    assert!(members_sets.contains(&vec![a, b]));
    assert!(members_sets.contains(&vec![a, b, c]));
}
