use std::error::Error;

use ling::feature::{
    Feature, FeatureId, Neighborhood, NeighborhoodId, NeighborhoodRef, SimilaritySpace,
};
use ling::mapping::{CompositionExample, CompositionLearner};
use ling::state::{LinkId, State, StateId, StateSpace, Type2Link};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let direction_neighborhood = NeighborhoodId(0);
    let mut feature_space = SimilaritySpace::new();
    feature_space.add_neighborhood(Neighborhood::new(
        direction_neighborhood,
        vec![FeatureId(0), FeatureId(1), FeatureId(2), FeatureId(3)],
    ));

    for feature in [FeatureId(0), FeatureId(1), FeatureId(2), FeatureId(3)] {
        feature_space.add_feature(Feature::new(
            feature,
            vec![NeighborhoodRef::from(direction_neighborhood)],
            vec![1.0],
        ));
    }

    let turn = StateId(10);
    let a = StateId(0);
    let b = StateId(1);
    let c = StateId(2);
    let d = StateId(3);

    let mut state_space = StateSpace::new();
    state_space.add_state(State::new(a, vec![FeatureId(0)]));
    state_space.add_state(State::new(b, vec![FeatureId(1)]));
    state_space.add_state(State::new(c, vec![FeatureId(2)]));
    state_space.add_state(State::new(d, vec![FeatureId(3)]));
    state_space.add_state(State::new(turn, Vec::new()));

    state_space.add_link(Type2Link::complete(
        LinkId(0),
        vec![a, b, c, d],
        vec![a],
        1.0,
    ));

    let mut learner = CompositionLearner::new(0.9);
    learner.add_example(CompositionExample {
        operator: turn,
        argument: a,
        result_link: LinkId(0),
    });

    if let Some(generated) =
        learner.transfer_single_target(turn, c, LinkId(1), &state_space, &feature_space)
    {
        log::info!(
            "generated link {:?}: {:?} -> {:?}",
            generated.id,
            generated.source,
            generated.target
        );
    }

    Ok(())
}
