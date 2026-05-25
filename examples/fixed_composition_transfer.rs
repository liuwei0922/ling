//! 固定复合迁移实验。
//!
//! 这个例子不是完整训练流程，而是验证一个较小的问题：当 S 上的相似空间和 S->T 输出
//! 映射都已经给定时，系统能否用两个已见组合迁移生成另外两个未见组合。
//!
//! 实验中 a、b、c、d 四个方向的状态被预先放在同一个相似邻域中。训练阶段只记录
//! `转a` 和 `转b` 两个组合；测试阶段请求 `转c` 和 `转d`。如果相似度超过阈值，
//! `CompositionLearner` 会把已见 2 型连线中的单目标替换为待测试状态，再通过 S->T
//! 输出映射得到最终输出特征。
//!
//! 可能结果：当前设置下通常会得到 train/test 都 100%。这并不表示模型已经学会了
//! "转"这个生成元，也不表示相似空间是从数据中学出来的；它只说明在这些结构预设成立
//! 时，复合迁移和 S->T 输出投影可以跑通。

use std::error::Error;

use rand::rngs::StdRng;
use rand::SeedableRng;

use ling::feature::FeatureId;
use ling::mapping::{
    CompositionExample, CompositionLearner, FeatureWeight, OutputMapping, StateOutputDistribution,
};
use ling::probability::SelectionMode;
use ling::state::{
    LinkId, Neighborhood, NeighborhoodId, NeighborhoodRef, State, StateId, StateSpace, Type2Link,
};

const SIMILARITY_THRESHOLD: f64 = 0.9;
const DIRECTION_NEIGHBORHOOD: NeighborhoodId = NeighborhoodId(0);

#[derive(Debug, Clone, Copy)]
struct DirectionCase {
    name: &'static str,
    state: StateId,
    input_feature: FeatureId,
    output_feature: FeatureId,
}

struct ReportContext<'a> {
    turn: StateId,
    learner: &'a CompositionLearner,
    state_space: &'a StateSpace,
    output_mapping: &'a OutputMapping,
    output_features: &'a [FeatureId],
    selection_mode: SelectionMode,
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let directions = [
        DirectionCase {
            name: "a",
            state: StateId(0),
            input_feature: FeatureId(0),
            output_feature: FeatureId(100),
        },
        DirectionCase {
            name: "b",
            state: StateId(1),
            input_feature: FeatureId(1),
            output_feature: FeatureId(101),
        },
        DirectionCase {
            name: "c",
            state: StateId(2),
            input_feature: FeatureId(2),
            output_feature: FeatureId(102),
        },
        DirectionCase {
            name: "d",
            state: StateId(3),
            input_feature: FeatureId(3),
            output_feature: FeatureId(103),
        },
    ];
    let turn = StateId(10);

    let mut state_space = build_state_space(turn, &directions);
    let output_mapping = build_output_mapping(&directions);
    let output_features: Vec<FeatureId> =
        directions.iter().map(|case| case.output_feature).collect();
    let mut learner = CompositionLearner::new(SIMILARITY_THRESHOLD);

    let train = &directions[..2];
    let test = &directions[2..];

    for (idx, case) in train.iter().enumerate() {
        let link_id = LinkId(idx);
        let source_members: Vec<StateId> = directions.iter().map(|d| d.state).collect();
        state_space.add_link(Type2Link::complete(
            link_id,
            DIRECTION_NEIGHBORHOOD,
            &source_members,
            vec![case.state],
            1.0,
        ));
        learner.add_example(CompositionExample {
            operator: turn,
            argument: case.state,
            result_link: link_id,
        });
    }

    log::info!("trained examples: {}", learner.examples.len());
    let report_context = ReportContext {
        turn,
        learner: &learner,
        state_space: &state_space,
        output_mapping: &output_mapping,
        output_features: &output_features,
        selection_mode: SelectionMode::Argmax,
    };
    report_split("train", train, &report_context);
    report_split("test", test, &report_context);

    Ok(())
}

fn build_state_space(turn: StateId, directions: &[DirectionCase]) -> StateSpace {
    let mut state_space = StateSpace::new();
    state_space.add_neighborhood(Neighborhood::new(
        DIRECTION_NEIGHBORHOOD,
        directions.iter().map(|case| case.state).collect(),
    ));

    for case in directions {
        state_space.add_state(State::new(
            case.state,
            vec![case.input_feature],
            vec![NeighborhoodRef::from(DIRECTION_NEIGHBORHOOD)],
            vec![1.0],
        ));
    }
    state_space.add_state(State::new(turn, vec![], vec![], vec![]));

    state_space
}

fn build_output_mapping(directions: &[DirectionCase]) -> OutputMapping {
    let mut output_mapping = OutputMapping::new();
    for case in directions {
        output_mapping.add_state_output(StateOutputDistribution::new(
            case.state,
            vec![FeatureWeight {
                feature: case.output_feature,
                weight: 1.0,
            }],
        ));
    }
    output_mapping
}

fn report_split(label: &str, cases: &[DirectionCase], context: &ReportContext<'_>) {
    let mut rng = StdRng::seed_from_u64(42);
    let mut correct = 0;
    for (idx, case) in cases.iter().enumerate() {
        let generated = context.learner.transfer_single_target(
            context.turn,
            case.state,
            LinkId(100 + idx),
            context.state_space,
        );
        let prediction = generated.as_ref().and_then(|link| {
            let output_distribution = context.output_mapping.link_output_distribution(link);
            context.output_mapping.select_feature(
                &output_distribution,
                context.selection_mode,
                Some(context.output_features),
                &mut rng,
            )
        });
        let is_correct = prediction == Some(case.output_feature);
        if is_correct {
            correct += 1;
        }

        log::info!(
            "{label:5} case {:>1}: predicted {:?}, expected {:?}, correct={}",
            case.name,
            prediction,
            case.output_feature,
            is_correct
        );
    }

    let accuracy = correct as f64 / cases.len() as f64;
    log::info!("{label:5} accuracy: {:.0}%", accuracy * 100.0);
}
