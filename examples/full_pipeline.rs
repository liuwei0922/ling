//! 完整流程实验：从训练"转"到泛化"转X"。
//!
//! 本实验验证端到端流程：
//! 1. 建立 4 个方向状态 (a,b,c,d) 和方向邻域。
//! 2. 训练"转"映射 {a,b,c,d} -> {a,b,c,d}，自动构造目标邻域。
//! 3. 建立 S->T 输出映射。
//! 4. 训练"转a"和"转b"两个具体组合。
//! 5. 测试"转c"和"转d"，验证系统能否通过相似空间泛化到未见组合。
//!
//! 关键机制：add_link 自动从 Type2Link 构造目标邻域（Born 归一化），
//! 使方向状态通过共享邻域获得相似性，从而支持组合迁移。

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
struct Direction {
    name: &'static str,
    state: StateId,
    input_feature: FeatureId,
    output_feature: FeatureId,
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let directions = [
        Direction {
            name: "a",
            state: StateId(0),
            input_feature: FeatureId(0),
            output_feature: FeatureId(100),
        },
        Direction {
            name: "b",
            state: StateId(1),
            input_feature: FeatureId(1),
            output_feature: FeatureId(101),
        },
        Direction {
            name: "c",
            state: StateId(2),
            input_feature: FeatureId(2),
            output_feature: FeatureId(102),
        },
        Direction {
            name: "d",
            state: StateId(3),
            input_feature: FeatureId(3),
            output_feature: FeatureId(103),
        },
    ];
    let turn = StateId(10);
    let dir_members: Vec<StateId> = directions.iter().map(|d| d.state).collect();
    let output_features: Vec<FeatureId> = directions.iter().map(|d| d.output_feature).collect();

    // ── Phase 0: 构建状态空间 ────────────────────────────────────────
    let mut state_space = StateSpace::new();

    // 方向邻域（手动预设，代表"方向"这个相似上下文）
    state_space.add_neighborhood(Neighborhood::new(
        DIRECTION_NEIGHBORHOOD,
        dir_members.clone(),
    ));
    for d in &directions {
        state_space.add_state(State::new(
            d.state,
            vec![d.input_feature],
            vec![NeighborhoodRef::from(DIRECTION_NEIGHBORHOOD)],
            vec![1.0],
        ));
    }
    state_space.add_state(State::new(turn, vec![], vec![], vec![]));

    log::info!(
        "Phase 0: 状态空间已构建，{} 个方向状态，1 个方向邻域",
        directions.len()
    );

    // ── Phase 1: 训练"转"映射 {a,b,c,d} -> {a,b,c,d} ──────────────────
    //
    // "转"是从方向集合到方向集合的映射，complete() 生成 4×4=16 条 Type1
    // 连线。add_link 自动为每个目标状态构造邻域（Born 归一化后置信度各 0.25）。
    state_space.add_link(Type2Link::complete(
        LinkId(0),
        DIRECTION_NEIGHBORHOOD,
        &dir_members,
        dir_members.clone(),
        1.0,
    ));

    log::info!(
        "Phase 1: 训练\"转\" {a:?}->{a:?}，自动生成 {} 个目标邻域",
        state_space.neighborhoods.len() - 1, // 减去方向邻域
        a = dir_members,
    );
    log::info!("  邻域总数: {}", state_space.neighborhoods.len());

    // ── Phase 2: 建立 S->T 输出映射 ────────────────────────────────────
    let mut output_mapping = OutputMapping::new();
    for d in &directions {
        output_mapping.add_state_output(StateOutputDistribution::new(
            d.state,
            vec![FeatureWeight {
                feature: d.output_feature,
                weight: 1.0,
            }],
        ));
    }

    log::info!("Phase 2: S->T 输出映射已建立");

    // ── Phase 3: 训练"转a"和"转b" ──────────────────────────────────────
    let train = &directions[..2];
    let test = &directions[2..];
    let mut learner = CompositionLearner::new(SIMILARITY_THRESHOLD);

    for (idx, d) in train.iter().enumerate() {
        let link_id = LinkId(10 + idx);
        state_space.add_link(Type2Link::complete(
            link_id,
            DIRECTION_NEIGHBORHOOD,
            &dir_members,
            vec![d.state],
            1.0,
        ));
        learner.add_example(CompositionExample {
            operator: turn,
            argument: d.state,
            result_link: link_id,
        });

        log::info!("Phase 3: 训练\"转{}\"，LinkId={:?}", d.name, link_id);
    }

    // 查看训练后方向状态的邻域置信度
    {
        let a_state = state_space.state(directions[0].state).unwrap();
        log::info!(
            "  State-a 邻域数: {}, 置信度: {:?}",
            a_state.neighborhoods.len(),
            a_state.confidences
        );
    }

    // ── Phase 4: 测试"转c"和"转d" ──────────────────────────────────────
    log::info!("Phase 4: 测试泛化...");
    let mut rng = StdRng::seed_from_u64(42);
    let mut all_correct = true;

    for (idx, d) in test.iter().enumerate() {
        let generated =
            learner.transfer_single_target(turn, d.state, LinkId(100 + idx), &state_space);

        let prediction = generated.as_ref().and_then(|link| {
            let dist = output_mapping.link_output_distribution(link);
            output_mapping.select_feature(
                &dist,
                SelectionMode::Argmax,
                Some(&output_features),
                &mut rng,
            )
        });

        let correct = prediction == Some(d.output_feature);
        if !correct {
            all_correct = false;
        }

        log::info!(
            "  转{}: predicted={:?}, expected={:?}, correct={}",
            d.name,
            prediction.map(|f| f.0),
            d.output_feature.0,
            correct,
        );
    }

    log::info!(
        "Phase 4 结果: {}",
        if all_correct {
            "全部泛化成功"
        } else {
            "存在泛化失败"
        }
    );

    Ok(())
}
