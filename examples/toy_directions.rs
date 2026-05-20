use std::error::Error;

use ling::core::amplitude::AmplitudeState;
use ling::engine::trainer::EvolutionaryOptimizer;
use ling::toy::{
    agent::{LossEvaluator, ToyAgent},
    data::{build_vocab, generate_dataset, generate_examples},
};

fn main() -> Result<(), Box<dyn Error>> {
    eprintln!("=== Ling: 4-Direction Toy Experiment ===");
    eprintln!();

    // ---- Dataset ----
    let (train_examples, test_examples) = generate_dataset();
    let vocab = build_vocab(&train_examples, &test_examples);
    eprintln!("Vocabulary: {:?}", vocab);
    eprintln!("Vocab size: {}", vocab.len());
    eprintln!();

    let train_data = generate_examples(&train_examples, &vocab);
    let test_data = generate_examples(&test_examples, &vocab);
    eprintln!(
        "Training examples: {} ({} commands × 4 directions)",
        train_data.len(),
        train_examples.len()
    );
    eprintln!(
        "Test examples: {} ({} commands × 4 directions)",
        test_data.len(),
        test_examples.len()
    );
    eprintln!();

    // ---- Agent ----
    let num_operators = 8;
    let mut rng = rand::thread_rng();
    let mut agent = ToyAgent::new(vocab.len(), num_operators, &mut rng);
    eprintln!(
        "Agent: {} features → {} operators → 4 directions",
        vocab.len(),
        num_operators
    );
    eprintln!("Parameters: {}", agent.num_params());
    eprintln!();

    // ---- Baseline (untrained) ----
    let train_acc = agent.compute_accuracy(&train_data);
    let test_acc = agent.compute_accuracy(&test_data);
    eprintln!("[Before training] Train acc: {:.2}%", train_acc * 100.0);
    eprintln!("[Before training] Test acc:  {:.2}%", test_acc * 100.0);
    eprintln!();

    // ---- Training ----
    let mut flat_params = agent.flatten_params();
    let evaluator = LossEvaluator::new(&mut agent, &train_data);
    let loss_fn = |params: &[f64]| -> f64 { evaluator.evaluate(params) };

    // Use evolutionary optimizer — 1 evaluation per step, accepts improvements
    let mut optimizer = EvolutionaryOptimizer::new(0.1, 2.0, 0.999);
    optimizer.train(&mut flat_params, &loss_fn, 20000, &mut rng, 1000);

    eprintln!();

    // ---- Restore trained parameters ----
    let mut trained_agent = ToyAgent::new(vocab.len(), num_operators, &mut rng);
    trained_agent.restore_params(&flat_params);

    // ---- Evaluation ----
    let final_train_loss = trained_agent.compute_loss(&train_data);
    let final_train_acc = trained_agent.compute_accuracy(&train_data);
    let final_test_loss = trained_agent.compute_loss(&test_data);
    let final_test_acc = trained_agent.compute_accuracy(&test_data);

    eprintln!("=== Results ===");
    eprintln!(
        "Train loss: {:.6}  |  Train acc: {:.2}%",
        final_train_loss,
        final_train_acc * 100.0
    );
    eprintln!(
        "Test loss:  {:.6}  |  Test acc:  {:.2}%",
        final_test_loss,
        final_test_acc * 100.0
    );
    eprintln!();

    // ---- Per-command breakdown on test set ----
    eprintln!("=== Test Command Breakdown ===");
    for test_ex in &test_examples {
        let features = ling::toy::data::encode_command(&test_ex.command, &vocab);
        let mut correct = 0;
        let mut total = 0;
        for dir_idx in 0..4 {
            let current_state = AmplitudeState::pure(dir_idx, 4);
            let output = trained_agent.forward(&features, &current_state);
            let predicted = output.argmax();
            let expected = test_ex.action.apply(ling::toy::data::Direction::from_index(dir_idx));
            if predicted == expected.to_index() {
                correct += 1;
            }
            total += 1;

            let probs = output.probabilities();
            let expected_p = probs[expected.to_index()];
            eprintln!(
                "  {:10} | {} → {} (pred={:?}, p={:.2})",
                test_ex.command,
                ling::toy::data::Direction::from_index(dir_idx).name(),
                expected.name(),
                ling::toy::data::Direction::from_index(predicted).name(),
                expected_p,
            );
        }
        eprintln!(
            "  {:10} | total: {}/{} correct ({:.0}%)",
            test_ex.command,
            correct,
            total,
            (correct as f64 / total as f64) * 100.0
        );
        eprintln!();
    }

    Ok(())
}
