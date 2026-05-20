use num_complex::Complex;
use ling::core::amplitude::{AmplitudeState, Operator, OperatorSuperposition};
use ling::core::activation::{Activation, BornRule, SigmoidGate, MagnitudeSoftmax};

#[test]
fn test_amplitude_state_pure() {
    let state = AmplitudeState::pure(2, 5);
    assert_eq!(state.size, 5);
    assert_eq!(state.amplitudes[2], Complex::new(1.0, 0.0));
    assert!(state.amplitudes.iter().enumerate().all(|(i, a)| {
        if i == 2 { a.norm_sqr() == 1.0 } else { a.norm_sqr() == 0.0 }
    }));
}

#[test]
fn test_amplitude_state_probabilities() {
    let mut state = AmplitudeState::new(3);
    state.amplitudes[0] = Complex::new(1.0, 0.0);
    let probs = state.probabilities();
    assert!((probs[0] - 1.0).abs() < 1e-10);
    assert!((probs[1] - 0.0).abs() < 1e-10);
    assert!((probs[2] - 0.0).abs() < 1e-10);
}

#[test]
fn test_amplitude_state_normalize() {
    let mut state = AmplitudeState::new(4);
    state.amplitudes[0] = Complex::new(2.0, 0.0);
    state.amplitudes[1] = Complex::new(0.0, 2.0);
    state.normalize();
    let norm_sq: f64 = state.amplitudes.iter().map(|a| a.norm_sqr()).sum();
    assert!((norm_sq - 1.0).abs() < 1e-10);
}

#[test]
fn test_operator_apply() {
    let state = AmplitudeState::pure(0, 2);
    let mut op = Operator::new(2);
    // Swap operator: |0⟩ ↔ |1⟩
    op.matrix[0][1] = Complex::new(1.0, 0.0);
    op.matrix[1][0] = Complex::new(1.0, 0.0);

    let result = op.apply(&state);
    assert!((result.amplitudes[0].norm_sqr() - 0.0).abs() < 1e-10);
    assert!((result.amplitudes[1].norm_sqr() - 1.0).abs() < 1e-10);
}

#[test]
fn test_operator_identity() {
    let state = AmplitudeState::pure(1, 3);
    let ident = Operator::identity(3);
    let result = ident.apply(&state);
    assert!((result.amplitudes[1].norm_sqr() - 1.0).abs() < 1e-10);
}

#[test]
fn test_superposition_interference_constructive() {
    let mut op1 = Operator::new(1);
    op1.matrix[0][0] = Complex::new(2.0, 0.0);
    let mut op2 = Operator::new(1);
    op2.matrix[0][0] = Complex::new(1.0, 0.0);

    let state = AmplitudeState::pure(0, 1);
    let sup = OperatorSuperposition::new(vec![op1, op2]);

    // Both positive: 1*2 + 1*1 = 3
    let result = sup.apply_superposition_real(&[1.0, 1.0], &state);
    assert!((result.amplitudes[0].re - 3.0).abs() < 1e-10);
}

#[test]
fn test_superposition_interference_destructive() {
    let mut op1 = Operator::new(1);
    op1.matrix[0][0] = Complex::new(2.0, 0.0);
    let mut op2 = Operator::new(1);
    op2.matrix[0][0] = Complex::new(1.0, 0.0);

    let state = AmplitudeState::pure(0, 1);
    let sup = OperatorSuperposition::new(vec![op1, op2]);

    // Opposite signs: 1*2 + (-1)*1 = 1
    let result = sup.apply_superposition_real(&[1.0, -1.0], &state);
    assert!((result.amplitudes[0].re - 1.0).abs() < 1e-10);
}

#[test]
fn test_superposition_interference_cancel() {
    let mut op1 = Operator::new(1);
    op1.matrix[0][0] = Complex::new(1.0, 0.0);
    let mut op2 = Operator::new(1);
    op2.matrix[0][0] = Complex::new(1.0, 0.0);

    let state = AmplitudeState::pure(0, 1);
    let sup = OperatorSuperposition::new(vec![op1, op2]);

    // Exact cancellation: 1*1 + (-1)*1 = 0
    let result = sup.apply_superposition_real(&[1.0, -1.0], &state);
    assert!((result.amplitudes[0].re - 0.0).abs() < 1e-10);
}

#[test]
fn test_born_rule_activation() {
    let born = BornRule;
    let amplitudes = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
    ];
    let probs = born.activate(&amplitudes);
    assert!((probs[0] - 0.5).abs() < 1e-10);
    assert!((probs[1] - 0.5).abs() < 1e-10);
}

#[test]
fn test_sigmoid_gate() {
    let gate = SigmoidGate::new(0.5, 0.1);
    let amplitudes = vec![
        Complex::new(1.0, 0.0), // mag=1.0 > threshold
        Complex::new(0.1, 0.0), // mag=0.1 < threshold
    ];
    let probs = gate.activate(&amplitudes);
    assert!(probs[0] > probs[1]);
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

#[test]
fn test_magnitude_softmax() {
    let sm = MagnitudeSoftmax::new(0.5);
    let amplitudes = vec![
        Complex::new(2.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    let probs = sm.activate(&amplitudes);
    assert!(probs[0] > probs[1]);
    assert!(probs[1] > probs[2]);
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

#[test]
fn test_operator_superposition_real_coeffs() {
    let n = 2;
    let mut op_a = Operator::new(n);
    op_a.matrix[0][0] = Complex::new(1.0, 0.0); // identity-like
    op_a.matrix[1][1] = Complex::new(1.0, 0.0);

    let mut op_b = Operator::new(n);
    op_b.matrix[0][0] = Complex::new(0.0, 0.0);
    op_b.matrix[1][1] = Complex::new(0.0, 0.0);

    let sup = OperatorSuperposition::new(vec![op_a, op_b]);
    let state = AmplitudeState::pure(0, n);
    let result = sup.apply_superposition_real(&[0.5, 0.5], &state);
    assert_eq!(result.argmax(), 0);
}
