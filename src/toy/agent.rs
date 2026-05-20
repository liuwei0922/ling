use rand::Rng;
use std::cell::RefCell;

use crate::core::{
    amplitude::{AmplitudeState, Operator, OperatorSuperposition},
    activation::BornRule,
};

/// A toy agent that learns to interpret commands via amplitude propagation.
///
/// Architecture:
/// 1. Command → bag-of-chars feature vector f
/// 2. f → operator coefficients β = W·f (with tanh nonlinearity)
/// 3. O_eff = Σ β_k · O_k (coherent superposition)
/// 4. |ψ'⟩ = O_eff · |current_state⟩
/// 5. Output = Born rule probabilities over directions
///
/// The operators O_k represent learned "concepts" like face_towards,
/// turn_left, etc. The mapping W learns which command patterns activate
/// which concept operators.
#[derive(Clone)]
pub struct ToyAgent {
    /// Feature dimension (vocabulary size).
    pub feature_dim: usize,
    /// Number of label directions (always 4).
    pub num_labels: usize,
    /// Number of concept operators.
    pub num_operators: usize,
    /// Operators O_k — each is a num_labels × num_labels real matrix.
    /// Stored as flat Vec for easy optimization: [op0_flat, op1_flat, ...]
    pub operators: Vec<f64>,
    /// Mapping W: feature_dim × num_operators, stored row-major.
    /// Maps command features to operator coefficients.
    pub weights: Vec<f64>,
    /// Activation function for output.
    pub activation: BornRule,
}

impl ToyAgent {
    pub fn new(feature_dim: usize, num_operators: usize, rng: &mut impl Rng) -> Self {
        let num_labels = 4;
        let num_op_params = num_operators * num_labels * num_labels;
        let num_weight_params = feature_dim * num_operators;

        // Initialize operators with small random values
        let mut operators = Vec::with_capacity(num_op_params);
        for _ in 0..num_op_params {
            operators.push(rng.gen_range(-0.5..0.5));
        }

        // Initialize weights with small random values
        let mut weights = Vec::with_capacity(num_weight_params);
        for _ in 0..num_weight_params {
            weights.push(rng.gen_range(-0.3..0.3));
        }

        Self {
            feature_dim,
            num_labels,
            num_operators,
            operators,
            weights,
            activation: BornRule,
        }
    }

    /// Total number of trainable parameters.
    pub fn num_params(&self) -> usize {
        self.operators.len() + self.weights.len()
    }

    /// Flatten all parameters into a single slice.
    pub fn flatten_params(&self) -> Vec<f64> {
        let mut flat = self.operators.clone();
        flat.extend(&self.weights);
        flat
    }

    /// Restore parameters from a flat slice.
    pub fn restore_params(&mut self, flat: &[f64]) {
        let op_len = self.operators.len();
        self.operators.copy_from_slice(&flat[..op_len]);
        self.weights.copy_from_slice(&flat[op_len..]);
    }

    /// Get a specific operator matrix by index.
    pub fn get_operator(&self, idx: usize) -> Vec<Vec<f64>> {
        let n = self.num_labels;
        let start = idx * n * n;
        let mut mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                mat[i][j] = self.operators[start + i * n + j];
            }
        }
        mat
    }

    /// Forward pass: given features and current state, produce output distribution.
    pub fn forward(&self, features: &[f64], current_state: &AmplitudeState) -> AmplitudeState {
        // Step 1: Compute operator coefficients β = tanh(W · f)
        let mut coefficients = vec![0.0; self.num_operators];
        for k in 0..self.num_operators {
            let mut sum = 0.0;
            for f in 0..self.feature_dim {
                sum += self.weights[f * self.num_operators + k] * features[f];
            }
            coefficients[k] = sum.tanh();
        }

        // Step 2: Build OperatorSuperposition from current operator matrices
        let n = self.num_labels;
        let mut operators: Vec<Operator> = Vec::with_capacity(self.num_operators);
        for k in 0..self.num_operators {
            let mut mat = vec![vec![num_complex::Complex::new(0.0, 0.0); n]; n];
            let start = k * n * n;
            for i in 0..n {
                for j in 0..n {
                    mat[i][j] = num_complex::Complex::new(self.operators[start + i * n + j], 0.0);
                }
            }
            operators.push(Operator { matrix: mat, size: n });
        }

        // Step 3: Apply superposition to current state
        let superposition = OperatorSuperposition::new(operators);
        superposition.apply_superposition_real(&coefficients, current_state)
    }

    /// Predict direction given a command feature vector and current direction index.
    pub fn predict(&self, features: &[f64], current_dir: usize) -> usize {
        let state = AmplitudeState::pure(current_dir, self.num_labels);
        let output = self.forward(features, &state);
        output.argmax()
    }

    /// Compute cross-entropy loss over a batch of examples.
    pub fn compute_loss(
        &self,
        examples: &[(Vec<f64>, usize, usize)],
    ) -> f64 {
        let mut total_loss = 0.0;
        let mut count = 0;
        for (features, current, target) in examples {
            let state = AmplitudeState::pure(*current, self.num_labels);
            let output = self.forward(features, &state);
            let probs = output.probabilities();
            let p = probs[*target].max(1e-15);
            total_loss -= p.ln();
            count += 1;
        }
        total_loss / count as f64
    }

    /// Compute accuracy over a batch.
    pub fn compute_accuracy(
        &self,
        examples: &[(Vec<f64>, usize, usize)],
    ) -> f64 {
        if examples.is_empty() {
            return 0.0;
        }
        let correct: usize = examples
            .iter()
            .filter(|(features, current, target)| {
                self.predict(features, *current) == *target
            })
            .count();
        correct as f64 / examples.len() as f64
    }
}

/// Compute loss from flat params without cloning the agent.
///
/// Uses RefCell for interior mutability — swaps params in place,
/// computes loss, then restores. Much faster than cloning.
pub struct LossEvaluator<'a> {
    agent: RefCell<&'a mut ToyAgent>,
    examples: &'a [(Vec<f64>, usize, usize)],
}

impl<'a> LossEvaluator<'a> {
    pub fn new(agent: &'a mut ToyAgent, examples: &'a [(Vec<f64>, usize, usize)]) -> Self {
        Self {
            agent: RefCell::new(agent),
            examples,
        }
    }

    pub fn evaluate(&self, params: &[f64]) -> f64 {
        let mut agent = self.agent.borrow_mut();
        // Save and restore
        let saved = (agent.operators.clone(), agent.weights.clone());
        let op_len = agent.operators.len();
        agent.operators.copy_from_slice(&params[..op_len]);
        agent.weights.copy_from_slice(&params[op_len..]);
        let loss = agent.compute_loss(self.examples);
        agent.operators = saved.0;
        agent.weights = saved.1;
        loss
    }
}
