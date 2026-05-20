use num_complex::Complex;
use rand::Rng;

/// How complex amplitudes are converted to output probabilities or decisions.
///
/// Different activation functions model different "observation" regimes:
/// - BornRule: standard quantum measurement, P(i) = |α_i|²
/// - SigmoidGate: soft gating, σ(|α_i|)
/// - SoftmaxGate: temperature-controlled competition
pub trait Activation: Send + Sync {
    /// Convert a vector of complex amplitudes to a probability distribution.
    fn activate(&self, amplitudes: &[Complex<f64>]) -> Vec<f64>;

    /// Sample an index from the activated distribution.
    fn sample(&self, amplitudes: &[Complex<f64>], rng: &mut impl Rng) -> usize {
        let probs = self.activate(amplitudes);
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r <= cumulative {
                return i;
            }
        }
        probs.len() - 1
    }
}

/// Standard Born rule: P(i) = |α_i|² / Σ|α_j|².
///
/// This is the quantum measurement postulate — probability is the
/// squared modulus of the amplitude.
#[derive(Clone)]
pub struct BornRule;

impl Activation for BornRule {
    fn activate(&self, amplitudes: &[Complex<f64>]) -> Vec<f64> {
        let norm_sq: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if norm_sq == 0.0 {
            return vec![0.0; amplitudes.len()];
        }
        amplitudes.iter().map(|a| a.norm_sqr() / norm_sq).collect()
    }
}

/// Sigmoid gating: P(i) ∝ σ(|α_i| - threshold), where σ is the logistic function.
///
/// Acts as a soft threshold — only amplitudes above the threshold contribute
/// significantly. Useful for sparse, selective activation.
#[derive(Clone)]
pub struct SigmoidGate {
    pub threshold: f64,
    pub temperature: f64,
}

impl SigmoidGate {
    pub fn new(threshold: f64, temperature: f64) -> Self {
        Self {
            threshold,
            temperature,
        }
    }
}

impl Activation for SigmoidGate {
    fn activate(&self, amplitudes: &[Complex<f64>]) -> Vec<f64> {
        let mut raw: Vec<f64> = amplitudes
            .iter()
            .map(|a| {
                let mag = a.norm();
                1.0 / (1.0 + (-(mag - self.threshold) / self.temperature).exp())
            })
            .collect();
        let sum: f64 = raw.iter().sum();
        if sum > 0.0 {
            for v in &mut raw {
                *v /= sum;
            }
        }
        raw
    }
}

/// Softmax over magnitudes: P(i) = exp(|α_i|/τ) / Σ exp(|α_j|/τ).
///
/// Temperature τ controls sharpness: τ → 0 gives argmax, τ → ∞ gives uniform.
#[derive(Clone)]
pub struct MagnitudeSoftmax {
    pub temperature: f64,
}

impl MagnitudeSoftmax {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }
}

impl Activation for MagnitudeSoftmax {
    fn activate(&self, amplitudes: &[Complex<f64>]) -> Vec<f64> {
        let max_mag = amplitudes
            .iter()
            .map(|a| a.norm())
            .fold(0.0f64, f64::max);
        let mut exps: Vec<f64> = amplitudes
            .iter()
            .map(|a| ((a.norm() - max_mag) / self.temperature).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        if sum > 0.0 {
            for v in &mut exps {
                *v /= sum;
            }
        }
        exps
    }
}
