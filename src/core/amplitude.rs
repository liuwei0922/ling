use num_complex::Complex;
use rand::Rng;

/// A quantum-like state: vector of complex amplitudes over a label set.
///
/// Each label |i⟩ carries a complex amplitude α_i, with Σ|α_i|² = 1.
/// The probability of observing label i under Born rule is |α_i|².
#[derive(Clone, Debug)]
pub struct AmplitudeState {
    pub amplitudes: Vec<Complex<f64>>,
    pub size: usize,
}

impl AmplitudeState {
    pub fn new(size: usize) -> Self {
        Self {
            amplitudes: vec![Complex::new(0.0, 0.0); size],
            size,
        }
    }

    /// A "pure" state where all amplitude is concentrated on one label.
    pub fn pure(label: usize, size: usize) -> Self {
        let mut s = Self::new(size);
        s.amplitudes[label] = Complex::new(1.0, 0.0);
        s
    }

    /// Compute Born probabilities P(i) = |α_i|² / Σ|α_j|².
    pub fn probabilities(&self) -> Vec<f64> {
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if norm_sq == 0.0 {
            return vec![0.0; self.size];
        }
        self.amplitudes.iter().map(|a| a.norm_sqr() / norm_sq).collect()
    }

    /// Sample a label from the Born distribution.
    pub fn sample(&self, rng: &mut impl Rng) -> usize {
        let probs = self.probabilities();
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r <= cumulative {
                return i;
            }
        }
        self.size - 1
    }

    /// Normalize in-place so Σ|α_i|² = 1.
    pub fn normalize(&mut self) {
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if norm_sq > 0.0 {
            let norm = norm_sq.sqrt();
            for a in &mut self.amplitudes {
                *a /= norm;
            }
        }
    }

    /// The label with the highest probability.
    pub fn argmax(&self) -> usize {
        let probs = self.probabilities();
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// A linear operator represented as a complex matrix: |ψ'⟩ = O|ψ⟩.
#[derive(Clone, Debug)]
pub struct Operator {
    pub matrix: Vec<Vec<Complex<f64>>>,
    pub size: usize,
}

impl Operator {
    pub fn new(size: usize) -> Self {
        Self {
            matrix: vec![vec![Complex::new(0.0, 0.0); size]; size],
            size,
        }
    }

    /// Create a random operator with Gaussian entries.
    pub fn random(size: usize, rng: &mut impl Rng, scale: f64) -> Self {
        let mut m = Self::new(size);
        for i in 0..size {
            for j in 0..size {
                let re = rng.gen_range(-scale..scale);
                let im = rng.gen_range(-scale..scale);
                m.matrix[i][j] = Complex::new(re, im);
            }
        }
        m
    }

    /// Apply operator to state: |ψ'⟩ = O|ψ⟩.
    pub fn apply(&self, state: &AmplitudeState) -> AmplitudeState {
        let mut result = AmplitudeState::new(self.size);
        for i in 0..self.size {
            result.amplitudes[i] = (0..self.size)
                .map(|j| self.matrix[i][j] * state.amplitudes[j])
                .fold(Complex::new(0.0, 0.0), |acc, x| acc + x);
        }
        result
    }

    /// Identity operator.
    pub fn identity(size: usize) -> Self {
        let mut m = Self::new(size);
        for i in 0..size {
            m.matrix[i][i] = Complex::new(1.0, 0.0);
        }
        m
    }
}

/// Coherent superposition of operators: O_eff = Σ β_k O_k.
///
/// Each β_k is a complex amplitude; operators combine via constructive
/// and destructive interference before being applied to a state.
#[derive(Clone, Debug)]
pub struct OperatorSuperposition {
    pub operators: Vec<Operator>,
}

impl OperatorSuperposition {
    pub fn new(operators: Vec<Operator>) -> Self {
        Self { operators }
    }

    pub fn num_operators(&self) -> usize {
        self.operators.len()
    }

    pub fn size(&self) -> usize {
        self.operators.first().map(|o| o.size).unwrap_or(0)
    }

    /// Combine operators with given coefficients and apply to state.
    ///
    /// Returns O_eff|ψ⟩ where O_eff = Σ_k coeffs[k] · O_k.
    pub fn apply_superposition(
        &self,
        coeffs: &[Complex<f64>],
        state: &AmplitudeState,
    ) -> AmplitudeState {
        assert_eq!(coeffs.len(), self.operators.len());
        let mut result = AmplitudeState::new(self.size());
        for (k, op) in self.operators.iter().enumerate() {
            let transformed = op.apply(state);
            for i in 0..self.size() {
                result.amplitudes[i] = result.amplitudes[i] + coeffs[k] * transformed.amplitudes[i];
            }
        }
        result
    }

    /// Same but with real-valued coefficients (imaginary part = 0).
    pub fn apply_superposition_real(
        &self,
        coeffs: &[f64],
        state: &AmplitudeState,
    ) -> AmplitudeState {
        let complex_coeffs: Vec<Complex<f64>> =
            coeffs.iter().map(|&c| Complex::new(c, 0.0)).collect();
        self.apply_superposition(&complex_coeffs, state)
    }
}
