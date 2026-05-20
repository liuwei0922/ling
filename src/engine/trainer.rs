use rand::Rng;

/// Simple gradient-free optimizer suited for small parameter spaces.
///
/// Uses finite-difference gradient estimation with momentum.
/// This is intentionally simple — the focus is on the representation,
/// not the optimization algorithm.
pub struct Trainer {
    pub learning_rate: f64,
    pub epsilon: f64, // finite-difference step
    pub momentum: f64,
    velocity: Vec<f64>,
}

impl Trainer {
    pub fn new(learning_rate: f64, epsilon: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            epsilon,
            momentum,
            velocity: Vec::new(),
        }
    }

    /// Estimate gradient via finite differences and update parameters.
    pub fn step<F>(&mut self, params: &mut [f64], loss_fn: &F)
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = params.len();
        if self.velocity.len() != n {
            self.velocity = vec![0.0; n];
        }

        let mut grad = vec![0.0; n];

        for i in 0..n {
            let original = params[i];
            params[i] = original + self.epsilon;
            let plus = loss_fn(params);
            params[i] = original - self.epsilon;
            let minus = loss_fn(params);
            params[i] = original; // restore

            grad[i] = (plus - minus) / (2.0 * self.epsilon);
        }

        // Update with momentum
        for i in 0..n {
            self.velocity[i] = self.momentum * self.velocity[i] + self.learning_rate * grad[i];
            params[i] -= self.velocity[i];
        }
    }

    /// Run a full training loop.
    pub fn train<F>(
        &mut self,
        params: &mut [f64],
        loss_fn: &F,
        epochs: usize,
        print_interval: usize,
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut losses = Vec::with_capacity(epochs / print_interval.max(1) + 1);
        for epoch in 0..epochs {
            self.step(params, loss_fn);
            if epoch % print_interval == 0 || epoch == epochs - 1 {
                let loss = loss_fn(params);
                losses.push(loss);
                if print_interval > 0 {
                    eprintln!("epoch {epoch}/{epochs}, loss = {loss:.6}");
                }
            }
        }
        losses
    }
}

/// Evolutionary strategy optimizer.
///
/// Alternative to finite-difference: perturbs parameters randomly,
/// accepts if loss improves, occasionally accepts worse solutions
/// (simulated annealing) to escape local minima.
pub struct EvolutionaryOptimizer {
    pub step_size: f64,
    pub temperature: f64, // annealing temperature
    pub decay: f64,       // temperature decay per step
}

impl EvolutionaryOptimizer {
    pub fn new(step_size: f64, temperature: f64, decay: f64) -> Self {
        Self {
            step_size,
            temperature,
            decay,
        }
    }

    /// One step of evolutionary optimization.
    pub fn step<F>(
        &mut self,
        params: &mut [f64],
        loss_fn: &F,
        rng: &mut impl Rng,
    ) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let current_loss = loss_fn(params);

        // Generate random perturbation
        let mut perturbation = vec![0.0; params.len()];
        for p in &mut perturbation {
            *p = rng.gen_range(-self.step_size..self.step_size);
        }

        // Apply
        for (p, delta) in params.iter_mut().zip(&perturbation) {
            *p += delta;
        }

        let new_loss = loss_fn(params);

        if new_loss < current_loss {
            // Accept improvement
            self.temperature *= self.decay;
            new_loss
        } else {
            // Maybe accept anyway (Boltzmann exploration)
            let acceptance = ((current_loss - new_loss) / self.temperature.max(1e-10)).exp();
            if rng.gen::<f64>() < acceptance {
                self.temperature *= self.decay;
                new_loss
            } else {
                // Revert
                for (p, delta) in params.iter_mut().zip(&perturbation) {
                    *p -= delta;
                }
                current_loss
            }
        }
    }

    pub fn train<F>(
        &mut self,
        params: &mut [f64],
        loss_fn: &F,
        epochs: usize,
        rng: &mut impl Rng,
        print_interval: usize,
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut losses = Vec::with_capacity(epochs / print_interval.max(1) + 1);
        for epoch in 0..epochs {
            let loss = self.step(params, loss_fn, rng);
            if epoch % print_interval == 0 || epoch == epochs - 1 {
                losses.push(loss);
                if print_interval > 0 {
                    eprintln!("epoch {epoch}/{epochs}, loss = {loss:.6}, temp = {:.4}", self.temperature);
                }
            }
        }
        losses
    }
}
