//! Amplitude distributions and selection functions.

use rand::Rng;
use std::collections::BTreeMap;

/// A probability assigned to one item after measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probability<T> {
    /// Measured item.
    pub item: T,
    /// Normalized probability.
    pub probability: f64,
}

/// How to pick an item from a probability distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    /// Deterministic: pick the highest-probability item (tie-break by smaller item).
    Argmax,
    /// Pick uniformly at random among items that share the highest probability.
    RandomTie,
    /// Sample from the full distribution proportionally to probability.
    Sample,
}

/// Sparse real-amplitude distribution.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AmplitudeDistribution<T> {
    amplitudes: BTreeMap<T, f64>,
}

impl<T> AmplitudeDistribution<T>
where
    T: Copy + Ord,
{
    /// Create an empty amplitude distribution.
    pub fn new() -> Self {
        Self {
            amplitudes: BTreeMap::new(),
        }
    }

    /// Add one amplitude contribution to an item.
    pub fn add(&mut self, item: T, amplitude: f64) {
        *self.amplitudes.entry(item).or_insert(0.0) += amplitude;
    }

    /// Convert amplitudes to probabilities with the Born rule.
    ///
    /// Returns an empty vector when the total squared amplitude is zero or
    /// negative, which signals that nothing was learned or amplitudes fully
    /// cancelled.
    pub fn probabilities(&self) -> Vec<Probability<T>> {
        let total = self
            .amplitudes
            .values()
            .map(|amplitude| amplitude * amplitude)
            .sum::<f64>();
        if total <= 0.0 {
            return Vec::new();
        }

        self.amplitudes
            .iter()
            .map(|(item, amplitude)| Probability {
                item: *item,
                probability: amplitude * amplitude / total,
            })
            .collect()
    }

    /// Select an item using the given mode, optionally restricted to a subset.
    pub fn select(
        &self,
        mode: SelectionMode,
        allowed_items: Option<&[T]>,
        rng: &mut impl Rng,
    ) -> Option<T> {
        match mode {
            SelectionMode::Argmax => self.select_argmax(allowed_items),
            SelectionMode::RandomTie => self.select_random_tie(allowed_items, rng),
            SelectionMode::Sample => self.sample(allowed_items, rng),
        }
    }

    /// Select the highest-probability item (deterministic tie-break: smaller item wins).
    pub fn select_argmax(&self, allowed_items: Option<&[T]>) -> Option<T> {
        self.probabilities()
            .into_iter()
            .filter(|entry| {
                allowed_items
                    .map(|allowed| allowed.contains(&entry.item))
                    .unwrap_or(true)
            })
            .max_by(|left, right| {
                left.probability
                    .partial_cmp(&right.probability)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.item.cmp(&right.item).reverse())
            })
            .map(|entry| entry.item)
    }

    /// Argmax selection that breaks ties uniformly at random.
    pub fn select_random_tie(&self, allowed_items: Option<&[T]>, rng: &mut impl Rng) -> Option<T> {
        let probs = self.probabilities();
        if probs.is_empty() {
            return None;
        }

        let filtered: Vec<Probability<T>> = probs
            .into_iter()
            .filter(|entry| {
                allowed_items
                    .map(|allowed| allowed.contains(&entry.item))
                    .unwrap_or(true)
            })
            .collect();

        if filtered.is_empty() {
            return None;
        }

        let max_prob = filtered
            .iter()
            .map(|entry| entry.probability)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let tied: Vec<T> = filtered
            .into_iter()
            .filter(|entry| entry.probability == max_prob)
            .map(|entry| entry.item)
            .collect();

        if tied.is_empty() {
            None
        } else {
            let idx = rng.gen_range(0..tied.len());
            Some(tied[idx])
        }
    }

    /// Sample an item proportionally to its probability.
    pub fn sample(&self, allowed_items: Option<&[T]>, rng: &mut impl Rng) -> Option<T> {
        let probs = self.probabilities();
        if probs.is_empty() {
            return None;
        }

        let filtered: Vec<Probability<T>> = probs
            .into_iter()
            .filter(|entry| {
                allowed_items
                    .map(|allowed| allowed.contains(&entry.item))
                    .unwrap_or(true)
            })
            .collect();

        if filtered.is_empty() {
            return None;
        }

        let threshold: f64 = rng.gen();
        let mut cumulative = 0.0;
        for entry in &filtered {
            cumulative += entry.probability;
            if threshold <= cumulative {
                return Some(entry.item);
            }
        }

        // Floating-point rounding: return the last item.
        filtered.last().map(|entry| entry.item)
    }
}
