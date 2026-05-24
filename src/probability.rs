//! Amplitude distributions and selection functions.

use std::cmp::Ordering;
use std::collections::BTreeMap;

/// A probability assigned to one item after measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probability<T> {
    /// Measured item.
    pub item: T,
    /// Normalized probability.
    pub probability: f64,
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

    /// Select the highest-probability item, optionally restricted to a subset.
    pub fn select_argmax(&self, allowed_items: Option<&[T]>) -> Option<T> {
        self.probabilities()
            .into_iter()
            .filter(|entry| {
                allowed_items
                    .map(|allowed| allowed.contains(&entry.item))
                    .unwrap_or(true)
            })
            .max_by(
                |left, right| match left.probability.total_cmp(&right.probability) {
                    Ordering::Equal => right.item.cmp(&left.item),
                    ordering => ordering,
                },
            )
            .map(|entry| entry.item)
    }
}
