/// A label in the concept space.
///
/// Each label represents a discovered "concept" — an element of S or a point
/// in the function space scr(F). Labels carry metadata for pruning decisions.
#[derive(Clone, Debug)]
pub struct Label {
    pub id: usize,
    /// Activation frequency count (how often this concept is used).
    pub frequency: usize,
    /// Average activation entropy across uses (lower = more focused).
    pub entropy: f64,
}

/// Pruning configuration for concept space reduction.
///
/// Three criteria:
/// - **Frequency**: remove labels rarely activated
/// - **Overlap**: merge labels that are too similar
/// - **Entropy**: remove labels that are too diffuse
#[derive(Clone, Debug)]
pub struct PruningConfig {
    /// Minimum frequency to keep a label.
    pub min_frequency: usize,
    /// Maximum similarity (above this, the more frequent label absorbs the other).
    pub max_similarity: f64,
    /// Maximum entropy (above this, the label is too diffuse).
    pub max_entropy: f64,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            min_frequency: 1,
            max_similarity: 0.85,
            max_entropy: 0.95,
        }
    }
}

/// A dynamic concept space with pruning heuristics.
///
/// Concepts start as an over-generated set of candidate labels. Over time,
/// pruning removes labels that are:
/// - Low-frequency (rarely activated)
/// - Overlapping with other labels (redundant)
/// - Too diffuse (high entropy — not a well-defined concept)
///
/// The surviving labels represent the "concept space" scr(F).
#[derive(Clone, Debug)]
pub struct ConceptSpace {
    pub labels: Vec<Label>,
    /// For overlap detection: similarity matrix between labels.
    /// similarity[i][j] ≈ how similar label i and j are.
    pub similarity: Vec<Vec<f64>>,
    pub next_id: usize,
}

impl ConceptSpace {
    pub fn new(initial_size: usize) -> Self {
        let labels: Vec<Label> = (0..initial_size)
            .map(|i| Label {
                id: i,
                frequency: 0,
                entropy: 0.0,
            })
            .collect();
        let n = labels.len();
        Self {
            labels,
            similarity: vec![vec![0.0; n]; n],
            next_id: initial_size,
        }
    }

    pub fn size(&self) -> usize {
        self.labels.len()
    }

    /// Add a new candidate concept label.
    pub fn add_label(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.labels.push(Label {
            id,
            frequency: 0,
            entropy: 0.0,
        });
        // Extend similarity matrix
        let n = self.labels.len();
        for row in &mut self.similarity {
            row.push(0.0);
        }
        self.similarity.push(vec![0.0; n]);
        id
    }

    /// Record activation of a label and update its entropy.
    pub fn record_activation(&mut self, label_idx: usize, activation_entropy: f64) {
        if label_idx < self.labels.len() {
            self.labels[label_idx].frequency += 1;
            let count = self.labels[label_idx].frequency as f64;
            self.labels[label_idx].entropy =
                self.labels[label_idx].entropy * (1.0 - 1.0 / count) + activation_entropy / count;
        }
    }

    /// Update similarity between two labels.
    pub fn update_similarity(&mut self, i: usize, j: usize, sim: f64) {
        if i < self.similarity.len() && j < self.similarity.len() {
            self.similarity[i][j] = sim;
            self.similarity[j][i] = sim;
        }
    }

    /// Prune the concept space using all three criteria.
    ///
    /// Returns a mapping from old label indices to new label indices
    /// (None means the label was removed).
    pub fn prune(&mut self, config: &PruningConfig) -> Vec<Option<usize>> {
        let n = self.labels.len();
        if n == 0 {
            return Vec::new();
        }

        let mut keep = vec![true; n];
        let mut merge_target: Vec<Option<usize>> = vec![None; n];

        // 1. Frequency pruning
        for i in 0..n {
            if self.labels[i].frequency < config.min_frequency && n > 1 {
                keep[i] = false;
            }
        }

        // 2. Overlap pruning: merge similar labels
        for i in 0..n {
            if !keep[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !keep[j] {
                    continue;
                }
                if self.similarity[i][j] > config.max_similarity {
                    if self.labels[i].frequency >= self.labels[j].frequency {
                        keep[j] = false;
                        merge_target[j] = Some(i);
                    } else {
                        keep[i] = false;
                        merge_target[i] = Some(j);
                        break;
                    }
                }
            }
        }

        // 3. Entropy pruning
        for i in 0..n {
            if !keep[i] {
                continue;
            }
            if self.labels[i].entropy > config.max_entropy && n > 1 {
                keep[i] = false;
            }
        }

        // Build mapping
        let mut old_to_new: Vec<Option<usize>> = vec![None; n];
        let mut new_idx = 0;
        for i in 0..n {
            if keep[i] {
                old_to_new[i] = Some(new_idx);
                new_idx += 1;
            } else if let Some(target) = merge_target[i] {
                old_to_new[i] = old_to_new[target];
            }
        }

        // Compact the space
        let new_labels: Vec<Label> = self
            .labels
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, l)| l.clone())
            .collect();

        let new_n = new_labels.len();
        let mut new_similarity = vec![vec![0.0; new_n]; new_n];
        let mut row = 0;
        for i in 0..n {
            if keep[i] {
                let mut col = 0;
                for j in 0..n {
                    if keep[j] {
                        new_similarity[row][col] = self.similarity[i][j];
                        col += 1;
                    }
                }
                row += 1;
            }
        }

        self.labels = new_labels;
        self.similarity = new_similarity;

        old_to_new
    }

    pub fn active_label_ids(&self) -> Vec<usize> {
        self.labels.iter().map(|l| l.id).collect()
    }
}
