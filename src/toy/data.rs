
/// Cardinal directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
}

impl Direction {
    pub fn from_index(i: usize) -> Self {
        match i % 4 {
            0 => Direction::North,
            1 => Direction::East,
            2 => Direction::South,
            3 => Direction::West,
            _ => unreachable!(),
        }
    }

    pub fn to_index(&self) -> usize {
        *self as usize
    }

    pub fn name(&self) -> &'static str {
        match self {
            Direction::North => "北",
            Direction::East => "东",
            Direction::South => "南",
            Direction::West => "西",
        }
    }

    pub fn all() -> [Direction; 4] {
        [Direction::North, Direction::East, Direction::South, Direction::West]
    }
}

/// A labeled action: one of the supported command types.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    FaceNorth,
    FaceEast,
    FaceSouth,
    FaceWest,
    TurnRight,
    TurnLeft,
    TurnAround,
}

impl Action {
    /// Apply this action given a current direction.
    pub fn apply(&self, current: Direction) -> Direction {
        match self {
            Action::FaceNorth => Direction::North,
            Action::FaceEast => Direction::East,
            Action::FaceSouth => Direction::South,
            Action::FaceWest => Direction::West,
            Action::TurnRight => Direction::from_index(current.to_index() + 1),
            Action::TurnLeft => Direction::from_index(current.to_index() + 3), // -1 mod 4
            Action::TurnAround => Direction::from_index(current.to_index() + 2),
        }
    }

    /// The target direction this action ultimately points to.
    /// For turn actions, returns None since result depends on current state.
    pub fn target_direction(&self) -> Option<Direction> {
        match self {
            Action::FaceNorth => Some(Direction::North),
            Action::FaceEast => Some(Direction::East),
            Action::FaceSouth => Some(Direction::South),
            Action::FaceWest => Some(Direction::West),
            _ => None,
        }
    }
}

/// A command string paired with its intended action.
#[derive(Debug, Clone)]
pub struct CommandExample {
    pub command: String,
    pub action: Action,
}

/// Generate the toy dataset.
///
/// Returns (training_examples, test_examples).
pub fn generate_dataset() -> (Vec<CommandExample>, Vec<CommandExample>) {
    // Training: each command type has a subset of possible phrasings
    let training = vec![
        // Face commands — explicit "面向X" pattern
        CommandExample { command: "面向北".into(), action: Action::FaceNorth },
        CommandExample { command: "面向东".into(), action: Action::FaceEast },
        CommandExample { command: "面向南".into(), action: Action::FaceSouth },
        CommandExample { command: "面向西".into(), action: Action::FaceWest },
        // Face commands — "向X走" pattern
        CommandExample { command: "向北走".into(), action: Action::FaceNorth },
        CommandExample { command: "向东走".into(), action: Action::FaceEast },
        CommandExample { command: "向南走".into(), action: Action::FaceSouth },
        CommandExample { command: "向西走".into(), action: Action::FaceWest },
        // Turn commands — full form
        CommandExample { command: "向右转".into(), action: Action::TurnRight },
        CommandExample { command: "向左转".into(), action: Action::TurnLeft },
        CommandExample { command: "转过身".into(), action: Action::TurnAround },
        // Turn commands — abbreviated form
        CommandExample { command: "右转".into(), action: Action::TurnRight },
        CommandExample { command: "左转".into(), action: Action::TurnLeft },
    ];

    // Test: unseen command phrasings (but same direction words)
    let test = vec![
        // Novel "看向X" pattern — "看" was never seen in training
        CommandExample { command: "看向北".into(), action: Action::FaceNorth },
        CommandExample { command: "看向东".into(), action: Action::FaceEast },
        CommandExample { command: "看向南".into(), action: Action::FaceSouth },
        CommandExample { command: "看向西".into(), action: Action::FaceWest },
        // Abbreviated "面X" pattern
        CommandExample { command: "面北".into(), action: Action::FaceNorth },
        CommandExample { command: "面东".into(), action: Action::FaceEast },
        // "朝X" pattern — "朝" never seen
        CommandExample { command: "朝北".into(), action: Action::FaceNorth },
        CommandExample { command: "朝东".into(), action: Action::FaceEast },
        // "向X" bare form
        CommandExample { command: "向北".into(), action: Action::FaceNorth },
        CommandExample { command: "向南".into(), action: Action::FaceSouth },
        // Turn — "掉头" (novel word, seen chars)
        CommandExample { command: "掉头".into(), action: Action::TurnAround },
        // "转过来" (novel phrase)
        CommandExample { command: "转过来".into(), action: Action::TurnAround },
    ];

    (training, test)
}

/// Build a character vocabulary from all examples.
pub fn build_vocab(training: &[CommandExample], test: &[CommandExample]) -> Vec<char> {
    let mut chars: Vec<char> = training
        .iter()
        .flat_map(|ex| ex.command.chars())
        .collect();
    // Also include test chars so feature space is consistent
    for ex in test {
        for c in ex.command.chars() {
            if !chars.contains(&c) {
                chars.push(c);
            }
        }
    }
    chars.sort();
    chars.dedup();
    chars
}

/// Encode a command string as a bag-of-characters vector.
pub fn encode_command(command: &str, vocab: &[char]) -> Vec<f64> {
    vocab
        .iter()
        .map(|&c| {
            if command.contains(c) {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Generate all possible (current_direction, command_example, target_direction) triples.
pub fn generate_examples(
    examples: &[CommandExample],
    vocab: &[char],
) -> Vec<(Vec<f64>, usize, usize)> {
    // Feature vector, current direction index, target direction index
    let mut result = Vec::new();
    for ex in examples {
        let features = encode_command(&ex.command, vocab);
        for current_dir in Direction::all() {
            let target = ex.action.apply(current_dir);
            result.push((features.clone(), current_dir.to_index(), target.to_index()));
        }
    }
    result
}
