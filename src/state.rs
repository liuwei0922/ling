//! State-space primitives.

pub mod link;
pub mod space;

pub use link::{LinkId, Type1Link, Type2Link};
pub use space::{Neighborhood, NeighborhoodId, NeighborhoodRef, State, StateId, StateSpace};
