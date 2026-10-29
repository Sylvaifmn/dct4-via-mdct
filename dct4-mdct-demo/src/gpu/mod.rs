//! GPU adaptation (§6): buffer pooling, stream partial order, and closed-form projections.
//!
//! The design's position is that porting this kit to a GPU is *not* an RTL translation
//! problem. Nothing about the folding theorem, the backend contract or the post-processing
//! changes on a GPU; what changes is the *scheduling*:
//!
//! * a per-block working set of two buffers (the folded `2N` word and the `N`-word output),
//! * a pool of `K` slots that bounds how many blocks may be in flight,
//! * `M` streams, each of which runs `FOLD -> MDCT -> POST` strictly in order,
//! * a partial order that the scheduler must respect (and that is exposed as a DAG).
//!
//! [`model::project`] gives the closed form; [`stream::simulate`] gives the event-driven
//! replay of the same plan. The test suite asserts the two agree to within one block latency,
//! which is what makes the closed form trustworthy as a design-time tool.
//!
//! ```text
//!   stream 0:  FOLD(0) -> MDCT(0) -> POST(0)      FOLD(2) -> ...
//!   stream 1:  FOLD(1) -> MDCT(1) -> POST(1)      FOLD(3) -> ...
//!                 ^ capacity = min(M, K/2) blocks in flight
//! ```

pub mod model;
pub mod pool;
pub mod stream;

pub use model::{bandwidth_cost, project, GpuProjection};
pub use pool::{BufferPool, PoolStats};
pub use stream::{
    simulate, BlockTiming, DependencyGraph, Node, Simulation, Stage, StageCost, StreamPlan,
};

use std::fmt;

/// A scheduling-plan or cost-model error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// `M = 0` streams cannot schedule anything.
    ZeroStreams,
    /// `B = 0` blocks is a no-op plan.
    ZeroBlocks,
    /// `K < 2` slots cannot hold one in-flight block (the GPU analogue of eq. 3's guard).
    SlotsTooSmall {
        /// Configured slot count.
        slots: usize,
        /// Slots required per in-flight block.
        per_block: usize,
    },
    /// A stage cost was negative, zero or non-finite.
    InvalidCost {
        /// The offending stage.
        stage: &'static str,
    },
    /// A model parameter was outside its domain.
    InvalidParameter {
        /// What was wrong.
        what: &'static str,
    },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanError::ZeroStreams => f.write_str("at least one stream is required"),
            PlanError::ZeroBlocks => f.write_str("at least one block is required"),
            PlanError::SlotsTooSmall { slots, per_block } => write!(
                f,
                "pool has {slots} slot(s) but an in-flight block needs {per_block}"
            ),
            PlanError::InvalidCost { stage } => {
                write!(f, "stage cost for {stage} must be finite and > 0")
            }
            PlanError::InvalidParameter { what } => write!(f, "invalid model parameter: {what}"),
        }
    }
}

impl std::error::Error for PlanError {}
