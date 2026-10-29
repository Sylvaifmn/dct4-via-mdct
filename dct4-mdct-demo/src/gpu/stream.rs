//! The stream partial order and the event-driven replay (§6).
//!
//! ## The partial order
//!
//! Three kinds of precedence constrain a schedule:
//!
//! 1. **Intra-block stage order:** `FOLD(b) < MDCT(b) < POST(b)`.
//! 2. **Stream serialisation:** block `b` and block `b + M` share a stream, so
//!    `FOLD(b) < FOLD(b + M)`.
//! 3. **Buffer reuse:** with `c = min(M, K/2)` blocks in flight, `POST(b) < FOLD(b + c)`.
//!
//! [`DependencyGraph`] materialises these as a DAG so a scheduler (or a reviewer) can inspect
//! them; [`DependencyGraph::critical_path`] gives the DAG lower bound on the makespan.
//!
//! ## The resource model used for replay
//!
//! * the MDCT backend is a **single shared unit**: the scarce resource being reused;
//! * `FOLD` and `POST` are elementwise stages with unlimited lanes (memory-bound on any real
//!   GPU, and never the binding constraint in this design);
//! * the buffer pool of [`crate::gpu::pool::BufferPool`] bounds the blocks in flight.
//!
//! Under those assumptions [`simulate`] is exact for the greedy in-order schedule, and
//! [`crate::gpu::model::project`] is a closed form for the same plan.

use super::pool::{BufferPool, PoolStats};
use super::PlanError;
use std::collections::BinaryHeap;
use std::fmt;

/// The pipeline stages, in execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// Inverse folding: permutation, sign inversion, zero fill.
    Fold,
    /// The external MDCT backend.
    Mdct,
    /// Scaling, sign, saturation and optional permutation.
    Post,
}

impl Stage {
    /// All stages, in execution order.
    pub const ALL: [Stage; 3] = [Stage::Fold, Stage::Mdct, Stage::Post];

    /// Zero-based position in the pipeline.
    pub fn index(self) -> usize {
        match self {
            Stage::Fold => 0,
            Stage::Mdct => 1,
            Stage::Post => 2,
        }
    }

    /// Short name.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Fold => "FOLD",
            Stage::Mdct => "MDCT",
            Stage::Post => "POST",
        }
    }
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Per-block cost of each stage, in an arbitrary time unit (seconds by convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageCost {
    /// Cost of the folding stage.
    pub fold: f64,
    /// Cost of one MDCT backend invocation (the shared unit).
    pub mdct: f64,
    /// Cost of the post stage.
    pub post: f64,
}

impl StageCost {
    /// Builds a cost model explicitly.
    pub fn new(fold: f64, mdct: f64, post: f64) -> Self {
        Self { fold, mdct, post }
    }

    /// A pipeline with identical stage costs, convenient for scaling studies.
    pub fn uniform(t: f64) -> Self {
        Self::new(t, t, t)
    }

    /// Total per-block latency, `t_fold + t_mdct + t_post`.
    pub fn sum(&self) -> f64 {
        self.fold + self.mdct + self.post
    }

    /// Cost of the slowest stage.
    pub fn max_stage(&self) -> f64 {
        self.fold.max(self.mdct).max(self.post)
    }

    /// Cost of one stage.
    pub fn get(&self, stage: Stage) -> f64 {
        match stage {
            Stage::Fold => self.fold,
            Stage::Mdct => self.mdct,
            Stage::Post => self.post,
        }
    }

    /// Checks that every stage is finite and positive.
    pub fn validate(&self) -> Result<(), PlanError> {
        let bad = |v: f64| !v.is_finite() || v <= 0.0;
        if bad(self.fold) {
            return Err(PlanError::InvalidCost { stage: "FOLD" });
        }
        if bad(self.mdct) {
            return Err(PlanError::InvalidCost { stage: "MDCT" });
        }
        if bad(self.post) {
            return Err(PlanError::InvalidCost { stage: "POST" });
        }
        Ok(())
    }
}

/// A scheduling plan: `M` streams, `B` blocks, `K` pool slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamPlan {
    /// Number of streams `M`.
    pub streams: usize,
    /// Number of blocks `B`.
    pub blocks: usize,
    /// Number of pool slots `K`.
    pub slots: usize,
    /// Slots consumed by one in-flight block (the folded `2N` word and the `N` output).
    pub slots_per_block: usize,
}

impl StreamPlan {
    /// A plan with the design's two slots per in-flight block.
    pub fn new(streams: usize, blocks: usize, slots: usize) -> Self {
        Self {
            streams,
            blocks,
            slots,
            slots_per_block: 2,
        }
    }

    /// Blocks that may be in flight simultaneously: `min(M, floor(K/2), B)`.
    pub fn capacity(&self) -> usize {
        let by_slots = self.slots / self.slots_per_block.max(1);
        self.streams.min(by_slots).min(self.blocks)
    }

    /// Validates the plan's shape.
    pub fn validate(&self) -> Result<(), PlanError> {
        if self.streams == 0 {
            return Err(PlanError::ZeroStreams);
        }
        if self.blocks == 0 {
            return Err(PlanError::ZeroBlocks);
        }
        if self.slots_per_block == 0 || self.slots < self.slots_per_block {
            return Err(PlanError::SlotsTooSmall {
                slots: self.slots,
                per_block: self.slots_per_block.max(1),
            });
        }
        Ok(())
    }
}

/// The schedule of one block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockTiming {
    /// Block index.
    pub block: usize,
    /// Stream the block was assigned to (`block % M`).
    pub stream: usize,
    /// Start of `FOLD`, i.e. when its buffer slots were reserved.
    pub start: f64,
    /// End of `FOLD`.
    pub fold_end: f64,
    /// Start of the shared MDCT unit for this block.
    pub mdct_start: f64,
    /// End of `MDCT`.
    pub mdct_end: f64,
    /// End of `POST`, when the slots are released.
    pub post_end: f64,
}

impl BlockTiming {
    /// End-to-end latency of this block.
    pub fn latency(&self) -> f64 {
        self.post_end - self.start
    }
}

/// The result of replaying a plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Simulation {
    /// Per-block timings, in issue order.
    pub timings: Vec<BlockTiming>,
    /// Time when the last block finishes.
    pub makespan: f64,
    /// Pool statistics observed during the replay.
    pub pool: PoolStats,
    /// Fraction of the pool's slot-time that was occupied.
    pub utilization: f64,
    /// Total time the shared MDCT unit was busy.
    pub mdct_busy: f64,
}

impl Simulation {
    /// Latency of the first block (`0.0` for an empty simulation).
    pub fn latency(&self) -> f64 {
        self.timings
            .first()
            .map(BlockTiming::latency)
            .unwrap_or(0.0)
    }

    /// Achieved throughput in blocks per unit time.
    pub fn throughput(&self) -> f64 {
        if self.makespan > 0.0 {
            self.timings.len() as f64 / self.makespan
        } else {
            0.0
        }
    }

    /// Fraction of the makespan during which the shared MDCT unit was busy.
    pub fn mdct_utilization(&self) -> f64 {
        if self.makespan > 0.0 {
            self.mdct_busy / self.makespan
        } else {
            0.0
        }
    }
}

/// Replays a plan under the resource model documented at the module level.
///
/// Blocks are issued in order, round-robin across streams. A block starts as soon as its
/// stream, its buffer slots and the shared MDCT unit allow; the schedule is deterministic.
pub fn simulate(plan: &StreamPlan, cost: &StageCost) -> Result<Simulation, PlanError> {
    plan.validate()?;
    cost.validate()?;

    let mut pool = BufferPool::new(plan.slots, plan.slots_per_block);
    let mut stream_free = vec![0.0_f64; plan.streams];
    let mut mdct_free = 0.0_f64;
    let mut timings = Vec::with_capacity(plan.blocks);
    let mut makespan = 0.0_f64;
    let mut busy_slot_time = 0.0_f64;

    for block in 0..plan.blocks {
        let stream = block % plan.streams;
        let not_before = stream_free[stream];
        let start = pool
            .earliest_start(not_before)
            .ok_or(PlanError::SlotsTooSmall {
                slots: plan.slots,
                per_block: plan.slots_per_block,
            })?;
        let fold_end = start + cost.fold;
        let mdct_start = fold_end.max(mdct_free);
        let mdct_end = mdct_start + cost.mdct;
        let post_end = mdct_end + cost.post;

        if !pool.occupy(not_before, start, post_end) {
            // Cannot happen: `start` was computed from the same pool state.
            return Err(PlanError::SlotsTooSmall {
                slots: plan.slots,
                per_block: plan.slots_per_block,
            });
        }
        mdct_free = mdct_end;
        stream_free[stream] = post_end;
        makespan = makespan.max(post_end);
        busy_slot_time += plan.slots_per_block as f64 * (post_end - start);

        timings.push(BlockTiming {
            block,
            stream,
            start,
            fold_end,
            mdct_start,
            mdct_end,
            post_end,
        });
    }

    let utilization = if makespan > 0.0 {
        busy_slot_time / (plan.slots as f64 * makespan)
    } else {
        0.0
    };

    Ok(Simulation {
        timings,
        makespan,
        pool: pool.stats(),
        utilization,
        mdct_busy: plan.blocks as f64 * cost.mdct,
    })
}

/// A node of the dependency DAG: one stage of one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Node {
    /// Block index.
    pub block: usize,
    /// Stage within the block.
    pub stage: Stage,
}

/// The stream partial order as a directed acyclic graph.
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    plan: StreamPlan,
    nodes: Vec<Node>,
    edges: Vec<(usize, usize)>,
}

impl DependencyGraph {
    /// Builds the DAG for a plan.
    pub fn build(plan: &StreamPlan) -> Self {
        let mut nodes = Vec::with_capacity(plan.blocks * Stage::ALL.len());
        for block in 0..plan.blocks {
            for stage in Stage::ALL {
                nodes.push(Node { block, stage });
            }
        }

        let idx = |block: usize, stage: Stage| block * Stage::ALL.len() + stage.index();
        let capacity = plan.capacity().max(1);
        let mut edges = Vec::new();
        for block in 0..plan.blocks {
            // 1. Intra-block stage order.
            edges.push((idx(block, Stage::Fold), idx(block, Stage::Mdct)));
            edges.push((idx(block, Stage::Mdct), idx(block, Stage::Post)));
            // 2. Stream serialisation.
            let next = block + plan.streams;
            if next < plan.blocks {
                edges.push((idx(block, Stage::Fold), idx(next, Stage::Fold)));
            }
            // 3. Buffer reuse: block `block + capacity` needs this block's slots.
            let reuse = block + capacity;
            if reuse < plan.blocks {
                edges.push((idx(block, Stage::Post), idx(reuse, Stage::Fold)));
            }
        }

        Self {
            plan: *plan,
            nodes,
            edges,
        }
    }

    /// The plan this graph describes.
    pub fn plan(&self) -> &StreamPlan {
        &self.plan
    }

    /// Nodes, ordered by `(block, stage)`.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Edges as `(from, to)` node indices.
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// True when the graph has no cycle.
    pub fn is_acyclic(&self) -> bool {
        self.topological_order().is_some()
    }

    /// A deterministic topological order, or `None` if the graph is cyclic.
    pub fn topological_order(&self) -> Option<Vec<usize>> {
        let n = self.nodes.len();
        let mut indegree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(u, v) in &self.edges {
            if u >= n || v >= n {
                return None;
            }
            adj[u].push(v);
            indegree[v] += 1;
        }
        // Reverse ordering of node ids keeps the pop order ascending.
        let mut heap = BinaryHeap::new();
        for (i, &d) in indegree.iter().enumerate() {
            if d == 0 {
                heap.push(std::cmp::Reverse(i));
            }
        }
        let mut order = Vec::with_capacity(n);
        while let Some(std::cmp::Reverse(i)) = heap.pop() {
            order.push(i);
            for &v in &adj[i] {
                indegree[v] -= 1;
                if indegree[v] == 0 {
                    heap.push(std::cmp::Reverse(v));
                }
            }
        }
        if order.len() == n {
            Some(order)
        } else {
            None
        }
    }

    /// Length of the longest path, i.e. the resource-free lower bound on the makespan.
    pub fn critical_path(&self, cost: &StageCost) -> f64 {
        let Some(order) = self.topological_order() else {
            return f64::INFINITY;
        };
        let mut finish = vec![0.0_f64; self.nodes.len()];
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for &(u, v) in &self.edges {
            preds[v].push(u);
        }
        let mut best = 0.0_f64;
        for &i in &order {
            let start = preds[i].iter().fold(0.0_f64, |m, &p| m.max(finish[p]));
            finish[i] = start + cost.get(self.nodes[i].stage);
            best = best.max(finish[i]);
        }
        best
    }

    /// Renders the DAG as Graphviz DOT, for design reviews.
    pub fn to_dot(&self) -> String {
        let mut s = String::from("digraph pipeline {\n  rankdir=LR;\n  node [shape=box];\n");
        for (i, node) in self.nodes.iter().enumerate() {
            s.push_str(&format!(
                "  n{i} [label=\"b{}/{}\"];\n",
                node.block,
                node.stage.name()
            ));
        }
        for &(u, v) in &self.edges {
            s.push_str(&format!("  n{u} -> n{v};\n"));
        }
        s.push_str("}\n");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_block_with_one_stream() {
        let plan = StreamPlan::new(1, 1, 2);
        let cost = StageCost::uniform(2.0);
        let sim = simulate(&plan, &cost).unwrap();
        assert_eq!(sim.timings.len(), 1);
        assert!((sim.latency() - 6.0).abs() < 1e-12);
        assert!((sim.makespan - 6.0).abs() < 1e-12);
        assert!((sim.throughput() - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn one_stream_serialises_everything() {
        let plan = StreamPlan::new(1, 4, 8);
        assert_eq!(plan.capacity(), 1);
        let cost = StageCost::uniform(1.0);
        let sim = simulate(&plan, &cost).unwrap();
        assert!((sim.makespan - 12.0).abs() < 1e-12);
        for w in sim.timings.windows(2) {
            assert!(w[1].start >= w[0].post_end - 1e-12);
        }
    }

    #[test]
    fn pool_bounds_the_in_flight_blocks() {
        // Plenty of streams, but K = 4 slots allow only two blocks in flight.
        let plan = StreamPlan::new(8, 8, 4);
        assert_eq!(plan.capacity(), 2);
        let cost = StageCost::uniform(1.0);
        let sim = simulate(&plan, &cost).unwrap();
        assert_eq!(sim.pool.peak_in_flight, 2);
        assert!(sim.pool.wait_events > 0, "blocks 2.. must wait for slots");
        // Pairs release at 3, then at 6 and 7, then 9 and 10, then 12 and 13.
        assert!((sim.makespan - 13.0).abs() < 1e-12, "{}", sim.makespan);
    }

    #[test]
    fn shared_mdct_unit_serialises() {
        // Two streams, plenty of slots, but only one MDCT unit.
        let plan = StreamPlan::new(2, 2, 8);
        let cost = StageCost::new(1.0, 5.0, 1.0);
        let sim = simulate(&plan, &cost).unwrap();
        assert!((sim.timings[1].mdct_start - sim.timings[0].mdct_end).abs() < 1e-12);
        assert!((sim.makespan - 12.0).abs() < 1e-12);
    }

    #[test]
    fn plan_validation() {
        assert_eq!(
            StreamPlan::new(0, 1, 2).validate(),
            Err(PlanError::ZeroStreams)
        );
        assert_eq!(
            StreamPlan::new(1, 0, 2).validate(),
            Err(PlanError::ZeroBlocks)
        );
        assert!(matches!(
            StreamPlan::new(1, 1, 1).validate(),
            Err(PlanError::SlotsTooSmall { .. })
        ));
        assert!(StageCost::uniform(f64::INFINITY).validate().is_err());
        assert!(StageCost::uniform(0.0).validate().is_err());
    }

    #[test]
    fn graph_is_acyclic_and_ordered() {
        let plan = StreamPlan::new(2, 6, 6);
        let g = DependencyGraph::build(&plan);
        assert!(g.is_acyclic());
        let order = g.topological_order().unwrap();
        assert_eq!(order.len(), 18);
        // Every edge points forward in the topological order.
        let mut pos = [0usize; 18];
        for (i, &node) in order.iter().enumerate() {
            pos[node] = i;
        }
        for &(u, v) in g.edges() {
            assert!(pos[u] < pos[v], "edge {u}->{v} violates the order");
        }
    }

    #[test]
    fn critical_path_is_a_lower_bound() {
        let plan = StreamPlan::new(2, 8, 6);
        let cost = StageCost::new(1.0, 3.0, 1.0);
        let g = DependencyGraph::build(&plan);
        let cp = g.critical_path(&cost);
        let sim = simulate(&plan, &cost).unwrap();
        assert!(sim.makespan >= cp - 1e-9);
        // Reuse/stream edges chain blocks 0 -> 2 -> 4 -> 6, five units each.
        assert!((cp - 20.0).abs() < 1e-12, "cp = {cp}");
    }

    #[test]
    fn dot_export() {
        let g = DependencyGraph::build(&StreamPlan::new(1, 2, 2));
        let dot = g.to_dot();
        assert!(dot.contains("digraph pipeline"));
        assert!(dot.contains("b0/FOLD"));
    }
}
