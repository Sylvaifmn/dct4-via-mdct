//! Closed-form GPU projection: latency, throughput, buffer and bandwidth (§6).
//!
//! Let
//!
//! * `t_f, t_m, t_p` be the per-block stage costs,
//! * `B` the batch size, `M` the number of streams, `K` the pool slots,
//! * `c = min(M, floor(K/2), B)` the number of blocks that can be in flight,
//! * `Sigma = t_f + t_m + t_p` the per-block latency.
//!
//! Then the model is
//!
//! ```text
//! latency        L = Sigma
//! period         P = max( t_m , Sigma / c )        steady-state period
//! mdct bound       = t_f + B*t_m + t_p             one shared MDCT unit, nothing buffered
//! smooth bound     = L + (B - c)^+ * P             saturated buffer pipeline
//! staircase bound  = L + floor((B-1)/c)*Sigma      waves of c blocks, staggered by t_m
//!                        + ((B-1) mod c)*t_m
//! makespan         = max( L, mdct, smooth, staircase )
//! throughput       = B / makespan
//! utilization      = B*Sigma / (c * makespan)
//! bytes/block      = 8N * bytes_per_sample         (FOLD 3N + MDCT 3N + POST 2N)
//! ```
//!
//! Three bounds are needed because the pipeline has three regimes, and a projection that
//! silently used only one of them would under-promise the makespan:
//!
//! * **mdct bound** — `B <= c`: nothing is buffered, the shared MDCT unit alone sets the pace.
//! * **smooth bound** — the buffer pipeline is saturated and slots are released evenly.
//! * **staircase bound** — the buffer pool releases slots in *waves* of `c` blocks, each wave
//!   held for `Sigma`, staggered by `t_m`. This regime appears whenever `t_f` or `t_p`
//!   dominates the MDCT unit, and it is where the smooth law alone under-estimates.
//!
//! The projection is **conservative**: the test suite replays a grid of plans with
//! [`super::stream::simulate`] and asserts `projection >= replay`. It is deliberately not an
//! equality: a design-time model that occasionally over-estimates by a wave is safe, while one
//! that under-estimates is not. The replay is the exact reference for a given plan.

use super::stream::{StageCost, StreamPlan};
use super::PlanError;

/// The closed-form projection of a plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuProjection {
    /// Blocks `B`.
    pub blocks: usize,
    /// Streams `M`.
    pub streams: usize,
    /// Pool slots `K`.
    pub slots: usize,
    /// Blocks in flight `c`.
    pub capacity: usize,
    /// Per-block latency `Sigma`.
    pub latency: f64,
    /// Steady-state period `P`.
    pub period: f64,
    /// End-to-end makespan for `B` blocks.
    pub makespan: f64,
    /// `B / makespan`, blocks per time unit.
    pub throughput: f64,
    /// `B*Sigma / (c*makespan)`, the fraction of the capacity-time area used.
    pub utilization: f64,
    /// Whether the MDCT unit is the binding resource.
    pub mdct_bound: bool,
    /// Whether the buffer pool / stream pipeline is the binding resource.
    pub pool_bound: bool,
    /// Bytes moved per block: `8N * bytes_per_sample`.
    pub bytes_per_block: u64,
    /// `bytes_per_block * throughput`, bytes per time unit.
    pub effective_bandwidth: f64,
}

impl GpuProjection {
    /// Renders the projection as a Markdown report.
    pub fn to_markdown(&self) -> String {
        format!(
            "| quantity | value |\n|---|---:|\n\
             | blocks B | {} |\n| streams M | {} |\n| slots K | {} |\n\
             | in flight c | {} |\n| latency | {:.4} |\n| period | {:.4} |\n\
             | makespan | {:.4} |\n| throughput | {:.4} block/unit |\n\
             | pool utilization | {:.1}% |\n| bottleneck | {} |\n\
             | bytes/block | {} |\n| effective bandwidth | {:.1} byte/unit |",
            self.blocks,
            self.streams,
            self.slots,
            self.capacity,
            self.latency,
            self.period,
            self.makespan,
            self.throughput,
            self.utilization * 100.0,
            if self.mdct_bound {
                "MDCT unit"
            } else {
                "buffer pool / streams"
            },
            self.bytes_per_block,
            self.effective_bandwidth
        )
    }
}

/// Projects the schedule of `plan` under `cost`.
///
/// `samples_per_block` is `N` and `bytes_per_sample` is the width of one sample, used only for
/// the bandwidth estimate.
pub fn project(
    plan: &StreamPlan,
    cost: &StageCost,
    samples_per_block: usize,
    bytes_per_sample: usize,
) -> Result<GpuProjection, PlanError> {
    plan.validate()?;
    cost.validate()?;
    if samples_per_block == 0 || bytes_per_sample == 0 {
        return Err(PlanError::InvalidParameter {
            what: "samples_per_block and bytes_per_sample must be positive",
        });
    }

    let capacity = plan.capacity();
    if capacity == 0 {
        return Err(PlanError::SlotsTooSmall {
            slots: plan.slots,
            per_block: plan.slots_per_block,
        });
    }
    let c = capacity as f64;
    let blocks = plan.blocks as f64;
    let sum = cost.sum();
    let latency = sum;
    let period = cost.mdct.max(sum / c);

    let extra = (plan.blocks.saturating_sub(capacity)) as f64;
    let mdct_bound = cost.fold + blocks * cost.mdct + cost.post;
    let smooth_bound = latency + extra * period;

    // Staircase: block `b` starts in wave `b / c` and its slots are held for `Sigma`; the
    // inner stagger between blocks of the same wave is the MDCT unit's occupancy.
    let last = plan.blocks - 1;
    let staircase_bound =
        latency + (last / capacity) as f64 * sum + (last % capacity) as f64 * cost.mdct;

    let makespan = latency
        .max(mdct_bound)
        .max(smooth_bound)
        .max(staircase_bound)
        .max(f64::MIN_POSITIVE);

    let throughput = blocks / makespan;
    let utilization = (blocks * sum / (c * makespan)).min(1.0);
    let bytes_per_block = 8u64
        .checked_mul(samples_per_block as u64)
        .and_then(|v| v.checked_mul(bytes_per_sample as u64))
        .ok_or(PlanError::InvalidParameter {
            what: "8*N*sizeof(sample) overflows u64",
        })?;

    Ok(GpuProjection {
        blocks: plan.blocks,
        streams: plan.streams,
        slots: plan.slots,
        capacity,
        latency,
        period,
        makespan,
        throughput,
        utilization,
        mdct_bound: mdct_bound >= smooth_bound - 1e-12
            && mdct_bound >= staircase_bound - 1e-12
            && mdct_bound >= latency - 1e-12,
        pool_bound: smooth_bound > mdct_bound + 1e-12 || staircase_bound > mdct_bound + 1e-12,
        bytes_per_block,
        effective_bandwidth: bytes_per_block as f64 * throughput,
    })
}

/// Derives a [`StageCost`] from bandwidth, launch overhead and a MAC-count model.
///
/// * `FOLD` moves `3N` samples (`N` in, `2N` out),
/// * `MDCT` moves `3N` samples (`2N` in, `N` out) and performs `4N log2(2N)` flops
///   (`2N` inputs, `log2(2N)` butterfly levels, two flops per butterfly),
/// * `POST` moves `2N` samples.
///
/// The launch overhead is charged once per stage, modelling the three kernels.
pub fn bandwidth_cost(
    n: usize,
    bytes_per_sample: usize,
    bandwidth_bytes_per_s: f64,
    launch_s: f64,
    flops_per_s: f64,
) -> Result<StageCost, PlanError> {
    if n == 0 || bytes_per_sample == 0 {
        return Err(PlanError::InvalidParameter {
            what: "n and bytes_per_sample must be positive",
        });
    }
    if !bandwidth_bytes_per_s.is_finite() || bandwidth_bytes_per_s <= 0.0 {
        return Err(PlanError::InvalidParameter {
            what: "bandwidth must be finite and > 0",
        });
    }
    if !flops_per_s.is_finite() || flops_per_s <= 0.0 {
        return Err(PlanError::InvalidParameter {
            what: "flops_per_s must be finite and > 0",
        });
    }
    if !launch_s.is_finite() || launch_s < 0.0 {
        return Err(PlanError::InvalidParameter {
            what: "launch_s must be finite and >= 0",
        });
    }

    let nf = n as f64;
    let bps = bytes_per_sample as f64;
    let transfer = |samples: f64| samples * bps / bandwidth_bytes_per_s;
    let mdct_flops = 4.0 * nf * (2.0 * nf).log2();

    Ok(StageCost::new(
        launch_s + transfer(3.0 * nf),
        launch_s + transfer(3.0 * nf) + mdct_flops / flops_per_s,
        launch_s + transfer(2.0 * nf),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::stream::simulate;

    fn grid() -> Vec<(StreamPlan, StageCost)> {
        let mut cases = Vec::new();
        let shapes = [
            (1usize, 4usize, 8usize),
            (2, 7, 4),
            (4, 8, 8),
            (8, 16, 4),
            (3, 10, 16),
            (2, 2, 8),
            (6, 6, 12),
            (1, 1, 2),
            (4, 1, 8),
            (16, 40, 8),
        ];
        let costs = [
            (1.0, 1.0, 1.0),
            (1.0, 4.0, 1.0),
            (2.0, 3.0, 1.0),
            (0.5, 2.0, 0.5),
            (3.0, 1.0, 2.0),
            (1.0, 5.0, 1.0),
            (4.0, 1.0, 1.0),
            (10.0, 1.0, 10.0),
            (0.1, 2.0, 0.1),
            (1.0, 0.2, 3.0),
        ];
        for (m, b, k) in shapes {
            for (f, md, p) in costs {
                cases.push((StreamPlan::new(m, b, k), StageCost::new(f, md, p)));
            }
        }
        cases
    }

    #[test]
    fn projection_is_a_conservative_bound_on_the_replay() {
        let mut worst_gap = 0.0_f64;
        for (plan, cost) in grid() {
            let p = project(&plan, &cost, 1024, 4).unwrap();
            let s = simulate(&plan, &cost).unwrap();
            assert!(
                p.makespan + 1e-9 >= s.makespan,
                "plan {plan:?} cost {cost:?}: projection {} < replay {}",
                p.makespan,
                s.makespan
            );
            // The projection must also be useful, not vacuous: never more than ~2 blocks of
            // latency above the replay.
            let gap = p.makespan - s.makespan;
            assert!(
                gap <= 2.0 * p.latency + 1e-9,
                "plan {plan:?} cost {cost:?}: projection {} too loose vs replay {}",
                p.makespan,
                s.makespan
            );
            worst_gap = worst_gap.max(gap / p.latency);
        }
        assert!(worst_gap < 2.0, "worst relative gap {worst_gap}");
    }

    #[test]
    fn projection_is_tight_for_balanced_plans() {
        // When the MDCT unit is the slowest stage the projection is exact.
        for (m, b, k) in [(1usize, 4usize, 8usize), (2, 7, 4), (4, 16, 8), (8, 32, 16)] {
            let cost = StageCost::new(1.0, 4.0, 1.0);
            let plan = StreamPlan::new(m, b, k);
            let p = project(&plan, &cost, 1024, 4).unwrap();
            let s = simulate(&plan, &cost).unwrap();
            assert!(
                (p.makespan - s.makespan).abs() < 1e-9,
                "plan {plan:?}: projection {} vs replay {}",
                p.makespan,
                s.makespan
            );
        }
    }

    #[test]
    fn single_block_numbers() {
        let plan = StreamPlan::new(1, 1, 2);
        let cost = StageCost::uniform(2.0);
        let p = project(&plan, &cost, 1024, 4).unwrap();
        assert!((p.latency - 6.0).abs() < 1e-12);
        assert!((p.makespan - 6.0).abs() < 1e-12);
        assert!((p.throughput - 1.0 / 6.0).abs() < 1e-12);
        assert_eq!(p.bytes_per_block, 8 * 1024 * 4);
        assert!((p.effective_bandwidth - p.bytes_per_block as f64 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn more_streams_never_hurt_before_the_pool_binds() {
        let cost = StageCost::new(1.0, 2.0, 1.0);
        let a = project(&StreamPlan::new(1, 16, 8), &cost, 512, 4).unwrap();
        let b = project(&StreamPlan::new(2, 16, 8), &cost, 512, 4).unwrap();
        assert!(b.makespan <= a.makespan);
        assert!(b.throughput >= a.throughput);
    }

    #[test]
    fn pool_binding_is_reported() {
        let cost = StageCost::new(10.0, 1.0, 10.0);
        let p = project(&StreamPlan::new(8, 32, 4), &cost, 256, 4).unwrap();
        assert_eq!(p.capacity, 2);
        assert!(p.pool_bound && !p.mdct_bound);
    }

    #[test]
    fn bandwidth_cost_is_positive_and_monotone() {
        let a = bandwidth_cost(1024, 4, 320e9, 3e-6, 8e12).unwrap();
        let b = bandwidth_cost(2048, 4, 320e9, 3e-6, 8e12).unwrap();
        assert!(a.validate().is_ok());
        assert!(b.fold > a.fold && b.mdct > a.mdct && b.post > a.post);
        assert!(a.mdct > a.fold);
        assert!(bandwidth_cost(0, 4, 1e9, 0.0, 1e9).is_err());
        assert!(bandwidth_cost(64, 4, 0.0, 0.0, 1e9).is_err());
        assert!(bandwidth_cost(64, 4, 1e9, -1.0, 1e9).is_err());
    }

    #[test]
    fn markdown_renders() {
        let p = project(&StreamPlan::new(2, 4, 4), &StageCost::uniform(1.0), 8, 4).unwrap();
        let md = p.to_markdown();
        assert!(md.contains("throughput"));
        assert!(md.contains("bytes/block"));
    }
}
