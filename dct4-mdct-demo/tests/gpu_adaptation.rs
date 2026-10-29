//! The GPU adaptation model (§6): pooling, the stream partial order, throughput and latency.

use dct4_mdct_kit::config::KitConfig;
use dct4_mdct_kit::gpu::{
    bandwidth_cost, project, simulate, BufferPool, DependencyGraph, Stage, StageCost, StreamPlan,
};
use dct4_mdct_kit::pipeline::{DctIvKit, GatePolicy};

#[test]
fn capacity_is_the_minimum_of_the_three_resources() {
    // min(M, floor(K/2), B)
    assert_eq!(StreamPlan::new(8, 100, 6).capacity(), 3);
    assert_eq!(StreamPlan::new(2, 100, 64).capacity(), 2);
    assert_eq!(StreamPlan::new(8, 5, 64).capacity(), 5);
    assert_eq!(StreamPlan::new(8, 100, 3).capacity(), 1);
}

#[test]
fn projection_bounds_the_replay_and_stays_useful() {
    let mut worst = 0.0_f64;
    for m in [1usize, 2, 3, 4, 8] {
        for k in [2usize, 4, 6, 8, 16] {
            for b in [1usize, 2, 5, 9, 17, 33] {
                for cost in [
                    StageCost::uniform(1.0),
                    StageCost::new(1.0, 4.0, 1.0),
                    StageCost::new(2.0, 3.0, 0.5),
                    StageCost::new(3.0, 1.0, 2.0),
                ] {
                    let plan = StreamPlan::new(m, b, k);
                    let p = project(&plan, &cost, 1024, 4).unwrap();
                    let s = simulate(&plan, &cost).unwrap();
                    assert!(
                        p.makespan + 1e-9 >= s.makespan,
                        "M={m} K={k} B={b} {cost:?}: {} < {}",
                        p.makespan,
                        s.makespan
                    );
                    let gap = p.makespan - s.makespan;
                    assert!(
                        gap <= 2.0 * p.latency + 1e-9,
                        "M={m} K={k} B={b} {cost:?}: gap {gap} vs latency {}",
                        p.latency
                    );
                    worst = worst.max(gap / p.latency);
                }
            }
        }
    }
    assert!(worst <= 2.0, "worst relative gap {worst}");
}

#[test]
fn dependency_graph_encodes_the_three_constraints() {
    let plan = StreamPlan::new(2, 6, 6);
    let g = DependencyGraph::build(&plan);
    let capacity = plan.capacity();
    assert_eq!(capacity, 2);
    let idx = |b: usize, s: Stage| b * 3 + s as usize;
    let has = |u: usize, v: usize| g.edges().contains(&(u, v));

    for b in 0..plan.blocks {
        // 1. intra-block stage order
        assert!(has(idx(b, Stage::Fold), idx(b, Stage::Mdct)));
        assert!(has(idx(b, Stage::Mdct), idx(b, Stage::Post)));
        // 2. same-stream serialisation: block b + M shares a stream with block b
        if b + plan.streams < plan.blocks {
            assert!(has(idx(b, Stage::Fold), idx(b + plan.streams, Stage::Fold)));
        }
        // 3. buffer reuse: block b + c needs the slots of block b
        if b + capacity < plan.blocks {
            assert!(has(idx(b, Stage::Post), idx(b + capacity, Stage::Fold)));
        }
    }

    // Every edge points from a lower to a higher node id (hence the graph is acyclic).
    for &(u, v) in g.edges() {
        assert!(u < v);
    }
    assert!(g.is_acyclic());
    assert_eq!(g.nodes().len(), 3 * plan.blocks);
    assert!(g.to_dot().contains("digraph"));
}

#[test]
fn critical_path_is_a_lower_bound_on_the_makespan() {
    for (m, b, k) in [(1usize, 4usize, 8usize), (2, 8, 6), (4, 16, 16)] {
        for cost in [
            StageCost::uniform(1.0),
            StageCost::new(1.0, 4.0, 1.0),
            StageCost::new(4.0, 1.0, 1.0),
        ] {
            let plan = StreamPlan::new(m, b, k);
            let cp = DependencyGraph::build(&plan).critical_path(&cost);
            let sim = simulate(&plan, &cost).unwrap();
            assert!(
                sim.makespan + 1e-9 >= cp,
                "M={m} B={b} K={k}: makespan {} < critical path {cp}",
                sim.makespan
            );
        }
    }
}

#[test]
fn slots_bound_the_blocks_in_flight() {
    for k in [2usize, 4, 6, 8] {
        let plan = StreamPlan::new(16, 40, k);
        let sim = simulate(&plan, &StageCost::uniform(1.0)).unwrap();
        assert_eq!(sim.pool.peak_in_flight, k / 2, "K={k}");
        let p = project(&plan, &StageCost::uniform(1.0), 1024, 4).unwrap();
        assert_eq!(p.capacity, k / 2);
    }
}

#[test]
fn one_slot_pair_is_a_serial_pipeline() {
    let plan = StreamPlan::new(8, 10, 2);
    assert_eq!(plan.capacity(), 1);
    let cost = StageCost::new(1.0, 2.0, 3.0);
    let sim = simulate(&plan, &cost).unwrap();
    assert!((sim.makespan - 10.0 * cost.sum()).abs() < 1e-12);
    for w in sim.timings.windows(2) {
        assert!(w[1].start + 1e-12 >= w[0].post_end);
    }
}

#[test]
fn more_streams_help_until_the_pool_binds() {
    // Uniform stages, so the buffer pool is the binding resource once c saturates.
    let cost = StageCost::uniform(1.0);
    let plan_for = |m| project(&StreamPlan::new(m, 32, 8), &cost, 1024, 4).unwrap();
    let p1 = plan_for(1);
    let p2 = plan_for(2);
    let p4 = plan_for(4);
    let p8 = plan_for(8);
    assert!(p2.makespan < p1.makespan);
    assert!(p4.makespan < p2.makespan);
    // K = 8 allows only 4 blocks in flight: the fifth stream buys nothing.
    assert!((p4.makespan - p8.makespan).abs() < 1e-12);
    assert_eq!(p8.capacity, 4);
}

#[test]
fn the_shared_mdct_unit_serialises_across_streams() {
    let plan = StreamPlan::new(4, 4, 16);
    let cost = StageCost::new(0.5, 5.0, 0.5);
    let sim = simulate(&plan, &cost).unwrap();
    for w in sim.timings.windows(2) {
        assert!(
            w[1].mdct_start + 1e-12 >= w[0].mdct_end,
            "the MDCT unit is shared"
        );
    }
    assert!(sim.mdct_utilization() > 0.5);
}

#[test]
fn bandwidth_account_matches_the_bytes_per_block() {
    let n = 1024;
    let bytes_per_sample = 4;
    let cost = bandwidth_cost(n, bytes_per_sample, 320e9, 3e-6, 8e12).unwrap();
    let p = project(&StreamPlan::new(4, 64, 6), &cost, n, bytes_per_sample).unwrap();

    // FOLD 3N + MDCT 3N + POST 2N = 8N samples per block.
    assert_eq!(p.bytes_per_block, 8 * n as u64 * bytes_per_sample as u64);
    assert!((p.effective_bandwidth - p.bytes_per_block as f64 * p.throughput).abs() < 1e-3);
    assert!(p.effective_bandwidth > 0.0);
    assert!(p.utilization > 0.0 && p.utilization <= 1.0);

    // The MDCT stage moves the same bytes as FOLD but also computes.
    assert!(cost.mdct > cost.fold);
    assert!(cost.fold > cost.post);
}

#[test]
fn utilization_is_always_a_fraction() {
    for m in 1..=6 {
        for k in 2..=8 {
            for b in [1usize, 7, 64] {
                let p = project(
                    &StreamPlan::new(m, b, k),
                    &StageCost::new(1.0, 2.0, 1.5),
                    512,
                    2,
                )
                .unwrap();
                assert!(
                    p.utilization >= 0.0 && p.utilization <= 1.0 + 1e-12,
                    "M={m} K={k} B={b}: utilization {}",
                    p.utilization
                );
            }
        }
    }
}

#[test]
fn a_t4_like_plan_is_projected_and_replayed() {
    // Rough T4-class numbers: 320 GB/s, 8 TFLOP/s, 3 us launch overhead, 1024-point blocks.
    let n = 1024;
    let cost = bandwidth_cost(n, 4, 320e9, 3e-6, 8e12).unwrap();
    let plan = StreamPlan::new(4, 128, 6);
    let p = project(&plan, &cost, n, 4).unwrap();
    let s = simulate(&plan, &cost).unwrap();

    // Launch-overhead dominated: three ~3 us stages per block, three blocks in flight.
    assert!(
        p.makespan <= 1e-3,
        "makespan {} s is implausible",
        p.makespan
    );
    assert!(p.throughput > 1e5, "throughput {} block/s", p.throughput);
    assert!(p.period > 3e-6);
    assert!(p.makespan + p.latency + 1e-12 >= s.makespan);
    assert!(p.latency > 0.0 && p.period > 0.0);
}

#[test]
fn pool_api_is_consistent_with_the_plan() {
    let mut pool = BufferPool::new(6, 2);
    assert_eq!(pool.max_in_flight(), 3);
    assert!(pool.occupy(0.0, 0.0, 4.0));
    assert!(pool.occupy(0.0, 0.0, 4.0));
    assert!(pool.occupy(0.0, 0.0, 4.0));
    assert_eq!(pool.earliest_start(0.0), Some(4.0));
    assert!(!pool.occupy(0.0, 1.0, 5.0), "no slots before t = 4");
    assert!(pool.occupy(0.0, 4.0, 8.0));
    assert_eq!(pool.stats().peak_in_flight, 3);
    assert_eq!(pool.stats().wait_events, 1);
}

#[test]
fn error_model_and_the_gpu_plan_agree_on_the_budget() {
    // The GPU projection decides *when* the blocks run; the error model decides *whether the
    // declared format can resolve them*. Both belong to the same acceptance record.
    let cfg = KitConfig::new(1024).with_formats(
        dct4_mdct_kit::fixed::QFormat::new(16, 14),
        dct4_mdct_kit::fixed::QFormat::new(17, 14),
        dct4_mdct_kit::fixed::QFormat::new(24, 14),
        dct4_mdct_kit::fixed::QFormat::new(24, 14),
    );
    let kit = DctIvKit::new(
        cfg,
        dct4_mdct_kit::backend::EmulatedMdct::ideal("ip")
            .with_format(dct4_mdct_kit::backend::SampleFormat::F32),
        GatePolicy::Enforce,
        None,
    )
    .unwrap();

    let bound = kit.error_bound(32.0);
    assert!(bound.total >= bound.backend && bound.backend > 0.0);
    assert!(bound.relative > 0.0);
    // f32 at N = 1024 cannot resolve a Q1.14 output word's LSB.
    assert!(kit.budget_violation(32.0).is_some());

    let cost = bandwidth_cost(1024, 4, 320e9, 3e-6, 8e12).unwrap();
    let p = project(&StreamPlan::new(4, 128, 6), &cost, 1024, 4).unwrap();
    assert!(p.throughput > 0.0);
}
