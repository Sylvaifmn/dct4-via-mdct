//! The GPU adaptation model (§6), printed as a design-time plan.
//!
//! ```text
//! cargo run --release --example gpu_plan
//! ```
//!
//! The closed-form projection and the event-driven replay are printed side by side; the last
//! section shows the stream partial order as a Graphviz DAG.

use dct4_mdct_kit::gpu::{
    bandwidth_cost, project, simulate, DependencyGraph, StageCost, StreamPlan,
};

fn main() {
    let n = 1024; // DCT-IV block length
    let bytes_per_sample = 4; // f32 staging buffers
    let bandwidth = 320e9; // bytes/s
    let launch = 3e-6; // seconds per kernel launch
    let flops = 8e12; // flop/s

    let cost = bandwidth_cost(n, bytes_per_sample, bandwidth, launch, flops)
        .expect("valid cost parameters");
    println!("Stage cost model for N = {n} (320 GB/s, 8 TFLOP/s, 3 us launch)");
    println!("-------------------------------------------------------------");
    println!(
        "  FOLD {:.3} us   MDCT {:.3} us   POST {:.3} us   (sum {:.3} us)",
        cost.fold * 1e6,
        cost.mdct * 1e6,
        cost.post * 1e6,
        cost.sum() * 1e6
    );
    println!(
        "  bytes per block = 8N samples = {} B\n",
        8 * n * bytes_per_sample
    );

    let blocks = 256;
    println!("Projection vs replay, B = {blocks} blocks");
    println!("--------------------------------------------------------------------");
    println!(
        "{:>3} {:>4} {:>5} {:>11} {:>11} {:>11} {:>10} {:>9}",
        "M", "K", "c", "latency us", "period us", "makespan us", "blocks/s", "util"
    );
    for (m, k) in [(1usize, 2usize), (2, 4), (4, 6), (8, 6), (8, 8), (16, 8)] {
        let plan = StreamPlan::new(m, blocks, k);
        let p = project(&plan, &cost, n, bytes_per_sample).unwrap();
        let s = simulate(&plan, &cost).unwrap();
        println!(
            "{:>3} {:>4} {:>5} {:>11.2} {:>11.2} {:>11.2} {:>10.0} {:>8.1}%",
            m,
            k,
            p.capacity,
            p.latency * 1e6,
            p.period * 1e6,
            p.makespan * 1e6,
            p.throughput,
            p.utilization * 100.0
        );
        assert!(p.makespan + 1e-12 >= s.makespan);
        println!(
            "      replay: makespan {:.2} us, peak in flight {}, waits {}, mdct busy {:.1}%",
            s.makespan * 1e6,
            s.pool.peak_in_flight,
            s.pool.wait_events,
            s.mdct_utilization() * 100.0
        );
    }

    println!("\nWhere the time goes (M = 8, K = 8)");
    println!("----------------------------------");
    let plan = StreamPlan::new(8, 8, 8);
    let sim = simulate(&plan, &cost).unwrap();
    println!(
        "{:>6} {:>8} {:>10} {:>10} {:>10}",
        "block", "stream", "fold end", "mdct end", "post end"
    );
    for t in &sim.timings {
        println!(
            "{:>6} {:>8} {:>10.2} {:>10.2} {:>10.2}",
            t.block,
            t.stream,
            t.fold_end * 1e6,
            t.mdct_end * 1e6,
            t.post_end * 1e6
        );
    }

    println!("\nStream partial order (M = 2, K = 4, B = 4) as DOT");
    println!("-------------------------------------------------");
    let graph = DependencyGraph::build(&StreamPlan::new(2, 4, 4));
    print!("{}", graph.to_dot());

    let balanced = StageCost::uniform(3.0e-6);
    println!("\nPipeline law check (M = 4, K = 6, B = 128, balanced stages)");
    println!("-----------------------------------------------------------");
    let p = project(&StreamPlan::new(4, 128, 6), &balanced, n, bytes_per_sample).unwrap();
    println!(
        "capacity c = {}, period P = max(t_m, Sigma/c) = max({:.2}, {:.2}) = {:.2} us",
        p.capacity,
        balanced.mdct * 1e6,
        balanced.sum() / p.capacity as f64 * 1e6,
        p.period * 1e6
    );
    println!(
        "makespan = {:.2} us, throughput = {:.0} blocks/s, effective bandwidth = {:.1} GB/s",
        p.makespan * 1e6,
        p.throughput,
        p.effective_bandwidth / 1e9
    );
}
