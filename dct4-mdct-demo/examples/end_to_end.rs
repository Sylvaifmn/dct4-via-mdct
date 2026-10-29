//! End-to-end demonstration of the indirect DCT-IV kit.
//!
//! ```text
//! cargo run --release --example end_to_end
//! ```
//!
//! It builds a kit around an emulated external MDCT IP, prints the compliance record, runs a
//! batch on both arithmetic paths, and compares the result with the analytic DCT-IV.

use dct4_mdct_kit::backend::{EmulatedMdct, OutputOrdering, SampleFormat};
use dct4_mdct_kit::config::{KitConfig, LengthSet};
use dct4_mdct_kit::dct4::dct4;
use dct4_mdct_kit::fixed::{QFormat, Q};
use dct4_mdct_kit::numerics;
use dct4_mdct_kit::pipeline::{DctIvKit, GatePolicy};
use dct4_mdct_kit::util::{max_abs, norm2, Rng};

fn main() {
    let n = 64;
    let batch = 4;

    // ------------------------------------------------------------------ configuration (eq. 3)
    let cfg = KitConfig::new(n)
        .with_batch(batch)
        .with_alpha(1024.0) // the IP's scalar gain: y = 1024 * M(x)
        .with_sign(1) // the IP's sign convention
        .with_formats(
            QFormat::new(16, 14), // Wu   : Q1.14 input
            QFormat::new(17, 14), // Win  : one guard bit for the fold
            QFormat::new(33, 14), // Wip  : sized below from alpha * sum|u|
            QFormat::new(33, 14), // Wout : Q18.14 kit output
        )
        .with_lengths(LengthSet::of(&[32, 64, 128]))
        .with_pool_slots(6);

    cfg.validate().expect("eq. 3 must hold");

    // The backend output is y = alpha * X, so Wip must cover alpha * max|X|, not max|X|.
    // sum_n |u[n]| is the cheapest rigorous bound on |X[k]|.
    let l1_bound = 0.75 * n as f64;
    println!(
        "Wip sizing: |alpha * X| <= alpha * sum|u| = {:.0}; required Wip = {} bits, configured {}",
        cfg.alpha * l1_bound,
        numerics::required_ip_width(cfg.alpha, l1_bound, cfg.ip.frac),
        cfg.ip
    );
    println!("== Configuration ==");
    println!(
        "N = {n}, B = {batch}, Wu = {}, Win = {}, Wip = {}, Wout = {}",
        cfg.input, cfg.guard, cfg.ip, cfg.output
    );
    println!(
        "alpha = {}, sigma = {}, S = sigma/alpha = {}\n",
        cfg.alpha,
        cfg.sign,
        cfg.signed_post_gain()
    );

    // ---------------------------------------------------------------- external backend (H1-H4)
    let backend = EmulatedMdct::ideal("acme-mdct-ip")
        .with_format(SampleFormat::Fixed(QFormat::new(24, 14)))
        .with_scale(1024.0)
        .with_lengths(LengthSet::of(&[32, 64, 128]));

    let mut kit = DctIvKit::new(cfg.clone(), backend, GatePolicy::Enforce, None)
        .expect("the compliance gate must accept this IP");
    println!("{}", kit.report().to_markdown());

    // ------------------------------------------------------------------------- float pipeline
    let mut rng = Rng::new(0x00DE_C0DE);
    let mut u = vec![0.0; batch * n];
    for block in 0..batch {
        for i in 0..n {
            // A deterministic, non-trivial test signal.
            let t = (i as f64 + 0.5) / n as f64;
            u[block * n + i] = 0.5 * (6.0 * std::f64::consts::PI * t + block as f64).sin()
                + 0.2 * rng.next_signed();
        }
    }

    let mut x = vec![0.0; batch * n];
    let stats = kit.transform_batch(&u, &mut x).expect("transform");
    println!("== Float path ==");
    println!(
        "blocks = {}, backend calls = {}, saturations = {}, ip clamps = {}, \
         fold overflows = {}, peak |X| = {:.4}",
        stats.blocks,
        stats.backend_calls,
        stats.saturations,
        stats.ip_clamps,
        stats.fold_overflows,
        stats.peak_abs
    );

    let mut worst = 0.0_f64;
    for block in 0..batch {
        let range = block * n..(block + 1) * n;
        let reference = dct4(&u[range.clone()]);
        let dev = max_abs(
            &x[range.clone()]
                .iter()
                .zip(&reference)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        );
        worst = worst.max(dev);
        println!("  block {block}: max |X - D_N(U)| = {dev:.3e}");
    }
    assert!(
        worst < 1e-10,
        "the indirect path must reproduce the definition"
    );

    // ------------------------------------------------------------------------- fixed pipeline
    println!("\n== Fixed-point path ==");
    let mut code_in = vec![Q::zero(cfg.input); n];
    for (q, v) in code_in.iter_mut().zip(&u[..n]) {
        *q = Q::from_f64(cfg.input, *v * 0.5);
    }
    let mut code_out = vec![Q::zero(cfg.output); n];
    let fstats = kit
        .transform_block_fixed(&code_in, &mut code_out)
        .expect("fixed transform");
    println!(
        "saturations = {}, ip clamps = {}, fold overflows = {}, peak |X| = {:.4}",
        fstats.saturations, fstats.ip_clamps, fstats.fold_overflows, fstats.peak_abs
    );
    assert_eq!(fstats.ip_clamps, 0, "Wip must cover alpha * max|X|");

    let unquantized: Vec<f64> = code_in.iter().map(|q| q.to_f64()).collect();
    let reference = dct4(&unquantized);
    let measured = max_abs(
        &code_out
            .iter()
            .zip(&reference)
            .map(|(q, r)| q.to_f64() - r)
            .collect::<Vec<_>>(),
    );
    let bound = kit.error_bound(norm2(&unquantized));
    println!("measured error        = {measured:.3e}");
    println!(
        "forward error bound   = {:.3e} (input {:.3e} + backend {:.3e} + post {:.3e})",
        bound.total, bound.input, bound.backend, bound.post
    );
    println!("relative error bound  = {:.3e}", bound.relative);
    println!(
        "implied SNR           = {:.1} dB",
        numerics::snr_db(n, norm2(&unquantized), bound.total)
    );
    assert!(
        measured <= bound.total,
        "the bound must contain the measurement"
    );

    // --------------------------------------------------------------------- permutation variant
    println!("\n== A permuted-output IP is admissible (H4 escape clause) ==");
    let perm: Vec<usize> = (1..n).chain(std::iter::once(0)).collect();
    let permuted = EmulatedMdct::ideal("acme-mdct-ip-reordered")
        .with_scale(1024.0)
        .with_ordering(OutputOrdering::Permutation(perm));
    let mut kit2 = DctIvKit::new(cfg, permuted, GatePolicy::Enforce, None).expect("gate");
    let x2 = kit2.transform(&u[..n]).expect("transform");
    let reference = dct4(&u[..n]);
    let dev = max_abs(
        &x2.iter()
            .zip(&reference)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>(),
    );
    println!("H4 decision: {:?}", kit2.report().ordering);
    println!("max deviation after POST un-permutes: {dev:.3e}");
    assert!(dev < 1e-10);

    println!("\nOK: the kit computes D_N through an MDCT backend it does not implement.");
}
