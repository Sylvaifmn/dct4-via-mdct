//! The compliance gate, shown as a bring-up decision sheet.
//!
//! ```text
//! cargo run --release --example compliance_sheet
//! ```
//!
//! Each candidate IP is probed, decided, and either accepted or refused with a classified
//! vulnerability and a remedy. This is the artefact a verification record would archive.

use dct4_mdct_kit::backend::{EmulatedMdct, OutputOrdering, SampleFormat, WindowKind};
use dct4_mdct_kit::compliance::{evaluate, ComplianceReport};
use dct4_mdct_kit::config::KitConfig;
use dct4_mdct_kit::fixed::QFormat;
use dct4_mdct_kit::pipeline::{DctIvKit, GatePolicy};

fn classify(name: &str, report: &ComplianceReport) {
    println!("\n### {name}");
    println!(
        "    format = {}, probe N = {}, tau_dtype = {:.3e}",
        report.format, report.probe_n, report.thresholds.tau_scale
    );
    println!(
        "    measured: alpha = {:.9}, spread = {:.3e}, phase err = {:.3e}, window dev = {:.3e}",
        report.measured.alpha,
        report.measured.scale_spread,
        report.measured.phase_error,
        report.measured.window_deviation
    );
    for check in &report.checks {
        println!(
            "    {} {:<26} {:<5} metric = {}",
            check.precondition.code(),
            check.precondition.title(),
            if check.passed { "pass" } else { "FAIL" },
            if check.metric.is_nan() {
                "n/a".to_string()
            } else {
                format!("{:.3e}", check.metric)
            }
        );
    }
    if report.accepted() {
        println!("    => ACCEPT");
    } else {
        println!("    => REJECT (backend_reject)");
        for v in &report.violations {
            println!("       - {:?}: {}", v.category, v.detail);
            println!("         remedy: {}", v.remedy);
        }
    }
}

fn main() {
    let cfg = KitConfig::new(32).with_formats(
        QFormat::new(16, 14),
        QFormat::new(17, 14),
        QFormat::new(24, 14),
        QFormat::new(24, 14),
    );

    println!("Compliance decision sheet for a 32-point DCT-IV kit");
    println!("==================================================");

    let candidates: Vec<(&str, EmulatedMdct)> = vec![
        ("golden IP (f64)", EmulatedMdct::ideal("golden")),
        (
            "sine-windowed IP",
            EmulatedMdct::ideal("windowed").with_window_kind(WindowKind::Sine),
        ),
        (
            "Hann-windowed IP",
            EmulatedMdct::ideal("windowed-hann").with_window_kind(WindowKind::Hann),
        ),
        (
            "phase-offset IP (+0.3 rad)",
            EmulatedMdct::ideal("phase").with_phase(0.3),
        ),
        (
            "per-bin drift IP (0.5%)",
            EmulatedMdct::ideal("drift").with_gain_drift(0.005),
        ),
        (
            "IP with a wrong declared gain",
            EmulatedMdct::ideal("liar")
                .with_scale(2.0)
                .with_declared_scale(4.0),
        ),
        (
            "reordered IP, permutation declared",
            EmulatedMdct::ideal("perm").with_ordering(OutputOrdering::Permutation(
                (1..32).chain(std::iter::once(0)).collect(),
            )),
        ),
        (
            "reordered IP, ordering undocumented",
            EmulatedMdct::ideal("shuffled").with_ordering(OutputOrdering::Unspecified),
        ),
        (
            "honest 12-bit fixed-point IP",
            EmulatedMdct::ideal("q12")
                .with_format(SampleFormat::Fixed(QFormat::new(13, 12)))
                .with_quant_bits(None, Some(13)),
        ),
        (
            "12-bit IP claiming f64",
            EmulatedMdct::ideal("q12-liar")
                .with_format(SampleFormat::F64)
                .with_quant_bits(None, Some(13)),
        ),
    ];

    for (name, mut backend) in candidates {
        match evaluate(&mut backend, &cfg, None) {
            Ok(report) => classify(name, &report),
            Err(e) => println!("\n### {name}\n    probing failed: {e}"),
        }
    }

    println!("\n\nRefusal at construction (`backend_reject`)");
    println!("==========================================");
    let backend = EmulatedMdct::ideal("windowed").with_window_kind(WindowKind::Sine);
    match DctIvKit::new(cfg, backend, GatePolicy::Enforce, None) {
        Ok(_) => println!("unexpectedly constructed"),
        Err(e) => println!("{e}"),
    }

    println!("\nBypass in effect (bring-up only, output must not ship)");
    println!("=====================================================");
    let backend = EmulatedMdct::ideal("windowed").with_window_kind(WindowKind::Sine);
    let kit = DctIvKit::new(KitConfig::new(8), backend, GatePolicy::Bypass, None).unwrap();
    println!(
        "enforced = {}, report accepted = {}",
        kit.is_enforced(),
        kit.report().accepted()
    );
}
