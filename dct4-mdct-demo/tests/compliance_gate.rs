//! The compliance gate: every `backend_reject` path, and the `tau_dtype` scale argument.
//!
//! These tests are the executable form of §4.2 and §8: a hard precondition fails if and only
//! if the corresponding decidable criterion fails, and the refusal is accompanied by a
//! classified vulnerability and a remedy.

use dct4_mdct_kit::backend::{EmulatedMdct, OutputOrdering, SampleFormat, WindowKind};
use dct4_mdct_kit::compliance::{
    evaluate, ComplianceReport, Precondition, Thresholds, Vulnerability,
};
use dct4_mdct_kit::config::KitConfig;
use dct4_mdct_kit::fixed::QFormat;
use dct4_mdct_kit::numerics;
use dct4_mdct_kit::pipeline::{DctIvKit, GatePolicy};

fn cfg(n: usize) -> KitConfig {
    KitConfig::new(n)
}

fn report(backend: &mut EmulatedMdct, c: &KitConfig) -> ComplianceReport {
    evaluate(backend, c, None).expect("probing must not fail")
}

#[test]
fn ideal_backend_passes_all_four() {
    let mut b = EmulatedMdct::ideal("golden-ip");
    let r = report(&mut b, &cfg(32));
    assert!(r.accepted(), "{}", r.to_markdown());
    assert_eq!(r.checks.len(), 4);
    for p in Precondition::ALL {
        let c = r.check(p).unwrap();
        assert!(c.passed, "{} should pass: {}", p.code(), c.detail);
    }
    assert!((r.measured.alpha - 1.0).abs() < 1e-12);
    assert!(r.measured.scale_spread < 1e-12);
    assert!(r.measured.phase_error < 1e-13);
}

#[test]
fn windowing_is_rejected_by_h1() {
    for kind in [WindowKind::Hann, WindowKind::Sine] {
        let mut b = EmulatedMdct::ideal("windowed").with_window_kind(kind);
        let r = report(&mut b, &cfg(32));
        assert!(!r.accepted(), "{kind:?} must be rejected");
        assert!(!r.check(Precondition::H1WindowFree).unwrap().passed);
        let v = r
            .violations
            .iter()
            .find(|v| v.precondition == Precondition::H1WindowFree)
            .expect("an H1 violation");
        assert_eq!(v.category, Vulnerability::WindowContamination);
        assert!(v.remedy.contains("taper"));
        assert!(v.detail.contains("implied window") || v.detail.contains("deviates"));
    }
}

#[test]
fn phase_offsets_are_rejected_by_h2() {
    for phase in [0.01, 0.3, -1.2] {
        let mut b = EmulatedMdct::ideal("phase").with_phase(phase);
        let r = report(&mut b, &cfg(16));
        assert!(!r.accepted(), "phase {phase} must be rejected");
        assert!(!r.check(Precondition::H2PhaseMatched).unwrap().passed);
        let v = r
            .violations
            .iter()
            .find(|v| v.precondition == Precondition::H2PhaseMatched)
            .expect("an H2 violation");
        assert_eq!(v.category, Vulnerability::PhaseMismatch);
        assert!(v.remedy.contains("N/2"));
    }
}

#[test]
fn per_bin_gain_drift_is_rejected_by_h3() {
    for drift in [0.001, 0.05] {
        let mut b = EmulatedMdct::ideal("drift").with_gain_drift(drift);
        let r = report(&mut b, &cfg(32));
        assert!(!r.accepted(), "drift {drift} must be rejected");
        let c = r.check(Precondition::H3ScalarScaling).unwrap();
        assert!(!c.passed);
        assert!(
            r.measured.scale_spread > r.thresholds.tau_scale,
            "spread {} vs tau {}",
            r.measured.scale_spread,
            r.thresholds.tau_scale
        );
        let v = r
            .violations
            .iter()
            .find(|v| v.precondition == Precondition::H3ScalarScaling)
            .expect("an H3 violation");
        assert_eq!(v.category, Vulnerability::ScalingDrift);
    }
}

#[test]
fn a_scale_that_does_not_match_the_contract_is_rejected() {
    let mut b = EmulatedMdct::ideal("scaled").with_scale(4.0);
    let mut c = cfg(16);
    c.alpha = 1.0;
    let r = report(&mut b, &c);
    assert!(!r.accepted());
    assert!(!r.check(Precondition::H3ScalarScaling).unwrap().passed);
}

#[test]
fn the_declared_scale_must_match_the_configured_one() {
    // The IP delivers the gain the kit compensates for, but its data sheet claims another
    // value: the descriptor and the configuration disagree, which is an H3 violation.
    let mut b = EmulatedMdct::ideal("liar")
        .with_scale(2.0)
        .with_declared_scale(3.0);
    let mut c = cfg(16);
    c.alpha = 2.0;
    let r = report(&mut b, &c);
    assert!(
        (r.measured.alpha - 2.0).abs() < 1e-12,
        "the probe measures the truth"
    );
    assert!(!r.accepted(), "{}", r.to_markdown());
    let v = r
        .violations
        .iter()
        .find(|v| v.precondition == Precondition::H3ScalarScaling)
        .expect("an H3 violation");
    assert!(v.detail.contains("declared alpha"), "detail: {}", v.detail);
}

#[test]
fn unspecified_ordering_fails_closed() {
    let mut b = EmulatedMdct::ideal("shuffled").with_ordering(OutputOrdering::Unspecified);
    let r = report(&mut b, &cfg(16));
    assert!(!r.accepted());
    assert!(!r.check(Precondition::H4NaturalOrdering).unwrap().passed);
    // H1-H3 are undecidable without a bin order, so they must fail closed rather than pass.
    for p in [
        Precondition::H1WindowFree,
        Precondition::H2PhaseMatched,
        Precondition::H3ScalarScaling,
    ] {
        let c = r.check(p).unwrap();
        assert!(!c.passed, "{} must fail closed", p.code());
        assert!(c.metric.is_nan());
    }
    assert!(r
        .violations
        .iter()
        .all(|v| v.category == Vulnerability::OrderingAmbiguity));
}

#[test]
fn a_declared_permutation_is_accepted_as_correctable() {
    for perm in [
        vec![1usize, 0, 3, 2],
        vec![3, 0, 2, 1],
        (1..16).chain(std::iter::once(0)).collect::<Vec<_>>(),
    ] {
        let n = perm.len();
        let mut b = EmulatedMdct::ideal("perm").with_ordering(OutputOrdering::Permutation(perm));
        let r = report(&mut b, &cfg(n));
        assert!(r.accepted(), "{}", r.to_markdown());
        assert!(r.check(Precondition::H4NaturalOrdering).unwrap().passed);
    }
}

#[test]
fn malformed_permutations_are_rejected() {
    for perm in [
        vec![0usize, 0, 1, 2],  // duplicate
        vec![0usize, 1, 2, 99], // out of range
        vec![1usize, 0],        // wrong length for N = 4
    ] {
        let mut b =
            EmulatedMdct::ideal("bad-perm").with_ordering(OutputOrdering::Permutation(perm));
        let r = report(&mut b, &cfg(4));
        assert!(!r.accepted());
        assert!(!r.check(Precondition::H4NaturalOrdering).unwrap().passed);
        assert_eq!(r.checks.iter().filter(|c| !c.passed).count(), 4);
    }
}

#[test]
fn rejection_is_surfaced_by_the_kit_with_remedies() {
    let backend = EmulatedMdct::ideal("windowed").with_window_kind(WindowKind::Sine);
    let err = DctIvKit::new(cfg(16), backend, GatePolicy::Enforce, None).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("backend_reject"), "message was: {msg}");
    assert!(msg.contains("H1"), "message was: {msg}");
    assert!(msg.contains("remedy"), "message was: {msg}");
}

#[test]
fn precision_is_judged_against_the_declared_format() {
    // The same 12-bit truncation is acceptable for a Q1.12 backend and unacceptable for a
    // backend that claims f64: tau_dtype is what makes that distinction decidable.
    let n = 16;
    let quantized = |fmt: SampleFormat| {
        EmulatedMdct::ideal("quantizing-ip")
            .with_format(fmt)
            .with_quant_bits(None, Some(13))
    };

    let mut honest = quantized(SampleFormat::Fixed(QFormat::new(13, 12)));
    let r_honest = report(&mut honest, &cfg(n));
    assert!(r_honest.accepted(), "{}", r_honest.to_markdown());

    let mut overclaiming = quantized(SampleFormat::F64);
    let r_overclaiming = report(&mut overclaiming, &cfg(n));
    assert!(!r_overclaiming.accepted());
    assert!(
        numerics::tau_dtype(SampleFormat::Fixed(QFormat::new(13, 12)), n)
            > 1_000.0 * numerics::tau_dtype(SampleFormat::F64, n)
    );
}

#[test]
fn thresholds_follow_the_format_precision() {
    let f64_th = Thresholds::for_format(SampleFormat::F64, 16, 3);
    let f32_th = Thresholds::for_format(SampleFormat::F32, 16, 3);
    let q12_th = Thresholds::for_format(SampleFormat::Fixed(QFormat::new(13, 12)), 16, 3);
    assert!(f64_th.tau_scale < f32_th.tau_scale);
    assert!(f32_th.tau_scale < q12_th.tau_scale);
    assert!(f64_th.eps_phase < f32_th.eps_phase);
    assert_eq!(f64_th.probe_vectors, 3);
}

#[test]
fn explicit_thresholds_are_used_verbatim() {
    let mut b = EmulatedMdct::ideal("golden");
    let tight = Thresholds {
        tau_scale: 1e-18,
        eps_phase: 1e-18,
        window_tol: 1e-18,
        alpha_tol: 1e-18,
        probe_vectors: 5,
    };
    let r = evaluate(&mut b, &cfg(16), Some(tight)).unwrap();
    assert_eq!(r.thresholds, tight);
    assert_eq!(r.thresholds.probe_vectors, 5);
    assert!(r.accepted(), "f64 arithmetic still passes 1e-18 thresholds");
}

#[test]
fn markdown_report_contains_the_audit_trail() {
    let mut b = EmulatedMdct::ideal("audit");
    let r = report(&mut b, &cfg(8));
    let md = r.to_markdown();
    for needle in [
        "Decision: **ACCEPT**",
        "| H1 |",
        "| H2 |",
        "| H3 |",
        "| H4 |",
        "Measured `alpha`",
        "tau_dtype",
    ] {
        assert!(md.contains(needle), "missing {needle:?} in:\n{md}");
    }
    assert!(r.gate().is_ok());

    let mut bad = EmulatedMdct::ideal("audit-bad").with_window_kind(WindowKind::Hann);
    let rb = report(&mut bad, &cfg(8));
    let mdb = rb.to_markdown();
    assert!(mdb.contains("REJECT"));
    assert!(mdb.contains("## Violations"));
    assert!(rb.gate().is_err());
}
