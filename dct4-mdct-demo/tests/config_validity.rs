//! The validity constraint of eq. 3, clause by clause.
//!
//! ```text
//! valid <=> (N mod 2 = 0) && (N in N_ip) && (Win >= Wu + 1) && (Wout >= Wip)
//! ```
//!
//! The kit must refuse an invalid configuration *before* touching a backend, and the refusal
//! must name the violated clause.

use dct4_mdct_kit::backend::EmulatedMdct;
use dct4_mdct_kit::config::{ConfigViolation, KitConfig, LengthSet};
use dct4_mdct_kit::error::KitError;
use dct4_mdct_kit::fixed::QFormat;
use dct4_mdct_kit::folding::fold_is_exact;
use dct4_mdct_kit::pipeline::{DctIvKit, GatePolicy};

fn err(cfg: &KitConfig) -> Vec<ConfigViolation> {
    cfg.validate().expect_err("configuration must be invalid")
}

#[test]
fn valid_configurations() {
    for n in [4usize, 6, 8, 12, 32, 256, 1024, 4096] {
        assert_eq!(KitConfig::new(n).validate(), Ok(()), "N = {n}");
        assert!(DctIvKit::new(
            KitConfig::new(n),
            EmulatedMdct::ideal("golden"),
            GatePolicy::Enforce,
            None
        )
        .is_ok());
    }
}

#[test]
fn clause1_n_must_be_even() {
    let cfg = KitConfig::new(31);
    let v = err(&cfg);
    assert!(v.contains(&ConfigViolation::OddLength { n: 31 }));
    let msg = KitError::InvalidConfig(v).to_string();
    assert!(msg.contains("odd"), "{msg}");
    assert!(msg.contains("clause 1"), "{msg}");
}

#[test]
fn clause2_n_must_be_in_the_backend_roster() {
    let cfg = KitConfig::new(24).with_lengths(LengthSet::of(&[8, 12, 16]));
    let v = err(&cfg);
    assert!(matches!(
        v.as_slice(),
        [ConfigViolation::LengthNotSupported { n: 24, .. }]
    ));
    assert!(v[0].to_string().contains("N_ip"));
}

#[test]
fn clause3_the_guard_bit_is_mandatory() {
    for missing in [0u8, 1] {
        let mut cfg = KitConfig::new(16);
        cfg.guard = QFormat::new(cfg.input.width - missing, cfg.input.frac);
        let v = err(&cfg);
        assert!(
            v.iter()
                .any(|x| matches!(x, ConfigViolation::MissingGuardBit { .. })),
            "Win = Wu - {missing} must be rejected: {v:?}"
        );
        assert!(v.iter().any(|x| x.to_string().contains("Win >= Wu + 1")));
    }
    // Wu + 1 is exactly admissible.
    let mut cfg = KitConfig::new(16);
    cfg.guard = QFormat::new(cfg.input.width + 1, cfg.input.frac);
    assert!(cfg.validate().is_ok());
}

#[test]
fn clause4_the_output_must_not_narrow_the_backend_word() {
    let mut cfg = KitConfig::new(16);
    cfg.output = QFormat::new(cfg.ip.width - 1, cfg.ip.frac);
    let v = err(&cfg);
    assert!(v
        .iter()
        .any(|x| matches!(x, ConfigViolation::OutputTooNarrow { .. })));
    assert!(v.iter().any(|x| x.to_string().contains("Wout >= Wip")));
    // Equal widths are admissible.
    let mut cfg = KitConfig::new(16);
    cfg.output = cfg.ip;
    assert!(cfg.validate().is_ok());
}

#[test]
fn structural_rules_are_enforced() {
    assert!(err(&KitConfig::new(8).with_alpha(0.0))
        .iter()
        .any(|v| matches!(v, ConfigViolation::InvalidAlpha { .. })));
    assert!(err(&KitConfig::new(8).with_alpha(-1.0))
        .iter()
        .any(|v| matches!(v, ConfigViolation::InvalidAlpha { .. })));
    assert!(err(&KitConfig::new(8).with_alpha(f64::NAN))
        .iter()
        .any(|v| matches!(v, ConfigViolation::InvalidAlpha { .. })));
    assert!(err(&KitConfig::new(8).with_sign(0))
        .iter()
        .any(|v| matches!(v, ConfigViolation::InvalidSign { .. })));
    assert!(err(&KitConfig::new(8).with_batch(0)).contains(&ConfigViolation::EmptyBatch));
    assert!(err(&KitConfig::new(8).with_pool_slots(0)).contains(&ConfigViolation::EmptyPool));

    // The folding stage is exact only when the fractional point does not move between Wu and
    // Win; a mismatched guard format is a structural error.
    let mut cfg = KitConfig::new(8);
    cfg.guard = QFormat::new(17, 13);
    assert!(err(&cfg)
        .iter()
        .any(|v| matches!(v, ConfigViolation::FractionMismatch { .. })));

    // A degenerate format is rejected.
    let mut cfg = KitConfig::new(8);
    cfg.ip = QFormat::new(1, 0);
    assert!(err(&cfg)
        .iter()
        .any(|v| matches!(v, ConfigViolation::InvalidFormat { .. })));
}

#[test]
fn all_violations_are_collected_not_just_the_first() {
    let mut cfg = KitConfig::new(9); // odd
    cfg.guard = QFormat::new(16, 14); // no guard bit
    cfg.output = QFormat::new(12, 14); // narrower than ip (and invalid: frac >= width)
    cfg.sign = 7;
    let v = err(&cfg);
    assert!(v.len() >= 5, "expected a full list, got {v:?}");
    let msg = KitError::InvalidConfig(v).to_string();
    assert!(msg.contains("violation(s)"), "{msg}");
}

#[test]
fn an_invalid_configuration_never_reaches_the_backend() {
    // The backend would panic if it were probed; the gate must reject first.
    let cfg = KitConfig::new(7);
    let e = DctIvKit::new(
        cfg,
        EmulatedMdct::ideal("golden"),
        GatePolicy::Enforce,
        None,
    )
    .unwrap_err();
    assert!(matches!(e, KitError::InvalidConfig(_)));
    assert!(e.to_string().contains("odd"));
}

#[test]
fn eq3_truth_table() {
    // Enumerate the four clauses over a small grid and compare `validate` with the predicate.
    for n in [4usize, 6, 7, 8] {
        for lengths in [LengthSet::any_even(), LengthSet::of(&[4, 8])] {
            for guard_extra in [0u8, 1, 2] {
                for out_slack in [-1i16, 0, 1] {
                    let mut cfg = KitConfig::new(n).with_lengths(lengths.clone());
                    cfg.guard = QFormat::new(cfg.input.width + guard_extra, cfg.input.frac);
                    let wout = (cfg.ip.width as i16 + out_slack).max(2) as u8;
                    cfg.output = QFormat::new(wout, (wout - 1).min(cfg.ip.frac));
                    let expected = n % 2 == 0
                        && n >= 4
                        && lengths.contains(n)
                        && cfg.guard.width > cfg.input.width
                        && cfg.output.width >= cfg.ip.width
                        && cfg.output.validate().is_ok();
                    assert_eq!(
                        cfg.validate().is_ok(),
                        expected,
                        "N={n} lengths={lengths} guard_extra={guard_extra} out_slack={out_slack}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_guard_bit_is_what_makes_the_fold_exact() {
    // Full-scale input at Wu: the fold sums two words, so Wu cannot hold the result.
    let wu = QFormat::new(16, 14);
    assert!(
        !fold_is_exact(16, wu, wu),
        "a Wu-wide folded word must overflow on a full-scale block"
    );
    let win = QFormat::new(17, 14);
    assert!(fold_is_exact(16, wu, win), "Wu + 1 absorbs the fold sum");
    // Exactly one extra bit, not two: 18 bits is enough but unnecessary.
    assert!(fold_is_exact(16, wu, QFormat::new(18, 14)));
}

#[test]
fn post_stage_scalars() {
    let cfg = KitConfig::new(8).with_alpha(4.0);
    assert_eq!(cfg.post_scale(), 0.25);
    assert_eq!(cfg.signed_post_gain(), 0.25);
    assert_eq!(cfg.clone().with_sign(-1).signed_post_gain(), -0.25);
    assert_eq!(cfg.folded_len(), 16);
    assert_eq!(cfg.with_batch(3).batch_samples(), 24);
}

#[test]
fn length_set_semantics() {
    let any = LengthSet::AnyEven;
    assert!(any.contains(4) && any.contains(1024) && !any.contains(3) && !any.contains(2));
    let explicit = LengthSet::of(&[32, 8, 8, 16]);
    assert_eq!(explicit.describe(), "{8, 16, 32}");
    assert!(explicit.contains(16) && !explicit.contains(64));
    assert!(explicit.to_string().contains("16"));
}
