//! End-to-end acceptance: the kit must reproduce the DCT-IV definition, on both arithmetic
//! paths, for every convention the post stage claims to absorb.

use dct4_mdct_kit::backend::{EmulatedMdct, OutputOrdering, SampleFormat};
use dct4_mdct_kit::config::{KitConfig, LengthSet};
use dct4_mdct_kit::dct4::dct4;
use dct4_mdct_kit::fixed::{QFormat, Q};
use dct4_mdct_kit::pipeline::{DctIvKit, GatePolicy};
use dct4_mdct_kit::util::{max_abs, norm2, Rng};

fn kit(n: usize, backend: EmulatedMdct) -> DctIvKit<EmulatedMdct> {
    DctIvKit::new(KitConfig::new(n), backend, GatePolicy::Enforce, None).unwrap()
}

#[test]
fn matches_the_definition_for_every_length() {
    for &n in &[4usize, 6, 8, 12, 16, 32, 64, 128, 256] {
        let mut k = kit(n, EmulatedMdct::ideal("golden"));
        let mut worst = 0.0_f64;
        for trial in 0..4u64 {
            let u = Rng::new(n as u64 * 31 + trial).signed_vec(n);
            let x = k.transform(&u).unwrap();
            let reference = dct4(&u);
            worst = worst.max(max_abs(
                &x.iter()
                    .zip(&reference)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>(),
            ));
        }
        assert!(worst < 1e-11, "N={n}: max deviation {worst:.3e}");
    }
}

#[test]
fn batch_equals_block_by_block() {
    let n = 32;
    let b = 5;
    let mut batched = DctIvKit::new(
        KitConfig::new(n).with_batch(b),
        EmulatedMdct::ideal("golden"),
        GatePolicy::Enforce,
        None,
    )
    .unwrap();
    let mut single = kit(n, EmulatedMdct::ideal("golden"));

    let u = Rng::new(7).signed_vec(b * n);
    let mut x = vec![0.0; b * n];
    let stats = batched.transform_batch(&u, &mut x).unwrap();
    assert_eq!(stats.blocks, b);
    assert_eq!(stats.backend_calls, b as u64);

    for block in 0..b {
        let range = block * n..(block + 1) * n;
        let reference = single.transform(&u[range.clone()]).unwrap();
        for i in 0..n {
            assert!((x[range.start + i] - reference[i]).abs() < 1e-12);
        }
    }
}

#[test]
fn scalar_scaling_and_sign_are_absorbed_by_post() {
    for alpha in [2f64.powi(15), 1024.0, 0.25, 3.0] {
        for sign in [1i8, -1] {
            let cfg = KitConfig::new(24).with_alpha(alpha).with_sign(sign);
            let backend = EmulatedMdct::ideal("scaled-ip").with_scale(alpha);
            let mut k = DctIvKit::new(cfg, backend, GatePolicy::Enforce, None).unwrap();
            let u = Rng::new(5).signed_vec(24);
            let x = k.transform(&u).unwrap();
            let reference: Vec<f64> = dct4(&u).iter().map(|v| sign as f64 * v).collect();
            let err = max_abs(
                &x.iter()
                    .zip(&reference)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>(),
            );
            assert!(
                err < 1e-10,
                "alpha={alpha} sign={sign}: post compensation failed, err={err:.3e}"
            );
        }
    }
}

#[test]
fn declared_permutation_is_undone_in_post() {
    let n = 16;
    // A derangement, so a no-op correction cannot pass by accident.
    let perm: Vec<usize> = (1..n).chain(std::iter::once(0)).collect();
    let backend =
        EmulatedMdct::ideal("permuted-ip").with_ordering(OutputOrdering::Permutation(perm));
    let mut k = kit(n, backend);
    let u = Rng::new(13).signed_vec(n);
    let x = k.transform(&u).unwrap();
    let reference = dct4(&u);
    assert!(
        max_abs(
            &x.iter()
                .zip(&reference)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>()
        ) < 1e-12
    );
}

#[test]
fn fixed_point_path_meets_the_forward_error_bound() {
    let n = 32;
    let cfg = KitConfig::new(n);
    let mut k = DctIvKit::new(
        cfg.clone(),
        EmulatedMdct::ideal("golden"),
        GatePolicy::Enforce,
        None,
    )
    .unwrap();
    let mut rng = Rng::new(99);
    let mut worst_ratio = 0.0_f64;

    for _ in 0..16 {
        let u: Vec<Q> = rng
            .signed_vec(n)
            .into_iter()
            .map(|v| Q::from_f64(cfg.input, v * 0.5))
            .collect();
        let mut x = vec![Q::zero(cfg.output); n];
        let stats = k.transform_block_fixed(&u, &mut x).unwrap();
        assert_eq!(
            stats.fold_overflows, 0,
            "the guard bit must prevent fold overflow"
        );
        assert_eq!(stats.saturations, 0, "the output word is wide enough");

        let unquantized: Vec<f64> = u.iter().map(|q| q.to_f64()).collect();
        let reference = dct4(&unquantized);
        let measured = max_abs(
            &x.iter()
                .zip(&reference)
                .map(|(q, r)| q.to_f64() - r)
                .collect::<Vec<_>>(),
        );
        let bound = k.error_bound(norm2(&unquantized)).total;
        assert!(
            measured <= bound,
            "measured {measured:.3e} exceeds the forward bound {bound:.3e}"
        );
        worst_ratio = worst_ratio.max(measured / bound);
    }
    assert!(worst_ratio < 1.0);
}

#[test]
fn saturation_is_counted_when_the_output_word_is_narrow() {
    // alpha = 1/1024 means the post stage multiplies by 1024. With a Q1.8 output word the
    // post stage must clamp, and the kit must count every clamp.
    let n = 16;
    let cfg = KitConfig::new(n).with_alpha(1.0 / 1024.0).with_formats(
        QFormat::new(16, 14),
        QFormat::new(17, 14),
        QFormat::new(16, 14),
        QFormat::new(16, 14),
    );
    let backend = EmulatedMdct::ideal("small-word-ip").with_scale(1.0 / 1024.0);
    let mut k = DctIvKit::new(cfg.clone(), backend, GatePolicy::Enforce, None).unwrap();

    // A full-scale constant block: the DCT-IV of ones has bins around 5 for N = 16, and the
    // post gain of 1024 pushes them far beyond the Q1.14 output range (+-2).
    let u = vec![Q::from_f64(cfg.input, 1.0); n];
    let mut x = vec![Q::zero(cfg.output); n];
    let stats = k.transform_block_fixed(&u, &mut x).unwrap();

    assert!(stats.saturations > 0, "the narrow output word must clamp");
    assert_eq!(stats.saturations, k.saturations());
    let max_code = cfg.output.max_raw();
    assert!(x.iter().any(|q| q.raw().abs() == max_code));
    assert_eq!(
        stats.fold_overflows, 0,
        "the guard bit absorbs the fold sum of ones"
    );
}

#[test]
fn missing_guard_bit_is_refused_before_any_transform() {
    let mut cfg = KitConfig::new(16);
    cfg.guard = QFormat::new(16, 14); // Win == Wu
    let err = DctIvKit::new(
        cfg,
        EmulatedMdct::ideal("golden"),
        GatePolicy::Enforce,
        None,
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Win >= Wu + 1"), "unexpected message: {msg}");
}

#[test]
fn bypass_mode_is_available_but_flagged() {
    let backend = EmulatedMdct::ideal("windowed-ip")
        .with_window_kind(dct4_mdct_kit::backend::WindowKind::Hann);
    let k = DctIvKit::new(KitConfig::new(16), backend, GatePolicy::Bypass, None).unwrap();
    assert!(!k.is_enforced());
    assert!(!k.report().accepted());
    assert!(k.report().to_markdown().contains("**REJECT**"));
}

#[test]
fn backend_length_roster_is_enforced() {
    let backend = EmulatedMdct::ideal("ip").with_lengths(LengthSet::of(&[8, 16, 32]));
    let mut k = kit(32, backend.clone());
    assert!(k.transform(&vec![0.25; 32]).is_ok());
    let err = DctIvKit::new(KitConfig::new(12), backend, GatePolicy::Enforce, None).unwrap_err();
    assert!(err.to_string().contains("N_ip"));
}

#[test]
fn format_metadata_does_not_change_the_mathematics() {
    // A backend may declare f32/fixed metadata; the kit's decision is numeric, not nominal.
    for fmt in [
        SampleFormat::F32,
        SampleFormat::F64,
        SampleFormat::I32,
        SampleFormat::Fixed(QFormat::new(24, 14)),
    ] {
        let mut k = kit(16, EmulatedMdct::ideal("ip").with_format(fmt));
        let u = Rng::new(17).signed_vec(16);
        let x = k.transform(&u).unwrap();
        let reference = dct4(&u);
        assert!(
            max_abs(
                &x.iter()
                    .zip(&reference)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>()
            ) < 1e-11
        );
        assert_eq!(k.report().format, fmt);
    }
}
