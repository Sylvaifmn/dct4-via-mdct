//! The mathematical obligations of §3, checked numerically and exactly.
//!
//! * **Theorem 1** (folding equivalence): `M_{2N->N}(x) = D_N(F(x))`.
//! * **Theorem 2** (inverse folding): `F(F^{-1}(u)) = u`, i.e. `F∘F^{-1} = Id`.
//! * **Corollary 1**: `P ∈ {0,±1}^{2N×N}`, `||P||_2 = 1`, no arithmetic rounding.
//! * **The chain** of eq. 2: `M(F^{-1}(u)) = D_N(u)`.

use dct4_mdct_kit::dct4::{dct4, Dct4Kernel};
use dct4_mdct_kit::folding::{fold, inverse_fold, InverseFoldMatrix};
use dct4_mdct_kit::mdct::{mdct, MdctKernel};
use dct4_mdct_kit::util::{max_abs, norm2, Rng};

const LENGTHS: [usize; 8] = [4, 6, 8, 12, 16, 32, 64, 128];

fn random_block(n: usize, seed: u64) -> Vec<f64> {
    Rng::new(seed).signed_vec(n)
}

#[test]
fn theorem1_folding_equivalence() {
    for &n in &LENGTHS {
        let kernel = MdctKernel::new(n);
        let direct = Dct4Kernel::new(n);
        for trial in 0..8u64 {
            let x = random_block(2 * n, 1000 * n as u64 + trial);
            let folded = fold(&x);

            let mut expect = vec![0.0; n];
            direct.apply(&folded, &mut expect);
            let mut got = vec![0.0; n];
            kernel.apply(&x, &mut got);

            let err = max_abs(
                &got.iter()
                    .zip(&expect)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>(),
            );
            assert!(
                err < 1e-11,
                "Theorem 1 failed for N={n} trial={trial}: max error {err:.3e}"
            );
        }
    }
}

#[test]
fn theorem2_inverse_folding() {
    for &n in &LENGTHS {
        for trial in 0..8u64 {
            let u = random_block(n, 2000 * n as u64 + trial);
            let x = inverse_fold(&u);
            assert_eq!(x.len(), 2 * n);
            // The middle N samples are structural zeros.
            for (m, v) in x.iter().enumerate().take(3 * n / 2).skip(n / 2) {
                assert_eq!(*v, 0.0, "zero-fill violated at m={m} for N={n}");
            }
            let back = fold(&x);
            let err = max_abs(&back.iter().zip(&u).map(|(a, b)| a - b).collect::<Vec<_>>());
            assert!(
                err == 0.0,
                "F∘F^{{-1}} must be exact, got {err:e} for N={n}"
            );
        }
    }
}

#[test]
fn corollary1_sparse_operator() {
    for &n in &LENGTHS {
        let p = InverseFoldMatrix::new(n);
        assert_eq!(p.nnz(), n, "P has exactly one non-zero per non-zero row");
        assert_eq!(p.rows(), 2 * n);
        assert_eq!(p.cols(), n);

        // Every entry is in {0, +1, -1}.
        for row in p.dense() {
            for v in row {
                assert!(v == 0.0 || v == 1.0 || v == -1.0);
            }
        }

        // ||P||_2 = 1  <=>  P^T P = I.
        assert!(p.is_isometry(0.0), "P is not an isometry for N={n}");

        // The sparse operator and the closed-form map agree.
        let u = random_block(n, 42 + n as u64);
        let via_matrix = p.apply(&u);
        let via_map = inverse_fold(&u);
        assert_eq!(via_matrix, via_map);

        // F∘P = Id, exactly.
        assert!(p.is_exact_inverse());
        assert!(p
            .apply_transpose(&via_map)
            .iter()
            .zip(&u)
            .all(|(a, b)| a == b));
    }
}

#[test]
fn fold_preserves_the_two_norm() {
    for &n in &LENGTHS {
        for trial in 0..4u64 {
            let u = random_block(n, 77 + trial + n as u64);
            let x = inverse_fold(&u);
            assert!(
                (norm2(&x) - norm2(&u)).abs() < 1e-12,
                "the fold is an isometric embedding for N={n}"
            );
        }
    }
}

#[test]
fn complete_chain_of_eq2() {
    // X = POST(MDCT IP(FOLD(U))) reduces to M_{2N->N}(F^{-1}(u)) = D_N(u).
    for &n in &LENGTHS {
        for trial in 0..4u64 {
            let u = random_block(n, 3000 + trial + 7 * n as u64);
            let x = inverse_fold(&u);
            let got = mdct(&x);
            let expect = dct4(&u);
            let err = max_abs(
                &got.iter()
                    .zip(&expect)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>(),
            );
            assert!(err < 1e-11, "chain identity failed for N={n}: {err:.3e}");
        }
    }
}

#[test]
fn integer_folding_is_exact_at_full_scale() {
    // Near the extremes of a 63-bit word, F and F^-1 still round-trip without loss.
    let n = 64;
    let huge: Vec<i64> = (0..n as i64)
        .map(|i| {
            if i % 2 == 0 {
                1i64 << 60
            } else {
                -(1i64 << 60)
            }
        })
        .collect();
    let x = inverse_fold(&huge);
    assert_eq!(fold(&x), huge);
}

#[test]
#[should_panic(expected = "multiple of 4")]
fn fold_rejects_a_non_multiple_of_four() {
    let _ = fold(&[0.0; 6]);
}

#[test]
#[should_panic(expected = "N must be even")]
fn inverse_fold_rejects_an_odd_length() {
    let _ = inverse_fold(&[0.0; 7]);
}
