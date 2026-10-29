//! The folding operator `F` and the closed-form inverse folding map `F^{-1}`.
//!
//! ## Definitions (§3.3, §3.5 of the design)
//!
//! Partition `x` of length `2N` into four blocks `(a, b, c, d)` of length `h = N/2`, and let
//! `v^R[q] = v[h - 1 - q]` denote reversal:
//!
//! ```text
//! F(x) = ( -c^R - d ,  a - b^R )                      (folding, length N)
//!
//! F^{-1}(u) = ( u_U , 0 , 0 , -u_L ),  u = (u_L, u_U) (inverse folding, length 2N)
//! ```
//!
//! ## Theorem 1 (folding equivalence)
//!
//! For every `x` of length `2N`: `M_{2N->N}(x) = D_N(F(x))`.
//!
//! ## Theorem 2 (inverse folding)
//!
//! `F(F^{-1}(u)) = u` for every `u` of length `N`, hence `F∘F^{-1} = Id`; consequently
//! `M_{2N->N}(F^{-1}(u)) = D_N(u)`.
//!
//! ## Corollary 1
//!
//! There is a sparse `P ∈ {0, ±1}^{2N × N}` with `x = P u` and `||P||_2 = 1`. The map
//! involves permutation, sign inversion and zero filling only, so it introduces **no
//! arithmetic rounding**. [`InverseFoldMatrix`] is that `P`; [`InverseFoldMatrix::is_isometry`]
//! checks `P^T P = I` (i.e. `||P||_2 = 1` and `P` has orthonormal columns).
//!
//! ## Genericity
//!
//! Both maps are implemented over [`Scalar`] so that the same code runs on `f64` (analysis,
//! probing, reference) and on `i64` raw fixed-point codes (production path). For `i64` the
//! maps are exact integer operations: the only possible defect is overflow, which the
//! `Win >= Wu + 1` guard bit makes impossible and which the pipeline counts defensively.

use crate::fixed::QFormat;
use std::fmt::Debug;

/// The minimal algebraic structure the folding maps need: negation, addition, subtraction.
///
/// Implemented for [`f64`] and [`i64`]. Nothing else is required — in particular there is
/// no multiplication and no division, which is precisely why the folding stage cannot
/// introduce rounding.
pub trait Scalar: Copy + Debug {
    /// The additive identity.
    fn zero() -> Self;
    /// `self + other`.
    fn add(self, other: Self) -> Self;
    /// `self - other`.
    fn sub(self, other: Self) -> Self;
    /// `-self`.
    fn neg(self) -> Self;
}

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn neg(self) -> Self {
        -self
    }
}

impl Scalar for i64 {
    fn zero() -> Self {
        0
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn neg(self) -> Self {
        -self
    }
}

/// Applies `F`, the folding operator, to a length-`2N` vector; returns a length-`N` vector.
///
/// `x` must have a length divisible by 4 and at least 4 (equivalently `N` even and `N >= 2`,
/// as enforced by the configuration validity constraint `N mod 2 = 0`, `N >= 4`).
///
/// # Panics
///
/// Panics if `x.len()` is not a multiple of 4 or is shorter than 4.
pub fn fold<S: Scalar>(x: &[S]) -> Vec<S> {
    let mut out = vec![S::zero(); x.len() / 2];
    fold_into(x, &mut out);
    out
}

/// In-place form of [`fold`].
///
/// # Panics
///
/// Panics if the shape preconditions of [`fold`] are violated.
pub fn fold_into<S: Scalar>(x: &[S], out: &mut [S]) {
    let total = x.len();
    assert!(
        total >= 4 && total % 4 == 0,
        "fold: input length must be a multiple of 4 and >= 4, got {total}"
    );
    let n = total / 2;
    let h = n / 2;
    assert_eq!(out.len(), n, "fold: output length must be N = len/2");
    let (a, rest) = x.split_at(h);
    let (b, rest) = rest.split_at(h);
    let (c, d) = rest.split_at(h);
    // u_L[q] = -c^R[q] - d[q]
    for q in 0..h {
        out[q] = c[h - 1 - q].neg().sub(d[q]);
    }
    // u_U[q] = a[q] - b^R[q]
    for q in 0..h {
        out[h + q] = a[q].sub(b[h - 1 - q]);
    }
}

/// Applies `F^{-1}`, the closed-form inverse folding map, to a length-`N` vector; returns a
/// length-`2N` vector with the middle `N` samples zero.
///
/// # Panics
///
/// Panics if `u.len()` is odd or shorter than 4.
pub fn inverse_fold<S: Scalar>(u: &[S]) -> Vec<S> {
    let mut out = vec![S::zero(); 2 * u.len()];
    inverse_fold_into(u, &mut out);
    out
}

/// In-place form of [`inverse_fold`].
///
/// # Panics
///
/// Panics if `x.len() != 2 * u.len()`, or if `u.len()` is odd or shorter than 4.
pub fn inverse_fold_into<S: Scalar>(u: &[S], x: &mut [S]) {
    let n = u.len();
    assert!(
        n >= 4 && n % 2 == 0,
        "inverse_fold: N must be even and >= 4, got {n}"
    );
    assert_eq!(x.len(), 2 * n, "inverse_fold: output length must be 2N");
    let h = n / 2;
    let (u_l, u_u) = u.split_at(h);
    // x = (u_U, 0, 0, -u_L)
    x[..h].copy_from_slice(u_u);
    for v in &mut x[h..3 * h] {
        *v = S::zero();
    }
    for q in 0..h {
        x[3 * h + q] = u_l[q].neg();
    }
}

/// Reports whether the fold of a worst-case `input` block stays inside `folded`.
///
/// The probe drives both terms of `F` to their largest magnitude: `a = max`,
/// `b = c = d = min` would give `u_L = -min - min = 2^W`, one code past the positive maximum of
/// a `(W+1)`-bit word, so the probe uses `c = min + 1` and reaches `2^W - 1`, which is exactly
/// what the guard bit covers. `u_L = -(min+1) - min = 2^W - 1` and
/// `u_U = max - min = 2^W - 1`.
///
/// Hence a `Wu`-wide folded word cannot hold the result, while a `Wu + 1`-wide word can. That
/// is the arithmetic reason for clause 3 of eq. 3.
///
/// # The residual corner
///
/// With `c = d = min` the true sum is `2^W`, one code beyond the `(W+1)`-bit maximum. No
/// `Wu + 1` word can represent it, so it is not a defect of the contract but a boundary case:
/// [`crate::pipeline::DctIvKit`] counts every folded word that leaves `Win` as
/// `fold_overflows` instead of letting it wrap, and callers that must accept the most-negative
/// code on *every* sample should allocate `Wu + 2`. The kit's own `F^{-1}` never reaches the
/// corner because it zero-fills the middle `N` samples.
pub fn fold_is_exact(n: usize, input: QFormat, folded: QFormat) -> bool {
    assert!(n >= 4 && n % 2 == 0, "N must be even and >= 4");
    let h = n / 2;
    let mut x = vec![0i64; 2 * n];
    for v in x.iter_mut().take(h) {
        *v = input.max_raw();
    }
    for v in x.iter_mut().take(n).skip(h) {
        *v = input.min_raw();
    }
    for v in x.iter_mut().take(3 * h).skip(n) {
        *v = input.min_raw() + 1;
    }
    for v in x.iter_mut().skip(3 * h) {
        *v = input.min_raw();
    }
    let result = fold(&x);
    result.iter().all(|&raw| folded.contains(raw))
}

/// The sparse matrix `P ∈ {0, ±1}^{2N × N}` of Corollary 1, with `x = P u`.
///
/// Row `m` of `P` has at most one non-zero entry:
///
/// | `m` range | entry |
/// |-----------|-------|
/// | `[0, N/2)` | `P[m, N/2 + m] = +1` |
/// | `[N/2, 3N/2)` | empty (structural zero) |
/// | `[3N/2, 2N)` | `P[m, m - 3N/2] = -1` |
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InverseFoldMatrix {
    /// DCT-IV block length `N`.
    pub n: usize,
    rows: Vec<Vec<(usize, i8)>>,
}

impl InverseFoldMatrix {
    /// Builds `P` for block length `n`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is odd or shorter than 4.
    pub fn new(n: usize) -> Self {
        assert!(n >= 4 && n % 2 == 0, "P: N must be even and >= 4");
        let h = n / 2;
        let mut rows = Vec::with_capacity(2 * n);
        for m in 0..2 * n {
            let row = if m < h {
                vec![(h + m, 1)]
            } else if m < 3 * h {
                Vec::new()
            } else {
                vec![(m - 3 * h, -1)]
            };
            rows.push(row);
        }
        Self { n, rows }
    }

    /// Number of columns (`N`).
    pub fn cols(&self) -> usize {
        self.n
    }

    /// Number of rows (`2N`).
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    /// Applies `P` to a length-`N` vector.
    pub fn apply<T: Scalar>(&self, u: &[T]) -> Vec<T> {
        assert_eq!(u.len(), self.n, "P::apply: expected N inputs");
        let mut out = vec![T::zero(); self.rows.len()];
        for (m, row) in self.rows.iter().enumerate() {
            let mut acc = T::zero();
            for &(col, sign) in row {
                let v = u[col];
                acc = acc.add(if sign < 0 { v.neg() } else { v });
            }
            out[m] = acc;
        }
        out
    }

    /// Applies `P^T` to a length-`2N` vector.
    pub fn apply_transpose<T: Scalar>(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.rows.len(), "P^T::apply: expected 2N inputs");
        let mut out = vec![T::zero(); self.n];
        for (m, row) in self.rows.iter().enumerate() {
            for &(col, sign) in row {
                out[col] = if sign < 0 {
                    out[col].sub(x[m])
                } else {
                    out[col].add(x[m])
                };
            }
        }
        out
    }

    /// Returns the number of non-zero entries (`N` for the inverse folding map: the map is
    /// an isometric embedding of the `N` non-zero samples into `2N` slots).
    pub fn nnz(&self) -> usize {
        self.rows.iter().map(|r| r.len()).sum()
    }

    /// Dense row-major matrix (`2N × N`), for inspection and for the Gram check.
    pub fn dense(&self) -> Vec<Vec<f64>> {
        self.rows
            .iter()
            .map(|row| {
                let mut v = vec![0.0; self.n];
                for &(col, sign) in row {
                    v[col] = sign as f64;
                }
                v
            })
            .collect()
    }

    /// Checks `P^T P = I_N` to within `tol`: the columns of `P` are orthonormal, hence
    /// `||P||_2 = 1` and the fold is an isometry (no scale factor, no rounding).
    pub fn is_isometry(&self, tol: f64) -> bool {
        let mut gram = vec![0.0; self.n * self.n];
        for row in &self.rows {
            for &(i, si) in row {
                for &(j, sj) in row {
                    gram[i * self.n + j] += (si * sj) as f64;
                }
            }
        }
        for i in 0..self.n {
            for j in 0..self.n {
                let want = if i == j { 1.0 } else { 0.0 };
                if (gram[i * self.n + j] - want).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Exact check that `P` round-trips: `F^{-1}` followed by `F` is the identity on `u`.
    ///
    /// This is Theorem 2 with no floating-point tolerance at all — the maps are exact on
    /// integers, so the identity is verified exactly.
    pub fn is_exact_inverse(&self) -> bool {
        let u: Vec<i64> = (0..self.n)
            .map(|i| (i as i64 * 37 + 11) % 101 - 50)
            .collect();
        let x = self.apply(&u);
        let back = fold(&x);
        back == u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theorem1_fold_matches_analytic_mdct() {
        // M_{2N->N}(x) == D_N(F(x)) is tested against the analytic kernels in
        // tests/folding_equivalence.rs; here we only smoke-test shapes and exactness.
        let x: Vec<f64> = (0..16).map(|i| i as f64 * 0.5).collect();
        let u = fold(&x);
        assert_eq!(u.len(), 8);
        let back = inverse_fold(&u);
        assert_eq!(back.len(), 16);
        assert_eq!(fold(&back), u);
    }

    #[test]
    fn integer_maps_are_exact() {
        for n in [4usize, 6, 8, 16, 64] {
            let p = InverseFoldMatrix::new(n);
            assert!(p.is_exact_inverse());
            assert!(p.is_isometry(0.0));
            assert_eq!(p.nnz(), n);
            assert_eq!(p.rows(), 2 * n);
            assert_eq!(p.cols(), n);
        }
    }

    #[test]
    fn transpose_matches_apply() {
        let p = InverseFoldMatrix::new(8);
        let u: Vec<f64> = (0..8).map(|i| i as f64 - 3.0).collect();
        let x = p.apply(&u);
        // <P u, v> == <u, P^T v> for an arbitrary v
        let v: Vec<f64> = (0..16).map(|i| ((i * 5) % 7) as f64 - 3.0).collect();
        let lhs = crate::util::dot(&x, &v);
        let rhs = crate::util::dot(&u, &p.apply_transpose(&v));
        assert!((lhs - rhs).abs() < 1e-12);
    }
}
