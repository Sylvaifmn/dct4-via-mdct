//! The reference DCT-IV of §3.2.
//!
//! ```text
//! D_N(u)[k] = sum_{n=0}^{N-1} u[n] * cos[ (pi/N) * (n + 1/2) * (k + 1/2) ],  k = 0..N-1
//! ```
//!
//! This is the *definition* the kit must reproduce. It is implemented here as an analytic
//! `O(N^2)` kernel for three purposes only:
//!
//! 1. it is the ground truth in the test suite and in the compliance probe (H2 compares the
//!    backend pointwise against it);
//! 2. it lets the kit self-report a measured deviation during bring-up;
//! 3. it documents the convention (the `+1/2` on both indices) that the backend must share.
//!
//! It is **not** the production path: the kit itself contains no MDCT/IMDCT core, and the
//! DCT-IV reference here is a verification oracle, not a fast transform.

use std::f64::consts::PI;

/// Kernel entry `cos[(pi/N)(i + 1/2)(k + 1/2)]`.
pub fn kernel_entry(n: usize, i: usize, k: usize) -> f64 {
    let scale = PI / n as f64;
    (scale * (i as f64 + 0.5) * (k as f64 + 0.5)).cos()
}

/// Pre-computed DCT-IV matrix for repeated application.
#[derive(Debug, Clone)]
pub struct Dct4Kernel {
    n: usize,
    /// Row-major `n x n` cosine table, row `k`, column `i`.
    table: Vec<f64>,
}

impl Dct4Kernel {
    /// Builds the kernel for block length `n >= 1`.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "DCT-IV length must be positive");
        let mut table = vec![0.0; n * n];
        for k in 0..n {
            for i in 0..n {
                table[k * n + i] = kernel_entry(n, i, k);
            }
        }
        Self { n, table }
    }

    /// Block length.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Row `k` of the kernel: the sensitivity of output bin `k` to every input sample.
    pub fn row(&self, k: usize) -> &[f64] {
        &self.table[k * self.n..(k + 1) * self.n]
    }

    #[allow(clippy::needless_range_loop)] // the kernel row is indexed by bin
    /// Applies `D_N` to `u`, writing `N` outputs to `out`.
    pub fn apply(&self, u: &[f64], out: &mut [f64]) {
        assert_eq!(u.len(), self.n, "dct4: expected N inputs");
        assert_eq!(out.len(), self.n, "dct4: expected N outputs");
        for k in 0..self.n {
            out[k] = crate::util::dot(self.row(k), u);
        }
    }
}

/// Computes `D_N(u)` straight from the definition.
pub fn dct4(u: &[f64]) -> Vec<f64> {
    let k = Dct4Kernel::new(u.len());
    let mut out = vec![0.0; u.len()];
    k.apply(u, &mut out);
    out
}

/// In-place [`dct4`].
pub fn dct4_into(u: &[f64], out: &mut [f64]) {
    Dct4Kernel::new(u.len()).apply(u, out);
}

/// The DCT-IV is its own inverse up to the factor `2/N`: `D_N(D_N(u)) = (N/2) u`.
///
/// Used as a self-check of the kernel convention in the test suite.
pub fn involution_factor(n: usize) -> f64 {
    n as f64 / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_is_symmetric() {
        let n = 8;
        let d = Dct4Kernel::new(n);
        for i in 0..n {
            for k in 0..n {
                let a = kernel_entry(n, i, k);
                let b = kernel_entry(n, k, i);
                assert!(
                    (a - b).abs() < 1e-13,
                    "kernel entry ({i},{k}) is not symmetric"
                );
                assert!((d.row(k)[i] - a).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn involution_property() {
        // D_N is involutive up to N/2: a strong check that the convention is right.
        let n = 12;
        let u: Vec<f64> = (0..n).map(|i| ((i * 3 + 1) as f64).sin()).collect();
        let y = dct4(&u);
        let z = dct4(&y);
        for i in 0..n {
            assert!((z[i] - (n as f64 / 2.0) * u[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn dc_input_is_not_dc_output() {
        // Sanity: DCT-IV of a constant is a decaying alternating sequence, not a spike.
        let n = 8;
        let u = vec![1.0; n];
        let y = dct4(&u);
        assert!(y.iter().all(|v| v.abs() > 1e-3));
        assert!(y[0] > 0.0 && y[1] < 0.0);
    }
}
