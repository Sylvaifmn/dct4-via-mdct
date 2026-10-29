//! The analytic MDCT of §3.2 — the definition the external backend must realise.
//!
//! ```text
//! M_{2N->N}(x)[k] = sum_{m=0}^{2N-1} x[m] * cos[ (pi/N) * (m + 1/2 + N/2) * (k + 1/2) ]
//! ```
//!
//! Note the **`+ N/2` phase term**. It is the single most common source of a silent
//! mismatch between a vendor IP and this kit, which is why H2 (phase-matched) is a hard
//! precondition and is decided by a pointwise comparison against [`MdctKernel::apply`].
//!
//! Like [`crate::dct4`], this module is a verification oracle (and the basis of the
//! compliance probe), not a fast transform: the kit never implements an MDCT core.

use std::f64::consts::PI;

/// Kernel entry `cos[(pi/N)(m + 1/2 + N/2)(k + 1/2)]`.
pub fn kernel_entry(n: usize, m: usize, k: usize) -> f64 {
    let scale = PI / n as f64;
    (scale * (m as f64 + 0.5 + n as f64 / 2.0) * (k as f64 + 0.5)).cos()
}

/// Pre-computed `2N -> N` MDCT matrix.
#[derive(Debug, Clone)]
pub struct MdctKernel {
    n: usize,
    /// Row-major `n x 2n` table, row `k`, column `m`.
    table: Vec<f64>,
}

impl MdctKernel {
    /// Builds the kernel for block length `n >= 1` (input length `2n`).
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "MDCT length must be positive");
        let mut table = vec![0.0; n * 2 * n];
        for k in 0..n {
            for m in 0..2 * n {
                table[k * 2 * n + m] = kernel_entry(n, m, k);
            }
        }
        Self { n, table }
    }

    /// Block length `N` (the input length is `2N`).
    pub fn n(&self) -> usize {
        self.n
    }

    /// Row `k` (length `2N`).
    pub fn row(&self, k: usize) -> &[f64] {
        &self.table[k * 2 * self.n..(k + 1) * 2 * self.n]
    }

    /// The column of the kernel at input index `m`: the backend response to an impulse
    /// `e_m`, i.e. the ideal probe column used by H1/H2/H3.
    pub fn column(&self, m: usize) -> Vec<f64> {
        assert!(m < 2 * self.n, "MDCT impulse index out of range");
        (0..self.n)
            .map(|k| self.table[k * 2 * self.n + m])
            .collect()
    }

    #[allow(clippy::needless_range_loop)] // the kernel row is indexed by bin
    /// Applies `M_{2N->N}` to `x` of length `2N`, writing `N` outputs to `out`.
    pub fn apply(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(x.len(), 2 * self.n, "mdct: expected 2N inputs");
        assert_eq!(out.len(), self.n, "mdct: expected N outputs");
        for k in 0..self.n {
            out[k] = crate::util::dot(self.row(k), x);
        }
    }
}

/// Computes `M_{2N->N}(x)` straight from the definition.
pub fn mdct(x: &[f64]) -> Vec<f64> {
    assert!(x.len() % 2 == 0, "mdct: input length must be 2N");
    let n = x.len() / 2;
    let k = MdctKernel::new(n);
    let mut out = vec![0.0; n];
    k.apply(x, &mut out);
    out
}

/// In-place [`mdct`].
pub fn mdct_into(x: &[f64], out: &mut [f64]) {
    assert!(x.len() % 2 == 0, "mdct: input length must be 2N");
    MdctKernel::new(x.len() / 2).apply(x, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_entry_matches_table() {
        let n = 8;
        let k = MdctKernel::new(n);
        for m in 0..2 * n {
            for b in 0..n {
                assert!((k.row(b)[m] - kernel_entry(n, m, b)).abs() < 1e-15);
            }
            let col = k.column(m);
            assert_eq!(col.len(), n);
            assert!((col[3] - kernel_entry(n, m, 3)).abs() < 1e-15);
        }
    }

    #[test]
    fn phase_shift_is_real() {
        // Dropping the +N/2 phase term is *not* a harmless convention change.
        let n = 8;
        let x: Vec<f64> = (0..2 * n).map(|i| ((i * 5 + 2) as f64).cos()).collect();
        let a = mdct(&x);
        let b: Vec<f64> = (0..n)
            .map(|b| {
                (0..2 * n)
                    .map(|m| x[m] * (PI / n as f64 * (m as f64 + 0.5) * (b as f64 + 0.5)).cos())
                    .sum()
            })
            .collect();
        let diff: f64 = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(
            diff > 1e-3,
            "phase mismatch should be visible, diff = {diff}"
        );
    }
}
