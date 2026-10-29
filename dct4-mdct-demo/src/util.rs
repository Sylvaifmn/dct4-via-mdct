//! Deterministic numerics helpers.
//!
//! The crate has no dependencies, so it ships a tiny xorshift PRNG for reproducible
//! probe vectors and a handful of array kernels used by the compliance stage and tests.

/// Deterministic xorshift64* generator.
///
/// Used for backend probe vectors so that compliance decisions are reproducible across
/// machines and runs — a requirement for a *decidable* acceptance criterion.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Creates a generator from a seed. A zero seed is remapped to a fixed non-zero state.
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    /// Next raw 64-bit word.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform sample in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // 53 significant bits, exactly representable.
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform sample in `[-1, 1)`.
    pub fn next_signed(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }

    /// Fills `out` with uniform samples in `[-1, 1)`.
    pub fn fill_signed(&mut self, out: &mut [f64]) {
        for v in out {
            *v = self.next_signed();
        }
    }

    /// Returns a vector of `len` uniform samples in `[-1, 1)`.
    pub fn signed_vec(&mut self, len: usize) -> Vec<f64> {
        let mut v = vec![0.0; len];
        self.fill_signed(&mut v);
        v
    }
}

/// Maximum absolute value of a slice (`0.0` for an empty slice).
pub fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// Euclidean (L2) norm.
pub fn norm2(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Euclidean inner product. Panics only on length mismatch in debug builds.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Natural log-free relative error: `max|a-b| / (1 + max|b|)`.
///
/// The `+1` in the denominator keeps the metric meaningful for near-silent input vectors,
/// which matters because the compliance probe must not report a huge relative error for a
/// vector whose reference output happens to be tiny.
pub fn rel_max_err(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
        / (1.0 + max_abs(b))
}

/// Median of a slice (copies and sorts; returns `0.0` when empty).
pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut c = v.to_vec();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = c.len() / 2;
    if c.len() % 2 == 0 {
        0.5 * (c[mid - 1] + c[mid])
    } else {
        c[mid]
    }
}

/// Arithmetic mean of a slice (`0.0` when empty).
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Unit impulse vector of length `n` with a one at `m`.
pub fn impulse(n: usize, m: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    if m < n {
        v[m] = 1.0;
    }
    v
}

/// Smallest even `n >= 4` (used to normalise probe lengths, since the folding stage needs
/// `N mod 2 = 0` and `N >= 4`).
pub fn even_at_least_four(n: usize) -> usize {
    let mut v = n.max(4);
    if v % 2 != 0 {
        v += 1;
    }
    v
}

/// Assertable closeness predicate with an absolute tolerance.
pub fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_bounded() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            let x = a.next_f64();
            assert_eq!(x, b.next_f64());
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn stats_helpers() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(mean(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(max_abs(&[-3.0, 2.0]), 3.0);
        assert!((norm2(&[3.0, 4.0]) - 5.0).abs() < 1e-15);
        assert_eq!(even_at_least_four(0), 4);
        assert_eq!(even_at_least_four(7), 8);
        assert_eq!(even_at_least_four(8), 8);
    }
}
