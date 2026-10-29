//! Numerical thresholds and the forward error model.
//!
//! ## `tau_dtype` — the acceptability threshold of a floating-point backend
//!
//! The design characterises the acceptability of a backend format by a numerical threshold
//! `tau_dtype`. A scale whose per-bin spread exceeds the format's own resolution cannot be
//! removed by the scalar post stage, so the tolerance has to scale with the format's
//! relative precision `eps` and with the length of the probe's inner products:
//!
//! ```text
//! tau_dtype = kappa * sqrt(n) * eps        (kappa = 8 by default)
//! ```
//!
//! A probe of `n` outputs is a sum of `~n` products, so its estimate of the scale carries a
//! relative uncertainty of about `sqrt(n) * eps`; `kappa` is the safety factor that keeps a
//! compliant backend on the safe side of the comparison. For `f64` and `n = 8` this is about
//! `2.5e-15`; for `f32` about `1.3e-6`; for `Q1.14` fixed point about `8e-4`.
//!
//! ## Forward error bound
//!
//! With `X = D_N(u)` and the pipeline `X = POST(MDCT IP(FOLD(u)))`, three independent terms
//! contribute:
//!
//! 1. **Input quantization.** The caller's `U` is a `Wu`/`Q` word: `|du| <= 2^{-(q+1)}`,
//!    hence `|dX_in[k]| <= N * 2^{-(q+1)}` (worst case over the `L1` kernel row).
//! 2. **Backend arithmetic.** For `gamma_{2N} = 2N*eps / (1 - 2N*eps)` and a window-free
//!    backend, `|dY[k]| <= gamma_{2N} * sum_m |x[m]| <= gamma_{2N} * sqrt(2N) * ||u||_2`
//!    (Cauchy–Schwarz; the fold is an isometry by Corollary 1, so `||x||_2 = ||u||_2`),
//!    scaled by `|S| = 1/alpha`.
//! 3. **Post quantization.** One rounding of the post stage: `2^{-(q_out+1)}`.
//!
//! Because `D_N^T D_N = (N/2) I`, the output norm is known exactly:
//! `||X||_2 = sqrt(N/2) ||u||_2`, which is what [`ForwardErrorBound::relative`] divides by.
//!
//! The bound is deliberately pessimistic (worst-case `L1` for the input term, `gamma_n`
//! for the accumulation); tests assert the *measured* deviation is below it.

use crate::backend::SampleFormat;
use crate::fixed::QFormat;

/// Safety factor `kappa` in `tau_dtype = kappa * sqrt(n) * eps`.
pub const SAFETY_FACTOR: f64 = 8.0;

/// Lower bound on any threshold, so that an exactly-present format cannot produce a zero
/// tolerance (which would make any real backend fail).
pub const TAU_FLOOR: f64 = 1e-18;

/// The mixed relative-precision threshold `tau_dtype` for a backend format and probe length.
pub fn tau_dtype(format: SampleFormat, n: usize) -> f64 {
    let n = n.max(1) as f64;
    (SAFETY_FACTOR * n.sqrt() * format.eps()).max(TAU_FLOOR)
}

/// Pointwise phase tolerance `eps_phase` for H2.
///
/// H2 compares a small-`N` transform pointwise, i.e. over an inner product of `2n` terms;
/// the tolerance is the same `kappa * sqrt(2n) * eps` scale, expressed relative to
/// `1 + max|ref|` (see [`crate::util::rel_max_err`]).
pub fn eps_phase(format: SampleFormat, n_probe: usize) -> f64 {
    tau_dtype(format, 2 * n_probe.max(1))
}

/// Tolerance on the *uniformity* of the implied impulse-response window (H1).
pub fn window_tolerance(format: SampleFormat, n_probe: usize) -> f64 {
    tau_dtype(format, 2 * n_probe.max(1))
}

/// Standard floating-point accumulation factor `gamma_n = n*eps / (1 - n*eps)`.
pub fn gamma(n: usize, eps: f64) -> f64 {
    let t = n as f64 * eps;
    if t >= 1.0 {
        f64::INFINITY
    } else {
        t / (1.0 - t)
    }
}

/// A forward error bound decomposed into its three contributions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardErrorBound {
    /// Contribution of the caller's input word length, `N * 2^{-(q+1)}`.
    pub input: f64,
    /// Contribution of the backend's arithmetic, `|S| * gamma_{2N} * sqrt(2N) * ||u||_2`.
    pub backend: f64,
    /// Contribution of the post stage's single rounding, `2^{-(q_out+1)}`.
    pub post: f64,
    /// Sum of the three terms (a rigorous upper bound on `max_k |X[k] - X_hat[k]|`).
    pub total: f64,
    /// `total` divided by the exact output norm `sqrt(N/2) * ||u||_2`.
    pub relative: f64,
}

/// Computes the forward error bound of the indirect pipeline.
///
/// * `n` — DCT-IV block length,
/// * `u_norm2` — `||u||_2` of the block being transformed,
/// * `ip` — the backend's sample format,
/// * `post_gain` — `|sigma / alpha|`, the magnitude of the post-stage scalar,
/// * `input` / `output` — the input and output word formats.
pub fn forward_error_bound(
    n: usize,
    u_norm2: f64,
    ip: SampleFormat,
    post_gain: f64,
    input: QFormat,
    output: QFormat,
) -> ForwardErrorBound {
    let eps_in = input.lsb();
    let input_term = n as f64 * 0.5 * eps_in;
    let backend_term =
        post_gain * gamma(2 * n, ip.eps()) * ((2 * n) as f64).sqrt() * u_norm2.max(0.0);
    let post_term = 0.5 * output.lsb();
    let total = input_term + backend_term + post_term;
    let out_norm = ((n as f64) / 2.0).sqrt() * u_norm2.max(f64::MIN_POSITIVE);
    ForwardErrorBound {
        input: input_term,
        backend: backend_term,
        post: post_term,
        total,
        relative: total / out_norm,
    }
}

/// The verdict of an error-budget check.
///
/// The budget is the half-LSB of the output word: if the backend's contribution exceeds it,
/// the backend's declared precision cannot support the declared output width, and the
/// `PrecisionInsufficiency` vulnerability applies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetAssessment {
    /// The backend's contribution to the output error.
    pub backend: f64,
    /// The half-LSB budget of the output word, `2^{-(q_out+1)}`.
    pub budget: f64,
    /// `budget - backend`; positive means the backend fits.
    pub margin: f64,
    /// `log2(budget / backend)`, how many bits of margin remain (negative on failure).
    pub margin_bits: f64,
    /// Whether the backend fits the budget.
    pub ok: bool,
}

/// Assesses whether the backend precision supports the output word length.
pub fn budget(
    n: usize,
    u_norm2: f64,
    ip: SampleFormat,
    post_gain: f64,
    output: QFormat,
) -> BudgetAssessment {
    let backend = post_gain * gamma(2 * n, ip.eps()) * ((2 * n) as f64).sqrt() * u_norm2.max(0.0);
    let budget = 0.5 * output.lsb();
    let margin = budget - backend;
    let margin_bits = if backend > 0.0 && budget > 0.0 {
        (budget / backend).log2()
    } else {
        f64::INFINITY
    };
    BudgetAssessment {
        backend,
        budget,
        margin,
        margin_bits,
        ok: backend <= budget,
    }
}

/// The smallest `Wip` that can hold `alpha * max|X|` with `frac` fractional bits.
///
/// The backend output is `y = alpha * X`, so the backend word must cover the *scaled* range,
/// not the DCT-IV's: a 24-bit IP with `alpha = 2^15` cannot hold 16-bit-scaled audio. This is
/// the dynamic-range complement of the `Wout >= Wip` width rule of eq. 3.
///
/// `max_abs_x` should be an upper bound on `|X[k]|`; the `L1` bound
/// `sum_n |u[n]|` is the cheapest rigorous one.
pub fn required_ip_width(alpha: f64, max_abs_x: f64, frac: u8) -> u8 {
    let bound = (alpha * max_abs_x).abs();
    let int_bits = if !bound.is_finite() {
        62
    } else if bound <= 1.0 {
        1
    } else {
        bound.log2().floor() as u32 + 1
    };
    let width = int_bits + frac as u32 + 1;
    width.clamp(2, 63) as u8
}

/// Smallest fractional-bit count whose LSB is at most `target` (an absolute tolerance).
///
/// Used when sizing a word from a system-level error target; returns `None` when `target` is
/// not a usable positive number.
pub fn frac_bits_for(target: f64) -> Option<u8> {
    if !target.is_finite() || target <= 0.0 {
        return None;
    }
    let q = (-(target).log2()).ceil();
    let q = q.clamp(0.0, 62.0) as u8;
    Some(q)
}

/// Signal-to-quantization-noise ratio in dB implied by an absolute error bound and the
/// exact output norm `sqrt(N/2) * ||u||_2`.
pub fn snr_db(n: usize, u_norm2: f64, error: f64) -> f64 {
    let out_norm = ((n as f64) / 2.0).sqrt() * u_norm2.max(f64::MIN_POSITIVE);
    if error <= 0.0 {
        return f64::INFINITY;
    }
    20.0 * (out_norm / error).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tau_ordering_and_floor() {
        assert!(tau_dtype(SampleFormat::F64, 8) < tau_dtype(SampleFormat::I32, 8));
        assert!(tau_dtype(SampleFormat::I32, 8) < tau_dtype(SampleFormat::F32, 8));
        let fixed = SampleFormat::Fixed(QFormat::new(16, 14));
        assert!((tau_dtype(fixed, 8) - 8.0 * 8f64.sqrt() * 2f64.powi(-14)).abs() < 1e-18);
        assert!(tau_dtype(SampleFormat::F64, 0) >= TAU_FLOOR);
    }

    #[test]
    fn gamma_is_tight_for_small_n() {
        let eps = SampleFormat::F64.eps();
        assert!((gamma(16, eps) - 16.0 * eps).abs() < 1e-26);
        assert_eq!(gamma(4, 0.5), f64::INFINITY);
    }

    #[test]
    fn bound_is_finite_and_dominated_by_the_backend() {
        let b = forward_error_bound(
            32,
            4.0,
            SampleFormat::F32,
            1.0,
            QFormat::new(16, 14),
            QFormat::new(24, 14),
        );
        assert!(b.total.is_finite());
        assert!(b.total >= b.post);
        assert!(b.relative > 0.0 && b.relative < 1.0);
    }

    #[test]
    fn error_scales_with_gain() {
        let a = forward_error_bound(
            16,
            1.0,
            SampleFormat::F32,
            1.0,
            QFormat::new(16, 14),
            QFormat::new(24, 14),
        );
        let b = forward_error_bound(
            16,
            1.0,
            SampleFormat::F32,
            4.0,
            QFormat::new(16, 14),
            QFormat::new(24, 14),
        );
        assert!((b.backend - 4.0 * a.backend).abs() < 1e-18);
    }

    #[test]
    fn f32_cannot_feed_a_24_bit_output() {
        // f32 relative precision ~6e-8; a Q9.14 output word has a half-LSB of ~3e-5.
        let ok = budget(16, 1.0, SampleFormat::F32, 1.0, QFormat::new(24, 14));
        assert!(
            ok.ok,
            "f32 fits a 14-fraction-bit output word at this length"
        );
        let bad = budget(32, 1.0, SampleFormat::F32, 1.0, QFormat::new(40, 30));
        assert!(!bad.ok, "f32 cannot support 30 fractional bits");
        assert!(bad.margin_bits < 0.0);
    }

    #[test]
    fn ip_width_sizing() {
        // alpha * X <= 13800 needs 14 integer bits, plus 14 fractional and one sign bit.
        assert_eq!(required_ip_width(1024.0, 13.5, 14), 29);
        assert_eq!(required_ip_width(1.0, 1.0, 14), 16);
        assert_eq!(required_ip_width(1.0, 0.0, 14), 16);
        assert!(required_ip_width(1e30, 1.0, 14) <= 63);
    }

    #[test]
    fn frac_bits_sizing() {
        assert_eq!(frac_bits_for(2f64.powi(-14)), Some(14));
        assert_eq!(frac_bits_for(0.0), None);
        assert_eq!(frac_bits_for(f64::NAN), None);
        assert_eq!(snr_db(8, 1.0, 0.0), f64::INFINITY);
        assert!(snr_db(8, 1.0, 1e-6) > 100.0);
    }
}
