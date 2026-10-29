//! A configurable **emulated** external MDCT IP.
//!
//! [`EmulatedMdct`] realises `y = alpha * M_{2N->N}(x)` and can inject exactly the defects
//! the H1–H4 contract exists to catch:
//!
//! | Defect | Builder | Precondition that must reject it |
//! |--------|---------|----------------------------------|
//! | Applies a window `w != 1` | [`EmulatedMdct::with_window_kind`] | H1 |
//! | Phase offset in the cosine argument | [`EmulatedMdct::with_phase`] | H2 |
//! | Per-bin gain drift | [`EmulatedMdct::with_gain_drift`] | H3 |
//! | Shuffled output bins | [`EmulatedMdct::with_ordering`] | H4 / correctable |
//! | Word-length truncation | [`EmulatedMdct::with_quant_bits`] | H3 precision budget |
//!
//! It is the vehicle for the compliance test suite: every `backend_reject` path in the kit
//! is exercised by a concrete, deterministic emulator rather than by a mock that merely
//! returns an error code.

use super::{BackendDescriptor, BackendError, MdctBackend, OutputOrdering, SampleFormat};
use crate::config::LengthSet;
use crate::mdct::MdctKernel;
use std::f64::consts::PI;

/// The analytic window shape an emulated IP is configured to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowKind {
    /// `w[m] = 1`: window-free (H1 compliant). The default.
    Rectangular,
    /// `w[m] = 0.5 - 0.5 cos(2 pi m / (2N - 1))`.
    Hann,
    /// `w[m] = sin(pi (m + 0.5) / (2N))`, the MDCT sine window.
    Sine,
}

impl WindowKind {
    /// Evaluates the window at `m` for a `2N`-length input.
    pub fn at(self, m: usize, two_n: usize) -> f64 {
        match self {
            WindowKind::Rectangular => 1.0,
            WindowKind::Hann => {
                if two_n <= 1 {
                    0.0
                } else {
                    0.5 - 0.5 * (2.0 * PI * m as f64 / (two_n - 1) as f64).cos()
                }
            }
            WindowKind::Sine => (PI * (m as f64 + 0.5) / two_n as f64).sin(),
        }
    }

    /// Materialises the window of length `2N`.
    pub fn samples(self, two_n: usize) -> Vec<f64> {
        (0..two_n).map(|m| self.at(m, two_n)).collect()
    }
}

/// An emulated external MDCT IP with configurable (defective) behaviour.
#[derive(Debug, Clone)]
pub struct EmulatedMdct {
    desc: BackendDescriptor,
    alpha: f64,
    window: WindowKind,
    phase: f64,
    gain_drift: f64,
    quant_in_bits: Option<u8>,
    quant_out_bits: Option<u8>,
    cache: Option<(usize, MdctKernel)>,
}

impl EmulatedMdct {
    /// An ideal IP: unit scale, no window, zero phase, no drift, no truncation.
    pub fn ideal(name: impl Into<String>) -> Self {
        Self {
            desc: BackendDescriptor::new(name, SampleFormat::F64, LengthSet::AnyEven),
            alpha: 1.0,
            window: WindowKind::Rectangular,
            phase: 0.0,
            gain_drift: 0.0,
            quant_in_bits: None,
            quant_out_bits: None,
            cache: None,
        }
    }

    /// Overrides the declared sample format (metadata only; the arithmetic stays `f64`).
    pub fn with_format(mut self, format: SampleFormat) -> Self {
        self.desc.format = format;
        self
    }

    /// Restricts the emulated IP to an explicit roster of lengths.
    pub fn with_lengths(mut self, lengths: LengthSet) -> Self {
        self.desc.supported_lengths = lengths;
        self
    }

    /// Sets the scalar scaling `alpha` and the matching declaration.
    pub fn with_scale(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self.desc.declared_scale = alpha;
        self
    }

    /// Overrides only the *declared* scale, leaving the arithmetic's actual `alpha` untouched.
    ///
    /// Models an IP whose data sheet disagrees with its silicon. H3 compares the declared
    /// value with the configured `alpha`, so this is a decidable violation.
    pub fn with_declared_scale(mut self, alpha: f64) -> Self {
        self.desc.declared_scale = alpha;
        self
    }

    /// Makes the IP apply a window, violating H1.
    pub fn with_window_kind(mut self, kind: WindowKind) -> Self {
        self.window = kind;
        self.desc.window_free = kind == WindowKind::Rectangular;
        self
    }

    /// Adds a phase offset (radians) inside the cosine argument, violating H2.
    ///
    /// This models the classic `+N/2` convention mismatch between an MDCT IP and a DCT-IV
    /// wrapper.
    pub fn with_phase(mut self, radians: f64) -> Self {
        self.phase = radians;
        self
    }

    /// Adds a per-bin gain drift, violating H3.
    ///
    /// Bin `k` is scaled by `alpha * (1 + drift * (k/(N-1) - 1/2))`, so the total spread of
    /// the effective scale over the band is `|drift|`.
    pub fn with_gain_drift(mut self, drift: f64) -> Self {
        self.gain_drift = drift;
        self
    }

    /// Truncates the input and/or output to `bits` magnitude bits at full scale 1.0.
    ///
    /// `bits = 15` means a quantization step of `2^-14`, i.e. a `Q1.14`-like word.
    pub fn with_quant_bits(mut self, input_bits: Option<u8>, output_bits: Option<u8>) -> Self {
        self.quant_in_bits = input_bits;
        self.quant_out_bits = output_bits;
        self
    }

    /// Declares a non-natural output ordering, violating (or, for a valid permutation,
    /// merely complicating) H4. The emulator really does emit the permuted order.
    pub fn with_ordering(mut self, ordering: OutputOrdering) -> Self {
        self.desc.ordering = ordering;
        self
    }

    /// Batch-friendly variant of [`EmulatedMdct::with_quant_bits`] for fixed-point emulation:
    /// a `Q(w, q)` output word has an LSB of `2^-q`, hence `q + 1` magnitude bits.
    pub fn with_fixed_output(mut self, frac: u8) -> Self {
        self.quant_out_bits = Some(frac.saturating_add(1));
        self.desc.format =
            SampleFormat::Fixed(crate::fixed::QFormat::new(frac.saturating_add(2), frac));
        self
    }

    /// The effective per-bin scale of the emulated IP (before ordering).
    pub fn effective_scale(&self, k: usize, n: usize) -> f64 {
        if self.gain_drift == 0.0 || n <= 1 {
            return self.alpha;
        }
        let t = k as f64 / (n - 1) as f64 - 0.5;
        self.alpha * (1.0 + self.gain_drift * t)
    }

    fn kernel(&mut self, n: usize) -> &MdctKernel {
        let stale = match &self.cache {
            Some((cached, _)) => *cached != n,
            None => true,
        };
        if stale {
            self.cache = Some((n, MdctKernel::new(n)));
        }
        &self.cache.as_ref().expect("cache just filled").1
    }
}

/// Rounds `v` to a step of `2^-(bits-1)`, i.e. `bits` magnitude bits at full scale 1.0.
fn quantize(v: f64, bits: Option<u8>) -> f64 {
    match bits {
        Some(b) if b > 0 && b < 60 => {
            let step = 2f64.powi(-(b as i32 - 1));
            (v / step).round() * step
        }
        _ => v,
    }
}

impl MdctBackend for EmulatedMdct {
    fn descriptor(&self) -> &BackendDescriptor {
        &self.desc
    }

    fn transform(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), BackendError> {
        if input.len() % 2 != 0 {
            return Err(BackendError::Shape {
                expected: input.len() + 1,
                got: input.len(),
            });
        }
        let n = input.len() / 2;
        if !self.desc.supports(n) {
            return Err(BackendError::UnsupportedLength {
                requested: n,
                supported: self.desc.supported_lengths.clone(),
            });
        }
        if output.len() != n {
            return Err(BackendError::Shape {
                expected: n,
                got: output.len(),
            });
        }

        // Snapshot the scalar configuration first: `kernel` borrows `self` mutably.
        let alpha = self.alpha;
        let phase = self.phase;
        let drift = self.gain_drift;
        let window = self.window;
        let quant_in = self.quant_in_bits;
        let quant_out = self.quant_out_bits;
        let two_n = 2 * n;

        let mut work = Vec::with_capacity(two_n);
        for (m, &x) in input.iter().enumerate() {
            work.push(quantize(x, quant_in) * window.at(m, two_n));
        }

        let kernel = self.kernel(n);
        let mut y = vec![0.0; n];
        #[allow(clippy::needless_range_loop)] // bin index drives both the row and the gain
        for k in 0..n {
            let theta = PI / n as f64;
            let mut acc = 0.0;
            for (m, &w) in work.iter().enumerate() {
                // Rebuild the kernel angle with the configured phase offset.
                let a = theta * (m as f64 + 0.5 + n as f64 / 2.0) * (k as f64 + 0.5) + phase;
                let c = if phase == 0.0 {
                    kernel.row(k)[m]
                } else {
                    a.cos()
                };
                acc += w * c;
            }
            let gain = if drift == 0.0 || n <= 1 {
                alpha
            } else {
                alpha * (1.0 + drift * (k as f64 / (n - 1) as f64 - 0.5))
            };
            y[k] = quantize(gain * acc, quant_out);
        }

        match self.desc.ordering.permutation() {
            Some(p) if p.len() == n && self.desc.ordering.is_bijection(n) => {
                for (i, &k) in p.iter().enumerate() {
                    output[i] = y[k];
                }
            }
            _ => {
                output.copy_from_slice(&y);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdct::mdct;

    #[test]
    fn ideal_matches_reference() {
        let mut ip = EmulatedMdct::ideal("ip");
        for n in [4usize, 8, 12] {
            let x = crate::util::Rng::new(n as u64).signed_vec(2 * n);
            let mut y = vec![0.0; n];
            ip.transform(&x, &mut y).unwrap();
            let z = mdct(&x);
            for k in 0..n {
                assert!((y[k] - z[k]).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn scale_is_applied() {
        let mut ip = EmulatedMdct::ideal("ip").with_scale(8.0);
        assert_eq!(ip.descriptor().declared_scale, 8.0);
        let x = crate::util::Rng::new(1).signed_vec(16);
        let mut y = vec![0.0; 8];
        ip.transform(&x, &mut y).unwrap();
        let z = mdct(&x);
        for k in 0..8 {
            assert!((y[k] - 8.0 * z[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn windowing_weights_columns_differently() {
        // A window scales column m by w[m]: invisible *within* one column, visible *across*
        // columns. That is exactly why H1 normalises the per-column gain by its median.
        let mut ip = EmulatedMdct::ideal("ip").with_window_kind(WindowKind::Sine);
        assert!(!ip.descriptor().window_free);
        let kernel = crate::mdct::MdctKernel::new(8);
        let gain = |ip: &mut EmulatedMdct, m: usize| {
            let y = ip.probe_impulse(8, m).unwrap();
            let a = kernel.column(m);
            crate::util::dot(&y, &a) / crate::util::dot(&a, &a)
        };
        let g0 = gain(&mut ip, 0);
        let g7 = gain(&mut ip, 7);
        let w0 = WindowKind::Sine.at(0, 16);
        let w7 = WindowKind::Sine.at(7, 16);
        assert!((g0 - w0).abs() < 1e-12, "column 0 gain {g0} != w[0] {w0}");
        assert!((g7 - w7).abs() < 1e-12, "column 7 gain {g7} != w[7] {w7}");
        assert!((g7 / g0 - w7 / w0).abs() < 1e-12);
        assert!(
            g7 / g0 > 3.0,
            "the sine window must vary strongly across columns"
        );
    }

    #[test]
    fn phase_offset_changes_the_sign_pattern() {
        let mut ip = EmulatedMdct::ideal("ip").with_phase(0.3);
        let x = crate::util::Rng::new(3).signed_vec(16);
        let mut y = vec![0.0; 8];
        ip.transform(&x, &mut y).unwrap();
        let z = mdct(&x);
        let err: f64 = y
            .iter()
            .zip(&z)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err > 1e-2);
    }

    #[test]
    fn permutation_is_emitted() {
        let perm = vec![2usize, 0, 3, 1];
        let mut ip =
            EmulatedMdct::ideal("ip").with_ordering(OutputOrdering::Permutation(perm.clone()));
        let x = crate::util::Rng::new(5).signed_vec(8);
        let mut y = vec![0.0; 4];
        ip.transform(&x, &mut y).unwrap();
        let z = mdct(&x);
        for (i, &k) in perm.iter().enumerate() {
            assert!((y[i] - z[k]).abs() < 1e-13);
        }
    }

    #[test]
    fn gain_drift_is_linear() {
        let ip = EmulatedMdct::ideal("ip").with_gain_drift(0.01);
        let lo = ip.effective_scale(0, 9);
        let hi = ip.effective_scale(8, 9);
        assert!((hi - lo - 0.01).abs() < 1e-15);
        let mid = ip.effective_scale(4, 9);
        assert!((mid - 1.0).abs() < 1e-15);
    }

    #[test]
    fn quantization_step_is_observable() {
        let mut ip = EmulatedMdct::ideal("ip").with_quant_bits(None, Some(8));
        let x = crate::util::Rng::new(7).signed_vec(16);
        let mut y = vec![0.0; 8];
        ip.transform(&x, &mut y).unwrap();
        let step = 2f64.powi(-7);
        for v in &y {
            let k = (v / step).round() * step;
            assert!((v - k).abs() < 1e-15);
            assert!((v - k).abs() <= step * 0.5 + 1e-15);
        }
    }
}
