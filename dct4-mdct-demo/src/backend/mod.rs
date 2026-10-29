//! The external MDCT backend boundary (§4.1, §4.2).
//!
//! The kit interfaces with an external IP that realises
//!
//! ```text
//! y = M_ip(x) = alpha * M_{2N->N}(x)
//! ```
//!
//! and it must do so under four **hard preconditions**:
//!
//! | Id | Precondition | Criterion (§4.2) |
//! |----|--------------|------------------|
//! | H1 | Window-free impulse response | `w_ip ≡ 1` |
//! | H2 | Phase-matched | small-`N` pointwise comparison `<= eps_phase` |
//! | H3 | Scalar scaling | `spread(alpha) <= tau_dtype` |
//! | H4 | Natural ordering | order `in {natural}` or correctable by a permutation |
//!
//! Failure of any one means `backend_reject`; see [`crate::compliance`].
//!
//! ## What a backend has to expose
//!
//! * [`MdctBackend::descriptor`] — declared metadata: name, `N_ip`, word format, ordering,
//!   declared scale.
//! * [`MdctBackend::transform`] — the blocking transform.
//! * [`MdctBackend::probe_impulse`] / [`MdctBackend::probe_vector`] — the probes the
//!   compliance stage uses. Both have default implementations on top of `transform`, so a
//!   minimal backend only implements the first two items.

pub mod direct;
pub mod emulated;

pub use direct::DirectMdct;
pub use emulated::{EmulatedMdct, WindowKind};

use crate::config::LengthSet;
use crate::fixed::QFormat;
use std::fmt;

/// The numeric format of the external backend's samples.
///
/// The relative precision `eps` of the format is what `tau_dtype` is derived from; see
/// [`crate::numerics::tau_dtype`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleFormat {
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary64.
    F64,
    /// Fixed point with an explicit `(width, frac)` format.
    Fixed(QFormat),
    /// Signed 32-bit integer with an implicit, backend-owned scale.
    I32,
}

impl SampleFormat {
    /// Relative precision of the format near unity.
    ///
    /// * `F32`: `2^-24`, * `F64`: `2^-53`,
    /// * `Fixed(w, q)`: `2^-q` (the LSB),
    /// * `I32`: `2^-31` (one sign bit, unit LSB).
    pub fn eps(&self) -> f64 {
        match self {
            SampleFormat::F32 => 2f64.powi(-24),
            SampleFormat::F64 => 2f64.powi(-53),
            SampleFormat::Fixed(f) => f.eps(),
            SampleFormat::I32 => 2f64.powi(-31),
        }
    }

    /// A short label used in reports.
    pub fn label(&self) -> String {
        match self {
            SampleFormat::F32 => "f32".to_string(),
            SampleFormat::F64 => "f64".to_string(),
            SampleFormat::Fixed(f) => format!("fixed({},{})", f.width, f.frac),
            SampleFormat::I32 => "i32".to_string(),
        }
    }
}

impl fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label())
    }
}

/// The order in which a backend emits frequency bins (precondition H4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputOrdering {
    /// Natural order: output index `k` carries frequency bin `k`.
    Natural,
    /// The value at output index `i` carries frequency bin `perm[i]`.
    ///
    /// The kit composes `perm^{-1}` into its post stage, so such a backend is admissible —
    /// it is "correctable by permutation", which is exactly the H4 escape clause.
    Permutation(Vec<usize>),
    /// The backend does not document its ordering. This is an H4 rejection: the kit cannot
    /// guarantee correctness without knowing how to undo the permutation.
    Unspecified,
}

impl OutputOrdering {
    /// The permutation `perm` such that `out[i] = alpha * M[perm[i]]`, if declared.
    pub fn permutation(&self) -> Option<&[usize]> {
        match self {
            OutputOrdering::Permutation(p) => Some(p),
            _ => None,
        }
    }

    /// True when the declared permutation is a bijection of `0..n`.
    ///
    /// A malformed permutation would make the correction map non-invertible, so it is
    /// treated as an H4 violation rather than a panic.
    pub fn is_bijection(&self, n: usize) -> bool {
        match self {
            OutputOrdering::Natural => true,
            OutputOrdering::Permutation(p) => {
                if p.len() != n {
                    return false;
                }
                let mut seen = vec![false; n];
                for &i in p {
                    if i >= n || seen[i] {
                        return false;
                    }
                    seen[i] = true;
                }
                true
            }
            OutputOrdering::Unspecified => false,
        }
    }

    /// The inverse permutation `inv[k] = perm^{-1}(k)`, i.e. where to read natural bin `k`.
    ///
    /// For [`OutputOrdering::Natural`] this is the identity; for a malformed permutation it
    /// returns `None`.
    pub fn inverse_permutation(&self, n: usize) -> Option<Vec<usize>> {
        if !self.is_bijection(n) {
            return None;
        }
        match self {
            OutputOrdering::Natural => Some((0..n).collect()),
            OutputOrdering::Permutation(p) => {
                let mut inv = vec![0usize; n];
                for (i, &k) in p.iter().enumerate() {
                    inv[k] = i;
                }
                Some(inv)
            }
            OutputOrdering::Unspecified => None,
        }
    }

    /// True when no post-stage correction is needed.
    pub fn is_identity(&self) -> bool {
        match self {
            OutputOrdering::Natural => true,
            OutputOrdering::Permutation(p) => p.iter().enumerate().all(|(i, &k)| i == k),
            OutputOrdering::Unspecified => false,
        }
    }
}

/// Declared metadata about an external MDCT backend.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendDescriptor {
    /// Human-readable identification, e.g. `"acme-mdct-ip-v2"`.
    pub name: String,
    /// `N_ip`, the supported DCT-IV lengths (equivalently, `2N`-input MDCT sizes).
    pub supported_lengths: LengthSet,
    /// Sample format of the backend.
    pub format: SampleFormat,
    /// Output ordering (H4).
    pub ordering: OutputOrdering,
    /// The backend's own declaration that it applies no window (H1). Declared, **not**
    /// trusted: H1 is decided by an impulse probe.
    pub window_free: bool,
    /// The backend's declared scalar scaling `alpha`.
    pub declared_scale: f64,
}

impl BackendDescriptor {
    /// A descriptor with no window, natural ordering and unit scale.
    pub fn new(name: impl Into<String>, format: SampleFormat, lengths: LengthSet) -> Self {
        Self {
            name: name.into(),
            supported_lengths: lengths,
            format,
            ordering: OutputOrdering::Natural,
            window_free: true,
            declared_scale: 1.0,
        }
    }

    /// Sets the declared output ordering.
    pub fn with_ordering(mut self, ordering: OutputOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    /// Sets the declared scalar scaling.
    pub fn with_scale(mut self, alpha: f64) -> Self {
        self.declared_scale = alpha;
        self
    }

    /// Sets the declared window-freeness.
    pub fn with_window_free(mut self, window_free: bool) -> Self {
        self.window_free = window_free;
        self
    }

    /// True when the backend advertises `n` in `N_ip`.
    pub fn supports(&self, n: usize) -> bool {
        self.supported_lengths.contains(n)
    }
}

/// Failures reported by an external backend.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendError {
    /// The backend cannot be driven at the requested length: `N not in N_ip`.
    UnsupportedLength {
        /// The requested DCT-IV length.
        requested: usize,
        /// The backend's supported set.
        supported: LengthSet,
    },
    /// A buffer had the wrong length for the requested block size.
    Shape {
        /// Required element count.
        expected: usize,
        /// Actual element count.
        got: usize,
    },
    /// The backend reported an internal fault (timeout, DMA error, ...).
    Internal(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::UnsupportedLength {
                requested,
                supported,
            } => {
                write!(f, "length {requested} not in N_ip = {supported}")
            }
            BackendError::Shape { expected, got } => {
                write!(f, "backend shape error: expected {expected}, got {got}")
            }
            BackendError::Internal(m) => write!(f, "backend internal fault: {m}"),
        }
    }
}

impl std::error::Error for BackendError {}

/// An external MDCT backend: `y = alpha * M_{2N->N}(x)`.
///
/// Implementors must be able to:
/// 1. describe themselves ([`MdctBackend::descriptor`]);
/// 2. transform a `2N` vector into `N` outputs.
///
/// The probe methods default to wrapping `transform`, so an implementation can be as small
/// as the two required methods.
pub trait MdctBackend {
    /// Declared metadata; the compliance stage verifies the declared parts it can probe.
    fn descriptor(&self) -> &BackendDescriptor;

    /// Computes `y = alpha * M_{2N->N}(x)`.
    ///
    /// `input` has length `2N`, `output` has length `N`.
    fn transform(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), BackendError>;

    /// Returns the backend's response to the impulse `e_m` of a length-`2n` input.
    fn probe_impulse(&mut self, n: usize, m: usize) -> Result<Vec<f64>, BackendError> {
        if m >= 2 * n {
            return Err(BackendError::Shape {
                expected: 2 * n,
                got: m,
            });
        }
        let mut x = vec![0.0; 2 * n];
        x[m] = 1.0;
        let mut y = vec![0.0; n];
        self.transform(&x, &mut y)?;
        Ok(y)
    }

    /// Returns the backend's response to an arbitrary length-`2n` input vector.
    fn probe_vector(&mut self, x: &[f64]) -> Result<Vec<f64>, BackendError> {
        if x.len() % 2 != 0 {
            return Err(BackendError::Shape {
                expected: x.len() + 1,
                got: x.len(),
            });
        }
        let n = x.len() / 2;
        let mut y = vec![0.0; n];
        self.transform(x, &mut y)?;
        Ok(y)
    }

    /// Applies the backend's declared output permutation to a natural-order vector, i.e.
    /// produces what the backend would emit for an input whose natural output is `y`.
    ///
    /// This is a convenience for tests and emulators; it never inspects real hardware.
    fn permute_declared(&self, y: &[f64]) -> Vec<f64> {
        let n = y.len();
        match self.descriptor().ordering.permutation() {
            Some(p) if p.len() == n => {
                let mut out = vec![0.0; n];
                for (i, &k) in p.iter().enumerate() {
                    if k < n {
                        out[i] = y[k];
                    }
                }
                out
            }
            _ => y.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_helpers() {
        let natural = OutputOrdering::Natural;
        assert!(natural.is_identity());
        assert_eq!(natural.inverse_permutation(4), Some(vec![0, 1, 2, 3]));

        let perm = OutputOrdering::Permutation(vec![2, 0, 3, 1]);
        assert!(perm.is_bijection(4));
        assert!(!perm.is_bijection(5));
        let inv = perm.inverse_permutation(4).unwrap();
        for (i, &k) in [2usize, 0, 3, 1].iter().enumerate() {
            assert_eq!(inv[k], i);
        }

        let dup = OutputOrdering::Permutation(vec![0, 0, 1, 2]);
        assert!(!dup.is_bijection(4));
        assert!(dup.inverse_permutation(4).is_none());

        let out = OutputOrdering::Permutation(vec![0, 1, 2, 9]);
        assert!(!out.is_bijection(4));

        assert!(!OutputOrdering::Unspecified.is_bijection(4));
        assert!(OutputOrdering::Unspecified.inverse_permutation(4).is_none());
    }

    #[test]
    fn format_precision_ordering() {
        // Two's-complement i32 with a unit LSB has 31 fractional bits near unity: finer than
        // f32 (24) and coarser than f64 (53).
        assert!(SampleFormat::F64.eps() < SampleFormat::I32.eps());
        assert!(SampleFormat::I32.eps() < SampleFormat::F32.eps());
        let f = SampleFormat::Fixed(QFormat::new(16, 14));
        assert!((f.eps() - 2f64.powi(-14)).abs() < 1e-20);
    }
}
