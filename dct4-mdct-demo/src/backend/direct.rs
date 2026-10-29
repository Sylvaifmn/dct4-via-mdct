//! A trusted, exact reference backend.
//!
//! [`DirectMdct`] evaluates the analytic MDCT definition in `f64`, with `alpha = 1`,
//! natural ordering and no window. It is the "golden" backend:
//!
//! * the default in tests and examples, where the point is the *folding* and the *contract*,
//!   not the IP;
//! * the bring-up baseline against which a real IP's compliance report can be compared.
//!
//! It is intentionally `O(N^2)` and CPU-only: a reference is supposed to be obviously
//! correct rather than fast.

use super::{BackendDescriptor, BackendError, MdctBackend, SampleFormat};
use crate::config::LengthSet;
use crate::mdct::MdctKernel;

/// A reference MDCT backend: exact analytic kernel, unit scale, natural order, no window.
#[derive(Debug, Clone)]
pub struct DirectMdct {
    desc: BackendDescriptor,
    cache: Option<(usize, MdctKernel)>,
}

impl DirectMdct {
    /// Builds a reference backend accepting every even `N >= 4`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            desc: BackendDescriptor::new(name, SampleFormat::F64, LengthSet::AnyEven),
            cache: None,
        }
    }

    /// Builds a reference backend restricted to an explicit roster of lengths.
    pub fn with_lengths(name: impl Into<String>, lengths: LengthSet) -> Self {
        Self {
            desc: BackendDescriptor::new(name, SampleFormat::F64, lengths),
            cache: None,
        }
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

impl MdctBackend for DirectMdct {
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
        self.kernel(n).apply(input, output);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdct::mdct;

    #[test]
    fn matches_the_analytic_definition() {
        let mut b = DirectMdct::new("ref");
        for n in [4usize, 6, 8, 16] {
            let mut rng = crate::util::Rng::new(n as u64);
            let x = rng.signed_vec(2 * n);
            let mut y = vec![0.0; n];
            b.transform(&x, &mut y).unwrap();
            let z = mdct(&x);
            for k in 0..n {
                assert!((y[k] - z[k]).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn rejects_bad_shapes_and_lengths() {
        let mut b = DirectMdct::new("ref");
        let mut y = vec![0.0; 8];
        assert!(matches!(
            b.transform(&[0.0; 7], &mut y),
            Err(BackendError::Shape { .. })
        ));
        let mut wrong_out = vec![0.0; 12];
        assert!(matches!(
            b.transform(&[0.0; 16], &mut wrong_out),
            Err(BackendError::Shape { .. })
        ));
        let mut b = DirectMdct::with_lengths("ip", LengthSet::of(&[8]));
        let mut y = vec![0.0; 16];
        assert!(matches!(
            b.transform(&[0.0; 32], &mut y),
            Err(BackendError::UnsupportedLength { .. })
        ));
    }
}
