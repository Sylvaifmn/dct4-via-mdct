//! Error types shared by the kit.
//!
//! The kit distinguishes three failure domains, mirroring §4.2 and §8.7 of the design:
//!
//! * [`BackendError`] — the external IP itself failed (shape, length, internal fault).
//! * [`ConfigViolation`] — the configuration violates the validity constraint (eq. 3).
//! * [`Violation`] — a hard precondition H1–H4 failed, i.e. `backend_reject`.
//!
//! Both violation sets are *collected* rather than fail-fast, so that a bring-up engineer
//! sees every reason for a refusal at once.

use crate::backend::BackendError;
use crate::compliance::Violation;
use crate::config::ConfigViolation;
use std::fmt;

/// Anything that can make a kit operation fail.
#[derive(Debug, Clone, PartialEq)]
pub enum KitError {
    /// The configuration violates eq. 3 (`valid`), see [`ConfigViolation`].
    InvalidConfig(Vec<ConfigViolation>),
    /// The backend failed one or more hard preconditions H1–H4 (`backend_reject`).
    BackendRejected(Vec<Violation>),
    /// The backend reported an error while executing a transform.
    Backend(BackendError),
    /// A slice had the wrong length.
    Shape {
        /// What the caller was expected to supply.
        context: &'static str,
        /// Required element count.
        expected: usize,
        /// Actual element count.
        got: usize,
    },
    /// An arithmetic domain was invalid (zero scale, negative length, ...).
    Numeric(String),
}

impl fmt::Display for KitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KitError::InvalidConfig(v) => {
                write!(f, "invalid configuration ({} violation(s)):", v.len())?;
                for x in v {
                    write!(f, "\n  - {x}")?;
                }
                Ok(())
            }
            KitError::BackendRejected(v) => {
                write!(
                    f,
                    "backend_reject: {} hard precondition(s) failed:",
                    v.len()
                )?;
                for x in v {
                    write!(
                        f,
                        "\n  - {} {:?}: {} (remedy: {})",
                        x.precondition.code(),
                        x.category,
                        x.detail,
                        x.remedy
                    )?;
                }
                Ok(())
            }
            KitError::Backend(e) => write!(f, "backend error: {e}"),
            KitError::Shape {
                context,
                expected,
                got,
            } => write!(
                f,
                "shape mismatch for {context}: expected {expected}, got {got}"
            ),
            KitError::Numeric(m) => write!(f, "numeric domain error: {m}"),
        }
    }
}

impl std::error::Error for KitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            KitError::Backend(e) => Some(e),
            _ => None,
        }
    }
}

impl From<BackendError> for KitError {
    fn from(e: BackendError) -> Self {
        KitError::Backend(e)
    }
}

impl From<ConfigViolation> for KitError {
    fn from(v: ConfigViolation) -> Self {
        KitError::InvalidConfig(vec![v])
    }
}
