//! The backend contract H1–H4 and its decidable compliance criteria (§4.2, §8).
//!
//! The kit never trusts a backend's self-description. [`evaluate`] probes the backend and
//! decides each hard precondition numerically:
//!
//! | Id | Precondition | Decidable criterion |
//! |----|--------------|---------------------|
//! | H1 | Window-free impulse response `w_ip ≡ 1` | The implied per-sample gain of every impulse column, normalised by its median, deviates by at most `window_tol`. |
//! | H2 | Phase-matched | On small `N`, `max_k |y[k] - alpha*ref[k]| / (1 + max|ref|) <= eps_phase` over several deterministic probe vectors. |
//! | H3 | Scalar scaling | `spread(alpha) = (max alpha_k - min alpha_k) / mean|alpha_k| <= tau_dtype`, and the measured and declared `alpha` agree with the configured one. |
//! | H4 | Natural ordering | Output order is natural, or the declared permutation is a valid bijection (then the kit inverts it in POST). |
//!
//! Every failure produces a [`Violation`] carrying a [`Vulnerability`] classification and a
//! concrete remedy (§10). [`ComplianceReport::gate`] is the `backend_reject` decision.
//!
//! ## Why the probes are shaped this way
//!
//! * **H1 is an impulse test, not a declaration.** A window multiplies column `m` of the
//!   transform matrix by `w[m]`, so it is invisible in a single column's shape; it is only
//!   visible *across* columns. Hence the normalised-gain comparison over all `m`.
//! * **H2 must be pointwise.** An overall energy error can hide a sign or phase flip; the
//!   pointwise comparison cannot.
//! * **H3 must be per-bin.** A scalar `alpha` is applied by POST; a per-bin gain cannot be.
//!   The spread of `alpha_k = y[k]/ref[k]` is therefore the decidable criterion.
//! * **Ordering is decided before H1–H3.** If the bin order is unknown, every per-bin
//!   statistic is meaningless, so H1–H3 fail closed rather than reporting noise as a defect.

use crate::backend::{BackendError, MdctBackend, OutputOrdering, SampleFormat};
use crate::config::KitConfig;
use crate::mdct::MdctKernel;
use crate::numerics;
use crate::util::{self, Rng};
use std::fmt;

/// The four hard preconditions of §4.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precondition {
    /// H1: the backend applies no analysis window.
    H1WindowFree,
    /// H2: the backend's cosine phase matches the design's kernel.
    H2PhaseMatched,
    /// H3: the backend's scaling is a single scalar.
    H3ScalarScaling,
    /// H4: the backend's output ordering is natural or permutable.
    H4NaturalOrdering,
}

impl Precondition {
    /// All four preconditions, in contract order.
    pub const ALL: [Precondition; 4] = [
        Precondition::H1WindowFree,
        Precondition::H2PhaseMatched,
        Precondition::H3ScalarScaling,
        Precondition::H4NaturalOrdering,
    ];

    /// Short identifier (`"H1"` … `"H4"`).
    pub fn code(self) -> &'static str {
        match self {
            Precondition::H1WindowFree => "H1",
            Precondition::H2PhaseMatched => "H2",
            Precondition::H3ScalarScaling => "H3",
            Precondition::H4NaturalOrdering => "H4",
        }
    }

    /// Human-readable statement of the precondition.
    pub fn title(self) -> &'static str {
        match self {
            Precondition::H1WindowFree => "window-free impulse response",
            Precondition::H2PhaseMatched => "phase-matched",
            Precondition::H3ScalarScaling => "scalar scaling",
            Precondition::H4NaturalOrdering => "natural ordering",
        }
    }
}

impl fmt::Display for Precondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.code(), self.title())
    }
}

/// Classification of the failure mode behind a rejected backend (§10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vulnerability {
    /// The backend tapers its input: `w_ip != 1`.
    WindowContamination,
    /// The cosine argument carries an extra (or missing) phase term.
    PhaseMismatch,
    /// The scale is not a single scalar over the band, or does not match the contract.
    ScalingDrift,
    /// The output permutation is unknown or malformed.
    OrderingAmbiguity,
    /// The backend's word length cannot support the configured output resolution.
    PrecisionInsufficiency,
}

impl Vulnerability {
    /// The remedy prescribed for this class of defect (§10).
    pub fn remedy(self) -> &'static str {
        match self {
            Vulnerability::WindowContamination => {
                "insert an inverse-taper stage before FOLD, or regenerate the IP from a \
                 window-free core; a per-sample window is not a scalar and cannot be undone in POST"
            }
            Vulnerability::PhaseMismatch => {
                "re-place the +N/2 phase term of the design's kernel, then re-run the small-N \
                 pointwise comparison (H2) before connecting the IP"
            }
            Vulnerability::ScalingDrift => {
                "restore a single scalar alpha for the whole band (or widen the IP word so the \
                 per-bin gain error falls below tau_dtype), then declare it in the descriptor"
            }
            Vulnerability::OrderingAmbiguity => {
                "obtain the vendor's output permutation and declare it in the descriptor (the \
                 kit inverts it in POST), or tap the IP's natural-order output"
            }
            Vulnerability::PrecisionInsufficiency => {
                "widen the backend format or reduce Wout: the declared precision cannot resolve \
                 the requested output word"
            }
        }
    }
}

/// A rejected hard precondition, with its classification and remedy.
#[derive(Debug, Clone, PartialEq)]
pub struct Violation {
    /// The precondition that failed.
    pub precondition: Precondition,
    /// The failure mode.
    pub category: Vulnerability,
    /// The measured evidence.
    pub detail: String,
    /// What to do about it.
    pub remedy: &'static str,
}

impl Violation {
    /// Builds a violation, filling in the remedy from its category.
    pub fn new(precondition: Precondition, category: Vulnerability, detail: String) -> Self {
        Self {
            precondition,
            category,
            detail,
            remedy: category.remedy(),
        }
    }
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {:?}: {} (remedy: {})",
            self.precondition.code(),
            self.category,
            self.detail,
            self.remedy
        )
    }
}

/// The numerical thresholds used to decide H1–H4.
///
/// The defaults are derived from the backend's declared format, so a `f32` IP is judged
/// against an `f32`-appropriate tolerance and a `Q1.14` IP against a `Q1.14` one. Override
/// them only for bring-up experiments, and record the override in the verification report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// H3: maximum admissible spread of the per-bin scale, `tau_dtype`.
    pub tau_scale: f64,
    /// H2: maximum admissible pointwise relative error.
    pub eps_phase: f64,
    /// H1: maximum admissible deviation of the implied window from unity.
    pub window_tol: f64,
    /// H3: maximum admissible relative disagreement between measured/declared/configured
    /// `alpha`.
    pub alpha_tol: f64,
    /// Number of deterministic probe vectors used by H2/H3.
    pub probe_vectors: usize,
}

impl Thresholds {
    /// Derives the thresholds from the declared format and probe length.
    pub fn for_format(format: SampleFormat, probe_n: usize, probe_vectors: usize) -> Self {
        let tau = numerics::tau_dtype(format, probe_n);
        Self {
            tau_scale: tau,
            eps_phase: numerics::eps_phase(format, probe_n),
            window_tol: numerics::window_tolerance(format, probe_n),
            alpha_tol: tau.max(8.0 * format.eps()),
            probe_vectors: probe_vectors.max(1),
        }
    }
}

impl Default for Thresholds {
    fn default() -> Self {
        Self::for_format(SampleFormat::F64, 8, 3)
    }
}

/// The measured properties of a probed backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    /// The scale estimated from the probes (natural order, averaged over bins and vectors).
    pub alpha: f64,
    /// `(max alpha_k - min alpha_k) / mean|alpha_k|` — the H3 statistic.
    pub scale_spread: f64,
    /// The H2 pointwise relative error.
    pub phase_error: f64,
    /// The H1 maximum deviation of the implied window from unity.
    pub window_deviation: f64,
    /// Maximum relative residual of an impulse column against a scalar multiple of the
    /// analytic column; a large value indicates a phase or ordering defect.
    pub column_residual: f64,
}

/// The outcome of one precondition check.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckOutcome {
    /// Which precondition was decided.
    pub precondition: Precondition,
    /// The decision.
    pub passed: bool,
    /// The measured statistic (`NaN` when the check could not be decided).
    pub metric: f64,
    /// The threshold the statistic was compared against.
    pub threshold: f64,
    /// A precise statement of what was measured.
    pub detail: String,
}

impl CheckOutcome {
    /// Renders the outcome as a Markdown table row.
    pub fn to_markdown_row(&self) -> String {
        let metric = if self.metric.is_nan() {
            "n/a".to_string()
        } else {
            format!("{:.3e}", self.metric)
        };
        format!(
            "| {} | {} | {} | {:.3e} | {} |",
            self.precondition.code(),
            self.precondition.title(),
            metric,
            self.threshold,
            if self.passed { "pass" } else { "**FAIL**" }
        )
    }
}

/// The complete, auditable result of probing a backend.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplianceReport {
    /// The backend's declared name.
    pub backend: String,
    /// The backend's declared format.
    pub format: SampleFormat,
    /// The backend's declared ordering.
    pub ordering: OutputOrdering,
    /// The probe length actually used (`N`).
    pub probe_n: usize,
    /// The thresholds used.
    pub thresholds: Thresholds,
    /// The measured properties.
    pub measured: Measured,
    /// One outcome per precondition, in contract order.
    pub checks: Vec<CheckOutcome>,
    /// All failures, i.e. the reason for `backend_reject` when non-empty.
    pub violations: Vec<Violation>,
}

impl ComplianceReport {
    /// True when every hard precondition is satisfied.
    pub fn accepted(&self) -> bool {
        self.violations.is_empty()
    }

    /// The `backend_reject` decision: `Ok(())` when admissible, `Err(failures)` otherwise.
    pub fn gate(&self) -> Result<(), Vec<Violation>> {
        if self.accepted() {
            Ok(())
        } else {
            Err(self.violations.clone())
        }
    }

    /// The outcome of one precondition, if it was decided.
    pub fn check(&self, p: Precondition) -> Option<&CheckOutcome> {
        self.checks.iter().find(|c| c.precondition == p)
    }

    /// Renders the report as a Markdown document, suitable for a verification record.
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "# Backend compliance report: `{}`\n\n",
            self.backend
        ));
        s.push_str(&format!(
            "- Decision: **{}**\n- Declared format: `{}`\n- Declared ordering: `{:?}`\n\
             - Probe length N: {}\n- Thresholds: tau_dtype = {:.3e}, eps_phase = {:.3e}, \
             window_tol = {:.3e}\n\n",
            if self.accepted() { "ACCEPT" } else { "REJECT" },
            self.format,
            self.ordering,
            self.probe_n,
            self.thresholds.tau_scale,
            self.thresholds.eps_phase,
            self.thresholds.window_tol
        ));
        s.push_str("| Id | Precondition | Measured | Threshold | Verdict |\n");
        s.push_str("|----|--------------|---------:|----------:|:-------:|\n");
        for c in &self.checks {
            s.push_str(&c.to_markdown_row());
            s.push('\n');
        }
        s.push_str(&format!(
            "\nMeasured `alpha` = {:.9}, spread = {:.3e}, phase error = {:.3e}, \
             window deviation = {:.3e}, column residual = {:.3e}.\n",
            self.measured.alpha,
            self.measured.scale_spread,
            self.measured.phase_error,
            self.measured.window_deviation,
            self.measured.column_residual
        ));
        if !self.violations.is_empty() {
            s.push_str("\n## Violations\n\n");
            for v in &self.violations {
                s.push_str(&format!(
                    "- **{} — {:?}**: {}\n  - remedy: {}\n",
                    v.precondition.code(),
                    v.category,
                    v.detail,
                    v.remedy
                ));
            }
        }
        s
    }
}

/// Picks the probe length: small for cost, but the deployment length when a permutation is
/// declared (so that the permutation is observable and can be inverted).
fn select_probe_n<B: MdctBackend>(backend: &B, cfg: &KitConfig) -> Result<usize, BackendError> {
    let desc = backend.descriptor();
    let want = match desc.ordering {
        OutputOrdering::Permutation(_) => cfg.n,
        _ => util::even_at_least_four(cfg.n.min(8)),
    };
    if desc.supports(want) {
        Ok(want)
    } else if desc.supports(cfg.n) {
        Ok(cfg.n)
    } else {
        Err(BackendError::UnsupportedLength {
            requested: cfg.n,
            supported: desc.supported_lengths.clone(),
        })
    }
}

/// Probes `backend` and decides H1–H4 against `cfg`.
///
/// This is the only path that may produce a [`ComplianceReport`]; [`crate::pipeline::DctIvKit`]
/// calls it once at construction and refuses service on rejection.
pub fn evaluate<B: MdctBackend>(
    backend: &mut B,
    cfg: &KitConfig,
    thresholds: Option<Thresholds>,
) -> Result<ComplianceReport, BackendError> {
    let probe_n = select_probe_n(backend, cfg)?;
    let desc = backend.descriptor().clone();
    let thresholds = thresholds.unwrap_or_else(|| Thresholds::for_format(desc.format, probe_n, 3));

    let mut checks = Vec::with_capacity(4);
    let mut violations = Vec::new();
    let mut measured = Measured {
        alpha: f64::NAN,
        scale_spread: f64::NAN,
        phase_error: f64::NAN,
        window_deviation: f64::NAN,
        column_residual: f64::NAN,
    };

    // ---- H4 first: without a decidable bin order, every per-bin statistic is meaningless.
    let ordering_ok = desc.ordering.is_bijection(cfg.n);
    let h4 = CheckOutcome {
        precondition: Precondition::H4NaturalOrdering,
        passed: ordering_ok,
        metric: if ordering_ok { 0.0 } else { f64::NAN },
        threshold: 0.0,
        detail: match &desc.ordering {
            OutputOrdering::Natural => "natural order declared and used".to_string(),
            OutputOrdering::Permutation(p) => format!(
                "declared permutation of {} bins is a valid bijection; POST will apply its inverse",
                p.len()
            ),
            OutputOrdering::Unspecified => {
                "output ordering is unspecified: the kit cannot guarantee correctness".to_string()
            }
        },
    };
    if !ordering_ok {
        let detail = match &desc.ordering {
            OutputOrdering::Permutation(p) => format!(
                "declared permutation is not a valid bijection of 0..{} (length {})",
                cfg.n,
                p.len()
            ),
            _ => "output ordering is unspecified; the per-bin contract cannot be verified"
                .to_string(),
        };
        violations.push(Violation::new(
            Precondition::H4NaturalOrdering,
            Vulnerability::OrderingAmbiguity,
            detail,
        ));
    }

    if ordering_ok {
        let kernel = MdctKernel::new(probe_n);
        let two_n = 2 * probe_n;
        let inv = desc.ordering.inverse_permutation(cfg.n);

        // Undo the declared permutation so that all per-bin statistics refer to natural order.
        let normalize = |y: Vec<f64>| -> Vec<f64> {
            match &inv {
                Some(inv) if inv.len() == y.len() => inv.iter().map(|&i| y[i]).collect(),
                _ => y,
            }
        };

        // ---- H1: implied window from all impulse columns.
        let mut gains = Vec::with_capacity(two_n);
        let mut residuals = Vec::with_capacity(two_n);
        for m in 0..two_n {
            let y = normalize(backend.probe_impulse(probe_n, m)?);
            let a = kernel.column(m);
            let grid = util::dot(&a, &a);
            let g = if grid > 0.0 {
                util::dot(&y, &a) / grid
            } else {
                0.0
            };
            let ny = util::norm2(&y);
            let residual = if ny > 0.0 {
                let mut d = y.clone();
                for (di, ai) in d.iter_mut().zip(&a) {
                    *di -= g * ai;
                }
                util::norm2(&d) / ny
            } else {
                1.0
            };
            gains.push(g.abs());
            residuals.push(residual);
        }
        let median_gain = util::median(&gains);
        let window_deviation = if median_gain > 0.0 {
            gains
                .iter()
                .map(|g| (g / median_gain - 1.0).abs())
                .fold(0.0_f64, f64::max)
        } else {
            f64::INFINITY
        };
        let column_residual = residuals.iter().fold(0.0_f64, |m, &r| m.max(r));
        measured.window_deviation = window_deviation;
        measured.column_residual = column_residual;

        let h1_pass = window_deviation <= thresholds.window_tol;
        checks.push(CheckOutcome {
            precondition: Precondition::H1WindowFree,
            passed: h1_pass,
            metric: window_deviation,
            threshold: thresholds.window_tol,
            detail: format!(
                "implied window over {} impulse columns: max deviation {:.3e} from unity \
                 (median gain {:.3e}, max column residual {:.3e})",
                two_n, window_deviation, median_gain, column_residual
            ),
        });
        if !h1_pass {
            violations.push(Violation::new(
                Precondition::H1WindowFree,
                Vulnerability::WindowContamination,
                format!(
                    "implied window deviates by {:.3e} (> {:.3e}); the backend is tapering its \
                     input, so M_ip is not proportional to M_{{2N->N}}",
                    window_deviation, thresholds.window_tol
                ),
            ));
        }

        // ---- H2 and H3 share the probe vectors.
        let mut rng = Rng::new(0x5EED_1234_ABCD_0001);
        let mut worst_phase = 0.0_f64;
        let mut ratios = Vec::new();
        for _ in 0..thresholds.probe_vectors {
            let x = rng.signed_vec(two_n);
            let mut reference = vec![0.0; probe_n];
            kernel.apply(&x, &mut reference);
            let y = normalize(backend.probe_vector(&x)?);

            // H2: pointwise comparison against the best scalar multiple of the reference.
            let grid = util::dot(&reference, &reference);
            let a = if grid > 0.0 {
                util::dot(&y, &reference) / grid
            } else {
                0.0
            };
            let max_diff = y
                .iter()
                .zip(&reference)
                .map(|(yi, ri)| (yi - a * ri).abs())
                .fold(0.0_f64, f64::max);
            let denom = 1.0 + a.abs() * util::max_abs(&reference);
            worst_phase = worst_phase.max(max_diff / denom);

            // H3: per-bin scale estimates, restricted to well-conditioned bins.
            let floor = 0.25 * util::max_abs(&reference);
            if floor > 0.0 {
                for k in 0..probe_n {
                    if reference[k].abs() >= floor {
                        ratios.push(y[k] / reference[k]);
                    }
                }
            }
        }
        measured.phase_error = worst_phase;

        let h2_pass = worst_phase <= thresholds.eps_phase;
        // Attribute downstream failures: a window makes every per-bin statistic wrong too.
        let h1_note = if h1_pass {
            ""
        } else {
            " [H1 also failed: this is likely a consequence of the window contamination]"
        };
        checks.push(CheckOutcome {
            precondition: Precondition::H2PhaseMatched,
            passed: h2_pass,
            metric: worst_phase,
            threshold: thresholds.eps_phase,
            detail: format!(
                "worst pointwise relative error over {} probe vectors: {:.3e}{}",
                thresholds.probe_vectors, worst_phase, h1_note
            ),
        });
        if !h2_pass {
            violations.push(Violation::new(
                Precondition::H2PhaseMatched,
                Vulnerability::PhaseMismatch,
                format!(
                    "pointwise error {:.3e} exceeds eps_phase = {:.3e}: the backend's cosine \
                     phase does not match (m + 1/2 + N/2)(k + 1/2)",
                    worst_phase, thresholds.eps_phase
                ),
            ));
        }

        // H3: spread of the per-bin scale.
        let (alpha_hat, spread) = if ratios.is_empty() {
            (f64::NAN, f64::INFINITY)
        } else {
            let mean_abs = util::mean(&ratios.iter().map(|r| r.abs()).collect::<Vec<_>>());
            let lo = ratios.iter().fold(f64::INFINITY, |m, &r| m.min(r));
            let hi = ratios.iter().fold(f64::NEG_INFINITY, |m, &r| m.max(r));
            (
                util::mean(&ratios),
                if mean_abs > 0.0 {
                    (hi - lo) / mean_abs
                } else {
                    f64::INFINITY
                },
            )
        };
        measured.alpha = alpha_hat;
        measured.scale_spread = spread;

        let alpha_mismatch = if alpha_hat.is_finite() && cfg.alpha != 0.0 {
            (alpha_hat / cfg.alpha - 1.0).abs()
        } else {
            f64::INFINITY
        };
        let declared_mismatch = if desc.declared_scale != 0.0 {
            (desc.declared_scale / cfg.alpha - 1.0).abs()
        } else {
            f64::INFINITY
        };
        let h3_pass = spread <= thresholds.tau_scale
            && alpha_mismatch <= thresholds.alpha_tol
            && declared_mismatch <= thresholds.alpha_tol;

        checks.push(CheckOutcome {
            precondition: Precondition::H3ScalarScaling,
            passed: h3_pass,
            metric: spread,
            threshold: thresholds.tau_scale,
            detail: format!(
                "measured alpha = {:.9} (configured {:.9}, mismatch {:.3e}; declared {:.9}, \
                 mismatch {:.3e}); per-bin spread {:.3e} over {} samples{}",
                alpha_hat,
                cfg.alpha,
                alpha_mismatch,
                desc.declared_scale,
                declared_mismatch,
                spread,
                ratios.len(),
                h1_note
            ),
        });
        if !h3_pass {
            let why = if spread > thresholds.tau_scale {
                format!(
                    "per-bin scale spread {:.3e} exceeds tau_dtype = {:.3e}: the scaling is not \
                     a single scalar over the band",
                    spread, thresholds.tau_scale
                )
            } else if alpha_mismatch > thresholds.alpha_tol {
                format!(
                    "measured alpha {:.9} disagrees with the configured alpha {:.9} \
                     (relative {:.3e} > {:.3e})",
                    alpha_hat, cfg.alpha, alpha_mismatch, thresholds.alpha_tol
                )
            } else {
                format!(
                    "declared alpha {:.9} disagrees with the configured alpha {:.9} \
                     (relative {:.3e} > {:.3e})",
                    desc.declared_scale, cfg.alpha, declared_mismatch, thresholds.alpha_tol
                )
            };
            violations.push(Violation::new(
                Precondition::H3ScalarScaling,
                Vulnerability::ScalingDrift,
                why,
            ));
        }

        checks.push(h4);
    } else {
        // Fail closed: H1–H3 cannot be decided without a decidable bin order.
        let undecidable = |p: Precondition| CheckOutcome {
            precondition: p,
            passed: false,
            metric: f64::NAN,
            threshold: match p {
                Precondition::H1WindowFree => thresholds.window_tol,
                Precondition::H2PhaseMatched => thresholds.eps_phase,
                _ => thresholds.tau_scale,
            },
            detail: "not decidable: the backend's output ordering is unspecified or malformed \
                     (see H4); per-bin statistics are meaningless"
                .to_string(),
        };
        for p in [
            Precondition::H1WindowFree,
            Precondition::H2PhaseMatched,
            Precondition::H3ScalarScaling,
        ] {
            checks.push(undecidable(p));
            violations.push(Violation::new(
                p,
                Vulnerability::OrderingAmbiguity,
                format!(
                    "{} is undecidable because the output ordering is not declared; \
                     fix H4 first",
                    p.code()
                ),
            ));
        }
        checks.push(h4);
    }

    // Keep the contract order regardless of evaluation order.
    checks.sort_by_key(|c| match c.precondition {
        Precondition::H1WindowFree => 0,
        Precondition::H2PhaseMatched => 1,
        Precondition::H3ScalarScaling => 2,
        Precondition::H4NaturalOrdering => 3,
    });

    Ok(ComplianceReport {
        backend: desc.name.clone(),
        format: desc.format,
        ordering: desc.ordering.clone(),
        probe_n,
        thresholds,
        measured,
        checks,
        violations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{EmulatedMdct, WindowKind};

    fn cfg(n: usize) -> KitConfig {
        KitConfig::new(n)
    }

    #[test]
    fn ideal_backend_is_accepted() {
        let mut b = EmulatedMdct::ideal("ideal");
        let r = evaluate(&mut b, &cfg(16), None).unwrap();
        assert!(r.accepted(), "{}", r.to_markdown());
        assert_eq!(r.checks.len(), 4);
        assert!((r.measured.alpha - 1.0).abs() < 1e-12);
    }

    #[test]
    fn windowing_is_rejected_by_h1() {
        let mut b = EmulatedMdct::ideal("win").with_window_kind(WindowKind::Hann);
        let r = evaluate(&mut b, &cfg(16), None).unwrap();
        assert!(!r.accepted());
        assert!(!r.check(Precondition::H1WindowFree).unwrap().passed);
        assert_eq!(r.violations[0].category, Vulnerability::WindowContamination);
    }

    #[test]
    fn phase_offset_is_rejected_by_h2() {
        let mut b = EmulatedMdct::ideal("phase").with_phase(0.05);
        let r = evaluate(&mut b, &cfg(16), None).unwrap();
        assert!(!r.accepted());
        assert!(!r.check(Precondition::H2PhaseMatched).unwrap().passed);
        assert!(r
            .violations
            .iter()
            .any(|v| v.category == Vulnerability::PhaseMismatch));
    }

    #[test]
    fn gain_drift_is_rejected_by_h3() {
        let mut b = EmulatedMdct::ideal("drift").with_gain_drift(0.01);
        let r = evaluate(&mut b, &cfg(16), None).unwrap();
        assert!(!r.accepted());
        assert!(!r.check(Precondition::H3ScalarScaling).unwrap().passed);
        assert!(r
            .violations
            .iter()
            .any(|v| v.category == Vulnerability::ScalingDrift));
    }

    #[test]
    fn unspecified_ordering_fails_closed() {
        let mut b = EmulatedMdct::ideal("shuffled").with_ordering(OutputOrdering::Unspecified);
        let r = evaluate(&mut b, &cfg(16), None).unwrap();
        assert!(!r.accepted());
        assert!(!r.check(Precondition::H4NaturalOrdering).unwrap().passed);
        assert!(!r.check(Precondition::H1WindowFree).unwrap().passed);
        assert!(r
            .violations
            .iter()
            .all(|v| v.category == Vulnerability::OrderingAmbiguity));
    }

    #[test]
    fn declared_permutation_is_accepted_and_reported() {
        let mut b = EmulatedMdct::ideal("perm")
            .with_ordering(OutputOrdering::Permutation(vec![3, 1, 0, 2]));
        let r = evaluate(&mut b, &cfg(4), None).unwrap();
        assert!(r.accepted(), "{}", r.to_markdown());
        assert!(matches!(r.ordering, OutputOrdering::Permutation(_)));
    }

    #[test]
    fn alpha_mismatch_is_rejected() {
        let mut b = EmulatedMdct::ideal("scaled").with_scale(2.0);
        let mut c = cfg(8);
        c.alpha = 1.0; // the kit expects unit scale, the IP delivers 2x
        let r = evaluate(&mut b, &c, None).unwrap();
        assert!(!r.accepted());
        assert!(r
            .violations
            .iter()
            .any(|v| v.category == Vulnerability::ScalingDrift));
    }

    #[test]
    fn malformed_permutation_is_rejected() {
        let mut b = EmulatedMdct::ideal("bad-perm")
            .with_ordering(OutputOrdering::Permutation(vec![0, 0, 1, 2]));
        let r = evaluate(&mut b, &cfg(4), None).unwrap();
        assert!(!r.accepted());
        assert!(r.checks.iter().filter(|c| !c.passed).count() == 4);
    }

    #[test]
    fn report_renders_markdown() {
        let mut b = EmulatedMdct::ideal("render");
        let r = evaluate(&mut b, &cfg(8), None).unwrap();
        let md = r.to_markdown();
        assert!(md.contains("ACCEPT"));
        assert!(md.contains("| H1 |"));
        assert!(md.contains("| H4 |"));
    }
}
