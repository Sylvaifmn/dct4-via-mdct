//! The three-stage pipeline `X = POST(MDCT IP(FOLD(U)))` (§4.1, eq. 2).
//!
//! ```text
//! U ──►[FOLD]──► x ──►[MDCT IP]──► y ──►[POST]──► X
//!        F^{-1}        y = a·M(x)        S = sigma/a, saturation, permutation
//! ```
//!
//! [`DctIvKit::new`] wires the compliance gate in front of the pipeline: a backend that fails
//! H1–H4 is refused with [`KitError::BackendRejected`] (`backend_reject`), so the transform
//! methods can only ever run on an admissible backend. [`GatePolicy::Bypass`] exists for
//! bring-up only and still records the violations in the report.
//!
//! ## Two arithmetic paths
//!
//! * [`DctIvKit::transform_block`] / [`DctIvKit::transform_batch`] — the analysis path, `f64`
//!   in and out. This is what a host-side reference or a GPU staging buffer uses.
//! * [`DctIvKit::transform_block_fixed`] / [`DctIvKit::transform_batch_fixed`] — the
//!   production path. FOLD runs on raw `i64` codes (exact, per Theorem 2 and Corollary 1),
//!   POST runs as a single rational multiply plus a fractional-point move in `i128`, with
//!   deterministic round-half-away-from-zero and counted saturation.
//!
//! ## The two numbers that catch real bugs
//!
//! [`DctIvKit::fold_overflows`] must stay zero. It is non-zero only if a folded word left the
//! `Win` range, i.e. if the guard bit promised by eq. 3 was not actually provided.
//! [`DctIvKit::saturations`] counts post-stage clamps: a non-zero value on a signal that is
//! in range means the configured `Wout` is too narrow for `alpha`, not that the input was hot.

use crate::backend::{BackendError, MdctBackend};
use crate::compliance::{
    self, ComplianceReport, Precondition, Thresholds, Violation, Vulnerability,
};
use crate::config::KitConfig;
use crate::dct4::Dct4Kernel;
use crate::error::KitError;
use crate::fixed::{self, QFormat, Q};
use crate::folding;
use crate::numerics::{self, BudgetAssessment, ForwardErrorBound};

/// Resolution of the post stage's rational multiplier, `2^-32`.
///
/// The post gain `sigma/alpha` is materialised as `mult / 2^32`; for every power-of-two
/// `alpha` a codec uses (`2^15`, `2^16`, ...) this is exact, and for other values the
/// multiplier's own rounding is far below the output LSB.
const POST_AUX_BITS: u32 = 32;

/// Largest post-gain magnitude the `i128` post stage accepts (`2^31`).
///
/// The post multiplier is `gain * 2^32`; multiplying it by a 63-bit word must stay below
/// `2^126`, hence the limit. The kit refuses at construction rather than clamping silently.
const MAX_POST_GAIN: f64 = 2_147_483_648.0; // 2^31

/// Whether the compliance gate is allowed to refuse service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatePolicy {
    /// Refuse to construct the kit unless every hard precondition H1–H4 passes (default).
    Enforce,
    /// Construct anyway, recording the violations. **Bring-up only**: the resulting
    /// transform is not compliant and its output must not be shipped.
    Bypass,
}

/// Per-call statistics of the pipeline.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BatchStats {
    /// Number of `N`-sample blocks processed.
    pub blocks: usize,
    /// Number of backend invocations.
    pub backend_calls: u64,
    /// Number of post-stage clamps (a non-zero value means `Wout` is too narrow).
    pub saturations: u64,
    /// Number of backend-output words that exceeded `Wip` (a non-zero value means the backend
    /// word is too narrow for `alpha * X`; see [`crate::numerics::required_ip_width`]).
    pub ip_clamps: u64,
    /// Number of folded words that left the `Win` range (must be zero).
    pub fold_overflows: u64,
    /// Peak absolute value written to the output, for headroom monitoring.
    pub peak_abs: f64,
}

impl BatchStats {
    fn accumulate(&mut self, other: BatchStats) {
        self.blocks += other.blocks;
        self.backend_calls += other.backend_calls;
        self.saturations += other.saturations;
        self.ip_clamps += other.ip_clamps;
        self.fold_overflows += other.fold_overflows;
        self.peak_abs = self.peak_abs.max(other.peak_abs);
    }
}

/// The post stage: `X[k] = S * y[perm^{-1}(k)]`, plus saturation and requantization.
#[derive(Debug, Clone)]
struct PostStage {
    gain: f64,
    out: QFormat,
    perm: Option<Vec<usize>>,
    mult: i128,
}

impl PostStage {
    fn new(cfg: &KitConfig, ordering: &crate::backend::OutputOrdering) -> Result<Self, KitError> {
        let gain = cfg.signed_post_gain();
        if !gain.is_finite() || gain.abs() > MAX_POST_GAIN || gain == 0.0 {
            return Err(KitError::Numeric(format!(
                "post gain sigma/alpha = {gain} is outside the representable range of the \
                 integer post stage (|gain| <= 2^53 and != 0)"
            )));
        }
        let mult_f = gain * 2f64.powi(POST_AUX_BITS as i32);
        let mult = mult_f.round();
        if !mult.is_finite() || mult.abs() < 1.0 {
            return Err(KitError::Numeric(format!(
                "post gain {gain} underflows the 32-bit multiplier resolution"
            )));
        }
        let perm = ordering
            .inverse_permutation(cfg.n)
            .filter(|p| p.iter().enumerate().any(|(i, &k)| i != k));
        Ok(Self {
            gain,
            out: cfg.output,
            perm,
            mult: mult as i128,
        })
    }

    /// Float path: gain and optional permutation, no saturation (the caller's domain is
    /// unbounded by design).
    fn apply_f64(&self, y: &[f64], x: &mut [f64]) -> f64 {
        let mut peak = 0.0_f64;
        for k in 0..x.len() {
            let src = match &self.perm {
                Some(p) => p[k],
                None => k,
            };
            let v = self.gain * y[src];
            peak = peak.max(v.abs());
            x[k] = v;
        }
        peak
    }

    /// Fixed path: one rational multiply, one fractional-point move, one saturation.
    /// Returns `(saturations, ip_clamps, peak)`.
    fn apply_fixed(&self, y: &[f64], ip: QFormat, x: &mut [Q]) -> (u64, u64, f64) {
        let ip_scale = 2f64.powi(ip.frac as i32);
        let lo = self.out.min_raw() as i128;
        let hi = self.out.max_raw() as i128;
        let mut saturations = 0u64;
        let mut ip_clamps = 0u64;
        let mut peak = 0.0_f64;
        for k in 0..x.len() {
            let src = match &self.perm {
                Some(p) => p[k],
                None => k,
            };
            let v = y[src] * ip_scale;
            let raw_ip = if v.is_nan() {
                0
            } else if v > ip.max_raw() as f64 || v < ip.min_raw() as f64 {
                ip_clamps += 1;
                v.round().clamp(ip.min_raw() as f64, ip.max_raw() as f64) as i64
            } else {
                v.round() as i64
            };
            let prod = fixed::round_mul_shift(raw_ip, self.mult, POST_AUX_BITS);
            let shifted = fixed::shift_frac(prod, ip.frac, self.out.frac);
            if shifted < lo || shifted > hi {
                saturations += 1;
            }
            let raw = fixed::saturate_i128(shifted, self.out);
            peak = peak.max((raw as f64) / 2f64.powi(self.out.frac as i32));
            x[k] = Q::from_raw(self.out, raw);
        }
        (saturations, ip_clamps, peak)
    }
}

/// A DCT-IV kit bound to one external MDCT backend and one configuration.
///
/// Construction validates eq. 3, checks that the backend supports `N`, probes the backend
/// against H1–H4, and (under [`GatePolicy::Enforce`]) refuses on rejection.
#[derive(Debug, Clone)]
pub struct DctIvKit<B: MdctBackend> {
    cfg: KitConfig,
    backend: B,
    report: ComplianceReport,
    enforced: bool,
    post: PostStage,
    backend_calls: u64,
    saturations: u64,
    ip_clamps: u64,
    fold_overflows: u64,
    fold_buf: Vec<f64>,
    y_buf: Vec<f64>,
    fold_raw: Vec<i64>,
    raw_in: Vec<i64>,
}

impl<B: MdctBackend> DctIvKit<B> {
    /// Validates the configuration, runs the compliance gate, and builds the kit.
    ///
    /// # Errors
    ///
    /// * [`KitError::InvalidConfig`] — eq. 3 (or a structural rule) is violated;
    /// * [`KitError::Backend`] — the backend cannot be driven at `N` or a probe failed;
    /// * [`KitError::BackendRejected`] — H1–H4 rejected the backend (`backend_reject`);
    /// * [`KitError::Numeric`] — the post-stage gain is not representable.
    pub fn new(
        cfg: KitConfig,
        mut backend: B,
        policy: GatePolicy,
        thresholds: Option<Thresholds>,
    ) -> Result<Self, KitError> {
        cfg.validate().map_err(KitError::InvalidConfig)?;

        {
            let desc = backend.descriptor();
            if !desc.supports(cfg.n) {
                return Err(KitError::Backend(BackendError::UnsupportedLength {
                    requested: cfg.n,
                    supported: desc.supported_lengths.clone(),
                }));
            }
        }

        let report = compliance::evaluate(&mut backend, &cfg, thresholds)?;
        if policy == GatePolicy::Enforce {
            if let Err(v) = report.gate() {
                return Err(KitError::BackendRejected(v));
            }
        }

        let post = PostStage::new(&cfg, &report.ordering)?;
        let n = cfg.n;
        Ok(Self {
            cfg,
            backend,
            report,
            enforced: policy == GatePolicy::Enforce,
            post,
            backend_calls: 0,
            saturations: 0,
            ip_clamps: 0,
            fold_overflows: 0,
            fold_buf: vec![0.0; 2 * n],
            y_buf: vec![0.0; n],
            fold_raw: vec![0; 2 * n],
            raw_in: vec![0; n],
        })
    }

    /// The configuration in force.
    pub fn config(&self) -> &KitConfig {
        &self.cfg
    }

    /// The compliance report produced at construction.
    pub fn report(&self) -> &ComplianceReport {
        &self.report
    }

    /// Whether the gate was enforcing (`false` for [`GatePolicy::Bypass`]).
    pub fn is_enforced(&self) -> bool {
        self.enforced
    }

    /// The backend, for backend-specific queries.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Total folded words that left the `Win` range since construction. Must be zero.
    pub fn fold_overflows(&self) -> u64 {
        self.fold_overflows
    }

    /// Total post-stage clamps since construction.
    pub fn saturations(&self) -> u64 {
        self.saturations
    }

    /// Total backend-output words that exceeded `Wip` since construction. Must be zero;
    /// otherwise widen `Wip` (or lower `alpha`) so the backend word covers `alpha * X`.
    pub fn ip_clamps(&self) -> u64 {
        self.ip_clamps
    }

    /// Total backend invocations since construction.
    pub fn backend_calls(&self) -> u64 {
        self.backend_calls
    }

    /// Transforms one `N`-sample block (`f64` path).
    pub fn transform_block(&mut self, u: &[f64], x: &mut [f64]) -> Result<BatchStats, KitError> {
        self.check_float_shape(u, x)?;
        let n = self.cfg.n;

        // FOLD: x = (u_U, 0, 0, -u_L) — permutation, sign, zero-fill; no rounding.
        folding::inverse_fold_into(u, &mut self.fold_buf);

        // MDCT IP: y = alpha * M(x).
        self.backend.transform(&self.fold_buf, &mut self.y_buf)?;
        self.backend_calls += 1;

        // POST: X = S * y with optional permutation correction.
        let peak = self.post.apply_f64(&self.y_buf, x);

        debug_assert_eq!(self.y_buf.len(), n);
        Ok(BatchStats {
            blocks: 1,
            backend_calls: 1,
            saturations: 0,
            ip_clamps: 0,
            fold_overflows: 0,
            peak_abs: peak,
        })
    }

    /// Transforms `B` blocks laid out row-major (`B x N`), `f64` path.
    pub fn transform_batch(&mut self, u: &[f64], x: &mut [f64]) -> Result<BatchStats, KitError> {
        let n = self.cfg.n;
        let b = self.cfg.batch;
        if u.len() != b * n {
            return Err(KitError::Shape {
                context: "U (B x N)",
                expected: b * n,
                got: u.len(),
            });
        }
        if x.len() != b * n {
            return Err(KitError::Shape {
                context: "X (B x N)",
                expected: b * n,
                got: x.len(),
            });
        }
        let mut total = BatchStats::default();
        for block in 0..b {
            let range = block * n..(block + 1) * n;
            let stats = self.transform_block(&u[range.clone()], &mut x[range])?;
            total.accumulate(stats);
        }
        Ok(total)
    }

    /// Transforms one block on the fixed-point path.
    ///
    /// `u` must be in the configured input format and `x` in the configured output format;
    /// the FOLD stage runs on raw integer codes, so it is exact.
    pub fn transform_block_fixed(&mut self, u: &[Q], x: &mut [Q]) -> Result<BatchStats, KitError> {
        let n = self.cfg.n;
        if u.len() != n {
            return Err(KitError::Shape {
                context: "U (N)",
                expected: n,
                got: u.len(),
            });
        }
        if x.len() != n {
            return Err(KitError::Shape {
                context: "X (N)",
                expected: n,
                got: x.len(),
            });
        }
        for q in u {
            if q.format != self.cfg.input {
                return Err(KitError::Numeric(format!(
                    "input word format {} does not match the configured {}",
                    q.format, self.cfg.input
                )));
            }
        }
        for q in x.iter() {
            if q.format != self.cfg.output {
                return Err(KitError::Numeric(format!(
                    "output word format {} does not match the configured {}",
                    q.format, self.cfg.output
                )));
            }
        }

        // FOLD on raw codes: exact integer permutation/sign/zero-fill.
        for (dst, q) in self.raw_in.iter_mut().zip(u) {
            *dst = q.raw();
        }
        folding::inverse_fold_into(&self.raw_in, &mut self.fold_raw);

        let input_scale = 2f64.powi(-(self.cfg.input.frac as i32));
        let guard = self.cfg.guard;
        let mut overflows = 0u64;
        for (i, &raw) in self.fold_raw.iter().enumerate() {
            if !guard.contains(raw) {
                overflows += 1;
            }
            // Exact: |raw| < 2^63 and the shift is a power of two.
            self.fold_buf[i] = raw as f64 * input_scale;
        }
        self.fold_overflows += overflows;

        self.backend.transform(&self.fold_buf, &mut self.y_buf)?;
        self.backend_calls += 1;

        let (saturations, ip_clamps, peak) = self.post.apply_fixed(&self.y_buf, self.cfg.ip, x);
        self.saturations += saturations;
        self.ip_clamps += ip_clamps;

        Ok(BatchStats {
            blocks: 1,
            backend_calls: 1,
            saturations,
            ip_clamps,
            fold_overflows: overflows,
            peak_abs: peak,
        })
    }

    /// Transforms `B` blocks laid out row-major (`B x N`), fixed-point path.
    pub fn transform_batch_fixed(&mut self, u: &[Q], x: &mut [Q]) -> Result<BatchStats, KitError> {
        let n = self.cfg.n;
        let b = self.cfg.batch;
        if u.len() != b * n || x.len() != b * n {
            return Err(KitError::Shape {
                context: "U/X (B x N)",
                expected: b * n,
                got: u.len().min(x.len()),
            });
        }
        let mut total = BatchStats::default();
        for block in 0..b {
            let range = block * n..(block + 1) * n;
            let stats = self.transform_block_fixed(&u[range.clone()], &mut x[range])?;
            total.accumulate(stats);
        }
        Ok(total)
    }

    /// Convenience wrapper around [`DctIvKit::transform_block`] that allocates the output.
    pub fn transform(&mut self, u: &[f64]) -> Result<Vec<f64>, KitError> {
        if u.len() != self.cfg.n {
            return Err(KitError::Shape {
                context: "U (N)",
                expected: self.cfg.n,
                got: u.len(),
            });
        }
        let mut x = vec![0.0; self.cfg.n];
        self.transform_block(u, &mut x)?;
        Ok(x)
    }

    /// Maximum absolute deviation of `x` from the analytic DCT-IV of `u`.
    ///
    /// This is a *verification* aid (bring-up, tests, acceptance records). It is not part of
    /// the production path: the kit itself contains no transform core.
    pub fn deviation_from_reference(&self, u: &[f64], x: &[f64]) -> f64 {
        if u.len() != x.len() {
            return f64::INFINITY;
        }
        let kernel = Dct4Kernel::new(u.len());
        let mut reference = vec![0.0; u.len()];
        kernel.apply(u, &mut reference);
        reference
            .iter()
            .zip(x)
            .map(|(r, v)| (r - v).abs())
            .fold(0.0_f64, f64::max)
    }

    /// The forward error bound for a block of norm `||u||_2`.
    pub fn error_bound(&self, u_norm2: f64) -> ForwardErrorBound {
        numerics::forward_error_bound(
            self.cfg.n,
            u_norm2,
            self.report.format,
            self.cfg.signed_post_gain().abs(),
            self.cfg.input,
            self.cfg.output,
        )
    }

    /// Whether the backend's declared precision can support the configured output word.
    pub fn assess_budget(&self, u_norm2: f64) -> BudgetAssessment {
        numerics::budget(
            self.cfg.n,
            u_norm2,
            self.report.format,
            self.cfg.signed_post_gain().abs(),
            self.cfg.output,
        )
    }

    /// A `PrecisionInsufficiency` violation when the backend cannot resolve `Wout`.
    ///
    /// This is the §7 error-model counterweight to H3: a backend can be perfectly *uniform*
    /// in scale and still be too coarse for the requested output word length.
    pub fn budget_violation(&self, u_norm2: f64) -> Option<Violation> {
        let b = self.assess_budget(u_norm2);
        if b.ok {
            None
        } else {
            Some(Violation::new(
                Precondition::H3ScalarScaling,
                Vulnerability::PrecisionInsufficiency,
                format!(
                    "backend error {:.3e} exceeds the output half-LSB budget {:.3e} \
                     ({:.1} bits short) in the declared format {}",
                    b.backend, b.budget, b.margin_bits, self.report.format
                ),
            ))
        }
    }

    fn check_float_shape(&self, u: &[f64], x: &[f64]) -> Result<(), KitError> {
        if u.len() != self.cfg.n {
            return Err(KitError::Shape {
                context: "U (N)",
                expected: self.cfg.n,
                got: u.len(),
            });
        }
        if x.len() != self.cfg.n {
            return Err(KitError::Shape {
                context: "X (N)",
                expected: self.cfg.n,
                got: x.len(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{EmulatedMdct, OutputOrdering, WindowKind};
    use crate::dct4::dct4;

    fn kit(n: usize) -> DctIvKit<EmulatedMdct> {
        DctIvKit::new(
            KitConfig::new(n),
            EmulatedMdct::ideal("ip"),
            GatePolicy::Enforce,
            None,
        )
        .unwrap()
    }

    #[test]
    fn matches_direct_dct4() {
        let mut k = kit(32);
        let mut rng = crate::util::Rng::new(9);
        for _ in 0..8 {
            let u = rng.signed_vec(32);
            let x = k.transform(&u).unwrap();
            let r = dct4(&u);
            for i in 0..32 {
                assert!((x[i] - r[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn scaled_backend_is_compensated() {
        let cfg = KitConfig::new(16).with_alpha(1024.0).with_sign(-1);
        let backend = EmulatedMdct::ideal("ip").with_scale(1024.0);
        let mut k = DctIvKit::new(cfg, backend, GatePolicy::Enforce, None).unwrap();
        let u = crate::util::Rng::new(3).signed_vec(16);
        let x = k.transform(&u).unwrap();
        let mut expect = dct4(&u);
        for v in &mut expect {
            *v = -*v;
        }
        for i in 0..16 {
            assert!((x[i] - expect[i]).abs() < 1e-9, "{} vs {}", x[i], expect[i]);
        }
    }

    #[test]
    fn permutation_is_corrected() {
        let n = 8;
        let perm: Vec<usize> = vec![3, 7, 0, 5, 1, 6, 2, 4];
        let backend = EmulatedMdct::ideal("ip").with_ordering(OutputOrdering::Permutation(perm));
        let mut k = DctIvKit::new(KitConfig::new(n), backend, GatePolicy::Enforce, None).unwrap();
        let u = crate::util::Rng::new(11).signed_vec(n);
        let x = k.transform(&u).unwrap();
        let r = dct4(&u);
        for i in 0..n {
            assert!((x[i] - r[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn windowed_backend_is_refused() {
        let backend = EmulatedMdct::ideal("ip").with_window_kind(WindowKind::Hann);
        let err =
            DctIvKit::new(KitConfig::new(16), backend, GatePolicy::Enforce, None).unwrap_err();
        match err {
            KitError::BackendRejected(v) => {
                assert!(v
                    .iter()
                    .any(|x| x.precondition == Precondition::H1WindowFree));
            }
            other => panic!("expected backend_reject, got {other:?}"),
        }
    }

    #[test]
    fn bypass_still_reports() {
        let backend = EmulatedMdct::ideal("ip").with_phase(0.2);
        let k = DctIvKit::new(KitConfig::new(16), backend, GatePolicy::Bypass, None).unwrap();
        assert!(!k.is_enforced());
        assert!(!k.report().accepted());
    }

    #[test]
    fn fixed_path_matches_float_path() {
        let mut k = kit(16);
        let inp = k.cfg.input;
        let out = k.cfg.output;
        let mut rng = crate::util::Rng::new(21);
        for _ in 0..8 {
            let u: Vec<Q> = rng
                .signed_vec(16)
                .into_iter()
                .map(|v| Q::from_f64(inp, v * 0.5))
                .collect();
            let mut xf = vec![Q::zero(out); 16];
            let stats = k.transform_block_fixed(&u, &mut xf).unwrap();
            assert_eq!(stats.fold_overflows, 0);
            assert_eq!(stats.saturations, 0);

            let uf: Vec<f64> = u.iter().map(|q| q.to_f64()).collect();
            let xr = k.transform(&uf).unwrap();
            for i in 0..16 {
                // Two quantization steps: the input word and the output word.
                let tol = 2.0 * (inp.lsb() + out.lsb()) + 1e-9;
                assert!(
                    (xf[i].to_f64() - xr[i]).abs() <= tol + out.lsb(),
                    "{} vs {}",
                    xf[i].to_f64(),
                    xr[i]
                );
            }
        }
    }

    #[test]
    fn shapes_and_formats_are_checked() {
        let mut k = kit(8);
        assert!(k.transform(&[0.0; 7]).is_err());
        let mut x = [0.0; 8];
        assert!(k.transform_block(&[0.0; 8], &mut x[..7]).is_err());
        let bad = QFormat::new(14, 12);
        let u = vec![Q::zero(bad); 8];
        let mut out = vec![Q::zero(k.cfg.output); 8];
        assert!(k.transform_block_fixed(&u, &mut out).is_err());
    }

    #[test]
    fn batch_layout() {
        let mut k = DctIvKit::new(
            KitConfig::new(8).with_batch(3),
            EmulatedMdct::ideal("ip"),
            GatePolicy::Enforce,
            None,
        )
        .unwrap();
        let u = vec![0.5; 24];
        let mut x = vec![0.0; 24];
        let stats = k.transform_batch(&u, &mut x).unwrap();
        assert_eq!(stats.blocks, 3);
        assert_eq!(stats.backend_calls, 3);
        let r = dct4(&[0.5; 8]);
        for b in 0..3 {
            for i in 0..8 {
                assert!((x[b * 8 + i] - r[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn budget_flags_a_coarse_backend() {
        let cfg = KitConfig::new(32).with_formats(
            QFormat::new(16, 14),
            QFormat::new(17, 14),
            QFormat::new(24, 14),
            QFormat::new(40, 30),
        );
        let backend = EmulatedMdct::ideal("ip").with_format(crate::backend::SampleFormat::F32);
        let k = DctIvKit::new(cfg, backend, GatePolicy::Enforce, None).unwrap();
        assert!(k.budget_violation(1.0).is_some());
    }
}
