//! Configuration and the validity constraint of eq. 3.
//!
//! ```text
//! valid <=> (N mod 2 = 0) && (N in N_ip) && (Win >= Wu + 1) && (Wout >= Wip)      (3)
//! ```
//!
//! The kit refuses to start on an invalid configuration. Beyond the four terms of eq. 3 the
//! validator also checks the things an implementation cannot get right without being told
//! (fractional-bit agreement, positive finite `alpha`, `sigma in {+1,-1}`, non-empty batch and
//! buffer pool). Each rejection names the violated clause and the concrete fix.

use crate::fixed::QFormat;
use std::fmt;

/// The set `N_ip` of block lengths an external backend supports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LengthSet {
    /// Every even length `>= 4`. Used by analytic/reference backends.
    AnyEven,
    /// An explicit roster, e.g. the hard-wired sizes of a fixed-function IP.
    Explicit(Vec<usize>),
}

impl LengthSet {
    /// The unrestricted set `{ N : N even, N >= 4 }`.
    pub fn any_even() -> Self {
        LengthSet::AnyEven
    }

    /// An explicit roster; duplicates are removed and the list is sorted.
    pub fn of(lengths: &[usize]) -> Self {
        let mut v = lengths.to_vec();
        v.sort_unstable();
        v.dedup();
        LengthSet::Explicit(v)
    }

    /// Membership test for `N in N_ip`.
    pub fn contains(&self, n: usize) -> bool {
        match self {
            LengthSet::AnyEven => n >= 4 && n % 2 == 0,
            LengthSet::Explicit(v) => v.contains(&n),
        }
    }

    /// Human-readable description used in diagnostics.
    pub fn describe(&self) -> String {
        match self {
            LengthSet::AnyEven => "{ N : N even, N >= 4 }".to_string(),
            LengthSet::Explicit(v) => {
                let list: Vec<String> = v.iter().map(|x| x.to_string()).collect();
                format!("{{{}}}", list.join(", "))
            }
        }
    }
}

impl fmt::Display for LengthSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

/// A violated clause of the configuration contract.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigViolation {
    /// `N mod 2 != 0`; the four-block partition of §3.3 needs `N/2` to be integral.
    OddLength {
        /// The rejected block length.
        n: usize,
    },
    /// `N < 4`; the kit is specified for `N >= 4`.
    TooShort {
        /// The rejected block length.
        n: usize,
    },
    /// `N not in N_ip`: the backend cannot be driven at this length.
    LengthNotSupported {
        /// The rejected block length.
        n: usize,
        /// The backend's supported set, rendered.
        supported: String,
    },
    /// `Win < Wu + 1`: the folding stage would overflow a folded word.
    MissingGuardBit {
        /// Width of the folded word.
        win: u8,
        /// Width of the input word.
        wu: u8,
    },
    /// `Wout < Wip`: the kit's output would narrow the backend's word.
    OutputTooNarrow {
        /// Width of the kit output.
        wout: u8,
        /// Width of the backend output.
        wip: u8,
    },
    /// A fixed-point format fails its own invariants.
    InvalidFormat {
        /// Which stage owns the format.
        stage: &'static str,
        /// Why the format is invalid.
        reason: &'static str,
    },
    /// The fractional bits of two stages that must align do not.
    FractionMismatch {
        /// Which stage owns the format.
        stage: &'static str,
        /// Expected fractional bits.
        expected: u8,
        /// Actual fractional bits.
        got: u8,
    },
    /// The declared backend scale is not a positive, finite, non-zero real.
    InvalidAlpha {
        /// The rejected value.
        alpha: f64,
    },
    /// The sign convention is neither `+1` nor `-1`.
    InvalidSign {
        /// The rejected value.
        sign: i8,
    },
    /// A zero-length batch cannot be scheduled.
    EmptyBatch,
    /// A buffer pool with zero slots cannot hold a single in-flight block.
    EmptyPool,
}

impl fmt::Display for ConfigViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigViolation::OddLength { n } => {
                write!(f, "N = {n} is odd (clause 1: N mod 2 = 0 required)")
            }
            ConfigViolation::TooShort { n } => {
                write!(f, "N = {n} is below the specified minimum N >= 4")
            }
            ConfigViolation::LengthNotSupported { n, supported } => {
                write!(f, "N = {n} is not in N_ip = {supported} (clause 2)")
            }
            ConfigViolation::MissingGuardBit { win, wu } => write!(
                f,
                "Win = {win} but Wu = {wu}: clause 3 requires Win >= Wu + 1 \
                 (the fold sums two words and needs one guard bit)"
            ),
            ConfigViolation::OutputTooNarrow { wout, wip } => write!(
                f,
                "Wout = {wout} but Wip = {wip}: clause 4 requires Wout >= Wip"
            ),
            ConfigViolation::InvalidFormat { stage, reason } => {
                write!(f, "invalid fixed-point format for {stage}: {reason}")
            }
            ConfigViolation::FractionMismatch {
                stage,
                expected,
                got,
            } => write!(
                f,
                "{stage} has Q{got} fractional bits but must match Q{expected}: the folding \
                 stage is exact only when the fractional point does not move"
            ),
            ConfigViolation::InvalidAlpha { alpha } => {
                write!(f, "alpha = {alpha} must be finite and > 0")
            }
            ConfigViolation::InvalidSign { sign } => {
                write!(f, "sign = {sign} must be +1 or -1")
            }
            ConfigViolation::EmptyBatch => f.write_str("batch size B must be >= 1"),
            ConfigViolation::EmptyPool => {
                f.write_str("pool slots K must be >= 1 (K/2 slots are needed per in-flight block)")
            }
        }
    }
}

/// The complete configuration of a DCT-IV kit instance.
///
/// Widths follow the design's symbols: `Wu` input, `Win` folded, `Wip` backend output,
/// `Wout` kit output; `Q` is carried per stage so that a rescaling is explicit.
#[derive(Debug, Clone, PartialEq)]
pub struct KitConfig {
    /// `N`, the DCT-IV block length (`N mod 2 = 0`, `N >= 4`).
    pub n: usize,
    /// `B`, the batch size.
    pub batch: usize,
    /// `Wu`: the format of the caller's input `U`.
    pub input: QFormat,
    /// `Win`: the format of the folded vector handed to the backend.
    pub guard: QFormat,
    /// `Wip`: the format of the backend's output.
    pub ip: QFormat,
    /// `Wout`: the format of the kit's output `X`.
    pub output: QFormat,
    /// `alpha`: the backend's scalar scaling `M_ip = alpha * M_B`.
    pub alpha: f64,
    /// `sigma`: the sign convention of the backend, `+1` or `-1`.
    pub sign: i8,
    /// `K`: number of GPU buffer slots in the adaptation model.
    pub pool_slots: usize,
    /// `N_ip`: the set of lengths the external backend accepts.
    pub lengths: LengthSet,
}

impl Default for KitConfig {
    fn default() -> Self {
        Self::new(32)
    }
}

impl KitConfig {
    /// A valid default configuration for block length `n`.
    ///
    /// Defaults: `Wu = Q1.14 (16 bit)`, `Win = Q1.14 (17 bit)`, `Wip = Wout = Q9.14 (24 bit)`,
    /// `alpha = 1`, `sigma = +1`, `B = 1`, `K = 6` slots, `N_ip = {even N >= 4}`.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            batch: 1,
            input: QFormat::new(16, 14),
            guard: QFormat::new(17, 14),
            ip: QFormat::new(24, 14),
            output: QFormat::new(24, 14),
            alpha: 1.0,
            sign: 1,
            pool_slots: 6,
            lengths: LengthSet::AnyEven,
        }
    }

    /// Sets the batch size.
    pub fn with_batch(mut self, batch: usize) -> Self {
        self.batch = batch;
        self
    }

    /// Sets the backend's supported length set `N_ip`.
    pub fn with_lengths(mut self, lengths: LengthSet) -> Self {
        self.lengths = lengths;
        self
    }

    /// Sets the backend scalar scaling `alpha` and the matching post-stage `S = 1/alpha`.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the sign convention `sigma`.
    pub fn with_sign(mut self, sign: i8) -> Self {
        self.sign = sign;
        self
    }

    /// Sets the buffer pool size `K`.
    pub fn with_pool_slots(mut self, k: usize) -> Self {
        self.pool_slots = k;
        self
    }

    /// Sets all four word formats.
    pub fn with_formats(
        mut self,
        input: QFormat,
        guard: QFormat,
        ip: QFormat,
        output: QFormat,
    ) -> Self {
        self.input = input;
        self.guard = guard;
        self.ip = ip;
        self.output = output;
        self
    }

    /// Length of the folded vector handed to the backend, `2N`.
    pub fn folded_len(&self) -> usize {
        2 * self.n
    }

    /// Number of samples in a batch, `B * N`.
    pub fn batch_samples(&self) -> usize {
        self.batch * self.n
    }

    /// Evaluates the validity constraint of eq. 3, plus the implementation-level checks.
    pub fn validate(&self) -> Result<(), Vec<ConfigViolation>> {
        let mut v = Vec::new();

        // Clause 1: N mod 2 = 0, and the specification's minimum N >= 4.
        if self.n % 2 != 0 {
            v.push(ConfigViolation::OddLength { n: self.n });
        }
        if self.n < 4 {
            v.push(ConfigViolation::TooShort { n: self.n });
        }
        // Clause 2: N in N_ip.
        if !self.lengths.contains(self.n) {
            v.push(ConfigViolation::LengthNotSupported {
                n: self.n,
                supported: self.lengths.describe(),
            });
        }
        // Clause 3: Win >= Wu + 1.
        if self.guard.width < self.input.width.saturating_add(1) {
            v.push(ConfigViolation::MissingGuardBit {
                win: self.guard.width,
                wu: self.input.width,
            });
        }
        // Clause 4: Wout >= Wip.
        if self.output.width < self.ip.width {
            v.push(ConfigViolation::OutputTooNarrow {
                wout: self.output.width,
                wip: self.ip.width,
            });
        }
        // Structural checks the paper leaves implicit.
        for (stage, fmt) in [
            ("input (Wu)", self.input),
            ("guard (Win)", self.guard),
            ("ip (Wip)", self.ip),
            ("output (Wout)", self.output),
        ] {
            if let Err(reason) = fmt.validate() {
                v.push(ConfigViolation::InvalidFormat { stage, reason });
            }
        }
        if self.guard.frac != self.input.frac {
            v.push(ConfigViolation::FractionMismatch {
                stage: "guard (Win)",
                expected: self.input.frac,
                got: self.guard.frac,
            });
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            v.push(ConfigViolation::InvalidAlpha { alpha: self.alpha });
        }
        if self.sign != 1 && self.sign != -1 {
            v.push(ConfigViolation::InvalidSign { sign: self.sign });
        }
        if self.batch == 0 {
            v.push(ConfigViolation::EmptyBatch);
        }
        if self.pool_slots == 0 {
            v.push(ConfigViolation::EmptyPool);
        }

        if v.is_empty() {
            Ok(())
        } else {
            Err(v)
        }
    }

    /// Convenience predicate form of [`KitConfig::validate`].
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// The post-stage scalar `S = 1/alpha` of eq. 2.
    pub fn post_scale(&self) -> f64 {
        1.0 / self.alpha
    }

    /// The signed post-stage gain `sigma/alpha`.
    pub fn signed_post_gain(&self) -> f64 {
        self.sign as f64 * self.post_scale()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        for n in [4, 6, 8, 12, 32, 1024] {
            assert_eq!(KitConfig::new(n).validate(), Ok(()));
        }
    }

    #[test]
    fn clause_violations_are_reported() {
        let c = KitConfig::new(7);
        let v = c.validate().unwrap_err();
        assert!(v.contains(&ConfigViolation::OddLength { n: 7 }));

        let c = KitConfig::new(2);
        let v = c.validate().unwrap_err();
        assert!(v.contains(&ConfigViolation::TooShort { n: 2 }));

        let c = KitConfig::new(12).with_lengths(LengthSet::of(&[8, 16]));
        assert!(matches!(
            c.validate().unwrap_err().as_slice(),
            [ConfigViolation::LengthNotSupported { n: 12, .. }]
        ));

        let mut c = KitConfig::new(8);
        c.guard = QFormat::new(16, 14); // Win == Wu, no guard bit
        assert!(matches!(
            c.validate().unwrap_err().as_slice(),
            [ConfigViolation::MissingGuardBit { .. }]
        ));

        let mut c = KitConfig::new(8);
        c.output = QFormat::new(20, 14); // Wout < Wip
        assert!(matches!(
            c.validate().unwrap_err().as_slice(),
            [ConfigViolation::OutputTooNarrow { .. }]
        ));
    }

    #[test]
    fn structural_violations_are_reported() {
        assert!(!KitConfig::new(8).with_alpha(0.0).is_valid());
        assert!(!KitConfig::new(8).with_alpha(f64::INFINITY).is_valid());
        assert!(!KitConfig::new(8).with_sign(0).is_valid());
        assert!(!KitConfig::new(8).with_batch(0).is_valid());
        assert!(!KitConfig::new(8).with_pool_slots(0).is_valid());

        let mut c = KitConfig::new(8);
        c.guard = QFormat::new(17, 12); // right width, wrong fraction
        assert!(matches!(
            c.validate().unwrap_err().as_slice(),
            [ConfigViolation::FractionMismatch { .. }]
        ));
    }

    #[test]
    fn post_gain() {
        let c = KitConfig::new(8).with_alpha(4.0);
        assert_eq!(c.post_scale(), 0.25);
        assert_eq!(c.signed_post_gain(), 0.25);
        let c = c.with_sign(-1);
        assert_eq!(c.signed_post_gain(), -0.25);
    }

    #[test]
    fn length_sets() {
        let s = LengthSet::AnyEven;
        assert!(s.contains(4) && s.contains(9 + 1) && !s.contains(9) && !s.contains(2));
        let s = LengthSet::of(&[32, 8, 8, 16]);
        assert_eq!(s.describe(), "{8, 16, 32}");
        assert!(s.contains(16) && !s.contains(64));
    }
}
