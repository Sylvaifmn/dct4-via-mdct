//! Fixed-point arithmetic with explicit word width `W` and fractional bits `Q`.
//!
//! The design's symbol table defines `Q` as the number of fractional bits and
//! `Wu, Win, Wip, Wout` as the input / folded / MDCT-output / kit-output bit widths.
//! A [`QFormat`] is the pair `(width, frac)`; the integer range is the two's-complement
//! range of `width` bits, so `width` includes the sign bit and `frac` counts fraction bits.
//!
//! ## Why the guard bit is mandatory
//!
//! The folding stage computes `u_L[q] = -c^R[q] - d[q]` and `u_U[q] = a[q] - b^R[q]`.
//! Each output is a **sum of two** input words, so it may need one extra magnitude bit.
//! The validity constraint of eq. 3, `Win >= Wu + 1`, is exactly this guard bit. A
//! configuration without it can overflow the folded word even though every input sample is
//! representable; [`crate::pipeline::DctIvKit`] therefore refuses it up front and also
//! counts any fold overflow at run time as a defence in depth.

use std::fmt;

/// A fixed-point format: `width` total bits (including sign) and `frac` fractional bits.
///
/// Invariants enforced by [`QFormat::validate`]: `2 <= width <= 63` and `frac < width`.
/// `width <= 63` keeps every value exactly representable in an [`i64`] and lets the
/// pipeline use `i128` for the single multiply of the post stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QFormat {
    /// Total word width in bits, sign bit included.
    pub width: u8,
    /// Number of fractional (least-significant) bits.
    pub frac: u8,
}

impl QFormat {
    /// Builds a format without validation; use [`QFormat::validate`] at configuration time.
    pub const fn new(width: u8, frac: u8) -> Self {
        Self { width, frac }
    }

    /// Checks the format invariants.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.width < 2 {
            return Err("width must be at least 2 bits (one sign bit, one magnitude bit)");
        }
        if self.width > 63 {
            return Err("width must not exceed 63 bits");
        }
        if self.frac >= self.width {
            return Err("fractional bits must be fewer than the total width");
        }
        Ok(())
    }

    /// Number of integer bits, sign excluded: `width - frac - 1`.
    pub fn int_bits(&self) -> u32 {
        (self.width as u32).saturating_sub(self.frac as u32 + 1)
    }

    /// Largest representable raw code.
    pub fn max_raw(&self) -> i64 {
        (1i64 << (self.width - 1)) - 1
    }

    /// Smallest representable raw code.
    pub fn min_raw(&self) -> i64 {
        -(1i64 << (self.width - 1))
    }

    /// Value of one least-significant bit, i.e. the quantization step.
    pub fn lsb(&self) -> f64 {
        2f64.powi(-(self.frac as i32))
    }

    /// Relative precision of the format: the representable resolution around unity.
    ///
    /// This is the `eps` used to derive `tau_dtype` (see [`crate::numerics::tau_dtype`]):
    /// the smallest non-zero difference between two words of this format near magnitude 1.
    pub fn eps(&self) -> f64 {
        self.lsb()
    }

    /// Full-scale magnitude (`2^int_bits`).
    pub fn full_scale(&self) -> f64 {
        2f64.powi(self.int_bits() as i32)
    }

    /// Clamps a raw code into this format's range.
    pub fn clamp_raw(&self, raw: i64) -> i64 {
        raw.clamp(self.min_raw(), self.max_raw())
    }

    /// True when `raw` is representable.
    pub fn contains(&self, raw: i64) -> bool {
        (self.min_raw()..=self.max_raw()).contains(&raw)
    }

    /// Requantizes a raw code from `self` to `to`, rounding half away from zero and
    /// saturating on overflow.
    pub fn requantize(&self, raw: i64, to: QFormat) -> i64 {
        let shift = to.frac as i32 - self.frac as i32;
        let scaled = if shift >= 0 {
            to_saturating_i128(raw as i128, shift as u32)
        } else {
            round_shift_div(raw as i128, (-shift) as u32)
        };
        saturate_i128(scaled, to)
    }
}

impl fmt::Display for QFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q{}.{} ({}-bit)", self.int_bits(), self.frac, self.width)
    }
}

/// Shifts left with saturation on magnitude overflow.
///
/// `i128::checked_shl` only bounds the shift count, not the value, so overflow would silently
/// wrap. This helper compares against `i128::MAX >> shift` and `i128::MIN >> shift` first.
fn to_saturating_i128(v: i128, shift: u32) -> i128 {
    if v == 0 || shift == 0 {
        return v;
    }
    if shift >= 127 {
        return if v < 0 { i128::MIN } else { i128::MAX };
    }
    let hi = i128::MAX >> shift;
    let lo = i128::MIN >> shift;
    if v > hi {
        i128::MAX
    } else if v < lo {
        i128::MIN
    } else {
        v << shift
    }
}

/// Rounds `v / 2^shift` half away from zero.
fn round_shift_div(v: i128, shift: u32) -> i128 {
    if shift == 0 {
        return v;
    }
    if shift > 120 {
        return if v < 0 { -1 } else { 0 };
    }
    let half = 1i128 << (shift - 1);
    if v >= 0 {
        (v + half) >> shift
    } else if v == i128::MIN {
        // No exact rounding is representable here; a plain arithmetic shift is monotone.
        v >> shift
    } else {
        -((-v + half) >> shift)
    }
}

/// Computes `round(raw * num / 2^den_shift)` without clamping.
///
/// The `i128` accumulator leaves room for the worst case the kit allows: a 63-bit word times
/// a `2^32` multiplier stays below `2^95`.
pub fn round_mul_shift(raw: i64, num: i128, den_shift: u32) -> i128 {
    round_shift_div(raw as i128 * num, den_shift)
}

/// Applies a fractional-point move to an unclamped `i128` accumulator, rounding half away
/// from zero. Moving right (fewer fractional bits) rounds; moving left is exact.
pub fn shift_frac(v: i128, from_frac: u8, to_frac: u8) -> i128 {
    let shift = to_frac as i32 - from_frac as i32;
    if shift >= 0 {
        to_saturating_i128(v, shift as u32)
    } else {
        round_shift_div(v, (-shift) as u32)
    }
}

/// Saturates an `i128` accumulator into a format's range.
pub fn saturate_i128(v: i128, fmt: QFormat) -> i64 {
    if v > fmt.max_raw() as i128 {
        fmt.max_raw()
    } else if v < fmt.min_raw() as i128 {
        fmt.min_raw()
    } else {
        v as i64
    }
}

/// A fixed-point word: a raw two's-complement integer plus the format that interprets it.
///
/// Arithmetic is saturating. Addition of two words of the same format is exact unless it
/// saturates; that is why the kit allocates one guard bit for the folding stage instead of
/// relying on saturation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q {
    /// Interpretation of [`Q::raw`].
    pub format: QFormat,
    raw: i64,
}

impl Q {
    /// Wraps a raw code, saturating it into the format's range.
    pub fn from_raw(format: QFormat, raw: i64) -> Self {
        Self {
            format,
            raw: format.clamp_raw(raw),
        }
    }

    /// Wraps a raw code that is already known to be representable.
    ///
    /// Returns `None` rather than saturating, so that callers can detect the loss of the
    /// guard bit explicitly (used by the folding stage's overflow counter).
    pub fn try_from_raw(format: QFormat, raw: i64) -> Option<Self> {
        if format.contains(raw) {
            Some(Self { format, raw })
        } else {
            None
        }
    }

    /// Quantizes a real value, rounding half away from zero, saturating at the format bounds.
    pub fn from_f64(format: QFormat, x: f64) -> Self {
        let scale = 2f64.powi(format.frac as i32);
        let scaled = x * scale;
        let raw = if scaled.is_nan() {
            0
        } else if scaled >= format.max_raw() as f64 {
            format.max_raw()
        } else if scaled <= format.min_raw() as f64 {
            format.min_raw()
        } else if scaled >= 0.0 {
            (scaled + 0.5).floor() as i64
        } else {
            (scaled - 0.5).ceil() as i64
        };
        Self { format, raw }
    }

    /// Raw integer code.
    pub fn raw(&self) -> i64 {
        self.raw
    }

    /// Real value of the word.
    pub fn to_f64(self) -> f64 {
        self.raw as f64 * 2f64.powi(-(self.format.frac as i32))
    }

    /// Zero word of a format.
    pub fn zero(format: QFormat) -> Self {
        Self { format, raw: 0 }
    }

    /// Rescales the fractional precision, rounding and saturating.
    pub fn requantize(self, out: QFormat) -> Self {
        Self {
            format: out,
            raw: self.format.requantize(self.raw, out),
        }
    }

    /// Headroom in bits between this word's magnitude and the format's maximum.
    ///
    /// Used by diagnostics: `0` means the word sits at full scale and the next addition
    /// needs the guard bit.
    pub fn headroom_bits(self) -> u32 {
        let mag = self.raw.unsigned_abs();
        if mag == 0 {
            return self.format.width as u32;
        }
        let used = 64 - mag.leading_zeros();
        (self.format.width as u32).saturating_sub(used)
    }
}

impl std::ops::Add for Q {
    type Output = Q;

    /// Saturating addition of two words of the same format.
    fn add(self, rhs: Q) -> Q {
        assert_eq!(
            self.format, rhs.format,
            "fixed-point addition across formats"
        );
        Q {
            format: self.format,
            raw: saturate_i128(self.raw as i128 + rhs.raw as i128, self.format),
        }
    }
}

impl std::ops::Sub for Q {
    type Output = Q;

    /// Saturating subtraction of two words of the same format.
    fn sub(self, rhs: Q) -> Q {
        assert_eq!(
            self.format, rhs.format,
            "fixed-point subtraction across formats"
        );
        Q {
            format: self.format,
            raw: saturate_i128(self.raw as i128 - rhs.raw as i128, self.format),
        }
    }
}

impl std::ops::Neg for Q {
    type Output = Q;

    /// Saturating negation.
    fn neg(self) -> Q {
        Q {
            format: self.format,
            raw: saturate_i128(-(self.raw as i128), self.format),
        }
    }
}

impl std::ops::Mul for Q {
    type Output = Q;

    /// Fixed-point multiply: `(self * rhs) >> frac`, rounded half away from zero.
    fn mul(self, rhs: Q) -> Q {
        assert_eq!(
            self.format, rhs.format,
            "fixed-point multiplication across formats"
        );
        let prod = self.raw as i128 * rhs.raw as i128;
        let shifted = round_shift_div(prod, self.format.frac as u32);
        Q {
            format: self.format,
            raw: saturate_i128(shifted, self.format),
        }
    }
}

impl Q {
    /// Multiplies by an exact real scalar with a rational multiplier `num / 2^den_shift`,
    /// rounding half away from zero and saturating.
    ///
    /// This is the post-stage operation `X = (sigma/alpha) * Y` evaluated without ever
    /// leaving integer arithmetic: the multiplier is materialised as an `i128` numerator
    /// with `den = 2^shift`, which is exact for the powers of two a codec uses.
    pub fn scale_rational(self, num: i128, den_shift: u32, out: QFormat) -> Self {
        let shifted = round_mul_shift(self.raw, num, den_shift);
        let raw = out.requantize_from_i128(shifted, self.format);
        Self { format: out, raw }
    }
}

impl QFormat {
    fn requantize_from_i128(&self, v: i128, from: QFormat) -> i64 {
        let shift = self.frac as i32 - from.frac as i32;
        let scaled = if shift >= 0 {
            to_saturating_i128(v, shift as u32)
        } else {
            round_shift_div(v, (-shift) as u32)
        };
        saturate_i128(scaled, *self)
    }
}

impl fmt::Display for Q {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}/{}]", self.to_f64(), self.raw, self.format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_geometry() {
        let f = QFormat::new(16, 14); // Q1.14
        assert_eq!(f.int_bits(), 1);
        assert_eq!(f.max_raw(), 32767);
        assert_eq!(f.min_raw(), -32768);
        assert!((f.lsb() - 2f64.powi(-14)).abs() < 1e-20);
        assert_eq!(f.validate(), Ok(()));
        assert!(QFormat::new(1, 0).validate().is_err());
        assert!(QFormat::new(16, 16).validate().is_err());
        assert!(QFormat::new(64, 10).validate().is_err());
    }

    #[test]
    fn round_trip_and_rounding() {
        let f = QFormat::new(16, 14);
        for x in [-1.5, -0.25, -0.0, 0.0, 0.25, 0.1, 1.0] {
            let q = Q::from_f64(f, x);
            assert!((q.to_f64() - x).abs() <= f.lsb() * 0.5 + 1e-15);
        }
        // half away from zero
        assert_eq!(Q::from_f64(f, 0.5 * f.lsb()).raw(), 1);
        assert_eq!(Q::from_f64(f, -0.5 * f.lsb()).raw(), -1);
        // saturation
        assert_eq!(Q::from_f64(f, 10.0).raw(), f.max_raw());
        assert_eq!(Q::from_f64(f, -10.0).raw(), f.min_raw());
        assert_eq!(Q::from_f64(f, f64::NAN).raw(), 0);
    }

    #[test]
    fn guard_bit_is_observable() {
        let wu = QFormat::new(8, 6); // Q1.6, range [-2.0, 1.984375]
        let c = Q::from_f64(wu, -1.0); // raw -64
        let d = Q::from_f64(wu, -1.0); // raw -64
                                       // u_L = -c^R - d = 1 + 1 = 2, which is *not* representable in Wu (max 1.984375).
        let folded = -c.raw() - d.raw();
        assert_eq!(folded, 128);
        assert!(!wu.contains(folded));
        // ... but it is representable with the mandatory guard bit, Win >= Wu + 1.
        let win = QFormat::new(9, 6);
        assert!(win.contains(folded));
        assert!((Q::from_raw(win, folded).to_f64() - 2.0).abs() < 1e-9);
        // Saturating into Wu would silently corrupt the block: that is the failure mode the
        // validity constraint exists to prevent.
        assert_eq!(Q::from_raw(wu, folded).raw(), wu.max_raw());
    }

    #[test]
    fn requantize_and_scale() {
        let a = QFormat::new(16, 14);
        let b = QFormat::new(20, 16);
        let q = Q::from_f64(a, 0.5);
        let r = q.requantize(b);
        assert!((r.to_f64() - 0.5).abs() < 1e-5);
        // X = 2 * q  (num = 2, den_shift = 0)
        let s = q.scale_rational(2, 0, b);
        assert!((s.to_f64() - 1.0).abs() < 1e-5);
        // multiplication saturates rather than wrapping
        let big = Q::from_f64(a, 1.9);
        assert_eq!((big * big).raw(), a.max_raw());
    }

    #[test]
    fn headroom() {
        let f = QFormat::new(16, 14);
        assert_eq!(Q::zero(f).headroom_bits(), 16);
        assert_eq!(Q::from_f64(f, 1.0).headroom_bits(), 1);
    }
}
