//! Bounded-Exp-Golomb ABRAC (BEGABRAC), per ISO/IEC 18181-1 committee
//! draft (2019-08-05) Annex D.7.1.
//!
//! BEGABRAC decodes a signed integer in a known `[lower, upper]` range
//! (with `lower <= 0 <= upper`) using a small set of adaptive bit
//! contexts maintained by the underlying [`Abrac`] coder. Each
//! BEGABRAC instance is parameterised by the maximum bit-depth `N` of
//! the values to be decoded (N = `ceil(log2(max(|lower|, |upper|))) + 1`
//! is sufficient).
//!
//! Per the spec the contexts are:
//!
//! * `ac_zero` — probability that the current value is exactly 0.
//! * `ac_sign` — sign bit when both `lower < 0` and `upper > 0`.
//! * `ac_exponent[0 .. N-1]` — unary-coded exponent of the magnitude.
//! * `ac_mantissa[0 .. N-1]` — explicit mantissa bits below the leading
//!   bit, decoded from MSB to LSB.
//!
//! The exponent contexts are seeded as a function of `init_ac_zero`
//! according to the `init_begabrac` procedure in D.7.1; the mantissa
//! contexts are seeded uniformly at `1024` (the spec value).
//!
//! See [`Begabrac::decode`] for the integer-decode procedure proper.

use crate::error::{JxlError as Error, Result};

use crate::abrac::Abrac;

/// A BEGABRAC integer-decoding context, owning its set of `ac_zero`,
/// `ac_sign`, `ac_exponent[..]` and `ac_mantissa[..]` adaptive chances.
#[derive(Debug, Clone)]
pub struct Begabrac {
    pub ac_zero: u32,
    pub ac_sign: u32,
    pub ac_exponent: Vec<u32>,
    pub ac_mantissa: Vec<u32>,
}

impl Begabrac {
    /// Build a fresh BEGABRAC with the documented `init_begabrac`
    /// initialisation.
    ///
    /// `n` is the maximum bit depth of values this BEGABRAC will decode.
    /// `init_ac_zero` is the seed `ac_zero` value (12-bit chance, in
    /// `[1, 4095]`).
    pub fn new(n: u32, init_ac_zero: u32) -> Self {
        let n = n.max(1) as usize;
        let mut ac_exponent = vec![0u32; n - 1];
        let mut c: u32 = 4096 - init_ac_zero;
        for slot in ac_exponent.iter_mut() {
            c = c.clamp(256, 3840);
            *slot = 4096 - c;
            c = (c.wrapping_mul(c) + 2048) >> 12;
        }
        let ac_mantissa = vec![1024u32; n];
        Self {
            ac_zero: init_ac_zero,
            ac_sign: 2048,
            ac_exponent,
            ac_mantissa,
        }
    }

    /// Decode one signed integer in `[lower, upper]` from `coder`,
    /// using and updating this context's adaptive chances.
    pub fn decode(&mut self, coder: &mut Abrac<'_>, lower: i32, upper: i32) -> Result<i32> {
        if !(lower <= 0 && 0 <= upper) {
            return Err(Error::InvalidData(
                "JXL BEGABRAC: range must contain zero".into(),
            ));
        }
        if coder.get_adaptive_bit(&mut self.ac_zero)? == 1 {
            return Ok(0);
        }
        // Sign decode.
        let sign: i32;
        if lower < 0 {
            if upper > 0 {
                sign = if coder.get_adaptive_bit(&mut self.ac_sign)? == 1 {
                    1
                } else {
                    -1
                };
            } else {
                sign = -1;
            }
        } else {
            sign = 1;
        }
        let max = if sign == 1 { upper } else { -lower } as i64;
        if max <= 0 {
            // Defensive: range was effectively zero on this side.
            return Err(Error::InvalidData(
                "JXL BEGABRAC: empty signed range on chosen side".into(),
            ));
        }
        let max_log2 = ilog2(max as u32);
        // Decode unary exponent.
        let mut exp: usize = 0;
        while exp < max_log2 as usize {
            if exp >= self.ac_exponent.len() {
                return Err(Error::InvalidData(
                    "JXL BEGABRAC: exponent context underflow".into(),
                ));
            }
            if coder.get_adaptive_bit(&mut self.ac_exponent[exp])? == 1 {
                break;
            }
            exp += 1;
        }
        // Decode mantissa: assemble v = 1 << exp, then for each lower bit
        // (high to low), conditionally set it if the candidate stays in
        // `[1, max]`.
        let mut v: i64 = 1i64 << exp as u32;
        if exp > 0 {
            for i in (0..exp).rev() {
                let one = v | (1i64 << i as u32);
                if one > max {
                    continue;
                }
                if i >= self.ac_mantissa.len() {
                    return Err(Error::InvalidData(
                        "JXL BEGABRAC: mantissa context underflow".into(),
                    ));
                }
                if coder.get_adaptive_bit(&mut self.ac_mantissa[i])? == 1 {
                    v = one;
                }
            }
        }
        Ok((v as i32) * sign)
    }
}

/// Integer base-2 logarithm, floor (matches the spec's `ilog2`).
/// `ilog2(0)` is treated as `0` for safety; callers shouldn't pass it.
fn ilog2(x: u32) -> u32 {
    if x == 0 {
        0
    } else {
        31 - x.leading_zeros()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Encoder-side BEGABRAC, used to build deterministic test fixtures.
    /// Mirrors the decoder's traversal step-for-step so the round trip
    /// is exact. NOT part of the public API.
    pub(crate) fn encode_one(
        enc: &mut crate::abrac::tests_enc::AbracEncoder,
        ctx: &mut Begabrac,
        value: i32,
        lower: i32,
        upper: i32,
    ) {
        // Mirrors `Begabrac::decode` step-for-step.
        let zero_bit = if value == 0 { 1 } else { 0 };
        enc.put_bit_adaptive(zero_bit, &mut ctx.ac_zero);
        if value == 0 {
            return;
        }
        let abs = value.unsigned_abs();
        let sign;
        if lower < 0 && upper > 0 {
            sign = if value > 0 { 1 } else { -1 };
            let bit = if sign == 1 { 1 } else { 0 };
            enc.put_bit_adaptive(bit, &mut ctx.ac_sign);
        } else {
            sign = if upper > 0 { 1 } else { -1 };
        }
        let _ = sign;
        let max = if value > 0 { upper } else { -lower } as u32;
        let max_log2 = if max == 0 {
            0
        } else {
            31 - max.leading_zeros()
        };
        let exp = if abs == 0 {
            0
        } else {
            31 - abs.leading_zeros()
        };
        // Unary exponent.
        let mut e = 0u32;
        while e < exp {
            enc.put_bit_adaptive(0, &mut ctx.ac_exponent[e as usize]);
            e += 1;
        }
        if exp < max_log2 {
            enc.put_bit_adaptive(1, &mut ctx.ac_exponent[exp as usize]);
        }
        // Mantissa.
        let mut v: u32 = 1u32 << exp;
        if exp > 0 {
            for i in (0..exp).rev() {
                let one = v | (1u32 << i);
                if one > max {
                    continue;
                }
                let bit = if (abs & (1u32 << i)) != 0 { 1 } else { 0 };
                if bit == 1 {
                    v = one;
                }
                enc.put_bit_adaptive(bit, &mut ctx.ac_mantissa[i as usize]);
            }
        }
    }

    #[test]
    fn round_trip_signed_range_basic() {
        use crate::abrac::tests_enc::AbracEncoder;
        let lower = -50;
        let upper = 50;
        let values = [0i32, 1, -1, 7, -7, 49, -50, 12, -12, 0, 0, 31];
        // Encode all.
        let mut enc = AbracEncoder::new();
        let mut ec = Begabrac::new(7, 1024);
        for &v in &values {
            encode_one(&mut enc, &mut ec, v, lower, upper);
        }
        let stream = enc.finish();
        // Decode.
        let mut dec = Abrac::new(&stream).unwrap();
        let mut dc = Begabrac::new(7, 1024);
        let mut decoded = Vec::new();
        for _ in 0..values.len() {
            decoded.push(dc.decode(&mut dec, lower, upper).unwrap());
        }
        assert_eq!(decoded, values);
        // Both contexts should have evolved identically.
        assert_eq!(ec.ac_zero, dc.ac_zero);
        assert_eq!(ec.ac_exponent, dc.ac_exponent);
    }

    #[test]
    fn round_trip_unsigned_range() {
        use crate::abrac::tests_enc::AbracEncoder;
        let lower = 0;
        let upper = 255;
        let values = [0i32, 1, 2, 100, 255, 0, 17, 200];
        let mut enc = AbracEncoder::new();
        let mut ec = Begabrac::new(9, 2048);
        for &v in &values {
            encode_one(&mut enc, &mut ec, v, lower, upper);
        }
        let stream = enc.finish();
        let mut dec = Abrac::new(&stream).unwrap();
        let mut dc = Begabrac::new(9, 2048);
        let mut decoded = Vec::new();
        for _ in 0..values.len() {
            decoded.push(dc.decode(&mut dec, lower, upper).unwrap());
        }
        assert_eq!(decoded, values);
    }

    #[test]
    fn rejects_range_not_containing_zero() {
        let stream = [0u8; 16];
        let mut dec = Abrac::new(&stream).unwrap();
        let mut ctx = Begabrac::new(5, 1024);
        // Range [1, 10] does not contain zero; spec assumption violated.
        assert!(ctx.decode(&mut dec, 1, 10).is_err());
    }

    #[test]
    fn ilog2_basic() {
        assert_eq!(ilog2(1), 0);
        assert_eq!(ilog2(2), 1);
        assert_eq!(ilog2(3), 1);
        assert_eq!(ilog2(4), 2);
        assert_eq!(ilog2(255), 7);
        assert_eq!(ilog2(256), 8);
    }
}
