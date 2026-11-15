//! How decoded samples are stored in a plane (D.3.5 `BitDepth`).
//!
//! Every decode path hands its frame over as byte planes in one of
//! three layouts: one byte per sample for integer depths through 8,
//! two little-endian bytes for 9 to 16, and four little-endian bytes
//! holding an IEEE binary32 value for floating-point samples
//! (`float_sample`, whatever the coded exponent / mantissa widths).
//! [`SampleStore`] is that choice, with the conversions between a
//! plane and the normalised float domain (`1.0` = the nominal maximum)
//! composition and the colour transforms work in.

use crate::error::{Error, Result};
use crate::metadata_fdis::BitDepthFdis;

/// The storage of one image's decoded samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SampleStore {
    /// Integers in `0 ..= 2^bits − 1`, `bits` in `1..=16`.
    Int { bits: u32 },
    /// Binary32 values; `1.0` is the nominal maximum, nothing is
    /// clamped.
    Float,
}

impl SampleStore {
    /// The storage for an image's `BitDepth`.
    pub(crate) fn of(depth: &BitDepthFdis) -> Result<Self> {
        if depth.float_sample {
            return Ok(Self::Float);
        }
        if depth.bits_per_sample == 0 || depth.bits_per_sample > 16 {
            return Err(Error::Unsupported(format!(
                "jxl decoder: integer bits_per_sample {} not supported (1..=16 only)",
                depth.bits_per_sample
            )));
        }
        Ok(Self::Int {
            bits: depth.bits_per_sample,
        })
    }

    /// Bytes per stored sample: 1, 2 or 4.
    pub(crate) fn bytes(self) -> usize {
        match self {
            Self::Int { bits } if bits <= 8 => 1,
            Self::Int { .. } => 2,
            Self::Float => 4,
        }
    }

    /// Whether the samples are floats.
    pub(crate) fn is_float(self) -> bool {
        self == Self::Float
    }

    /// Store the normalised sample `v` at index `i` of `data`:
    /// integers are clamped to the nominal range and rounded, floats
    /// are kept as they are.
    #[inline]
    pub(crate) fn put(self, data: &mut [u8], i: usize, v: f32) {
        match self {
            Self::Int { bits } => {
                let max = ((1u32 << bits) - 1) as f32;
                let q = (v.clamp(0.0, 1.0) * max).round() as u32;
                if bits <= 8 {
                    data[i] = q as u8;
                } else {
                    data[2 * i..2 * i + 2].copy_from_slice(&(q as u16).to_le_bytes());
                }
            }
            Self::Float => data[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes()),
        }
    }

    /// The normalised value of sample `i` of `data`.
    #[inline]
    pub(crate) fn get(self, data: &[u8], i: usize) -> f32 {
        match self {
            Self::Int { bits } => {
                let v = if bits <= 8 {
                    u32::from(data[i])
                } else {
                    u32::from(u16::from_le_bytes([data[2 * i], data[2 * i + 1]]))
                };
                v as f32 / ((1u32 << bits) - 1) as f32
            }
            Self::Float => f32::from_le_bytes([
                data[4 * i],
                data[4 * i + 1],
                data[4 * i + 2],
                data[4 * i + 3],
            ]),
        }
    }

    /// A plane of `n` zero samples.
    pub(crate) fn zeroed(self, n: usize) -> Vec<u8> {
        vec![0u8; n * self.bytes()]
    }
}

/// The value of a Modular integer sample under a floating-point
/// `BitDepth` (D.3.5): one sign bit, `exp_bits` exponent bits with a
/// bias of `2^(exp_bits − 1) − 1`, the rest mantissa, read like an
/// ISO/IEC 60559 number — subnormals with a zero exponent, infinities
/// and NaN with the highest one (the standard leaves those
/// unspecified).
#[inline]
pub(crate) fn float_from_bits(v: i32, bits: u32, exp_bits: u32) -> f32 {
    if bits == 32 && exp_bits == 8 {
        return f32::from_bits(v as u32);
    }
    let raw = if bits >= 32 {
        v as u32
    } else {
        v as u32 & ((1u32 << bits) - 1)
    };
    let mant_bits = bits - exp_bits - 1;
    let sign = (raw >> (bits - 1)) & 1;
    let exp = (raw >> mant_bits) & ((1u32 << exp_bits) - 1);
    let mant = raw & ((1u32 << mant_bits) - 1);
    let bias = (1i32 << (exp_bits - 1)) - 1;
    let frac = f64::from(mant) / (1u64 << mant_bits) as f64;
    let magnitude = if exp == (1u32 << exp_bits) - 1 {
        if mant == 0 { f64::INFINITY } else { f64::NAN }
    } else if exp == 0 {
        frac * 2f64.powi(1 - bias)
    } else {
        (1.0 + frac) * 2f64.powi(exp as i32 - bias)
    };
    (if sign == 1 { -magnitude } else { magnitude }) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_round_trip_normalised_values() {
        for (store, bytes) in [
            (SampleStore::Int { bits: 8 }, 1),
            (SampleStore::Int { bits: 5 }, 1),
            (SampleStore::Int { bits: 12 }, 2),
            (SampleStore::Int { bits: 16 }, 2),
            (SampleStore::Float, 4),
        ] {
            assert_eq!(store.bytes(), bytes);
            let mut data = store.zeroed(3);
            store.put(&mut data, 0, 0.0);
            store.put(&mut data, 1, 1.0);
            store.put(&mut data, 2, 0.5);
            assert_eq!(store.get(&data, 0), 0.0);
            assert_eq!(store.get(&data, 1), 1.0);
            assert!((store.get(&data, 2) - 0.5).abs() <= 0.5 / 31.0);
        }
        // Integers clamp, floats do not.
        let int = SampleStore::Int { bits: 8 };
        let mut d = int.zeroed(2);
        int.put(&mut d, 0, -3.0);
        int.put(&mut d, 1, 7.0);
        assert_eq!(d, [0, 255]);
        let mut d = SampleStore::Float.zeroed(1);
        SampleStore::Float.put(&mut d, 0, -3.5);
        assert_eq!(SampleStore::Float.get(&d, 0), -3.5);
    }

    #[test]
    fn half_floats_read_like_iso_60559() {
        // binary16: 1 sign, 5 exponent, 10 mantissa bits.
        let half = |bits: u16| float_from_bits(i32::from(bits), 16, 5);
        assert_eq!(half(0x0000), 0.0);
        assert_eq!(half(0x3C00), 1.0);
        assert_eq!(half(0xC000), -2.0);
        assert_eq!(half(0x3555), 0.333_251_95);
        assert_eq!(half(0x7BFF), 65504.0);
        assert_eq!(half(0x0001), 2f32.powi(-24));
        assert_eq!(half(0x0400), 2f32.powi(-14));
        assert!(half(0x7C00).is_infinite() && half(0x7E00).is_nan());
        // binary32 passes through bit for bit.
        for v in [0.0f32, 1.0, -0.15625, 1.0e-40, f32::MAX] {
            assert_eq!(float_from_bits(v.to_bits() as i32, 32, 8), v);
        }
        // A 24-bit float with 7 exponent bits: bias 63.
        let one = (63u32 << 16) as i32;
        assert_eq!(float_from_bits(one, 24, 7), 1.0);
        assert_eq!(float_from_bits(one | (1 << 15), 24, 7), 1.5);
    }
}
