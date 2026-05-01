//! HybridUintConfig — FDIS Annex D.3.7 (Listing D.7, p. 67).
//!
//! Each clustered distribution carries one of these so that
//! `DecodeHybridVarLenUint` (D.3.6) can convert a token to an unsigned
//! integer with a shared MSB / LSB / extra-bits split.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Decoded `HybridUintConfig` (D.3.7).
///
/// `split = 1 << split_exponent`. Tokens below `split` are returned as
/// the integer value directly; tokens at or above `split` carry a
/// dynamic number of MSB-significant bits, LSB bits, and extra-bits
/// payload as defined in `ReadUint` (D.3.6 Listing D.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridUintConfig {
    pub split_exponent: u32,
    pub msb_in_token: u32,
    pub lsb_in_token: u32,
    pub split: u32,
}

/// Maximum permitted `log_alphabet_size`. FDIS D.3.1 caps
/// `use_prefix_code == 0` at 15 and `use_prefix_code == 1` at
/// `5 + u(2)` i.e. at most 8; this constant is the ceiling we accept
/// for either path.
pub const LOG_ALPHABET_SIZE_MAX: u32 = 15;

impl HybridUintConfig {
    /// FDIS Listing D.7. `log_alphabet_size` is capped at
    /// [`LOG_ALPHABET_SIZE_MAX`] before any `u(n)` reads to make the
    /// implicit `nbits` ceil(log2(...)) safe.
    pub fn read(br: &mut BitReader<'_>, log_alphabet_size: u32) -> Result<Self> {
        if log_alphabet_size > LOG_ALPHABET_SIZE_MAX {
            return Err(Error::InvalidData(
                "JXL HybridUintConfig: log_alphabet_size out of range".into(),
            ));
        }

        // ceil(log2(log_alphabet_size + 1)) is the width of the
        // `split_exponent` field. log_alphabet_size in [0, 15] gives a
        // width in [0, 4].
        let split_exp_bits = ceil_log2(log_alphabet_size + 1);
        let split_exponent = br.read_bits(split_exp_bits)?;
        if split_exponent > log_alphabet_size {
            return Err(Error::InvalidData(
                "JXL HybridUintConfig: split_exponent > log_alphabet_size".into(),
            ));
        }

        let mut msb_in_token = 0u32;
        let mut lsb_in_token = 0u32;
        if split_exponent != log_alphabet_size {
            let nbits = ceil_log2(split_exponent + 1);
            msb_in_token = br.read_bits(nbits)?;
            if msb_in_token > split_exponent {
                return Err(Error::InvalidData(
                    "JXL HybridUintConfig: msb_in_token > split_exponent".into(),
                ));
            }
            let nbits = ceil_log2(split_exponent - msb_in_token + 1);
            lsb_in_token = br.read_bits(nbits)?;
        }
        if msb_in_token + lsb_in_token > split_exponent {
            return Err(Error::InvalidData(
                "JXL HybridUintConfig: msb+lsb > split_exponent".into(),
            ));
        }
        // split_exponent <= log_alphabet_size <= 15, so 1 << split_exponent fits.
        let split = 1u32 << split_exponent;
        Ok(Self {
            split_exponent,
            msb_in_token,
            lsb_in_token,
            split,
        })
    }

    /// `ReadUint(config, token)` per FDIS D.3.6 Listing D.6.
    ///
    /// `token` is the raw symbol decoded from the entropy-coded stream;
    /// the routine returns the actual unsigned integer it represents,
    /// reading any extra `n` bits from `br`. `n` is bounded by
    /// `30 - (msb + lsb)` which itself is bounded by
    /// `30 - 0 = 30 < 32`, so the result fits in `u32` without wraparound
    /// for any sensible bitstream — but we still guard with checked math
    /// in the shift to make malicious input fail safely.
    pub fn read_uint(&self, br: &mut BitReader<'_>, token: u32) -> Result<u32> {
        if token < self.split {
            return Ok(token);
        }
        let total_in_token = self
            .msb_in_token
            .checked_add(self.lsb_in_token)
            .ok_or_else(|| Error::InvalidData("JXL ReadUint: msb+lsb overflow".into()))?;
        // n = split_exponent + ((token - split) >> total_in_token).
        // We've already verified split_exponent <= 15 and
        // msb+lsb <= split_exponent, so n stays bounded by ~31 for any
        // practically-reachable token.
        let above = token - self.split;
        let n_extra = above >> total_in_token;
        let n = self
            .split_exponent
            .checked_add(n_extra)
            .ok_or_else(|| Error::InvalidData("JXL ReadUint: n overflow".into()))?;
        if n >= 32 {
            return Err(Error::InvalidData(
                "JXL ReadUint: extra-bits count >= 32".into(),
            ));
        }

        let lsb_mask = (1u32 << self.lsb_in_token).wrapping_sub(1);
        let lsb = token & lsb_mask;
        let mut tok = token >> self.lsb_in_token;
        let msb_mask = (1u32 << self.msb_in_token).wrapping_sub(1);
        tok &= msb_mask;
        tok |= 1u32 << self.msb_in_token;

        let extra = br.read_bits(n)?;
        // (((token << n) | extra) << lsb_in_token) | lsb.
        let shifted = tok
            .checked_shl(n)
            .ok_or_else(|| Error::InvalidData("JXL ReadUint: shift overflow".into()))?;
        let combined = (shifted | extra)
            .checked_shl(self.lsb_in_token)
            .ok_or_else(|| Error::InvalidData("JXL ReadUint: lsb shift overflow".into()))?;
        Ok(combined | lsb)
    }

    /// In-tree encoder for `read_uint`'s inverse — used only by unit
    /// tests so that a round-trip can be verified without dragging in
    /// a reference encoder.
    ///
    /// Returns `(token, extra_bits, n_extra)`: the entropy-coded token
    /// the decoder will see, plus `n_extra` bits of payload to follow.
    /// Inverse of [`Self::read_uint`].
    #[cfg(test)]
    pub(super) fn encode_uint(&self, value: u32) -> (u32, u32, u32) {
        if value < self.split {
            return (value, 0, 0);
        }
        let total_in_token = self.msb_in_token + self.lsb_in_token;

        // Find the smallest `n` (>= split_exponent) such that the
        // decode formula
        //   value = ((((1 << msb) | top_msb) << n) | extra_bits) << lsb | lsb_bits
        // can represent `value`. This is `n = floor(log2(value >> lsb)) - msb`.
        let lsb_bits = value & ((1u32 << self.lsb_in_token).wrapping_sub(1));
        let v = value >> self.lsb_in_token;
        // `v` has its top bit at position `top_bit_pos`.
        let top_bit_pos = 31 - v.leading_zeros();
        // Decoder reconstructs `tok = (1 << msb) | top_msb_part`, then
        // `(tok << n)` carries the leading 1 to bit `(msb + n)`. So
        // `n = top_bit_pos - msb`.
        let n = top_bit_pos - self.msb_in_token;
        debug_assert!(n >= self.split_exponent);
        let n_above = n - self.split_exponent;
        // Bits below the leading 1 in v break into `msb_in_token` MSB
        // bits (go into the token) and `n` extra-payload bits.
        let below_leading_1 = v ^ (1u32 << top_bit_pos);
        let extra_bits = below_leading_1 & ((1u32 << n).wrapping_sub(1));
        let msb_part = below_leading_1 >> n;
        let token =
            self.split + ((n_above << total_in_token) | (msb_part << self.lsb_in_token) | lsb_bits);
        (token, extra_bits, n)
    }
}

/// `ceil(log2(x))` clamped at 0 for `x <= 1`.
fn ceil_log2(x: u32) -> u32 {
    if x <= 1 {
        0
    } else {
        32 - (x - 1).leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    #[test]
    fn ceil_log2_matches_spec() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(9), 4);
        assert_eq!(ceil_log2(16), 4);
        assert_eq!(ceil_log2(17), 5);
    }

    #[test]
    fn split_exponent_equal_to_log_alphabet_size_skips_msb_lsb() {
        // log_alphabet_size = 8 → ceil(log2(9)) = 4 bits for split_exp.
        // We pick split_exponent = 8 (binary 1000); since this equals
        // log_alphabet_size, msb/lsb are not read.
        let bytes = pack_lsb(&[(8, 4)]);
        let mut br = BitReader::new(&bytes);
        let cfg = HybridUintConfig::read(&mut br, 8).unwrap();
        assert_eq!(cfg.split_exponent, 8);
        assert_eq!(cfg.msb_in_token, 0);
        assert_eq!(cfg.lsb_in_token, 0);
        assert_eq!(cfg.split, 256);
    }

    #[test]
    fn read_uint_below_split_returns_token() {
        let cfg = HybridUintConfig {
            split_exponent: 4,
            msb_in_token: 0,
            lsb_in_token: 0,
            split: 16,
        };
        // No extra bits needed.
        let mut br = BitReader::new(&[]);
        assert_eq!(cfg.read_uint(&mut br, 0).unwrap(), 0);
        assert_eq!(cfg.read_uint(&mut br, 15).unwrap(), 15);
    }

    #[test]
    fn read_uint_round_trip_simple_config() {
        // split_exponent = 4, msb = 2, lsb = 1.
        // msb+lsb = 3 <= split_exponent = 4. split = 16.
        //
        // Below `split` (=16) the token IS the value. Above split, the
        // smallest encodable value is `((1 << msb) << split_exponent) << lsb`
        // = ((4 << 4) << 1) = 128, so we test values that cleanly
        // partition into the two regimes.
        let cfg = HybridUintConfig {
            split_exponent: 4,
            msb_in_token: 2,
            lsb_in_token: 1,
            split: 16,
        };
        // Values strictly below split: encoded as raw token.
        for value in [0u32, 1, 7, 15] {
            let mut br = BitReader::new(&[]);
            assert_eq!(cfg.read_uint(&mut br, value).unwrap(), value);
        }
        // Values at or above the smallest above-split representation.
        for value in [128u32, 129, 130, 200, 1000, 10_000, 100_000] {
            let (token, extra, n) = cfg.encode_uint(value);
            let bits_to_pack: Vec<(u32, u32)> = if n == 0 { Vec::new() } else { vec![(extra, n)] };
            let bytes = pack_lsb(&bits_to_pack);
            let mut br = BitReader::new(&bytes);
            let decoded = cfg.read_uint(&mut br, token).unwrap();
            assert_eq!(decoded, value, "round-trip failed for {value}");
        }
    }

    #[test]
    fn read_uint_round_trip_no_extras_config() {
        // split_exponent = 8, msb = 0, lsb = 0 — every above-split
        // token unambiguously identifies its value via raw extra bits.
        let cfg = HybridUintConfig {
            split_exponent: 8,
            msb_in_token: 0,
            lsb_in_token: 0,
            split: 256,
        };
        for value in [0u32, 1, 100, 255, 256, 257, 1000, 0x7FFF_FFFF] {
            let (token, extra, n) = cfg.encode_uint(value);
            let bits_to_pack: Vec<(u32, u32)> = if n == 0 { Vec::new() } else { vec![(extra, n)] };
            let bytes = pack_lsb(&bits_to_pack);
            let mut br = BitReader::new(&bytes);
            let decoded = cfg.read_uint(&mut br, token).unwrap();
            assert_eq!(decoded, value, "round-trip failed for {value}");
        }
    }

    #[test]
    fn read_uint_truncates_n_at_32() {
        // Construct a degenerate token that would require n >= 32.
        let cfg = HybridUintConfig {
            split_exponent: 0,
            msb_in_token: 0,
            lsb_in_token: 0,
            split: 1,
        };
        // token = 0xFFFFFFFF, msb+lsb = 0, so n = 0 + (token-1) >> 0 = huge.
        let mut br = BitReader::new(&[0u8; 4]);
        assert!(cfg.read_uint(&mut br, 0xFFFF_FFFF).is_err());
    }

    #[test]
    fn malicious_huge_token_rejected() {
        // Construct a config that would ask for n > 31 extra bits.
        // split_exponent = 1, msb=0, lsb=0, total_in_token=0.
        // For token = 0xFFFF_FFFF, n = 1 + (0xFFFF_FFFF - 2) = ~ 2^32.
        let cfg = HybridUintConfig {
            split_exponent: 1,
            msb_in_token: 0,
            lsb_in_token: 0,
            split: 2,
        };
        let mut br = BitReader::new(&[0u8; 4]);
        assert!(cfg.read_uint(&mut br, 0xFFFF_FFFF).is_err());
    }

    #[test]
    fn read_rejects_oversized_log_alphabet_size() {
        let mut br = BitReader::new(&[0u8; 4]);
        assert!(HybridUintConfig::read(&mut br, 16).is_err());
        assert!(HybridUintConfig::read(&mut br, 30).is_err());
    }
}
