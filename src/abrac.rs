//! Adaptive binary range coder (ABRAC), per ISO/IEC 18181-1 committee
//! draft (2019-08-05) Annex D.7. ABRAC is the binary building block from
//! which the MABEGABRAC context model used by the Modular sub-bitstream
//! is constructed.
//!
//! Bit layout (D.7):
//!
//! * State: two unsigned 32-bit integers `range` and `low`.
//! * Initial state: `range = 1 << 24`; `low` is filled from three byte
//!   reads (`u(8) << 16 | u(8) << 8 | u(8)`).
//! * "Chances" are 12-bit unsigned integers in `[1, 4095]`, denominator
//!   `4096`. They represent `Pr(bit == 1)` (per the draft text).
//! * `Normalize` reads one byte at a time when `range <= 1 << 16`.
//!
//! The ABRAC stream is byte-oriented. The `BitReader` underlying the
//! containing decoder must be byte-aligned (`pu0()`) at the position where
//! the ABRAC stream begins; from that point the ABRAC reader consumes
//! bytes directly.
//!
//! `get_adaptive_bit(ac)` is the chance-adaptive variant: after reading a
//! bit, the chance `ac` is updated by the documented exponential-moving-
//! average rule (`ac += (4096 - ac) >> 5` on a `1` bit, `ac -= ac >> 5`
//! on a `0` bit).

use crate::error::{JxlError as Error, Result};

/// Decoder state for the base ABRAC range coder (D.7).
///
/// Constructed from a byte slice positioned at the first byte of the
/// ABRAC stream; the caller is responsible for byte-aligning the
/// underlying bitstream first.
#[derive(Debug)]
pub struct Abrac<'a> {
    bytes: &'a [u8],
    pos: usize,
    range: u32,
    low: u32,
}

impl<'a> Abrac<'a> {
    /// Initialise the range coder by reading 3 bytes of the initial
    /// `low` register, as specified in D.7.
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 3 {
            return Err(Error::InvalidData(
                "JXL ABRAC: stream too short for initial state".into(),
            ));
        }
        let low = ((bytes[0] as u32) << 16) | ((bytes[1] as u32) << 8) | (bytes[2] as u32);
        Ok(Self {
            bytes,
            pos: 3,
            range: 1 << 24,
            low,
        })
    }

    /// Number of bytes already consumed (including the 3 initial bytes).
    pub fn bytes_consumed(&self) -> usize {
        self.pos
    }

    fn next_byte(&mut self) -> Result<u32> {
        if self.pos >= self.bytes.len() {
            // The spec is silent on EOF beyond the encoded stream; in
            // practice readers fall through to a zero pad. We surface
            // this as an explicit error so misuse is caught early.
            return Err(Error::InvalidData(
                "JXL ABRAC: unexpected end of stream during normalisation".into(),
            ));
        }
        let b = self.bytes[self.pos] as u32;
        self.pos += 1;
        Ok(b)
    }

    fn normalise(&mut self) -> Result<()> {
        while self.range <= (1u32 << 16) {
            self.low = (self.low << 8) | self.next_byte()?;
            self.range <<= 8;
        }
        Ok(())
    }

    /// `get_bit(c)` per D.7. `chance` is in `[1, 4095]`, representing the
    /// probability that the bit is `1`, with denominator `4096`.
    ///
    /// Returns the next decoded bit (0 or 1).
    pub fn get_bit(&mut self, chance: u32) -> Result<u32> {
        debug_assert!((1..=4095).contains(&chance));
        // Carry the multiply through u64 so the (range * chance) product
        // doesn't wrap to zero when `range` is at or near `1 << 24` (the
        // initial range value). The shift then drops the >>12 we need to
        // recover the chance fraction, leaving `nc` safely back in u32.
        let nc = (((self.range as u64) * (chance as u64)) >> 12) as u32;
        let bit;
        if self.low >= self.range - nc {
            bit = 1;
            self.low -= self.range - nc;
            self.range = nc;
        } else {
            bit = 0;
            self.range -= nc;
        }
        self.normalise()?;
        Ok(bit)
    }

    /// `get_adaptive_bit(ac)` per D.7. `ac` is mutated in place
    /// according to the documented exponential-moving-average rule.
    pub fn get_adaptive_bit(&mut self, ac: &mut u32) -> Result<u32> {
        let bit = self.get_bit(*ac)?;
        if bit == 1 {
            *ac = ac.saturating_add((4096 - *ac) >> 5);
        } else {
            *ac = ac.saturating_sub(*ac >> 5);
        }
        Ok(bit)
    }
}

#[cfg(test)]
pub(crate) mod tests_enc {
    /// Encoder counterpart to `Abrac::get_bit`, used to build test
    /// fixtures end-to-end. This implements a textbook range encoder
    /// matching the decoder math in D.7. It is not part of the public
    /// API; the crate is decoder-only.
    pub(crate) struct AbracEncoder {
        out: Vec<u8>,
        low: u64,
        range: u64,
        // count of pending 0xff bytes for carry handling
        cache: u8,
        cache_size: u32,
        first: bool,
    }

    impl AbracEncoder {
        pub(crate) fn new() -> Self {
            Self {
                out: Vec::new(),
                low: 0,
                range: 1u64 << 24,
                cache: 0,
                cache_size: 1, // first byte is held back for carry
                first: true,
            }
        }

        pub(crate) fn put_bit(&mut self, bit: u32, chance: u32) {
            // Mirror the decoder's `nc = (range * chance) >> 12` exactly.
            // The intermediate product can exceed u32 (e.g. range = 1<<24,
            // chance = 1024 gives 1<<34) so do the math in u64.
            let nc = (self.range * (chance as u64)) >> 12;
            if bit == 1 {
                self.low += self.range - nc;
                self.range = nc;
            } else {
                self.range -= nc;
            }
            while self.range <= (1u64 << 16) {
                self.shift_low();
                self.range <<= 8;
            }
        }

        pub(crate) fn put_bit_adaptive(&mut self, bit: u32, ac: &mut u32) {
            self.put_bit(bit, *ac);
            if bit == 1 {
                *ac = ac.saturating_add((4096 - *ac) >> 5);
            } else {
                *ac = ac.saturating_sub(*ac >> 5);
            }
        }

        fn shift_low(&mut self) {
            // emit the byte that just left the top of `low` (with carry
            // propagation through cache_size pending 0xff bytes).
            if self.low < 0xff_0000 || self.low >= 0x100_0000 {
                let carry = (self.low >> 24) as u8;
                if !self.first {
                    self.out.push(self.cache.wrapping_add(carry));
                } else {
                    self.first = false;
                }
                for _ in 0..self.cache_size.saturating_sub(1) {
                    self.out.push(0xffu8.wrapping_add(carry));
                }
                self.cache = ((self.low >> 16) & 0xff) as u8;
                self.cache_size = 1;
            } else {
                self.cache_size += 1;
            }
            self.low = (self.low & 0xffff) << 8;
        }

        pub(crate) fn finish(mut self) -> Vec<u8> {
            // flush remaining bits.
            for _ in 0..3 {
                self.shift_low();
            }
            // emit final cache byte.
            if !self.first {
                self.out.push(self.cache);
            }
            for _ in 0..self.cache_size.saturating_sub(1) {
                self.out.push(0xff);
            }
            self.out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tests_enc::AbracEncoder;
    use super::*;

    #[test]
    fn round_trip_single_bits_static_chance() {
        // Encode a simple sequence with chance=2048 (50/50) and decode it.
        let bits = [1u32, 0, 1, 1, 0, 0, 1, 0];
        let mut enc = AbracEncoder::new();
        for &b in &bits {
            enc.put_bit(b, 2048);
        }
        let stream = enc.finish();
        let mut dec = Abrac::new(&stream).unwrap();
        let mut decoded = Vec::new();
        for _ in 0..bits.len() {
            decoded.push(dec.get_bit(2048).unwrap());
        }
        assert_eq!(decoded, bits);
    }

    #[test]
    fn round_trip_skewed_chance() {
        // chance=4000 → strongly biased to '1'. Make sure both 0 and 1
        // decode correctly even when one has tiny probability mass.
        let bits = [1u32, 1, 1, 0, 1, 1, 0, 1];
        let chance = 4000;
        let mut enc = AbracEncoder::new();
        for &b in &bits {
            enc.put_bit(b, chance);
        }
        let stream = enc.finish();
        let mut dec = Abrac::new(&stream).unwrap();
        let mut decoded = Vec::new();
        for _ in 0..bits.len() {
            decoded.push(dec.get_bit(chance).unwrap());
        }
        assert_eq!(decoded, bits);
    }

    #[test]
    fn round_trip_adaptive() {
        // Encoder must use the same chance trajectory as the decoder for
        // a clean round trip.
        let bits = [1u32, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1];
        let mut enc = AbracEncoder::new();
        let mut enc_ac: u32 = 2048;
        for &b in &bits {
            enc.put_bit(b, enc_ac);
            if b == 1 {
                enc_ac += (4096 - enc_ac) >> 5;
            } else {
                enc_ac -= enc_ac >> 5;
            }
        }
        let stream = enc.finish();
        let mut dec = Abrac::new(&stream).unwrap();
        let mut dec_ac: u32 = 2048;
        let mut decoded = Vec::new();
        for _ in 0..bits.len() {
            decoded.push(dec.get_adaptive_bit(&mut dec_ac).unwrap());
        }
        assert_eq!(decoded, bits);
        assert_eq!(dec_ac, enc_ac);
    }

    #[test]
    fn rejects_truncated_initial_state() {
        let stream = [0u8; 2];
        assert!(Abrac::new(&stream).is_err());
    }

    #[test]
    fn rejects_unexpected_eof() {
        // Only 3 bytes → just enough for `low`, but `normalise()` will
        // fail when the first decode forces a refill.
        let stream = [0u8, 0, 0];
        let mut dec = Abrac::new(&stream).unwrap();
        // chance=2048 splits range exactly; with low=0 and range=1<<24
        // the first bit decodes as 0, then range = 1<<23 (still > 1<<16,
        // no normalise yet). Iterate until normalise triggers.
        let mut err_seen = false;
        for _ in 0..40 {
            if dec.get_bit(2048).is_err() {
                err_seen = true;
                break;
            }
        }
        assert!(err_seen, "expected normalise EOF after enough bits drained");
    }
}
