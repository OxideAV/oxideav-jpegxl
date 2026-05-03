//! LSB-first bit writer used by the JPEG XL encoder.
//!
//! Mirror of [`crate::bitreader::BitReader`]: each byte is filled bit
//! 0 → bit 7, multi-bit fields are emitted least-significant bit first.
//! Round-trip with the bit reader is a hard invariant — every test in
//! the encoder modules pairs `BitWriter::write_bits(v, n)` against a
//! subsequent `BitReader::read_bits(n)` and asserts equality.

use oxideav_core::{Error, Result};

/// LSB-first bit writer.
///
/// `out` accumulates whole bytes. The current partial byte is held in
/// `out.last()` (or pushed lazily when the first bit lands at bit_pos=0).
pub struct BitWriter {
    out: Vec<u8>,
    bit_pos: u8,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            bit_pos: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            out: Vec::with_capacity(cap),
            bit_pos: 0,
        }
    }

    /// Total bits emitted so far.
    pub fn bits_written(&self) -> usize {
        if self.bit_pos == 0 {
            self.out.len() * 8
        } else {
            (self.out.len() - 1) * 8 + self.bit_pos as usize
        }
    }

    /// Write a single bit (0 or 1).
    pub fn write_bit(&mut self, bit: u32) {
        if self.bit_pos == 0 {
            self.out.push(0);
        }
        let last = self.out.len() - 1;
        self.out[last] |= ((bit & 1) as u8) << self.bit_pos;
        self.bit_pos = (self.bit_pos + 1) & 7;
    }

    /// Write the lowest `n` bits of `value`, LSB-first. `n` must be
    /// `<= 32`.
    pub fn write_bits(&mut self, value: u32, n: u32) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if n > 32 {
            return Err(Error::other("BitWriter::write_bits: n > 32 not supported"));
        }
        for i in 0..n {
            self.write_bit((value >> i) & 1);
        }
        Ok(())
    }

    /// Pad the current byte with zero bits so the next write starts on
    /// a byte boundary. No-op if already aligned. Mirror of
    /// [`crate::bitreader::BitReader::pu0`].
    pub fn pad_to_byte(&mut self) {
        while self.bit_pos != 0 {
            self.write_bit(0);
        }
    }

    /// Write JXL `U8()` per FDIS §9.2.5: 1-bit zero flag, otherwise
    /// 3-bit magnitude `n` followed by `u(n)` extra bits with implicit
    /// leading 1.
    pub fn write_u8_value(&mut self, value: u32) -> Result<()> {
        if value == 0 {
            self.write_bit(0);
            return Ok(());
        }
        if value > 256 {
            return Err(Error::other(
                "BitWriter::write_u8_value: value > 256 not representable",
            ));
        }
        self.write_bit(1);
        // Find n such that value = (1 << n) + extra, 0 <= extra < (1 << n),
        // and n in [0, 7].
        let n = if value == 1 {
            0
        } else {
            31 - (value - 1).leading_zeros()
        };
        let extra = value - (1u32 << n);
        self.write_bits(n, 3)?;
        self.write_bits(extra, n)?;
        Ok(())
    }

    /// Write a JXL `U32` field using the matching distribution from a
    /// `[U32WriteDist; 4]` table. The selector chooses the entry that
    /// can represent `value`; the encoder errors if no entry fits.
    pub fn write_u32(&mut self, dists: [U32WriteDist; 4], value: u32) -> Result<()> {
        for (sel, dist) in dists.iter().enumerate() {
            if let Some((nbits, raw)) = dist.encode(value) {
                self.write_bits(sel as u32, 2)?;
                if nbits > 0 {
                    self.write_bits(raw, nbits)?;
                }
                return Ok(());
            }
        }
        Err(Error::other(
            "BitWriter::write_u32: value not representable in any of the four distributions",
        ))
    }

    /// Write a JXL `U64()` per FDIS §9.2.3 — minimal encoding for the
    /// values the encoder uses (sel=0 for 0, sel=1 for 1..16, sel=2
    /// for 17..272, sel=3 for larger).
    pub fn write_u64(&mut self, value: u64) -> Result<()> {
        if value == 0 {
            self.write_bits(0, 2)?;
            return Ok(());
        }
        if (1..=16).contains(&value) {
            // sel=1 → BitsOffset(4, 1); raw = value - 1.
            self.write_bits(1, 2)?;
            self.write_bits((value - 1) as u32, 4)?;
            return Ok(());
        }
        if (17..=272).contains(&value) {
            // sel=2 → BitsOffset(8, 17); raw = value - 17.
            self.write_bits(2, 2)?;
            self.write_bits((value - 17) as u32, 8)?;
            return Ok(());
        }
        // sel=3: u(12) followed by 8-bit chunks gated on a 1-bit "more"
        // flag, plus optional 4-bit final chunk at shift==60.
        self.write_bits(3, 2)?;
        self.write_bits((value & 0xfff) as u32, 12)?;
        let mut v = value >> 12;
        let mut shift: u32 = 12;
        while v != 0 {
            self.write_bit(1); // more
            if shift == 60 {
                self.write_bits((v & 0xf) as u32, 4)?;
                return Ok(());
            }
            self.write_bits((v & 0xff) as u32, 8)?;
            v >>= 8;
            shift += 8;
        }
        self.write_bit(0); // stop
        Ok(())
    }

    /// Take the underlying byte buffer. After calling, the writer is
    /// effectively empty; callers must not write more bits.
    pub fn finish(self) -> Vec<u8> {
        self.out
    }

    /// Borrow the underlying byte buffer (e.g. to take its length).
    pub fn as_bytes(&self) -> &[u8] {
        &self.out
    }
}

/// Encoder-side mirror of [`crate::bitreader::U32Dist`]. Each variant
/// describes how a single distribution slot in a `read_u32` table maps
/// values to (raw, nbits) pairs.
#[derive(Copy, Clone, Debug)]
pub enum U32WriteDist {
    /// Single literal value; encodes only `value == v`.
    Val(u32),
    /// `n` raw bits; encodes any `value < (1 << n)`.
    Bits(u32),
    /// `n` raw bits + offset; encodes any `offset <= value < offset + (1 << n)`.
    BitsOffset(u32, u32),
}

impl U32WriteDist {
    fn encode(&self, value: u32) -> Option<(u32, u32)> {
        match *self {
            U32WriteDist::Val(v) => {
                if value == v {
                    Some((0, 0))
                } else {
                    None
                }
            }
            U32WriteDist::Bits(n) => {
                if (n == 32) || value < (1u32 << n) {
                    Some((n, value))
                } else {
                    None
                }
            }
            U32WriteDist::BitsOffset(n, off) => {
                if value < off {
                    return None;
                }
                let raw = value - off;
                if (n == 32) || raw < (1u32 << n) {
                    Some((n, raw))
                } else {
                    None
                }
            }
        }
    }
}

/// `PackSigned(s)` — inverse of `unpack_signed(u)` from
/// `crate::bitreader`. `unpack_signed` maps even u → u/2 and odd u →
/// -(u+1)/2, so the inverse is: nonneg s → 2s, neg s → -2s - 1.
pub fn pack_signed(s: i32) -> u32 {
    if s >= 0 {
        (s as u32) << 1
    } else {
        (((-(s + 1)) as u32) << 1) | 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::{unpack_signed, BitReader, U32Dist};

    #[test]
    fn round_trip_single_bit() {
        let mut bw = BitWriter::new();
        for &b in &[1u32, 0, 1, 1, 0, 1, 0, 0, 1] {
            bw.write_bit(b);
        }
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        for &b in &[1u32, 0, 1, 1, 0, 1, 0, 0, 1] {
            assert_eq!(br.read_bit().unwrap(), b);
        }
    }

    #[test]
    fn round_trip_multi_bit_fields() {
        let mut bw = BitWriter::new();
        bw.write_bits(0xDEAD, 16).unwrap();
        bw.write_bits(0x5, 4).unwrap();
        bw.write_bits(0xBEEF, 16).unwrap();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(16).unwrap(), 0xDEAD);
        assert_eq!(br.read_bits(4).unwrap(), 0x5);
        assert_eq!(br.read_bits(16).unwrap(), 0xBEEF);
    }

    #[test]
    fn pad_to_byte_after_partial_byte() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b11, 2).unwrap();
        bw.pad_to_byte();
        let bytes = bw.finish();
        assert_eq!(bytes.len(), 1);
        // Bits 0..1 = 1, bits 2..7 = 0.
        assert_eq!(bytes[0], 0b0000_0011);
    }

    #[test]
    fn u8_value_round_trip() {
        for v in [0u32, 1, 2, 3, 5, 7, 8, 15, 16, 31, 100, 255, 256] {
            let mut bw = BitWriter::new();
            bw.write_u8_value(v).unwrap();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            assert_eq!(br.read_u8_value().unwrap(), v, "round-trip failed for {v}");
        }
    }

    #[test]
    fn u32_round_trip() {
        let dists_w = [
            U32WriteDist::Val(0),
            U32WriteDist::Val(1),
            U32WriteDist::BitsOffset(4, 2),
            U32WriteDist::BitsOffset(12, 1),
        ];
        let dists_r = [
            U32Dist::Val(0),
            U32Dist::Val(1),
            U32Dist::BitsOffset(4, 2),
            U32Dist::BitsOffset(12, 1),
        ];
        for v in [0u32, 1, 2, 5, 17, 100, 1000, 4097] {
            let mut bw = BitWriter::new();
            bw.write_u32(dists_w, v).unwrap();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            assert_eq!(
                br.read_u32(dists_r).unwrap(),
                v,
                "round-trip failed for {v}"
            );
        }
    }

    #[test]
    fn u64_round_trip() {
        for v in [0u64, 1, 5, 16, 17, 100, 272, 273, 4096, 1_000_000, u64::MAX] {
            let mut bw = BitWriter::new();
            bw.write_u64(v).unwrap();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            assert_eq!(br.read_u64().unwrap(), v, "round-trip failed for {v}");
        }
    }

    #[test]
    fn pack_signed_round_trip() {
        for s in -300..=300 {
            assert_eq!(unpack_signed(pack_signed(s)), s);
        }
    }
}
