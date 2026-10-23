//! Bit writer — the inverse of [`crate::bitreader::BitReader`] for the
//! ISO/IEC 18181-1 B.2 field types (`u(n)`, `U32`, `U64`, `Bool`,
//! `ZeroPadToByte`).
//!
//! Bits are packed least-significant first into successive bytes
//! (B.2.1): the first bit written lands in bit 0 of byte 0.

use crate::bitreader::U32Dist;
use crate::error::{Error, Result};

/// LSB-first bit sink.
#[derive(Debug, Default, Clone)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Bits used in the last byte of `bytes` (0 when byte-aligned).
    bit_pos: u32,
}

impl BitWriter {
    /// An empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total bits written so far.
    pub fn bit_len(&self) -> usize {
        self.bytes.len() * 8
            - if self.bit_pos == 0 {
                0
            } else {
                (8 - self.bit_pos) as usize
            }
    }

    /// `u(n)`: the low `n` bits of `value`, LSB first (`n <= 32`).
    pub fn write_bits(&mut self, n: u32, value: u32) {
        debug_assert!(n <= 32);
        debug_assert!(
            n == 32 || value >> n == 0,
            "value {value} does not fit in {n} bits"
        );
        for i in 0..n {
            let bit = ((value >> i) & 1) as u8;
            if self.bit_pos == 0 {
                self.bytes.push(0);
            }
            let last = self.bytes.len() - 1;
            self.bytes[last] |= bit << self.bit_pos;
            self.bit_pos = (self.bit_pos + 1) & 7;
        }
    }

    /// `Bool()`.
    pub fn write_bool(&mut self, b: bool) {
        self.write_bits(1, b as u32);
    }

    /// `U32(d0, d1, d2, d3)`: the first distribution able to carry
    /// `value` is selected (B.2.2). `Err` when none can.
    pub fn write_u32(&mut self, dists: [U32Dist; 4], value: u32) -> Result<()> {
        for (sel, d) in dists.iter().enumerate() {
            match *d {
                U32Dist::Val(v) if v == value => {
                    self.write_bits(2, sel as u32);
                    return Ok(());
                }
                U32Dist::Bits(n) if n == 32 || value >> n == 0 => {
                    self.write_bits(2, sel as u32);
                    self.write_bits(n, value);
                    return Ok(());
                }
                U32Dist::BitsOffset(n, off)
                    if value >= off && (n == 32 || (value - off) >> n == 0) =>
                {
                    self.write_bits(2, sel as u32);
                    self.write_bits(n, value - off);
                    return Ok(());
                }
                _ => {}
            }
        }
        Err(Error::invalid(format!(
            "JXL writer: value {value} fits no U32 distribution of {dists:?}"
        )))
    }

    /// `U64()` (B.2.3).
    pub fn write_u64(&mut self, value: u64) {
        if value == 0 {
            self.write_bits(2, 0);
        } else if value <= 16 {
            self.write_bits(2, 1);
            self.write_bits(4, (value - 1) as u32);
        } else if value <= 272 {
            self.write_bits(2, 2);
            self.write_bits(8, (value - 17) as u32);
        } else {
            self.write_bits(2, 3);
            self.write_bits(12, (value & 0xFFF) as u32);
            let mut rest = value >> 12;
            let mut shift = 12;
            while rest != 0 {
                self.write_bits(1, 1);
                if shift == 60 {
                    self.write_bits(4, (rest & 0xF) as u32);
                    return;
                }
                self.write_bits(8, (rest & 0xFF) as u32);
                rest >>= 8;
                shift += 8;
            }
            self.write_bits(1, 0);
        }
    }

    /// `Enum(...)` (B.2.6): `U32(0, 1, 2 + u(4), 18 + u(6))`.
    pub fn write_enum(&mut self, value: u32) -> Result<()> {
        self.write_u32(
            [
                U32Dist::Val(0),
                U32Dist::Val(1),
                U32Dist::BitsOffset(4, 2),
                U32Dist::BitsOffset(6, 18),
            ],
            value,
        )
    }

    /// `ZeroPadToByte()` (B.2.7).
    pub fn zero_pad_to_byte(&mut self) {
        self.bit_pos = 0;
    }

    /// Append whole bytes (the writer must be byte-aligned).
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        debug_assert_eq!(self.bit_pos, 0);
        self.bit_pos = 0;
        self.bytes.extend_from_slice(bytes);
    }

    /// The bytes written so far (padded with zero bits to a byte).
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Byte length once padded.
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
}

/// `PackSigned`: the inverse of [`crate::bitreader::unpack_signed`].
pub fn pack_signed(v: i32) -> u32 {
    if v >= 0 {
        (v as u32) << 1
    } else {
        ((-(v as i64)) as u32 * 2) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::{BitReader, unpack_signed};

    #[test]
    fn bits_round_trip_through_the_reader() {
        let mut w = BitWriter::new();
        w.write_bits(3, 0b101);
        w.write_bool(true);
        w.write_bits(13, 0x1ABC);
        w.write_u64(0);
        w.write_u64(16);
        w.write_u64(17);
        w.write_u64(272);
        w.write_u64(273);
        w.write_u64(u64::MAX);
        w.write_u64(1 << 40);
        w.write_enum(13).unwrap();
        w.write_enum(40).unwrap();
        w.zero_pad_to_byte();
        w.write_bytes(&[0xAA]);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(3).unwrap(), 0b101);
        assert!(r.read_bool().unwrap());
        assert_eq!(r.read_bits(13).unwrap(), 0x1ABC);
        for v in [0u64, 16, 17, 272, 273, u64::MAX, 1 << 40] {
            assert_eq!(r.read_u64().unwrap(), v, "U64 {v}");
        }
        let en = [
            U32Dist::Val(0),
            U32Dist::Val(1),
            U32Dist::BitsOffset(4, 2),
            U32Dist::BitsOffset(6, 18),
        ];
        assert_eq!(r.read_u32(en).unwrap(), 13);
        assert_eq!(r.read_u32(en).unwrap(), 40);
        r.pu0().unwrap();
        assert_eq!(r.read_bits(8).unwrap(), 0xAA);
    }

    #[test]
    fn u32_picks_the_first_fitting_distribution() {
        let d = [
            U32Dist::Bits(10),
            U32Dist::BitsOffset(14, 1024),
            U32Dist::BitsOffset(22, 17408),
            U32Dist::BitsOffset(30, 4211712),
        ];
        for v in [
            0u32,
            1023,
            1024,
            17407,
            17408,
            4211711,
            4211712,
            4211712 + (1 << 30) - 1,
        ] {
            let mut w = BitWriter::new();
            w.write_u32(d, v).unwrap();
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.read_u32(d).unwrap(), v);
        }
        let mut w = BitWriter::new();
        assert!(w.write_u32([U32Dist::Val(1); 4], 2).is_err());
    }

    #[test]
    fn pack_signed_inverts_unpack_signed() {
        for v in [-70000, -2, -1, 0, 1, 2, 70000, i32::MAX, i32::MIN + 1] {
            assert_eq!(unpack_signed(pack_signed(v)), v);
        }
    }
}
