//! LSB-first bit reader used by the JPEG XL codestream.
//!
//! JPEG XL packs bits least-significant-first inside each byte: bit 0 of a
//! byte is read before bit 7, and multi-bit fields are assembled with the
//! first bit read becoming the least-significant bit of the field.

use oxideav_core::{Error, Result};

pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    pub fn bits_read(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    pub fn bytes_consumed(&self) -> usize {
        self.byte_pos + if self.bit_pos == 0 { 0 } else { 1 }
    }

    pub fn read_bit(&mut self) -> Result<u32> {
        if self.byte_pos >= self.data.len() {
            return Err(Error::InvalidData("unexpected end of JXL bitstream".into()));
        }
        let b = (self.data[self.byte_pos] >> self.bit_pos) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Ok(b as u32)
    }

    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(Error::InvalidData(
                "cannot read more than 32 bits at once".into(),
            ));
        }
        let mut out: u32 = 0;
        for i in 0..n {
            let bit = self.read_bit()?;
            out |= bit << i;
        }
        Ok(out)
    }

    /// Peek up to 16 bits ahead without advancing the read cursor.
    ///
    /// Required by the ANS distribution decoder (FDIS D.3.4): the
    /// `kLogCountLut` lookup is keyed off `u(7)` worth of LSB-first bits,
    /// then the bitstream is advanced by `kLogCountLut[h][0]` bits (which
    /// is between 3 and 7), so a separate peek + advance step is needed.
    /// Bits past EOF are read as zero — the caller must validate the
    /// derived advance against the actual remaining bit budget.
    pub fn peek_bits(&self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 16 {
            return Err(Error::InvalidData(
                "JXL peek_bits(): cannot peek more than 16 bits at once".into(),
            ));
        }
        let mut out: u32 = 0;
        let mut byte_pos = self.byte_pos;
        let mut bit_pos = self.bit_pos;
        for i in 0..n {
            let bit = if byte_pos >= self.data.len() {
                // Past EOF: treat as zero. Caller must validate the
                // subsequent `advance_bits` against actual data length.
                0
            } else {
                ((self.data[byte_pos] >> bit_pos) & 1) as u32
            };
            out |= bit << i;
            bit_pos += 1;
            if bit_pos == 8 {
                bit_pos = 0;
                byte_pos += 1;
            }
        }
        Ok(out)
    }

    /// Advance the read cursor by exactly `n` bits, validating against
    /// EOF. Used as the matching `advance` for [`peek_bits`].
    pub fn advance_bits(&mut self, n: u32) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let total_bits_remaining =
            (self.data.len() * 8).saturating_sub(self.byte_pos * 8 + self.bit_pos as usize);
        if (n as usize) > total_bits_remaining {
            return Err(Error::InvalidData(
                "JXL advance_bits(): unexpected end of bitstream".into(),
            ));
        }
        let new_bit = self.bit_pos as u32 + n;
        self.byte_pos += (new_bit / 8) as usize;
        self.bit_pos = (new_bit % 8) as u8;
        Ok(())
    }

    /// Total bits still available behind the read cursor. Used by
    /// allocation-sizing checks to bound `Vec::with_capacity` against the
    /// real input length.
    pub fn bits_remaining(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.byte_pos * 8 + self.bit_pos as usize)
    }

    /// JXL `U8()` per 9.2.5: 1-bit "is zero" flag, otherwise 3-bit
    /// magnitude `n` followed by `u(n)` extra bits with implicit
    /// leading 1.
    pub fn read_u8_value(&mut self) -> Result<u32> {
        if self.read_bit()? == 0 {
            return Ok(0);
        }
        let n = self.read_bits(3)?;
        Ok(self.read_bits(n)? + (1u32 << n))
    }

    pub fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_bit()? != 0)
    }

    /// Read a JXL `U32` field: a 2-bit selector chooses one of four
    /// `distributions`, where each entry is either a literal value
    /// (`U32Dist::Val`) or a variable-width integer with a base offset
    /// (`U32Dist::BitsOffset(nbits, offset)`).
    pub fn read_u32(&mut self, dists: [U32Dist; 4]) -> Result<u32> {
        let sel = self.read_bits(2)?;
        match dists[sel as usize] {
            U32Dist::Val(v) => Ok(v),
            U32Dist::Bits(n) => self.read_bits(n),
            U32Dist::BitsOffset(n, off) => Ok(self.read_bits(n)? + off),
        }
    }

    /// `pu0()` per A.3.2.4: skip to the next byte boundary; the skipped
    /// bits MUST all be zero, otherwise the codestream is ill-formed.
    pub fn pu0(&mut self) -> Result<()> {
        if self.bit_pos == 0 {
            return Ok(());
        }
        let n = 8 - self.bit_pos;
        let v = self.read_bits(n as u32)?;
        if v != 0 {
            return Err(Error::InvalidData(
                "JXL pu0(): non-zero padding bits before byte boundary".into(),
            ));
        }
        Ok(())
    }

    /// `Varint()` per A.3.1.5: read a 7-bit-per-byte little-endian
    /// variable-length unsigned integer of up to 63 bits.
    pub fn read_varint(&mut self) -> Result<u64> {
        let mut value: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            let b = self.read_bits(8)? as u64;
            value |= (b & 0x7f) << shift;
            if b <= 127 {
                break;
            }
            shift += 7;
            if shift >= 63 {
                return Err(Error::InvalidData("JXL Varint(): shift overflow".into()));
            }
        }
        Ok(value)
    }

    /// `pu()` per A.3.1.2: read enough bits to align to the next byte
    /// boundary (returning their value). Unlike [`pu0`] this does NOT
    /// require the skipped bits to be zero.
    pub fn pu(&mut self) -> Result<u32> {
        if self.bit_pos == 0 {
            return Ok(0);
        }
        let n = 8 - self.bit_pos;
        self.read_bits(n as u32)
    }

    /// Borrow the underlying byte slice (used by entropy coders that
    /// switch from bit-level to byte-level reads after a `pu0()`).
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// `U64()` per FDIS 18181-1 §9.2.3 (Listing 9.1).
    ///
    /// 2-bit selector:
    ///   * 0 → value = 0
    ///   * 1 → value = `BitsOffset(4, 1)`
    ///   * 2 → value = `BitsOffset(8, 17)`
    ///   * 3 → value = `u(12)` followed by zero or more 8-bit
    ///     continuations gated on a 1-bit "more" flag, plus an optional
    ///     final 4-bit chunk when shift would otherwise reach 60.
    ///
    /// Bounds: shift never advances past 60 (the listing's terminating
    /// branch caps the final chunk at 4 bits) so the result fits in 64
    /// bits unconditionally.
    pub fn read_u64(&mut self) -> Result<u64> {
        let sel = self.read_bits(2)?;
        match sel {
            0 => Ok(0),
            1 => Ok(self.read_bits(4)? as u64 + 1),
            2 => Ok(self.read_bits(8)? as u64 + 17),
            _ => {
                let mut value: u64 = self.read_bits(12)? as u64;
                let mut shift: u32 = 12;
                loop {
                    if self.read_bit()? == 0 {
                        break;
                    }
                    if shift == 60 {
                        let chunk = self.read_bits(4)? as u64;
                        value |= chunk << shift;
                        break;
                    }
                    let chunk = self.read_bits(8)? as u64;
                    value |= chunk << shift;
                    shift += 8;
                }
                Ok(value)
            }
        }
    }

    /// `F16()` per FDIS 18181-1 §9.2.6 (Listing 9.5).
    ///
    /// Reads 16 bits and interprets them as IEEE-754 binary16. Per the
    /// listing's normative rule the biased exponent must not be 31
    /// (NaN/infinity rejected); this is enforced.
    pub fn read_f16(&mut self) -> Result<f32> {
        let bits16 = self.read_bits(16)? as u16;
        let biased_exp = (bits16 >> 10) & 0x1F;
        if biased_exp == 31 {
            return Err(Error::InvalidData(
                "JXL F16(): biased_exp == 31 (NaN/Inf disallowed)".into(),
            ));
        }
        Ok(interpret_as_f16(bits16))
    }
}

/// Decode an IEEE-754 binary16 bit pattern as an `f32`. NaN/Inf values
/// are decoded as the corresponding `f32` representation, but
/// [`BitReader::read_f16`] rejects them upstream.
pub fn interpret_as_f16(bits16: u16) -> f32 {
    let sign = (bits16 >> 15) & 1;
    let exp = ((bits16 >> 10) & 0x1F) as i32;
    let mant = (bits16 & 0x3FF) as u32;
    let f32_bits: u32 = if exp == 0 && mant == 0 {
        (sign as u32) << 31
    } else if exp == 0 {
        // Subnormal: convert by shifting until normal.
        let mut m = mant;
        let mut e: i32 = -14;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = (e + 127) as u32;
        ((sign as u32) << 31) | (f32_exp << 23) | (m << 13)
    } else if exp == 31 {
        // NaN/Inf: emit float NaN/Inf with same sign.
        ((sign as u32) << 31) | (0xFF << 23) | (mant << 13)
    } else {
        let f32_exp = (exp - 15 + 127) as u32;
        ((sign as u32) << 31) | (f32_exp << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// `UnpackSigned(u)` per FDIS §5.2: `u/2` if u is even, `-(u+1)/2` if u
/// is odd. Used by `Customxy` (A.4) and a handful of other JXL fields
/// that store signed integers as ZigZag-encoded unsigned ones.
pub fn unpack_signed(u: u32) -> i32 {
    if u & 1 == 0 {
        (u >> 1) as i32
    } else {
        -((u >> 1) as i32 + 1)
    }
}

#[derive(Copy, Clone, Debug)]
pub enum U32Dist {
    Val(u32),
    Bits(u32),
    BitsOffset(u32, u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_bits_lsb_first() {
        // Byte 0xB4 = 1011_0100 binary; LSB-first: 0,0,1,0,1,1,0,1
        let data = [0xB4];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bit().unwrap(), 0);
        assert_eq!(br.read_bit().unwrap(), 0);
        assert_eq!(br.read_bit().unwrap(), 1);
        assert_eq!(br.read_bit().unwrap(), 0);
        assert_eq!(br.read_bits(4).unwrap(), 0b1011);
    }

    #[test]
    fn crosses_byte_boundary() {
        // bytes: 0x3C 0x5A
        // LSB-first read of 12 bits assembles low byte first into LSB of result.
        let data = [0x3C, 0x5A];
        let mut br = BitReader::new(&data);
        let v = br.read_bits(12).unwrap();
        assert_eq!(v, 0x3C | ((0x5A & 0x0F) << 8));
    }

    #[test]
    fn u32_selector_val() {
        let data = [0b0000_0000];
        let mut br = BitReader::new(&data);
        let v = br
            .read_u32([
                U32Dist::Val(7),
                U32Dist::Val(8),
                U32Dist::Val(9),
                U32Dist::Val(10),
            ])
            .unwrap();
        assert_eq!(v, 7);
    }

    #[test]
    fn read_zero_bits_is_noop() {
        let data = [0xAA];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(0).unwrap(), 0);
        assert_eq!(br.bits_read(), 0);
    }

    #[test]
    fn read_more_than_32_bits_rejected() {
        let data = [0xFF; 8];
        let mut br = BitReader::new(&data);
        assert!(br.read_bits(33).is_err());
    }

    #[test]
    fn read_full_32_bits_round_trips() {
        // Bits LSB-first: 0xDEADBEEF in field order. Encode by writing the
        // value low-byte-first (LSB inside each byte).
        let v: u32 = 0xDEAD_BEEF;
        let bytes = v.to_le_bytes();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(32).unwrap(), v);
        // After consuming 32 bits we should be exactly 4 bytes in.
        assert_eq!(br.bits_read(), 32);
        assert_eq!(br.bytes_consumed(), 4);
    }

    #[test]
    fn eof_returns_invalid_data() {
        let data = [0u8; 1];
        let mut br = BitReader::new(&data);
        let _ = br.read_bits(8).unwrap();
        let err = br.read_bit().unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn bytes_consumed_tracks_partial_byte() {
        let data = [0xF0, 0x0F];
        let mut br = BitReader::new(&data);
        let _ = br.read_bits(4).unwrap();
        // 4 bits read out of byte 0 → still consuming byte 0.
        assert_eq!(br.bytes_consumed(), 1);
        let _ = br.read_bits(4).unwrap();
        assert_eq!(br.bytes_consumed(), 1);
        let _ = br.read_bit().unwrap();
        assert_eq!(br.bytes_consumed(), 2);
    }

    #[test]
    fn pu0_passes_at_byte_boundary() {
        let data = [0x00];
        let mut br = BitReader::new(&data);
        br.pu0().unwrap();
        assert_eq!(br.bits_read(), 0);
    }

    #[test]
    fn pu0_passes_when_padding_zero() {
        // Read 3 bits of zero, then pu0 should consume the rest with no error.
        let data = [0x00];
        let mut br = BitReader::new(&data);
        let _ = br.read_bits(3).unwrap();
        br.pu0().unwrap();
        assert_eq!(br.bits_read(), 8);
    }

    #[test]
    fn pu0_rejects_nonzero_padding() {
        // Byte 0xF0 = bits 0..=3 are 0, bits 4..=7 are 1. After reading 3
        // zero bits we are at bit pos 3; pu0 must read bits 3..=7 = 0,1,1,1,1
        // which has value 0b11110 != 0 → error.
        let data = [0xF0];
        let mut br = BitReader::new(&data);
        let _ = br.read_bits(3).unwrap();
        assert!(br.pu0().is_err());
    }

    #[test]
    fn read_varint_single_byte() {
        // 0x42 = 0b0100_0010 → top bit clear, value = 0x42.
        let data = [0x42];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_varint().unwrap(), 0x42);
        assert_eq!(br.bytes_consumed(), 1);
    }

    #[test]
    fn read_varint_multi_byte() {
        // Encode 300 (0x12C). Two bytes: 0x80 | (0x2C) = 0xAC, then 0x02.
        // Decoded: (0xAC & 0x7F) | (0x02 << 7) = 0x2C | 0x100 = 0x12C.
        let data = [0xAC, 0x02];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_varint().unwrap(), 300);
        assert_eq!(br.bytes_consumed(), 2);
    }

    #[test]
    fn peek_then_advance_matches_read() {
        let data = [0xB4, 0x5A];
        let mut br1 = BitReader::new(&data);
        let mut br2 = BitReader::new(&data);
        let p = br1.peek_bits(7).unwrap();
        br1.advance_bits(7).unwrap();
        let r = br2.read_bits(7).unwrap();
        assert_eq!(p, r);
        assert_eq!(br1.bits_read(), 7);
    }

    #[test]
    fn peek_past_eof_returns_zero_bits() {
        // Two-byte input, peek 16 bits at offset 8 → upper byte is real,
        // bits past the end (none here) would be zero.
        let data = [0xAB, 0xCD];
        let br = BitReader::new(&data);
        let v = br.peek_bits(16).unwrap();
        assert_eq!(v, 0xCDAB);
    }

    #[test]
    fn advance_bits_rejects_past_eof() {
        let data = [0u8; 1];
        let mut br = BitReader::new(&data);
        assert!(br.advance_bits(9).is_err());
    }

    #[test]
    fn read_u8_value_zero() {
        // bit0 = 0 → value 0.
        let data = [0u8];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_u8_value().unwrap(), 0);
    }

    #[test]
    fn read_u8_value_three() {
        // Decode value 3 with the JXL `U8()` LSB-first read order:
        //   u(1) = 1                    → not-zero flag
        //   u(3) = 1                    → n = 1
        //   u(1) = 1                    → extra; value = 1 + (1<<1) = 3
        // LSB-first packing: bit0=1, bit1=1, bit2=0, bit3=0, bit4=1.
        // → byte = 0b00010011 = 0x13.
        // (The spec's section 9.2.5 example "bits 10011 in value 3"
        // lists the same five bits in MSB-first display order.)
        let data = [0x13];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_u8_value().unwrap(), 3);
    }

    #[test]
    fn u64_selector_zero() {
        // 2-bit selector = 0 → value = 0 unconditionally.
        let data = [0u8];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_u64().unwrap(), 0);
    }

    #[test]
    fn u64_selector_one_bits_offset_4_1() {
        // sel = 1 → BitsOffset(4, 1). Pick raw bits = 5 → value = 6.
        // LSB-first: sel bits are bit0=1, bit1=0; payload bits are 1,0,1,0.
        let mut bw = BitWriterTest::new();
        bw.w(1, 2); // sel
        bw.w(5, 4); // raw
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_u64().unwrap(), 6);
    }

    #[test]
    fn u64_selector_three_no_continuation() {
        // sel = 3, then u(12) = 0xABC, then 0-bit terminator.
        let mut bw = BitWriterTest::new();
        bw.w(3, 2);
        bw.w(0xABC, 12);
        bw.w(0, 1);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_u64().unwrap(), 0xABC);
    }

    #[test]
    fn u64_selector_three_one_continuation() {
        // sel = 3, u(12) = 0x123, 1-bit "more", u(8) = 0xCD, 0-bit "stop".
        // value = 0x123 | (0xCD << 12) = 0x000C_D123.
        let mut bw = BitWriterTest::new();
        bw.w(3, 2);
        bw.w(0x123, 12);
        bw.w(1, 1);
        bw.w(0xCD, 8);
        bw.w(0, 1);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_u64().unwrap(), 0x000C_D123);
    }

    #[test]
    fn f16_zero_and_one() {
        // 0x0000 → +0.0 ; 0x3C00 → 1.0 ; 0xC000 → -2.0
        let bytes = pack_u16(&[0x0000, 0x3C00, 0xC000]);
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_f16().unwrap(), 0.0);
        assert_eq!(br.read_f16().unwrap(), 1.0);
        assert_eq!(br.read_f16().unwrap(), -2.0);
    }

    #[test]
    fn f16_rejects_inf() {
        // 0x7C00 → +inf ; biased_exp == 31, must error.
        let bytes = pack_u16(&[0x7C00]);
        let mut br = BitReader::new(&bytes);
        assert!(br.read_f16().is_err());
    }

    #[test]
    fn unpack_signed_round_trip() {
        assert_eq!(unpack_signed(0), 0);
        assert_eq!(unpack_signed(1), -1);
        assert_eq!(unpack_signed(2), 1);
        assert_eq!(unpack_signed(3), -2);
        assert_eq!(unpack_signed(4), 2);
        assert_eq!(unpack_signed(5), -3);
    }

    /// LSB-first bit writer for tests.
    struct BitWriterTest {
        out: Vec<u8>,
        bit_pos: u8,
    }
    impl BitWriterTest {
        fn new() -> Self {
            Self {
                out: Vec::new(),
                bit_pos: 0,
            }
        }
        fn w(&mut self, value: u32, n: u32) {
            for i in 0..n {
                if self.bit_pos == 0 {
                    self.out.push(0);
                }
                let bit = ((value >> i) & 1) as u8;
                let last = self.out.len() - 1;
                self.out[last] |= bit << self.bit_pos;
                self.bit_pos = (self.bit_pos + 1) % 8;
            }
        }
        fn finish(self) -> Vec<u8> {
            self.out
        }
    }

    fn pack_u16(values: &[u16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in values {
            out.push((v & 0xFF) as u8);
            out.push((v >> 8) as u8);
        }
        out
    }

    #[test]
    fn u32_selector_bits_offset() {
        // Two-bit selector = 2 → BitsOffset(3, 1). LSB-first we need bits 0,1
        // to be "01" (selector=2, since bit-pos-0 is LSB of selector) → wait:
        // selector is read with read_bits(2), which assembles the two bits
        // LSB-first. To get selector value 2 (binary 10), we need bit0=0,
        // bit1=1. Then we read 3 bits as the value; pick 0b101 = 5.
        // So the raw byte bits LSB→MSB are: 0,1,1,0,1,... → 0b0010110 (MSB view) = 0x16? Let's just compute.
        // bits LSB→MSB in the byte: b0=0, b1=1, b2=1, b3=0, b4=1 → byte = 0b0001_0110 = 0x16
        let data = [0x16];
        let mut br = BitReader::new(&data);
        let v = br
            .read_u32([
                U32Dist::Val(100),
                U32Dist::Val(200),
                U32Dist::BitsOffset(3, 1),
                U32Dist::Val(300),
            ])
            .unwrap();
        assert_eq!(v, 5 + 1);
    }
}
