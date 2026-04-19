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
