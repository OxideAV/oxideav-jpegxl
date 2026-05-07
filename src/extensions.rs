//! `Extensions` bundle — FDIS 18181-1 §A.5 (Table A.14).
//!
//! Appears as a tail field in `ImageMetadata`, `FrameHeader`, and
//! `RestorationFilter`. The decoded form captures the bitmask of
//! present extensions and, for each, the number of bits it occupies in
//! the codestream. The actual extension *payload* is read from the
//! caller's bitstream by calling [`Extensions::skip_payload`] — this
//! crate does not interpret any extensions (Annex H is open-ended), but
//! must skip them to keep the caller's bit cursor aligned with later
//! fields.
//!
//! Allocation bound: `extensions` is at most a 64-bit mask, so we never
//! allocate more than 64 entries. `extension_bits[i]` is a `U64` value
//! whose payload bit-count we cap against the bit reader's remaining
//! input length before any actual read.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Decoded `Extensions` bundle — bitmask + per-extension payload sizes.
///
/// `extensions == 0` is the common case (the bundle defaulted to zero),
/// in which case `extension_bits` is empty and [`Self::skip_payload`]
/// is a no-op.
#[derive(Debug, Clone, Default)]
pub struct Extensions {
    /// Bitmask of present extensions: bit `i` set means extension with
    /// `ext_i = i` is present. `i` ranges over `[0, 63)`.
    pub mask: u64,
    /// `extension_bits[k]` = number of payload bits for the `k`-th
    /// present extension (in ascending bit-index order). Length equals
    /// `mask.count_ones()`.
    pub extension_bits: Vec<u64>,
}

impl Extensions {
    /// Read the `extensions` mask + per-extension bit counts per Table
    /// A.14. Does NOT read the payload bits — call
    /// [`Self::skip_payload`] for that.
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let mask = br.read_u64()?;
        if mask == 0 {
            return Ok(Self::default());
        }
        let n_ext = mask.count_ones() as usize;
        // Each extension_bits is a U64() — minimum cost 2 bits (sel=0).
        // Refuse a mask whose claimed entries exceed our bit budget.
        if n_ext.saturating_mul(2) > br.bits_remaining() {
            return Err(Error::InvalidData(
                "JXL Extensions: mask asserts more entries than input could supply".into(),
            ));
        }
        let mut extension_bits = Vec::with_capacity(n_ext);
        for _ in 0..n_ext {
            extension_bits.push(br.read_u64()?);
        }
        Ok(Self {
            mask,
            extension_bits,
        })
    }

    /// Sum of `extension_bits[i]` — total payload bits we still need to
    /// consume after the mask + per-extension bit-count fields.
    pub fn payload_bits(&self) -> u64 {
        self.extension_bits.iter().copied().sum()
    }

    /// Skip the extension payload (`sum(extension_bits)` bits). The
    /// caller is responsible for invoking this *immediately* after the
    /// containing bundle finishes its other fields, so that subsequent
    /// fields land on the correct bit boundary.
    pub fn skip_payload(&self, br: &mut BitReader<'_>) -> Result<()> {
        let total = self.payload_bits();
        if total > br.bits_remaining() as u64 {
            return Err(Error::InvalidData(
                "JXL Extensions: payload exceeds remaining input".into(),
            ));
        }
        // Skip in 32-bit chunks so we can use `read_bits` (which caps at
        // 32 bits per call). For payloads < 32 bits the loop runs once.
        let mut left = total;
        while left > 0 {
            let take = left.min(32) as u32;
            // Reading bits past our cap is impossible at this point —
            // we just verified bits_remaining covers `total`.
            let _ = br.read_bits(take)?;
            left -= take as u64;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build LSB-first bit packing for tests.
    struct Bw(Vec<u8>, u8);
    impl Bw {
        fn new() -> Self {
            Self(Vec::new(), 0)
        }
        fn w(&mut self, value: u64, n: u32) {
            for i in 0..n {
                if self.1 == 0 {
                    self.0.push(0);
                }
                let bit = ((value >> i) & 1) as u8;
                let last = self.0.len() - 1;
                self.0[last] |= bit << self.1;
                self.1 = (self.1 + 1) % 8;
            }
        }
    }

    #[test]
    fn empty_mask_is_default() {
        // U64() sel=0 → mask = 0.
        let mut bw = Bw::new();
        bw.w(0, 2);
        let mut br = BitReader::new(&bw.0);
        let ext = Extensions::read(&mut br).unwrap();
        assert_eq!(ext.mask, 0);
        assert!(ext.extension_bits.is_empty());
        assert_eq!(ext.payload_bits(), 0);
        ext.skip_payload(&mut br).unwrap();
    }

    #[test]
    fn single_extension_with_payload_round_trip() {
        // mask = 1 (one extension at index 0), extension_bits[0] = 8 (=
        // BitsOffset(4, 1) with raw 7), payload = 8 zero bits.
        let mut bw = Bw::new();
        // mask = U64(): sel=1 → BitsOffset(4, 1) → raw=0 → value=1.
        bw.w(1, 2); // sel=1
        bw.w(0, 4); // raw=0 → value=1
                    // extension_bits[0] = U64() sel=1 → raw=7 → value=8.
        bw.w(1, 2);
        bw.w(7, 4);
        // 8 payload bits (all zero).
        bw.w(0, 8);
        let mut br = BitReader::new(&bw.0);
        let ext = Extensions::read(&mut br).unwrap();
        assert_eq!(ext.mask, 1);
        assert_eq!(ext.extension_bits, vec![8]);
        assert_eq!(ext.payload_bits(), 8);
        ext.skip_payload(&mut br).unwrap();
    }

    #[test]
    fn two_extensions_payload_consumed() {
        // mask = 0b101 (extensions 0 and 2), extension_bits = [4, 12].
        // payload = 16 bits (split is irrelevant since we skip).
        let mut bw = Bw::new();
        // mask via sel=2 → BitsOffset(8, 17) → raw = 5 - 17 = ... no.
        // Easier: sel=1 BitsOffset(4,1), raw = 4 → value = 5 = 0b101.
        bw.w(1, 2);
        bw.w(4, 4);
        // ext0_bits = sel=1 raw=3 → 4
        bw.w(1, 2);
        bw.w(3, 4);
        // ext1_bits = sel=1 raw=11 → 12
        bw.w(1, 2);
        bw.w(11, 4);
        // 16 payload bits.
        bw.w(0, 16);
        let mut br = BitReader::new(&bw.0);
        let ext = Extensions::read(&mut br).unwrap();
        assert_eq!(ext.mask, 0b101);
        assert_eq!(ext.extension_bits, vec![4, 12]);
        // Track input bits exactly: mask = 2+4 = 6, ext0_bits = 2+4 = 6,
        // ext1_bits = 2+4 = 6, payload = 16 → total 34 bits. The packer
        // emits whole bytes (40 bits) so 6 trailing zero bits remain.
        ext.skip_payload(&mut br).unwrap();
        assert_eq!(br.bits_remaining(), 6);
    }

    #[test]
    fn malicious_mask_with_no_input_rejected() {
        // U64 selector sel=3, u(12) = 0xFFF, then 1-bit "more" but no
        // continuation bytes available — but we want a mask claiming
        // many extensions and no follow-on input.
        // Use sel=1 raw=14 → mask = 15, then no bits left → must error.
        let bytes = vec![0b0011_1001_u8]; // sel=1, raw=14
        let mut br = BitReader::new(&bytes);
        // mask = 15 (4 extensions). bits_remaining after mask read = 0.
        // n_ext = 4, 4 * 2 = 8 > 0 → must reject.
        let res = Extensions::read(&mut br);
        assert!(res.is_err());
    }
}
