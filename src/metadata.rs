//! JPEG XL codestream header parsing.
//!
//! Covers the fixed preamble of every JXL codestream:
//!   * 2-byte signature (`FF 0A`)
//!   * `SizeHeader` (image dimensions, possibly implied via aspect ratio)
//!   * the first fields of `ImageMetadata`: the `all_default` bit and, if
//!     clear, the `extra_fields` / orientation / preview / animation /
//!     bit-depth `all_default` flags. Full ColorEncoding decoding is not
//!     attempted here.
//!
//! Bit layout mirrors the reference libjxl implementation
//! (`lib/jxl/headers.cc`, `lib/jxl/image_metadata.cc`): LSB-first bit
//! packing, `U32` fields with 2-bit selectors, and `Bundle::AllDefault`
//! shortcut bits.

use oxideav_core::{Error, Result};

use crate::bitreader::{BitReader, U32Dist};
use crate::container::{detect, extract_codestream, Signature, RAW_CODESTREAM_SIGNATURE};

/// The seven fixed aspect ratios a `SizeHeader` can reference.
///
/// `(numerator, denominator)` applied as `xsize = ysize * num / den`
/// (integer truncation). Index 0 means "ratio not used, xsize is
/// transmitted explicitly".
pub const FIXED_ASPECT_RATIOS: [(u32, u32); 7] = [
    (1, 1),
    (12, 10),
    (4, 3),
    (3, 2),
    (16, 9),
    (5, 4),
    (2, 1),
];

/// Decoded JXL image size, with the raw encoding flags preserved for
/// diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeHeader {
    pub width: u32,
    pub height: u32,
    /// `true` if the height (and possibly width) was transmitted using the
    /// 5-bit "divisible by 8, at most 256" short form.
    pub small: bool,
    /// Aspect ratio selector 1..=7 if xsize was implied; 0 if xsize was
    /// transmitted explicitly.
    pub ratio: u8,
}

/// High-level flags pulled from the `ImageMetadata` preamble. Only the
/// fields this crate currently decodes are present; fuller parsing lands
/// with the actual pixel pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageMetadata {
    /// True when the bundle's `all_default` bit was set: orientation is 1,
    /// no preview/animation, 8-bit unsigned samples, sRGB, no XYB encoding.
    pub all_default: bool,
    /// EXIF-style orientation (1 = identity).
    pub orientation: u8,
    pub have_preview: bool,
    pub have_animation: bool,
    pub have_intrinsic_size: bool,
    /// True if the XYB opsin color transform is applied to the stored samples.
    pub xyb_encoded: bool,
    pub bit_depth: BitDepth,
    /// Number of declared extra channels (alpha, depth, spot colour, ...).
    pub num_extra_channels: u32,
}

/// Sample format of the stored image (not of any post-processing output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitDepth {
    pub floating_point: bool,
    pub bits_per_sample: u32,
    /// Only meaningful when `floating_point` is true.
    pub exponent_bits: u32,
}

impl BitDepth {
    pub const DEFAULT_U8: Self = Self {
        floating_point: false,
        bits_per_sample: 8,
        exponent_bits: 0,
    };
}

/// Full result of a best-effort header parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Headers {
    pub signature: Signature,
    pub size: SizeHeader,
    pub metadata: ImageMetadata,
}

/// Parse the JXL signature, `SizeHeader`, and `ImageMetadata` preamble
/// from the start of `input`. `input` may be either a raw codestream or
/// an ISOBMFF-wrapped JXL file; the wrapper is unpacked internally.
pub fn parse_headers(input: &[u8]) -> Result<Headers> {
    let signature = detect(input).ok_or_else(|| {
        Error::InvalidData("not a JPEG XL file: signature mismatch".into())
    })?;
    let codestream = extract_codestream(input)?;
    let cs = &*codestream;
    if cs.len() < 2 || cs[..2] != RAW_CODESTREAM_SIGNATURE {
        return Err(Error::InvalidData(
            "JXL codestream missing FF 0A signature".into(),
        ));
    }
    let mut br = BitReader::new(&cs[2..]);
    let size = parse_size_header(&mut br)?;
    let metadata = parse_image_metadata(&mut br)?;
    Ok(Headers { signature, size, metadata })
}

/// Decode the JXL `SizeHeader` bundle.
pub fn parse_size_header(br: &mut BitReader<'_>) -> Result<SizeHeader> {
    let small = br.read_bool()?;
    let height = if small {
        (br.read_bits(5)? + 1) * 8
    } else {
        br.read_u32([
            U32Dist::BitsOffset(9, 1),
            U32Dist::BitsOffset(13, 1),
            U32Dist::BitsOffset(18, 1),
            U32Dist::BitsOffset(30, 1),
        ])?
    };
    let ratio = br.read_bits(3)? as u8;
    let width = if ratio == 0 {
        if small {
            (br.read_bits(5)? + 1) * 8
        } else {
            br.read_u32([
                U32Dist::BitsOffset(9, 1),
                U32Dist::BitsOffset(13, 1),
                U32Dist::BitsOffset(18, 1),
                U32Dist::BitsOffset(30, 1),
            ])?
        }
    } else {
        let (num, den) = FIXED_ASPECT_RATIOS[(ratio - 1) as usize];
        ((height as u64 * num as u64) / den as u64) as u32
    };
    if width == 0 || height == 0 {
        return Err(Error::InvalidData(
            "JXL SizeHeader: zero-dimensional image".into(),
        ));
    }
    Ok(SizeHeader { width, height, small, ratio })
}

fn parse_bit_depth(br: &mut BitReader<'_>) -> Result<BitDepth> {
    let floating_point = br.read_bool()?;
    if !floating_point {
        let bits_per_sample = br.read_u32([
            U32Dist::Val(8),
            U32Dist::Val(10),
            U32Dist::Val(12),
            U32Dist::BitsOffset(6, 1),
        ])?;
        if bits_per_sample > 31 {
            return Err(Error::InvalidData(format!(
                "JXL BitDepth: invalid integer bits_per_sample {bits_per_sample}"
            )));
        }
        Ok(BitDepth {
            floating_point: false,
            bits_per_sample,
            exponent_bits: 0,
        })
    } else {
        let bits_per_sample = br.read_u32([
            U32Dist::Val(32),
            U32Dist::Val(16),
            U32Dist::Val(24),
            U32Dist::BitsOffset(6, 1),
        ])?;
        let exponent_bits = br.read_bits(4)? + 1;
        if !(2..=8).contains(&exponent_bits) {
            return Err(Error::InvalidData(format!(
                "JXL BitDepth: invalid exponent_bits_per_sample {exponent_bits}"
            )));
        }
        let mantissa_bits = bits_per_sample as i64 - exponent_bits as i64 - 1;
        if !(2..=23).contains(&mantissa_bits) {
            return Err(Error::InvalidData(format!(
                "JXL BitDepth: invalid float bits_per_sample {bits_per_sample}"
            )));
        }
        Ok(BitDepth {
            floating_point: true,
            bits_per_sample,
            exponent_bits,
        })
    }
}

fn skip_size_header(br: &mut BitReader<'_>) -> Result<()> {
    parse_size_header(br).map(|_| ())
}

/// Decode the `ImageMetadata` fields this crate currently understands.
///
/// When the bundle's `all_default` bit is set (the common case — plain
/// sRGB 8-bit images) we return the defaults directly. Otherwise we walk
/// the extra-fields block (orientation + preview/animation/intrinsic-size
/// flags + their sub-bundles) and the bit-depth / extra-channel count,
/// then stop before ColorEncoding (which this crate does not yet decode).
pub fn parse_image_metadata(br: &mut BitReader<'_>) -> Result<ImageMetadata> {
    let all_default = br.read_bool()?;
    if all_default {
        return Ok(ImageMetadata {
            all_default: true,
            orientation: 1,
            have_preview: false,
            have_animation: false,
            have_intrinsic_size: false,
            xyb_encoded: true,
            bit_depth: BitDepth::DEFAULT_U8,
            num_extra_channels: 0,
        });
    }

    let extra_fields = br.read_bool()?;
    let mut orientation: u8 = 1;
    let mut have_intrinsic_size = false;
    let mut have_preview = false;
    let mut have_animation = false;
    if extra_fields {
        orientation = (br.read_bits(3)? as u8) + 1;
        have_intrinsic_size = br.read_bool()?;
        if have_intrinsic_size {
            skip_size_header(br)?;
        }
        have_preview = br.read_bool()?;
        if have_preview {
            return Err(Error::Unsupported(
                "jxl: preview header parsing not yet implemented".into(),
            ));
        }
        have_animation = br.read_bool()?;
        if have_animation {
            return Err(Error::Unsupported(
                "jxl: animation header parsing not yet implemented".into(),
            ));
        }
    }

    let bit_depth_all_default = br.read_bool()?;
    let bit_depth = if bit_depth_all_default {
        BitDepth::DEFAULT_U8
    } else {
        parse_bit_depth(br)?
    };

    let _modular_16_bit_buffer_sufficient = br.read_bool()?;

    let num_extra_channels = br.read_u32([
        U32Dist::Val(0),
        U32Dist::Val(1),
        U32Dist::BitsOffset(4, 2),
        U32Dist::BitsOffset(12, 1),
    ])?;

    // ExtraChannelInfo, ColorEncoding and ToneMapping follow in the
    // bitstream; none of them are required to expose xyb_encoded/size so
    // we defer their decoding to the real pixel pipeline.
    let xyb_encoded = false;

    Ok(ImageMetadata {
        all_default: false,
        orientation,
        have_preview,
        have_animation,
        have_intrinsic_size,
        xyb_encoded,
        bit_depth,
        num_extra_channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_signature(codestream_bytes: &[u8]) -> Vec<u8> {
        let mut v = vec![0xFF, 0x0A];
        v.extend_from_slice(codestream_bytes);
        v
    }

    /// Encoder helper: append `value` as `n` LSB-first bits into a bit buffer.
    struct BitWriter {
        out: Vec<u8>,
        bit_pos: u8,
    }

    impl BitWriter {
        fn new() -> Self { Self { out: Vec::new(), bit_pos: 0 } }

        fn write_bits(&mut self, value: u32, n: u32) {
            for i in 0..n {
                let bit = ((value >> i) & 1) as u8;
                if self.bit_pos == 0 {
                    self.out.push(0);
                }
                let last = self.out.len() - 1;
                self.out[last] |= bit << self.bit_pos;
                self.bit_pos = (self.bit_pos + 1) % 8;
            }
        }

        fn finish(self) -> Vec<u8> { self.out }
    }

    #[test]
    fn parses_small_square_8x8() {
        // small=1; ysize_div8_minus_1=0 (→ 8); ratio=1 (1:1, xsize=ysize);
        // all_default=1.
        let mut bw = BitWriter::new();
        bw.write_bits(1, 1);  // small
        bw.write_bits(0, 5);  // ysize_div8_minus_1 → ysize = 8
        bw.write_bits(1, 3);  // ratio = 1 (square)
        bw.write_bits(1, 1);  // ImageMetadata.all_default
        let bits = bw.finish();
        let full = with_signature(&bits);
        let h = parse_headers(&full).unwrap();
        assert_eq!(h.signature, Signature::RawCodestream);
        assert_eq!(h.size.width, 8);
        assert_eq!(h.size.height, 8);
        assert_eq!(h.size.ratio, 1);
        assert!(h.size.small);
        assert!(h.metadata.all_default);
        assert_eq!(h.metadata.bit_depth, BitDepth::DEFAULT_U8);
    }

    #[test]
    fn parses_small_16x24_explicit_ratio() {
        // small=1; height=24 → ysize_div8_minus_1 = 2; ratio=0 (explicit);
        // width=16 → xsize_div8_minus_1 = 1; all_default=1.
        let mut bw = BitWriter::new();
        bw.write_bits(1, 1);
        bw.write_bits(2, 5);
        bw.write_bits(0, 3);
        bw.write_bits(1, 5);
        bw.write_bits(1, 1);
        let full = with_signature(&bw.finish());
        let h = parse_headers(&full).unwrap();
        assert_eq!((h.size.width, h.size.height), (16, 24));
        assert_eq!(h.size.ratio, 0);
    }

    #[test]
    fn parses_large_non_multiple_of_8() {
        // small=0; ysize via selector 0: BitsOffset(9,1) → raw bits = 99, ysize = 100.
        // ratio=0, xsize via same selector = 149.
        let mut bw = BitWriter::new();
        bw.write_bits(0, 1);
        bw.write_bits(0, 2);
        bw.write_bits(99, 9);
        bw.write_bits(0, 3);
        bw.write_bits(0, 2);
        bw.write_bits(149, 9);
        bw.write_bits(1, 1);
        let full = with_signature(&bw.finish());
        let h = parse_headers(&full).unwrap();
        assert_eq!((h.size.width, h.size.height), (150, 100));
        assert!(!h.size.small);
    }

    #[test]
    fn parses_implicit_aspect_ratio_16_9() {
        // small=1, ysize = 72 → div8_minus_1 = 8, ratio=5 (16:9) → xsize=128.
        let mut bw = BitWriter::new();
        bw.write_bits(1, 1);
        bw.write_bits(8, 5);
        bw.write_bits(5, 3);
        bw.write_bits(1, 1);
        let full = with_signature(&bw.finish());
        let h = parse_headers(&full).unwrap();
        assert_eq!((h.size.width, h.size.height), (128, 72));
        assert_eq!(h.size.ratio, 5);
    }

    #[test]
    fn parses_metadata_non_default_u16() {
        // small=1, 8x8 square, all_default=0, extra_fields=0,
        // bit_depth_all_default=0 → floating_point=0, bits_per_sample selector=3
        // (BitsOffset(6,1), bits=15 → 16), modular_16_bit=1, num_extra_channels=0.
        let mut bw = BitWriter::new();
        bw.write_bits(1, 1);
        bw.write_bits(0, 5);
        bw.write_bits(1, 3);
        bw.write_bits(0, 1);  // all_default
        bw.write_bits(0, 1);  // extra_fields
        bw.write_bits(0, 1);  // bit_depth all_default
        bw.write_bits(0, 1);  // floating_point = false
        bw.write_bits(3, 2);  // selector 3 → BitsOffset(6,1)
        bw.write_bits(15, 6); // bits_per_sample = 15+1 = 16
        bw.write_bits(1, 1);  // modular_16_bit_buffer_sufficient
        bw.write_bits(0, 2);  // num_extra_channels: selector 0 → Val(0)
        let full = with_signature(&bw.finish());
        let h = parse_headers(&full).unwrap();
        assert!(!h.metadata.all_default);
        assert_eq!(h.metadata.bit_depth.bits_per_sample, 16);
        assert!(!h.metadata.bit_depth.floating_point);
        assert_eq!(h.metadata.orientation, 1);
        assert_eq!(h.metadata.num_extra_channels, 0);
    }

    #[test]
    fn rejects_missing_signature() {
        let err = parse_headers(&[0xDE, 0xAD]).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }
}
