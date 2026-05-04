//! JPEG XL (JXL) codec — decoder-side header parsing.
//!
//! JPEG XL is ISO/IEC 18181 (final specification 2022). It supersedes
//! classic JPEG with a modal design that separates a "VarDCT" path
//! (variable-size DCT + LF/HF subbands, quality-competitive with AVIF
//! and modern JPEG) from a "Modular" path (grid-of-pixels predictor +
//! MA-tree range coder, strong at lossless + non-photo material).
//!
//! This crate currently ships:
//!
//! * Container + signature detection for both JXL wrappings:
//!   raw codestream (`FF 0A`) and ISOBMFF-wrapped
//!   (`00 00 00 0C 4A 58 4C 20 0D 0A 87 0A`), including extraction of
//!   the codestream from `jxlc` / `jxlp` boxes.
//! * An LSB-first [`bitreader::BitReader`] matching the reference
//!   bit packing used by the codestream.
//! * Parsing of the codestream preamble: [`metadata::SizeHeader`] and the
//!   [`metadata::ImageMetadata`] fields up to `num_extra_channels`
//!   (bit depth, orientation, preview/animation flags). Fuller
//!   ColorEncoding + ToneMapping decoding is deferred.
//! * Modular sub-bitstream pixel decode (per the 2019 committee draft,
//!   Annexes C.9 + D.7), made of:
//!   - [`abrac::Abrac`] — the bit-level adaptive range coder (D.7);
//!   - [`begabrac::Begabrac`] — bounded-Exp-Golomb integer coder over a
//!     known signed range (D.7.1);
//!   - [`matree::MaTree`] — the meta-adaptive decision tree that picks
//!     a per-context BEGABRAC for each pixel (D.7.2 / D.7.3);
//!   - [`predictors`] — the five named pixel predictors (Zero, Average,
//!     Gradient, Left, Top) from C.9.3.1;
//!   - [`modular`] — the channel header parser plus the per-pixel
//!     property + predictor + entropy decode loop.
//!
//! Programs that only need probe-level information (dimensions, bit
//! depth) should call [`probe`] directly; programs that want to drive
//! the per-channel Modular decode end-to-end should instantiate
//! [`modular::decode_single_channel`] against a hand-built fixture
//! (unit tests in `modular` show the expected wire format).
//!
//! ## FDIS 18181-1:2021 layer
//!
//! In addition to the committee-draft pipeline above, the FDIS layer
//! is being built up additively across rounds:
//!
//! * Round 1: [`ans`] — FDIS Annex D entropy decoder (prefix codes,
//!   ANS, distribution clustering, hybrid integer coding).
//! * Round 2: [`extensions`] — A.5 Extensions; [`metadata_fdis`] —
//!   full A.6 ImageMetadata refresh including ColorEncoding,
//!   ToneMapping, ExtraChannelInfo, AnimationHeader, OpsinInverseMatrix,
//!   PreviewHeader; [`frame_header`] — C.2 FrameHeader bundle including
//!   Passes, BlendingInfo, RestorationFilter; [`toc`] — C.3 TOC with
//!   Lehmer-code permutation decoder driven by the round-1 ANS layer;
//!   [`ans::cluster::read_general_clustering`] — D.3.5 general path.
//! * Round 3 onwards: GlobalModular wiring + cjxl fixture decode.
//!
//! ## Standalone vs registry-integrated
//!
//! The crate's default `registry` Cargo feature pulls in `oxideav-core`
//! and exposes the [`Decoder`](oxideav_core::Decoder) /
//! [`Encoder`](oxideav_core::Encoder) trait surface plus a
//! [`registry::register`] entry point. Disable the feature
//! (`default-features = false`) for an `oxideav-core`-free build that
//! still exposes the standalone [`decode_one_frame`] /
//! [`encoder::encode_one_frame`] API plus the underlying `container` /
//! `metadata` / `metadata_fdis` / `frame_header` / `toc` / `lf_global`
//! modules and the crate-local [`JxlImage`] / [`JxlError`] types.

pub mod abrac;
pub mod ans;
pub mod ans_encoder;
pub mod begabrac;
pub mod bitreader;
pub mod bitwriter;
pub mod container;
pub mod encoder;
pub mod error;
pub mod extensions;
pub mod frame_header;
pub mod global_modular;
pub mod image;
pub mod lf_global;
pub mod matree;
pub mod metadata;
pub mod metadata_fdis;
pub mod modular;
pub mod modular_fdis;
pub mod predictors;
pub mod toc;
pub mod transforms;

#[cfg(feature = "registry")]
pub mod registry;

pub use container::{detect, extract_codestream, wrap_codestream, Signature};
pub use error::{JxlError, Result};
pub use image::{JxlImage, JxlPixelFormat, JxlPlane};
pub use metadata::{parse_headers, BitDepth, Headers, ImageMetadata, SizeHeader};

use crate::bitreader::BitReader;
use crate::error::JxlError as Error;
use crate::frame_header::{FrameDecodeParams, FrameHeader};
use crate::lf_global::LfGlobal;
use crate::metadata_fdis::{ColourSpace, ImageMetadataFdis, SizeHeaderFdis};
use crate::toc::Toc;

/// Public codec id string. Matches the aggregator feature name `jpegxl`.
pub const CODEC_ID_STR: &str = "jpegxl";

// Registry-gated re-exports — the framework integration surface
// (Decoder/Encoder traits, `register` entry point, `JxlDecoder` /
// `JxlEncoder` wrappers) lives behind the default-on `registry`
// feature so image-library callers can build the crate without
// dragging in `oxideav-core`.
#[cfg(feature = "registry")]
pub use registry::{make_decoder, make_encoder, register, JxlDecoder, JxlEncoder};

/// Decode the entire JXL packet (raw codestream OR ISOBMFF-wrapped) and
/// return the first frame as a [`JxlImage`]. Round-3 envelope.
pub fn decode_one_frame(input: &[u8], pts: Option<i64>) -> Result<JxlImage> {
    let sig = container::detect(input)
        .ok_or_else(|| Error::InvalidData("jxl decoder: no JXL signature".into()))?;
    match sig {
        container::Signature::RawCodestream => decode_codestream(&input[2..], pts),
        container::Signature::Isobmff => {
            let codestream_owned = container::extract_codestream(input)?;
            decode_codestream(&codestream_owned, pts)
        }
    }
}

fn decode_codestream(codestream: &[u8], pts: Option<i64>) -> Result<JxlImage> {
    let mut br = BitReader::new(codestream);

    // 1. SizeHeader (FDIS A.3).
    let size = SizeHeaderFdis::read(&mut br)?;

    // 2. ImageMetadata (FDIS A.6).
    let metadata = ImageMetadataFdis::read(&mut br)?;

    // 3. ICC profile is gated on `metadata.colour_encoding.want_icc`. We
    //    reject any frame that wants an ICC profile in round 3 — the
    //    Annex B decoder is not wired.
    if metadata.colour_encoding.want_icc {
        return Err(Error::Unsupported(
            "jxl decoder (round 3): want_icc=true (Annex B ICC stream) not yet wired".into(),
        ));
    }

    // 4. Byte-align before frame data per FDIS 6.3.
    br.pu0()?;

    // 5. FrameHeader (FDIS C.2).
    let fh_params = FrameDecodeParams {
        xyb_encoded: metadata.xyb_encoded,
        num_extra_channels: metadata.num_extra_channels,
        have_animation: metadata.have_animation,
        have_animation_timecodes: metadata
            .animation
            .map(|a| a.have_timecodes)
            .unwrap_or(false),
        image_width: size.width,
        image_height: size.height,
    };
    let fh = FrameHeader::read(&mut br, &fh_params)?;

    // 6. TOC (FDIS C.3) — entries byte-aligned per spec.
    let toc = Toc::read(&mut br, &fh)?;

    // 7. Single-group frames have a single TOC entry containing all
    //    frame data. Round 3 only handles that case.
    if toc.entries.len() != 1 {
        return Err(Error::Unsupported(format!(
            "jxl decoder (round 3): TOC with {} entries; only single-group frames supported",
            toc.entries.len()
        )));
    }
    // Diagnostic on unhandled features.
    if fh.encoding != crate::frame_header::Encoding::Modular {
        return Err(Error::Unsupported(format!(
            "jxl decoder (round 3): encoding {:?} not supported (Modular only)",
            fh.encoding
        )));
    }
    if fh.width == 0 || fh.height == 0 {
        return Err(Error::InvalidData("jxl decoder: zero-dim frame".into()));
    }

    // 8. LfGlobal (FDIS C.4) — for a single-group Modular frame the TOC
    //    points at one section that begins with LfGlobal and contains
    //    nothing else (no LfGroup / HfGlobal / PassGroup follow).
    let lf_global = LfGlobal::read(&mut br, &fh, &metadata)?;

    // 9. Map the decoded modular image to a JxlImage. Only Grey
    //    8-bit-per-sample is wired in round 3.
    if metadata.colour_encoding.colour_space != ColourSpace::Grey {
        return Err(Error::Unsupported(format!(
            "jxl decoder (round 3): colour_space {:?} not supported (Grey only)",
            metadata.colour_encoding.colour_space
        )));
    }
    if metadata.bit_depth.float_sample {
        return Err(Error::Unsupported(
            "jxl decoder (round 3): float bit depth not supported".into(),
        ));
    }
    if metadata.bit_depth.bits_per_sample != 8 {
        return Err(Error::Unsupported(format!(
            "jxl decoder (round 3): bits_per_sample {} not supported (8 only)",
            metadata.bit_depth.bits_per_sample
        )));
    }
    let img = lf_global.global_modular.image;
    if img.channels.len() != 1 {
        return Err(Error::Unsupported(format!(
            "jxl decoder (round 3): {} channels not supported (1 only)",
            img.channels.len()
        )));
    }
    let desc = img.descs[0];
    let w = desc.width as usize;
    let h = desc.height as usize;
    let mut bytes = Vec::with_capacity(w * h);
    for &v in img.channels[0].iter() {
        bytes.push(v.clamp(0, 255) as u8);
    }
    let plane = JxlPlane {
        stride: w,
        data: bytes,
    };
    Ok(JxlImage {
        width: fh.width,
        height: fh.height,
        pixel_format: JxlPixelFormat::Gray8,
        planes: vec![plane],
        pts,
    })
}

/// FDIS-side `Headers` returned by [`probe_fdis`]. Mirrors the
/// committee-draft [`Headers`] but uses the FDIS bundle types.
#[derive(Debug, Clone)]
pub struct HeadersFdis {
    pub signature: container::Signature,
    pub size: SizeHeaderFdis,
    pub metadata: ImageMetadataFdis,
}

/// FDIS-side probe: parse SizeHeader + full A.6 ImageMetadata.
pub fn probe_fdis(input: &[u8]) -> Result<HeadersFdis> {
    let signature = container::detect(input)
        .ok_or_else(|| Error::InvalidData("jxl probe: no JXL signature".into()))?;
    match signature {
        container::Signature::RawCodestream => probe_fdis_codestream(&input[2..], signature),
        container::Signature::Isobmff => {
            let codestream_owned = container::extract_codestream(input)?;
            probe_fdis_codestream(&codestream_owned, signature)
        }
    }
}

fn probe_fdis_codestream(
    codestream: &[u8],
    signature: container::Signature,
) -> Result<HeadersFdis> {
    let mut br = BitReader::new(codestream);
    let size = SizeHeaderFdis::read(&mut br)?;
    let metadata = ImageMetadataFdis::read(&mut br)?;
    Ok(HeadersFdis {
        signature,
        size,
        metadata,
    })
}

/// Inspect a JXL file (raw codestream or ISOBMFF-wrapped) and return the
/// signature type + parsed `SizeHeader` + `ImageMetadata` preamble.
///
/// This is the main API users can reach today: it covers identification,
/// dimensions and sample format without needing an actual decoder.
pub fn probe(input: &[u8]) -> Result<Headers> {
    parse_headers(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_rejects_non_jxl() {
        let err = probe(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]).unwrap_err();
        assert!(matches!(err, JxlError::InvalidData(_)));
    }

    #[test]
    fn probe_accepts_minimal_raw_codestream() {
        // small=1, 8x8 square (ratio=1), all_default=1 → 10 bits total.
        // LSB-first packing: byte0 holds bits 0..=7, byte1 holds bits 8..=9.
        // bit0=1, bits1..=5=0, bits6..=8=001 (ratio=1), bit9=1 (all_default)
        // → byte0 = 0b01000001 = 0x41, byte1 = 0b00000010 = 0x02.
        let input = [0xFF, 0x0A, 0x41, 0x02];
        let h = probe(&input).unwrap();
        assert_eq!(h.size.width, 8);
        assert_eq!(h.size.height, 8);
        assert!(h.metadata.all_default);
    }
}
