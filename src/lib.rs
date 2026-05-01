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
//! The integrated registered decoder is not yet wired: the registered
//! `make_decoder` reports [`Error::Unsupported`] because the surrounding
//! codestream framing (FrameHeader + TOC + frame-byte alignment) is not
//! yet wired to the per-channel path. Programs that only need
//! probe-level information (dimensions, bit depth) should call
//! [`probe`] directly; programs that want to drive the per-channel
//! Modular decode end-to-end should instantiate
//! [`modular::decode_single_channel`] against a hand-built fixture
//! (unit tests in `modular` show the expected wire format).
//!
//! Follow-up work (tracked for the eventual landing PR):
//!
//! * FrameHeader (C.2) + TOC (C.3) + the modular-image header
//!   (`max_extra_properties`, transforms) so that whole-image
//!   codestreams (not just isolated channels) decode.
//! * Squeeze inverse transform (I.3) for multi-resolution Modular
//!   images.
//! * VarDCT-path decoder (variable-size DCT + LF/HF, Chroma-from-Luma,
//!   Gaborish smoothing, EPF) — out of scope for this round.
//! * MABrotli / MAANS entropy coders (the 2019 committee draft's
//!   `entropy_coder` ∈ {1, 2}); only MABEGABRAC (`entropy_coder == 0`)
//!   is implemented today.
//!
//! ## FDIS 18181-1:2021 entropy layer
//!
//! In addition to the committee-draft pipeline above, the [`ans`]
//! module ships the FDIS 18181-1:2021 Annex D entropy decoder
//! (prefix codes, ANS, distribution clustering, hybrid integer
//! coding). It is **additive**: the registered `make_decoder` does
//! not yet drive it; future rounds will replace the committee-draft
//! Modular entry point with the FDIS path that wires through
//! FrameHeader + TOC into [`ans`].

pub mod abrac;
pub mod ans;
pub mod begabrac;
pub mod bitreader;
pub mod container;
pub mod matree;
pub mod metadata;
pub mod modular;
pub mod predictors;

pub use container::{detect, extract_codestream, Signature};
pub use metadata::{parse_headers, BitDepth, Headers, ImageMetadata, SizeHeader};

use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Error, Result};
use oxideav_core::{CodecInfo, CodecRegistry, Decoder, Encoder};

/// Public codec id string. Matches the aggregator feature name `jpegxl`.
pub const CODEC_ID_STR: &str = "jpegxl";

/// Register the JPEG XL decoder stub. The encoder slot is intentionally
/// left unregistered: the crate is decoder-side only.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("jpegxl_headers_only")
        .with_lossy(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder),
    );
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Err(Error::Unsupported("jxl decode not yet implemented".into()))
}

/// Inspect a JXL file (raw codestream or ISOBMFF-wrapped) and return the
/// signature type + parsed `SizeHeader` + `ImageMetadata` preamble.
///
/// This is the main API users can reach today: it covers identification,
/// dimensions and sample format without needing an actual decoder.
pub fn probe(input: &[u8]) -> Result<Headers> {
    parse_headers(input)
}

/// Encoder slot, always rejected. Exposed for completeness so callers
/// that wire an `Encoder` factory by codec id get a clean `Unsupported`
/// error instead of `CodecNotFound`.
pub fn make_encoder(_params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Err(Error::Unsupported(
        "jxl encode is out of scope for this crate".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_reports_unsupported() {
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        match reg.make_decoder(&params) {
            Err(Error::Unsupported(msg)) => {
                assert!(msg.contains("jxl decode not yet implemented"), "{msg}");
            }
            Err(other) => panic!("expected Error::Unsupported, got {other:?}"),
            Ok(_) => panic!("expected Error::Unsupported, got a live decoder"),
        }
    }

    #[test]
    fn probe_rejects_non_jxl() {
        let err = probe(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
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

    #[test]
    fn encoder_factory_rejects_cleanly() {
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        assert!(matches!(make_encoder(&params), Err(Error::Unsupported(_))));
    }
}
