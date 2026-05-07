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
//! * GlobalModular wiring (C.4.8) so the FDIS path can actually drive
//!   the Modular sub-bitstream end-to-end.
//! * Squeeze inverse transform (I.3) for multi-resolution Modular
//!   images.
//! * VarDCT-path decoder (variable-size DCT + LF/HF, Chroma-from-Luma,
//!   Gaborish smoothing, EPF) — out of scope for this round.
//! * MABrotli / MAANS entropy coders (the 2019 committee draft's
//!   `entropy_coder` ∈ {1, 2}); only MABEGABRAC (`entropy_coder == 0`)
//!   is implemented today.
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
//! * Round 3 (planned): GlobalModular wiring + cjxl fixture decode.
//!
//! ## Round-1 (2024-spec) status (this commit)
//!
//! `make_decoder` returns a live decoder ([`JxlDecoder`]) that handles
//! the simplest end-to-end Modular bitstreams:
//!
//! * Raw codestream OR ISOBMFF wrapping.
//! * Grey (1 plane) OR RGB (3 planes), 8 bits per sample (integer).
//! * Single-group, single-pass frame (`num_groups == 1 &&
//!   num_passes == 1`).
//! * `nb_transforms` arbitrary at the *parse* level (TransformInfo
//!   bundles per H.7 are decoded for any nb_transforms > 0); inverse
//!   application of Palette / Squeeze defers to round 2 with a clean
//!   `Error::Unsupported` exit point. RCT (no channel-list change)
//!   passes through the layout step.
//! * Multi-leaf MA tree evaluated end-to-end (decision-node
//!   `property[k] > value` traversal per H.4.1).
//! * `use_global_tree` is honoured.
//! * No Patches / Splines / NoiseParameters — those are LfGlobal
//!   features round 2 will land alongside the VarDCT path.
//! * No ICC profile (Annex E.4).
//! * Predictor 6 (Annex H.5 Self-correcting) only resolved at the
//!   (0, 0) origin; full WP defers to round 2.
//!
//! The acceptance fixture for round 1 is `pixel-1x1.jxl` (1×1 RGB
//! lossless, 22 B): decodes to R=255 G=0 B=0 matching its
//! `expected.png`.
//!
//! Anything outside this envelope returns
//! [`Error::Unsupported`](oxideav_core::Error::Unsupported) at the
//! relevant gate point. Wider coverage (VarDCT, Squeeze inverse,
//! Palette inverse, ICC, full WP predictor 6) lands in round 2+.
//!
//! ## Round-6 (2024-spec) additions
//!
//! * **Annex E.4 ICC profile decode** ([`icc`]): the 7-state-equivalent
//!   entropy-coded ICC byte stream (41 pre-clustered distributions +
//!   `IccContext(i, b1, b2)` 41-context function) is decoded into the
//!   final ICC profile bytes per E.4.3 (header), E.4.4 (tag list) and
//!   E.4.5 (main content). When `metadata.colour_encoding.want_icc ==
//!   true` the bit-position is now correctly advanced past the ICC
//!   stream rather than failing with `Error::Unsupported` outright;
//!   the decoded bytes are validated for the "acsp" magic at offset 36
//!   but are not yet propagated to `oxideav_core::VideoFrame` (which
//!   has no ICC slot in 0.1.x).
//! * **G.2 LfGroup / G.4 PassGroup type scaffolding** ([`lf_group`],
//!   [`pass_group`]): typed bundles + per-group rectangle geometry +
//!   `(minshift, maxshift)` computation per pass. Per-LfGroup and
//!   per-PassGroup decode itself is not yet wired (round-7 follow-up
//!   gated on the GlobalModular `nb_meta_channels`-aware refactor —
//!   see `lf_group` crate-level docs).
//! * Multi-LfGroup / multi-group / multi-pass / VarDCT frames fail
//!   with precise round-7-targeting error messages instead of the
//!   round-3 generic "TOC with N entries" rejection.

pub mod abrac;
pub mod ans;
pub mod begabrac;
pub mod bitreader;
pub mod container;
pub mod extensions;
pub mod frame_header;
pub mod global_modular;
pub mod icc;
pub mod lf_global;
pub mod lf_group;
pub mod matree;
pub mod metadata;
pub mod metadata_fdis;
pub mod modular;
pub mod modular_fdis;
pub mod pass_group;
pub mod predictors;
pub mod toc;

pub use container::{detect, extract_codestream, Signature};
pub use metadata::{parse_headers, BitDepth, Headers, ImageMetadata, SizeHeader};

use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Error, Result};
use oxideav_core::{
    CodecInfo, CodecRegistry, Decoder, Encoder, Frame, Packet, RuntimeContext, VideoFrame,
    VideoPlane,
};

use crate::bitreader::BitReader;
use crate::frame_header::{FrameDecodeParams, FrameHeader};
use crate::lf_global::LfGlobal;
use crate::metadata_fdis::{ColourSpace, ImageMetadataFdis, SizeHeaderFdis};
use crate::toc::Toc;

/// Public codec id string. Matches the aggregator feature name `jpegxl`.
pub const CODEC_ID_STR: &str = "jpegxl";

/// Register the JPEG XL decoder stub into the supplied
/// [`CodecRegistry`]. The encoder slot is intentionally left
/// unregistered: the crate is decoder-side only and currently
/// retired-pending-cleanroom (see crate-level docs).
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("jpegxl_headers_only")
        .with_lossy(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder),
    );
}

/// Unified entry point: install the JPEG XL codec into a
/// [`RuntimeContext`].
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

oxideav_core::register!("jpegxl", register);

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let codec_id = params.codec_id.clone();
    Ok(Box::new(JxlDecoder {
        codec_id,
        pending: None,
        eof: false,
    }))
}

/// Round-1 (2024-spec) JXL decoder. Drives `decode_one_frame` per packet.
///
/// Limitations (round 1):
/// * Only Modular-encoded frames (kModular, not kVarDCT).
/// * Grey (1ch) OR RGB (3ch) only — XYB / YCbCr defer.
/// * Single-group, single-pass frames.
/// * Inverse Palette / Squeeze transforms defer (parsing + RCT
///   layout pass-through is wired).
/// * Predictor 6 (Self-correcting) only at (0, 0) origin.
/// * No Patches / Splines / Noise / ICC profile.
///
/// Anything outside this envelope returns `Error::Unsupported` from a
/// well-defined point in the bitstream rather than panicking.
struct JxlDecoder {
    codec_id: CodecId,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for JxlDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "jxl decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let vf = decode_one_frame(&pkt.data, pkt.pts)?;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Decode the ICC stream (Annex E.4) at the current bit position and
/// return the resulting ICC profile bytes.
///
/// The caller has already verified that
/// `metadata.colour_encoding.want_icc == true`. Round 6 wires the
/// decode end-to-end; the returned bytes are valid per E.4.3..E.4.5 if
/// `Ok`. The function also performs a minimal ICC.1 sanity check —
/// for outputs >= 40 bytes the magic "acsp" must be at offset 36 —
/// because the predicted-header rule in E.4.3 forces those bytes when
/// the encoded delta is zero, but a malformed delta could shift them.
fn decode_icc_stream_at(br: &mut BitReader<'_>) -> Result<Vec<u8>> {
    let encoded = icc::decode_encoded_icc_stream(br)?;
    let profile = icc::reconstruct_icc_profile(&encoded)?;
    if profile.len() >= 40 && &profile[36..40] != b"acsp" {
        return Err(Error::InvalidData(format!(
            "JXL ICC: decoded profile lacks 'acsp' magic at offset 36 (got {:02X?})",
            &profile[36..40]
        )));
    }
    Ok(profile)
}

/// Decode the entire JXL packet (raw codestream OR ISOBMFF-wrapped) and
/// return the first frame as a [`VideoFrame`]. Round-3 envelope.
pub fn decode_one_frame(input: &[u8], pts: Option<i64>) -> Result<VideoFrame> {
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

fn decode_codestream(codestream: &[u8], pts: Option<i64>) -> Result<VideoFrame> {
    let mut br = BitReader::new(codestream);

    // 1. SizeHeader (FDIS A.3).
    let size = SizeHeaderFdis::read(&mut br)?;

    // 2. ImageMetadata (FDIS A.6).
    let metadata = ImageMetadataFdis::read(&mut br)?;

    // 3. ICC profile (Annex E.4) — round-6 lands the decoder. The
    //    decoded ICC bytes are validated (must contain "acsp" magic at
    //    offset 36 if length >= 40) but not currently propagated to
    //    `VideoFrame` because `oxideav_core::VideoFrame` has no ICC
    //    slot. The decode is still run because (a) it advances the
    //    bit reader past the ICC stream so subsequent FrameHeader /
    //    TOC parsing finds the right bit offset, and (b) it gives a
    //    direct `Error::InvalidData` if the codestream's ICC stream
    //    is malformed.
    if metadata.colour_encoding.want_icc {
        let _icc_bytes = decode_icc_stream_at(&mut br)?;
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
    //    frame data. Round 6 still only handles that case but reports
    //    the precise reason (multi-group vs multi-pass vs VarDCT) so
    //    round-7 can target each gate individually.
    let num_groups = fh.num_groups();
    let num_lf_groups = fh.num_lf_groups();
    if num_lf_groups > 1 {
        return Err(crate::lf_group::unsupported_multi_lf_group_error(
            num_lf_groups,
            fh.encoding,
        ));
    }
    if num_groups > 1 || fh.passes.num_passes > 1 {
        return Err(crate::pass_group::unsupported_multi_group_error(
            num_groups,
            fh.passes.num_passes,
        ));
    }
    if toc.entries.len() != 1 {
        return Err(Error::Unsupported(format!(
            "jxl decoder (round 6): TOC with {} entries unexpected for single-group/single-pass frame",
            toc.entries.len()
        )));
    }
    // Diagnostic on unhandled features.
    if fh.encoding != crate::frame_header::Encoding::Modular {
        return Err(Error::Unsupported(format!(
            "jxl decoder (round 6): encoding {:?} not supported (Modular only — VarDCT round-8+ work)",
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

    // 9. Map the decoded modular image to a VideoFrame.
    //
    // Round-1 (2024-spec) supports:
    //   - Grey colour_space (single channel, 1 plane)
    //   - RGB colour_space (3 channels → 3 planes in R/G/B order)
    //   - 8-bit integer bit depth
    //
    // Other colour spaces (XYB, YCbCr) and float bit depths fall in
    // later rounds.
    if metadata.bit_depth.float_sample {
        return Err(Error::Unsupported(
            "jxl decoder (round 1): float bit depth not supported".into(),
        ));
    }
    if metadata.bit_depth.bits_per_sample != 8 {
        return Err(Error::Unsupported(format!(
            "jxl decoder (round 1): bits_per_sample {} not supported (8 only)",
            metadata.bit_depth.bits_per_sample
        )));
    }
    let img = lf_global.global_modular.image;
    let n_chans = img.channels.len();
    let expected_chans = match metadata.colour_encoding.colour_space {
        ColourSpace::Grey => 1,
        ColourSpace::Rgb => 3,
        _ => {
            return Err(Error::Unsupported(format!(
                "jxl decoder (round 1): colour_space {:?} not supported (Grey/RGB only)",
                metadata.colour_encoding.colour_space
            )));
        }
    };
    if n_chans != expected_chans {
        return Err(Error::Unsupported(format!(
            "jxl decoder (round 1): {} channels but colour_space wants {}",
            n_chans, expected_chans
        )));
    }
    let mut planes: Vec<VideoPlane> = Vec::with_capacity(n_chans);
    for (i, ch_data) in img.channels.iter().enumerate() {
        let desc = img.descs[i];
        let w = desc.width as usize;
        let h = desc.height as usize;
        let mut bytes = Vec::with_capacity(w * h);
        for &v in ch_data.iter() {
            bytes.push(v.clamp(0, 255) as u8);
        }
        planes.push(VideoPlane {
            stride: w,
            data: bytes,
        });
        // Sanity check height while we're here.
        debug_assert_eq!(planes[i].data.len(), w * h);
    }
    Ok(VideoFrame { pts, planes })
}

/// FDIS-side `Headers` returned by [`probe_fdis`]. Mirrors the
/// committee-draft [`Headers`] but uses the FDIS bundle types.
#[derive(Debug, Clone)]
pub struct HeadersFdis {
    pub signature: container::Signature,
    pub size: SizeHeaderFdis,
    pub metadata: ImageMetadataFdis,
}

/// FDIS-side probe: parse SizeHeader + full A.6 ImageMetadata. Falls
/// back to the committee-draft probe if the FDIS path errors (so that
/// container detection still works on edge cases the committee-draft
/// path tolerates).
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
    fn decoder_factory_returns_live_decoder() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let dec = ctx
            .codecs
            .first_decoder(&params)
            .expect("expected live decoder");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
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
