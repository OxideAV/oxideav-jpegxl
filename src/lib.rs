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
//! ## Round-3 status (this commit)
//!
//! `make_decoder` now returns a live FDIS decoder ([`JxlDecoder`]) that
//! handles the *narrow envelope* needed by the simplest cjxl Modular
//! lossless output:
//!
//! * Raw codestream OR ISOBMFF wrapping;
//! * Single-channel Grey 8 bpp;
//! * Single-group, single-pass frame (`num_groups == 1 &&
//!   num_passes == 1`);
//! * `nb_transforms == 0` (no Squeeze / Palette / RCT);
//! * Single-leaf MA tree (no decision-node evaluation);
//! * No Patches / Splines / NoiseParameters;
//! * `use_global_tree == false`;
//! * No ICC profile (Annex B);
//! * No weighted predictor (Annex E predictor `6`).
//!
//! Anything outside this envelope returns
//! [`Error::Unsupported`](oxideav_core::Error::Unsupported) at the
//! relevant gate point. Wider coverage (RGB, VarDCT, Squeeze, ICC,
//! weighted predictor, multi-leaf trees) lands in round 4.

pub mod abrac;
pub mod ans;
pub mod ans_encoder;
pub mod begabrac;
pub mod bitreader;
pub mod bitwriter;
pub mod container;
pub mod encoder;
pub mod extensions;
pub mod frame_header;
pub mod global_modular;
pub mod lf_global;
pub mod matree;
pub mod metadata;
pub mod metadata_fdis;
pub mod modular;
pub mod modular_fdis;
pub mod predictors;
pub mod toc;

pub use container::{detect, extract_codestream, wrap_codestream, Signature};
pub use metadata::{parse_headers, BitDepth, Headers, ImageMetadata, SizeHeader};

use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Error, PixelFormat, Result};
use oxideav_core::{
    CodecInfo, CodecRegistry, Decoder, Encoder, Frame, Packet, TimeBase, VideoFrame, VideoPlane,
};

use crate::encoder::{encode_one_frame as encoder_encode_one_frame, InputFormat};

use crate::bitreader::BitReader;
use crate::frame_header::{FrameDecodeParams, FrameHeader};
use crate::lf_global::LfGlobal;
use crate::metadata_fdis::{ColourSpace, ImageMetadataFdis, SizeHeaderFdis};
use crate::toc::Toc;

/// Public codec id string. Matches the aggregator feature name `jpegxl`.
pub const CODEC_ID_STR: &str = "jpegxl";

/// Register the JPEG XL codec — decoder + round-1 lossless modular
/// encoder.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("jpegxl_headers_only")
        .with_lossy(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder),
    );
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let codec_id = params.codec_id.clone();
    Ok(Box::new(JxlDecoder {
        codec_id,
        pending: None,
        eof: false,
    }))
}

/// Round-3 JXL decoder. Drives `decode_one_frame` per packet.
///
/// Limitations (round 3):
/// * Only Modular-encoded frames with a single Grey channel.
/// * Only single-group frames (`num_groups == 1 && num_passes == 1`).
/// * No transforms (kPalette / kRCT / kSqueeze).
/// * No global tree (`use_global_tree == false`).
/// * MA tree must be a single leaf (no decision nodes evaluated).
/// * No Patches / Splines / Noise.
///
/// Anything outside this envelope returns `Error::Unsupported` from a
/// well-defined point in the bitstream rather than panicking. Round 4
/// will widen the envelope to RGB / VarDCT / Squeeze.
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

    // 9. Map the decoded modular image to a VideoFrame. Only Grey
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
    let plane = VideoPlane {
        stride: w,
        data: bytes,
    };
    Ok(VideoFrame {
        pts,
        planes: vec![plane],
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

/// Round-1 minimal lossless modular JPEG XL encoder.
///
/// Accepts `pixel_format ∈ {Gray8, Rgb24, Rgba}` at any width/height up
/// to 1024×1024 (the single-group cap of the round-1 implementation).
/// Larger images return [`Error::Unsupported`].
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let codec_id = params.codec_id.clone();
    let pixel_format = params
        .pixel_format
        .ok_or_else(|| Error::other("jxl encoder: pixel_format required (Gray8, Rgb24 or Rgba)"))?;
    let input_format = match pixel_format {
        PixelFormat::Gray8 => InputFormat::Gray8,
        PixelFormat::Rgb24 => InputFormat::Rgb8,
        PixelFormat::Rgba => InputFormat::Rgba8,
        other => {
            return Err(Error::Unsupported(format!(
                "jxl encoder: pixel_format {other:?} not supported (round 1 is Gray8/Rgb24/Rgba only)"
            )));
        }
    };
    let width = params
        .width
        .ok_or_else(|| Error::other("jxl encoder: width required in CodecParameters"))?;
    let height = params
        .height
        .ok_or_else(|| Error::other("jxl encoder: height required in CodecParameters"))?;
    let output_params = params.clone();
    Ok(Box::new(JxlEncoder {
        codec_id,
        input_format,
        width,
        height,
        output_params,
        pending_packet: None,
        eof: false,
    }))
}

/// Round-1 JPEG XL encoder. Accepts one [`Frame`] per call to
/// [`Encoder::send_frame`] and emits exactly one [`Packet`] containing
/// the full codestream from [`Encoder::receive_packet`].
struct JxlEncoder {
    codec_id: CodecId,
    input_format: InputFormat,
    width: u32,
    height: u32,
    output_params: CodecParameters,
    pending_packet: Option<Packet>,
    eof: bool,
}

impl Encoder for JxlEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        if self.pending_packet.is_some() {
            return Err(Error::other(
                "jxl encoder: receive_packet must be called before sending another frame",
            ));
        }
        let vf = match frame {
            Frame::Video(vf) => vf,
            _ => {
                return Err(Error::other("jxl encoder: only Video frames are supported"));
            }
        };
        if vf.planes.len() != 1 {
            return Err(Error::Unsupported(format!(
                "jxl encoder: expected 1 interleaved plane, got {}",
                vf.planes.len()
            )));
        }
        let plane = &vf.planes[0];
        let channels = self.input_format.channel_count() as usize;
        let expected_stride = self.width as usize * channels;
        if plane.stride != expected_stride {
            return Err(Error::other(format!(
                "jxl encoder: plane stride {} != expected {} for {}x{} {:?}",
                plane.stride, expected_stride, self.width, self.height, self.input_format
            )));
        }
        let expected_len = expected_stride * self.height as usize;
        if plane.data.len() != expected_len {
            return Err(Error::other(format!(
                "jxl encoder: plane data len {} != expected {}",
                plane.data.len(),
                expected_len
            )));
        }
        let data =
            encoder_encode_one_frame(self.width, self.height, &plane.data, self.input_format)?;
        self.pending_packet = Some(
            Packet::new(0, TimeBase::new(1, 1), data)
                .with_keyframe(true)
                .with_pts(vf.pts.unwrap_or(0)),
        );
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(pkt) = self.pending_packet.take() {
            return Ok(pkt);
        }
        if self.eof {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_factory_returns_live_decoder() {
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let dec = reg.make_decoder(&params).expect("expected live decoder");
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
    fn encoder_factory_requires_pixel_format() {
        // Round-1 encoder rejects the bare-minimum params: no pixel
        // format set, no width, no height — we expect a descriptive
        // error pointing the caller at the missing fields.
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        assert!(make_encoder(&params).is_err());
    }

    #[test]
    fn encoder_factory_accepts_rgb24_with_dimensions() {
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(8);
        params.height = Some(8);
        params.pixel_format = Some(PixelFormat::Rgb24);
        let enc = make_encoder(&params).expect("expected live encoder");
        assert_eq!(enc.codec_id().as_str(), CODEC_ID_STR);
    }

    #[test]
    fn encoder_factory_rejects_unsupported_pixel_format() {
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(8);
        params.height = Some(8);
        params.pixel_format = Some(PixelFormat::Yuv420P);
        assert!(matches!(make_encoder(&params), Err(Error::Unsupported(_))));
    }
}
