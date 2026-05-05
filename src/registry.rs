//! `oxideav-core` integration: `Decoder` / `Encoder` trait impls,
//! `Frame` / `Error` conversions, and the [`register`] entry point.
//!
//! Gated behind the default-on `registry` Cargo feature. With the
//! feature off the rest of the crate still exposes the standalone
//! [`crate::decode_one_frame`] / [`crate::encoder::encode_one_frame`]
//! API plus the underlying `container` / `metadata` / `frame_header` /
//! ... modules and [`crate::JxlImage`] / [`crate::JxlError`] types —
//! none of which depend on `oxideav-core`.

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, ContainerRegistry,
    Decoder, Encoder, Error as CoreError, Frame, Packet, PixelFormat, Result as CoreResult,
    RuntimeContext, TimeBase, VideoFrame, VideoPlane,
};

use crate::encoder::{encode_one_frame as encoder_encode_one_frame, InputFormat};
use crate::error::JxlError;
use crate::image::{JxlImage, JxlPixelFormat};
use crate::{decode_one_frame, CODEC_ID_STR};

/// Convert a [`JxlError`] into the framework-shared `oxideav_core::Error`
/// so trait impls in this crate can use `?` on errors returned by the
/// framework-free decode/encode functions.
impl From<JxlError> for CoreError {
    fn from(e: JxlError) -> Self {
        match e {
            JxlError::InvalidData(s) => CoreError::InvalidData(s),
            JxlError::Unsupported(s) => CoreError::Unsupported(s),
            JxlError::Eof => CoreError::Eof,
            JxlError::NeedMore => CoreError::NeedMore,
            JxlError::Other(s) => CoreError::Other(s),
        }
    }
}

impl From<JxlPixelFormat> for PixelFormat {
    fn from(p: JxlPixelFormat) -> Self {
        match p {
            JxlPixelFormat::Gray8 => PixelFormat::Gray8,
            JxlPixelFormat::Rgb24 => PixelFormat::Rgb24,
            JxlPixelFormat::Rgba => PixelFormat::Rgba,
        }
    }
}

impl From<JxlImage> for Frame {
    fn from(img: JxlImage) -> Self {
        let planes = img
            .planes
            .into_iter()
            .map(|p| VideoPlane {
                stride: p.stride,
                data: p.data,
            })
            .collect();
        Frame::Video(VideoFrame {
            pts: img.pts,
            planes,
        })
    }
}

/// Register the JPEG XL codec — decoder + round-1 lossless modular
/// encoder.
pub fn register_codecs(reg: &mut CodecRegistry) {
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

/// Unified registration entry point: install both the JPEG XL codec
/// factories and the `.jxl` extension hint into a [`RuntimeContext`].
///
/// This is the preferred entry point for new code — it matches the
/// convention every sibling crate now follows. Direct callers that
/// only need one of the two sub-registries can keep using
/// [`register_codecs`] / [`register_containers`].
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
    register_containers(&mut ctx.containers);
}

oxideav_core::register!("jpegxl", register);

/// Register the `.jxl` file extension so the framework's container
/// resolver can route JPEG XL files to this codec.
///
/// This crate does not yet ship a JXL demuxer (a `.jxl` file is a
/// single codestream / ISOBMFF blob, decoded directly by
/// [`decode_one_frame`](crate::decode_one_frame)), so only the
/// extension hint is registered here.
pub fn register_containers(reg: &mut ContainerRegistry) {
    reg.register_extension("jxl", CODEC_ID_STR);
}

/// Decoder factory used by the registry. Returned as a boxed
/// [`Decoder`] so the framework can type-erase it.
pub fn make_decoder(params: &CodecParameters) -> CoreResult<Box<dyn Decoder>> {
    let codec_id = params.codec_id.clone();
    Ok(Box::new(JxlDecoder {
        codec_id,
        pending: None,
        eof: false,
    }))
}

/// Round-3 JXL decoder. Drives [`crate::decode_one_frame`] per packet.
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
pub struct JxlDecoder {
    codec_id: CodecId,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for JxlDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> CoreResult<()> {
        if self.pending.is_some() {
            return Err(CoreError::other(
                "jxl decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> CoreResult<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(CoreError::Eof)
            } else {
                Err(CoreError::NeedMore)
            };
        };
        let img = decode_one_frame(&pkt.data, pkt.pts)?;
        Ok(img.into())
    }

    fn flush(&mut self) -> CoreResult<()> {
        self.eof = true;
        Ok(())
    }
}

/// Round-1 minimal lossless modular JPEG XL encoder factory.
///
/// Accepts `pixel_format ∈ {Gray8, Rgb24, Rgba}` at any width/height up
/// to 1024×1024 (the single-group cap of the round-1 implementation).
/// Larger images return [`CoreError::Unsupported`].
pub fn make_encoder(params: &CodecParameters) -> CoreResult<Box<dyn Encoder>> {
    let codec_id = params.codec_id.clone();
    let pixel_format = params.pixel_format.ok_or_else(|| {
        CoreError::other("jxl encoder: pixel_format required (Gray8, Rgb24 or Rgba)")
    })?;
    let input_format = match pixel_format {
        PixelFormat::Gray8 => InputFormat::Gray8,
        PixelFormat::Rgb24 => InputFormat::Rgb8,
        PixelFormat::Rgba => InputFormat::Rgba8,
        other => {
            return Err(CoreError::Unsupported(format!(
                "jxl encoder: pixel_format {other:?} not supported (round 1 is Gray8/Rgb24/Rgba only)"
            )));
        }
    };
    let width = params
        .width
        .ok_or_else(|| CoreError::other("jxl encoder: width required in CodecParameters"))?;
    let height = params
        .height
        .ok_or_else(|| CoreError::other("jxl encoder: height required in CodecParameters"))?;
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
pub struct JxlEncoder {
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

    fn send_frame(&mut self, frame: &Frame) -> CoreResult<()> {
        if self.pending_packet.is_some() {
            return Err(CoreError::other(
                "jxl encoder: receive_packet must be called before sending another frame",
            ));
        }
        let vf = match frame {
            Frame::Video(vf) => vf,
            _ => {
                return Err(CoreError::other(
                    "jxl encoder: only Video frames are supported",
                ));
            }
        };
        if vf.planes.len() != 1 {
            return Err(CoreError::Unsupported(format!(
                "jxl encoder: expected 1 interleaved plane, got {}",
                vf.planes.len()
            )));
        }
        let plane = &vf.planes[0];
        let channels = self.input_format.channel_count() as usize;
        let expected_stride = self.width as usize * channels;
        if plane.stride != expected_stride {
            return Err(CoreError::other(format!(
                "jxl encoder: plane stride {} != expected {} for {}x{} {:?}",
                plane.stride, expected_stride, self.width, self.height, self.input_format
            )));
        }
        let expected_len = expected_stride * self.height as usize;
        if plane.data.len() != expected_len {
            return Err(CoreError::other(format!(
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

    fn receive_packet(&mut self) -> CoreResult<Packet> {
        if let Some(pkt) = self.pending_packet.take() {
            return Ok(pkt);
        }
        if self.eof {
            Err(CoreError::Eof)
        } else {
            Err(CoreError::NeedMore)
        }
    }

    fn flush(&mut self) -> CoreResult<()> {
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
        register_codecs(&mut reg);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let dec = reg.first_decoder(&params).expect("expected live decoder");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
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
        assert!(matches!(
            make_encoder(&params),
            Err(CoreError::Unsupported(_))
        ));
    }

    #[test]
    fn register_containers_resolves_jxl_extension_case_insensitive() {
        let mut reg = ContainerRegistry::new();
        register_containers(&mut reg);
        assert_eq!(reg.container_for_extension("jxl"), Some(CODEC_ID_STR));
        assert_eq!(reg.container_for_extension("JXL"), Some(CODEC_ID_STR));
        assert_eq!(reg.container_for_extension("Jxl"), Some(CODEC_ID_STR));
        assert_eq!(reg.container_for_extension("png"), None);
    }

    #[test]
    fn register_via_runtime_context_installs_codec_factory() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let dec = ctx
            .codecs
            .first_decoder(&params)
            .expect("jxl decoder factory");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
        // The unified entry point also wires the .jxl extension hint
        // through the same call, so the consumer doesn't need a
        // separate `register_containers` invocation.
        assert_eq!(
            ctx.containers.container_for_extension("jxl"),
            Some(CODEC_ID_STR)
        );
    }
}
