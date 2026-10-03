//! `oxideav-core` registry adapter (the `registry` feature).
//!
//! Everything here is a thin layer over the standalone API in
//! [`crate::api`]:
//!
//! * [`register`] / [`register_codecs`] install the decoder into a
//!   `RuntimeContext` / `CodecRegistry`; [`make_decoder`] /
//!   [`make_encoder`] are the factories (the encoder one always reports
//!   `Unsupported` — the crate is decoder-only).
//! * `From<JxlImage> for VideoFrame`, [`JxlImage::from_video_frame`] and
//!   `TryFrom<(&VideoFrame, &CodecParameters)>` bridge the contract
//!   image and the framework frame (one packed plane; colour signal and
//!   significant-bits side-channels stamped).
//! * `From<JxlError> for oxideav_core::Error`.
//! * The deprecated planar wrappers ([`decode_one_frame`],
//!   [`decode_all_frames`], [`decode_vardct_frame_from_codestream`]).

use std::collections::VecDeque;

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, ColorPrimaries,
    ColorSignal, Decoder, Encoder, Frame, MatrixCoefficients, Packet, PixelFormat, RuntimeContext,
    TransferCharacteristics, VideoFrame, VideoPlane,
};

use crate::error::{JxlError, Result};
use crate::image::{ColorInfo, ColorRange, JxlImage, JxlPixelFormat, Plane, RawFrame};
use crate::CODEC_ID_STR;

impl From<JxlError> for oxideav_core::Error {
    fn from(e: JxlError) -> Self {
        match e {
            JxlError::InvalidData(s) => oxideav_core::Error::InvalidData(s),
            JxlError::Unsupported(s) => oxideav_core::Error::Unsupported(s),
            JxlError::LimitExceeded(s) => oxideav_core::Error::ResourceExhausted(s),
            JxlError::Io(e) => oxideav_core::Error::Io(e),
            JxlError::Eof => oxideav_core::Error::Eof,
            JxlError::NeedMore => oxideav_core::Error::NeedMore,
            JxlError::Other(s) => oxideav_core::Error::other(s),
        }
    }
}

// ---- pixel formats -------------------------------------------------------

/// The framework label of a crate layout (1:1 by name).
pub fn to_core_pixel_format(pf: JxlPixelFormat) -> PixelFormat {
    match pf {
        JxlPixelFormat::Gray8 => PixelFormat::Gray8,
        JxlPixelFormat::Ya8 => PixelFormat::Ya8,
        JxlPixelFormat::Rgb24 => PixelFormat::Rgb24,
        JxlPixelFormat::Rgba => PixelFormat::Rgba,
        JxlPixelFormat::Gray16Le => PixelFormat::Gray16Le,
        JxlPixelFormat::Ya16Le => PixelFormat::Ya16Le,
        JxlPixelFormat::Rgb48Le => PixelFormat::Rgb48Le,
        JxlPixelFormat::Rgba64Le => PixelFormat::Rgba64Le,
    }
}

/// The crate layout of a framework label; `Unsupported` for labels this
/// crate never produces.
pub fn from_core_pixel_format(pf: PixelFormat) -> Result<JxlPixelFormat> {
    Ok(match pf {
        PixelFormat::Gray8 => JxlPixelFormat::Gray8,
        PixelFormat::Ya8 => JxlPixelFormat::Ya8,
        PixelFormat::Rgb24 => JxlPixelFormat::Rgb24,
        PixelFormat::Rgba => JxlPixelFormat::Rgba,
        PixelFormat::Gray16Le => JxlPixelFormat::Gray16Le,
        PixelFormat::Ya16Le => JxlPixelFormat::Ya16Le,
        PixelFormat::Rgb48Le => JxlPixelFormat::Rgb48Le,
        PixelFormat::Rgba64Le => JxlPixelFormat::Rgba64Le,
        other => {
            return Err(JxlError::unsupported(format!(
                "JPEG XL: pixel format {other:?} is not a JPEG XL output layout"
            )))
        }
    })
}

impl From<JxlPixelFormat> for PixelFormat {
    fn from(pf: JxlPixelFormat) -> Self {
        to_core_pixel_format(pf)
    }
}

impl TryFrom<PixelFormat> for JxlPixelFormat {
    type Error = JxlError;
    fn try_from(pf: PixelFormat) -> Result<Self> {
        from_core_pixel_format(pf)
    }
}

// ---- colour --------------------------------------------------------------

/// [`ColorInfo`] as the framework's [`ColorSignal`].
pub fn to_color_signal(c: &ColorInfo) -> ColorSignal {
    let range = match c.range {
        ColorRange::Unspecified => oxideav_core::ColorRange::Unspecified,
        ColorRange::Limited => oxideav_core::ColorRange::Limited,
        ColorRange::Full => oxideav_core::ColorRange::Full,
    };
    ColorSignal::new(
        range,
        ColorPrimaries(c.primaries),
        TransferCharacteristics(c.transfer),
        MatrixCoefficients(c.matrix),
    )
}

/// The inverse of [`to_color_signal`].
pub fn from_color_signal(s: &ColorSignal) -> ColorInfo {
    let range = match s.range {
        oxideav_core::ColorRange::Limited => ColorRange::Limited,
        oxideav_core::ColorRange::Full => ColorRange::Full,
        _ => ColorRange::Unspecified,
    };
    ColorInfo::new(range, s.primaries.0, s.transfer.0, s.matrix.0)
}

// ---- frame bridge --------------------------------------------------------

fn image_into_video_frame(mut image: JxlImage, pts: Option<i64>) -> VideoFrame {
    let plane = if image.planes.is_empty() {
        VideoPlane {
            stride: 0,
            data: Vec::new(),
        }
    } else {
        let p = image.planes.swap_remove(0);
        VideoPlane {
            stride: p.stride,
            data: p.data,
        }
    };
    let mut frame = VideoFrame {
        pts,
        planes: vec![plane],
    };
    stamp_frame_side_channels(&mut frame, &image);
    frame
}

fn stamp_frame_side_channels(frame: &mut VideoFrame, image: &JxlImage) {
    // JPEG XL always carries a ColourEncoding (defaults to sRGB), so the
    // signal is stamped on every frame.
    if image.color.is_specified() {
        frame.set_color_signal(to_color_signal(&image.color));
    }
    let storage = (image.format.bytes_per_sample() * 8) as u8;
    if image.bits_per_sample != storage {
        frame.set_significant_bits(vec![image.bits_per_sample]);
    }
}

impl From<JxlImage> for VideoFrame {
    fn from(image: JxlImage) -> Self {
        image_into_video_frame(image, None)
    }
}

impl From<&JxlImage> for VideoFrame {
    fn from(image: &JxlImage) -> Self {
        image_into_video_frame(image.clone(), None)
    }
}

impl JxlImage {
    /// Rebuild a [`JxlImage`] from a framework frame; dimensions and
    /// layout come from `params`, the colour signal and significant
    /// bits from the frame's side-channels.
    pub fn from_video_frame(frame: &VideoFrame, params: &CodecParameters) -> Result<Self> {
        let width = params
            .width
            .ok_or_else(|| JxlError::invalid("JPEG XL: CodecParameters.width missing"))?;
        let height = params
            .height
            .ok_or_else(|| JxlError::invalid("JPEG XL: CodecParameters.height missing"))?;
        let pix = from_core_pixel_format(params.pixel_format.unwrap_or(PixelFormat::Rgba))?;
        let plane = frame
            .image_planes()
            .first()
            .ok_or_else(|| JxlError::invalid("JPEG XL: frame has no image plane"))?;
        let mut img = JxlImage::new(
            width,
            height,
            pix,
            vec![Plane::new(plane.stride, plane.data.clone())],
        )?;
        if let Some(sig) = frame.color_signal() {
            img.color = from_color_signal(&sig);
        }
        if let Some(bits) = frame.significant_bits().and_then(|b| b.first().copied()) {
            img = img.with_bits_per_sample(bits);
        }
        Ok(img)
    }
}

impl TryFrom<(&VideoFrame, &CodecParameters)> for JxlImage {
    type Error = JxlError;
    fn try_from((frame, params): (&VideoFrame, &CodecParameters)) -> Result<Self> {
        JxlImage::from_video_frame(frame, params)
    }
}

// ---- registry ------------------------------------------------------------

/// Register the JPEG XL decoder into a [`CodecRegistry`]. No encoder is
/// registered: the crate is decoder-only.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video(CODEC_ID_STR)
        .with_lossy(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder),
    );
}

/// Install the JPEG XL codec into a [`RuntimeContext`] (fleet
/// signature).
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

oxideav_core::register!("jpegxl", register);

/// Decoder factory: one whole JPEG XL file per packet, every presented
/// frame returned in order (animations yield several frames per
/// packet).
pub fn make_decoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Decoder>> {
    Ok(Box::new(JxlDecoder {
        codec_id: params.codec_id.clone(),
        pending: None,
        ready: VecDeque::new(),
        eof: false,
    }))
}

/// Encoder factory: always `Unsupported` (no JPEG XL encoder in this
/// crate). Exposed so callers wiring an `Encoder` by codec id get a
/// clean `Unsupported` instead of `CodecNotFound`.
pub fn make_encoder(_params: &CodecParameters) -> oxideav_core::Result<Box<dyn Encoder>> {
    Err(oxideav_core::Error::Unsupported(
        "JPEG XL encoding is not implemented".into(),
    ))
}

/// Registered decoder: [`crate::decode_all`] per packet, frames handed
/// out one per `receive_frame` in the contract's packed native layout
/// (`Gray8` / `Ya8` / `Rgb24` / `Rgba` and their 16-bit forms) with the
/// colour-signal side-channel stamped.
struct JxlDecoder {
    codec_id: CodecId,
    pending: Option<Packet>,
    ready: VecDeque<VideoFrame>,
    eof: bool,
}

impl Decoder for JxlDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> oxideav_core::Result<()> {
        if self.pending.is_some() || !self.ready.is_empty() {
            return Err(oxideav_core::Error::other(
                "jxl decoder: drain receive_frame before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> oxideav_core::Result<Frame> {
        if let Some(pkt) = self.pending.take() {
            let frames = crate::decode_all(&pkt.data)?;
            // `pts` is applied to the first frame only; later frames of
            // an animation carry `None` (their spacing is the frame
            // header's tick duration, not a stream timebase).
            let mut pts = pkt.pts;
            for f in frames {
                self.ready
                    .push_back(image_into_video_frame(f.image, pts.take()));
            }
        }
        match self.ready.pop_front() {
            Some(vf) => Ok(Frame::Video(vf)),
            None if self.eof => Err(oxideav_core::Error::Eof),
            None => Err(oxideav_core::Error::NeedMore),
        }
    }

    fn flush(&mut self) -> oxideav_core::Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ---- deprecated planar wrappers ---------------------------------------------

fn raw_to_video_frame(raw: RawFrame) -> VideoFrame {
    VideoFrame {
        pts: raw.pts,
        planes: raw
            .planes
            .into_iter()
            .map(|p| VideoPlane {
                stride: p.stride,
                data: p.data,
            })
            .collect(),
    }
}

/// Decode the first frame as per-channel planes (grey: 1 plane, RGB: 3,
/// extra channels after), 1 byte per sample for depths ≤ 8 and 2 bytes
/// little-endian otherwise.
#[deprecated(note = "use oxideav_jpegxl::decode (IMAGE_CRATE_API)")]
pub fn decode_one_frame(input: &[u8], pts: Option<i64>) -> oxideav_core::Result<VideoFrame> {
    Ok(raw_to_video_frame(crate::decode_planar(input, pts)?))
}

/// Decode every presented frame as per-channel planes (see
/// [`decode_one_frame`] for the layout).
#[deprecated(note = "use oxideav_jpegxl::decode_all (IMAGE_CRATE_API)")]
pub fn decode_all_frames(input: &[u8], pts: Option<i64>) -> oxideav_core::Result<Vec<VideoFrame>> {
    Ok(crate::decode_all_planar(input, pts)?
        .into_iter()
        .map(raw_to_video_frame)
        .collect())
}

/// Historical alias of [`decode_one_frame`].
#[deprecated(note = "use oxideav_jpegxl::decode (IMAGE_CRATE_API)")]
pub fn decode_vardct_frame_from_codestream(
    input: &[u8],
    pts: Option<i64>,
) -> oxideav_core::Result<VideoFrame> {
    Ok(raw_to_video_frame(crate::decode_planar(input, pts)?))
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
    fn encoder_factory_rejects_cleanly() {
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        assert!(matches!(
            make_encoder(&params),
            Err(oxideav_core::Error::Unsupported(_))
        ));
    }

    #[test]
    fn frame_bridge_round_trips() {
        let img = JxlImage::packed(
            2,
            1,
            JxlPixelFormat::Rgb48Le,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        )
        .unwrap()
        .with_color(ColorInfo::srgb())
        .with_bits_per_sample(12);
        let vf: VideoFrame = (&img).into();
        assert_eq!(vf.image_planes().len(), 1);
        assert_eq!(vf.significant_bits(), Some(&[12u8][..]));
        assert!(vf.color_signal().is_some());
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(2);
        params.height = Some(1);
        params.pixel_format = Some(PixelFormat::Rgb48Le);
        let back = JxlImage::from_video_frame(&vf, &params).unwrap();
        assert_eq!(back, img);
        let via_try: JxlImage = (&vf, &params).try_into().unwrap();
        assert_eq!(via_try, img);
    }

    #[test]
    fn pixel_formats_map_by_name() {
        for pf in [
            JxlPixelFormat::Gray8,
            JxlPixelFormat::Ya8,
            JxlPixelFormat::Rgb24,
            JxlPixelFormat::Rgba,
            JxlPixelFormat::Gray16Le,
            JxlPixelFormat::Ya16Le,
            JxlPixelFormat::Rgb48Le,
            JxlPixelFormat::Rgba64Le,
        ] {
            let core: PixelFormat = pf.into();
            assert_eq!(format!("{core:?}"), format!("{pf:?}"));
            assert_eq!(JxlPixelFormat::try_from(core).unwrap(), pf);
        }
        assert!(JxlPixelFormat::try_from(PixelFormat::Yuv420P).is_err());
    }
}
