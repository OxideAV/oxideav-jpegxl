//! `oxideav-core` registry adapter (the `registry` feature).
//!
//! Everything here is a thin layer over the standalone API in
//! [`crate::api`]:
//!
//! * [`register`] / [`register_codecs`] install the decoder and the
//!   Modular encoder into a `RuntimeContext` / `CodecRegistry`;
//!   [`make_decoder`] / [`make_encoder`] are the factories. The encoder
//!   declares the layouts it codes ([`ACCEPTED_PIXEL_FORMATS`]) and the
//!   [`JxlEncoderOptions`] schema (`quality`, `container`,
//!   `group_size_shift`); anything else is a typed `Unsupported`.
//! * Metadata blobs (`oxideav_core::blob`): the decoder attaches the
//!   codestream ICC profile and the `Exif` / `xml ` box payloads to the
//!   first frame as `BlobKind::ICC` / `EXIF` / `XMP`; the encoder reads
//!   them back from the frame (a frame blob overrides the stream blob
//!   of the same kind on `CodecParameters::blobs`) and writes the ICC
//!   into the codestream, Exif / XMP as boxes.
//! * `From<JxlImage> for VideoFrame`, [`JxlImage::from_video_frame`] and
//!   `TryFrom<(&VideoFrame, &CodecParameters)>` bridge the contract
//!   image and the framework frame (one packed plane; colour signal,
//!   significant-bits and blob side-channels stamped).
//! * `From<JxlError> for oxideav_core::Error`.
//! * The deprecated planar wrappers ([`decode_one_frame`],
//!   [`decode_all_frames`], [`decode_vardct_frame_from_codestream`]).

use std::collections::VecDeque;

use oxideav_core::{
    BlobKind, CodecCapabilities, CodecId, CodecInfo, CodecOptionsStruct, CodecParameters,
    CodecRegistry, ColorPrimaries, ColorSignal, Decoder, Encoder, Frame, MatrixCoefficients,
    MetadataBlob, OptionField, OptionKind, OptionValue, Packet, PixelFormat, RuntimeContext,
    TimeBase, TransferCharacteristics, VideoFrame, VideoPlane,
};

use crate::error::{JxlError, Result};
use crate::image::{ColorInfo, ColorRange, JxlImage, JxlPixelFormat, Metadata, Plane, RawFrame};
use crate::options::EncodeOptions;
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
        JxlPixelFormat::YuvJ444P => PixelFormat::YuvJ444P,
        JxlPixelFormat::YuvJ422P => PixelFormat::YuvJ422P,
        JxlPixelFormat::YuvJ420P => PixelFormat::YuvJ420P,
        JxlPixelFormat::Yuv440P => PixelFormat::Yuv440P,
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
        PixelFormat::YuvJ444P => JxlPixelFormat::YuvJ444P,
        PixelFormat::YuvJ422P => JxlPixelFormat::YuvJ422P,
        PixelFormat::YuvJ420P => JxlPixelFormat::YuvJ420P,
        PixelFormat::Yuv440P => JxlPixelFormat::Yuv440P,
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
    // One plane for the packed layouts, `Y` / `Cb` / `Cr` for the planar
    // JPEG-transcode layouts — the framework label's own plane order.
    let planes: Vec<VideoPlane> = if image.planes.is_empty() {
        vec![VideoPlane {
            stride: 0,
            data: Vec::new(),
        }]
    } else {
        std::mem::take(&mut image.planes)
            .into_iter()
            .map(|p| VideoPlane {
                stride: p.stride,
                data: p.data,
            })
            .collect()
    };
    let mut frame = VideoFrame { pts, planes };
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
    let blobs = blobs_of_metadata(&image.metadata);
    if !blobs.is_empty() {
        frame.set_blobs(blobs);
    }
}

/// The ICC / Exif / XMP of a [`Metadata`] as framework blobs, in that
/// order (the `oxideav_core::blob` payload conventions: a complete
/// profile, a TIFF-structured Exif block, the XMP packet).
pub fn blobs_of_metadata(meta: &Metadata) -> Vec<MetadataBlob> {
    let mut out = Vec::new();
    if let Some(icc) = &meta.icc {
        out.push(MetadataBlob::new(BlobKind::ICC, icc.clone()));
    }
    if let Some(exif) = &meta.exif {
        out.push(MetadataBlob::new(BlobKind::EXIF, exif.clone()));
    }
    if let Some(xmp) = &meta.xmp {
        out.push(MetadataBlob::new(BlobKind::XMP, xmp.clone()).with_mime("application/rdf+xml"));
    }
    out
}

/// [`Metadata`] from framework blobs: the first `icc` / `exif` / `xmp`
/// blob of `frame` (per-frame), else of `stream` (stream-level) —
/// a frame blob overrides the stream blob of the same kind. Other
/// kinds are not carried by JPEG XL and are ignored.
pub fn metadata_of_blobs(frame: &[MetadataBlob], stream: &[MetadataBlob]) -> Metadata {
    let pick = |kind: &BlobKind| -> Option<Vec<u8>> {
        frame
            .iter()
            .chain(stream.iter())
            .find(|b| b.is(kind))
            .map(|b| b.data.clone())
    };
    Metadata::new()
        .with_icc(pick(&BlobKind::ICC))
        .with_exif(pick(&BlobKind::EXIF))
        .with_xmp(pick(&BlobKind::XMP))
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
        let image_planes = frame.image_planes();
        let want = pix.plane_count();
        if image_planes.len() < want {
            return Err(JxlError::invalid(format!(
                "JPEG XL: frame carries {} image plane(s), {pix:?} needs {want}",
                image_planes.len()
            )));
        }
        let planes = image_planes[..want]
            .iter()
            .map(|p| Plane::new(p.stride, p.data.clone()))
            .collect();
        let mut img = JxlImage::new(width, height, pix, planes)?;
        if let Some(sig) = frame.color_signal() {
            img.color = from_color_signal(&sig);
        }
        if let Some(bits) = frame.significant_bits().and_then(|b| b.first().copied()) {
            img = img.with_bits_per_sample(bits);
        }
        img.metadata = metadata_of_blobs(&frame.blobs(), &params.blobs);
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

/// The layouts the registry encoder codes, in ladder order (colour
/// first, 8-bit before 16-bit, grey last): every one is written as it
/// is and decodes to the same label.
pub const ACCEPTED_PIXEL_FORMATS: &[PixelFormat] = &[
    PixelFormat::Rgb24,
    PixelFormat::Rgba,
    PixelFormat::Rgb48Le,
    PixelFormat::Rgba64Le,
    PixelFormat::Gray8,
    PixelFormat::Ya8,
    PixelFormat::Gray16Le,
    PixelFormat::Ya16Le,
];

/// Register the JPEG XL decoder and encoder into a [`CodecRegistry`].
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video(CODEC_ID_STR)
        .with_lossy(true)
        .with_lossless(true)
        .with_intra_only(true)
        .with_pixel_formats(ACCEPTED_PIXEL_FORMATS.to_vec());
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder)
            .encoder_options::<JxlEncoderOptions>(),
    );
}

/// Register the `jpegxl` container of [`crate::demux`] (demuxer, muxer,
/// probe and the `.jxl` extension).
pub fn register_containers(reg: &mut oxideav_core::ContainerRegistry) {
    crate::demux::register(reg);
}

/// Install the JPEG XL codec and container into a [`RuntimeContext`]
/// (fleet signature).
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
    register_containers(&mut ctx.containers);
}

oxideav_core::register!("jpegxl", register);

/// Decoder option key selecting how frames are released
/// ([`PACING_PACKET`]).
pub const OPTION_PACING: &str = "pacing";
/// `pacing = packet`: one frame per packet received. The first packet
/// carries the whole file and releases its first presented frame;
/// every further packet (normally zero-length, as the `jpegxl` demuxer
/// emits them) releases the next frame and lends it its `pts`;
/// `flush` releases whatever is left. Without the option every frame
/// of the file comes out of the packet that carried it.
pub const PACING_PACKET: &str = "packet";

/// Decoder factory: one whole JPEG XL file per packet, every presented
/// frame returned in order — all of them from the one packet by
/// default, one per packet with `pacing = packet` ([`PACING_PACKET`],
/// what the `jpegxl` demuxer asks for so animation frames carry the
/// container's timing).
pub fn make_decoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Decoder>> {
    let paced = match params.options.get(OPTION_PACING) {
        None => false,
        Some(v) if v == PACING_PACKET => true,
        Some("all") => false,
        Some(other) => {
            return Err(oxideav_core::Error::invalid(format!(
            "jxl decoder: option {OPTION_PACING}={other:?} is not \"{PACING_PACKET}\" or \"all\""
        )))
        }
    };
    Ok(Box::new(JxlDecoder {
        codec_id: params.codec_id.clone(),
        pending: None,
        ready: VecDeque::new(),
        paced,
        credits: VecDeque::new(),
        eof: false,
    }))
}

/// Encoder options (the `quality` / `container` / `group_size_shift`
/// keys of `CodecParameters::options`).
///
/// | key | kind | default | meaning |
/// |---|---|---|---|
/// | `quality` | u32 `0..=100` | `100` | `100` = lossless Modular; below it the residual quantiser of [`crate::encoder::quantizer_for_quality`] |
/// | `container` | bool | `false` | always write the 18181-2 box file (a bare codestream otherwise, unless Exif / XMP blobs need boxes) |
/// | `group_size_shift` | u32 `0..=3` | `1` | `group_dim = 128 << shift` |
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct JxlEncoderOptions {
    pub quality: u32,
    pub container: bool,
    pub group_size_shift: u32,
}

impl Default for JxlEncoderOptions {
    fn default() -> Self {
        Self {
            quality: 100,
            container: false,
            group_size_shift: 1,
        }
    }
}

impl JxlEncoderOptions {
    /// Same as [`Default::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set `quality`.
    pub fn with_quality(mut self, quality: u32) -> Self {
        self.quality = quality;
        self
    }

    /// Builder: set `container`.
    pub fn with_container(mut self, container: bool) -> Self {
        self.container = container;
        self
    }

    /// Builder: set `group_size_shift`.
    pub fn with_group_size_shift(mut self, shift: u32) -> Self {
        self.group_size_shift = shift;
        self
    }

    fn to_encode_options(&self) -> oxideav_core::Result<EncodeOptions> {
        if self.quality > 100 {
            return Err(oxideav_core::Error::invalid(
                "jxl encoder: quality must be in 0..=100",
            ));
        }
        if self.group_size_shift > 3 {
            return Err(oxideav_core::Error::invalid(
                "jxl encoder: group_size_shift must be in 0..=3",
            ));
        }
        Ok(EncodeOptions::new()
            .with_quality(self.quality as u8)
            .with_container(self.container)
            .with_group_size_shift(self.group_size_shift as u8))
    }
}

impl CodecOptionsStruct for JxlEncoderOptions {
    const SCHEMA: &'static [OptionField] = &[
        OptionField {
            name: "quality",
            kind: OptionKind::U32,
            default: OptionValue::U32(100),
            help: "0..=100; 100 = lossless Modular, lower quantises the Modular residuals",
        },
        OptionField {
            name: "container",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "always write the ISO/IEC 18181-2 box file instead of a bare codestream",
        },
        OptionField {
            name: "group_size_shift",
            kind: OptionKind::U32,
            default: OptionValue::U32(1),
            help: "0..=3; group_dim = 128 << shift",
        },
    ];

    fn apply(&mut self, key: &str, value: &OptionValue) -> oxideav_core::Result<()> {
        match key {
            "quality" => self.quality = value.as_u32()?,
            "container" => self.container = value.as_bool()?,
            "group_size_shift" => self.group_size_shift = value.as_u32()?,
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// Encoder factory: one still JPEG XL file per frame received (a bare
/// codestream, or the box file when the frame's / stream's Exif / XMP
/// blobs need boxes or `container` is set). `params.pixel_format` must
/// be one of [`ACCEPTED_PIXEL_FORMATS`] (default `Rgba`) — anything
/// else is `Unsupported`, never converted. The frame's `pts` is kept on
/// the packet; an animation is assembled by the `jpegxl` muxer from one
/// packet per picture.
pub fn make_encoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Encoder>> {
    let opts: JxlEncoderOptions = oxideav_core::parse_options(&params.options)
        .map_err(|e| oxideav_core::Error::invalid(format!("jxl encoder: {e}")))?;
    let encode_opts = opts.to_encode_options()?;
    let fmt = params.pixel_format.unwrap_or(PixelFormat::Rgba);
    if !ACCEPTED_PIXEL_FORMATS.contains(&fmt) {
        return Err(oxideav_core::Error::Unsupported(format!(
            "jxl encoder: pixel format {fmt:?} is not an accepted input layout ({ACCEPTED_PIXEL_FORMATS:?})"
        )));
    }
    let (Some(w), Some(h)) = (params.width, params.height) else {
        return Err(oxideav_core::Error::invalid(
            "jxl encoder: width and height are required",
        ));
    };
    if w == 0 || h == 0 {
        return Err(oxideav_core::Error::invalid("jxl encoder: zero dimension"));
    }
    let mut output = params.clone();
    output.codec_id = CodecId::new(CODEC_ID_STR);
    output.pixel_format = Some(fmt);
    output.extradata.clear();
    output.tag = None;
    Ok(Box::new(JxlEncoder {
        params: output,
        opts: encode_opts,
        ready: VecDeque::new(),
        eof: false,
    }))
}

struct JxlEncoder {
    params: CodecParameters,
    opts: EncodeOptions,
    ready: VecDeque<Packet>,
    eof: bool,
}

impl Encoder for JxlEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.params
    }

    fn send_frame(&mut self, frame: &Frame) -> oxideav_core::Result<()> {
        let Frame::Video(vf) = frame else {
            return Err(oxideav_core::Error::invalid(
                "jxl encoder: a video frame is required",
            ));
        };
        if self.eof {
            return Err(oxideav_core::Error::invalid(
                "jxl encoder: send_frame after flush",
            ));
        }
        let image = JxlImage::from_video_frame(vf, &self.params)?;
        let bytes = crate::encode(&image, &self.opts)?;
        let mut pkt = Packet::new(0, TimeBase::new(1, 1), bytes);
        pkt.pts = vf.pts;
        pkt.dts = vf.pts;
        pkt.flags.keyframe = true;
        self.ready.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> oxideav_core::Result<Packet> {
        match self.ready.pop_front() {
            Some(p) => Ok(p),
            None if self.eof => Err(oxideav_core::Error::Eof),
            None => Err(oxideav_core::Error::NeedMore),
        }
    }

    fn flush(&mut self) -> oxideav_core::Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Registered decoder: [`crate::decode_all`] per packet, frames handed
/// out one per `receive_frame` in the contract's packed native layout
/// (`Gray8` / `Ya8` / `Rgb24` / `Rgba` and their 16-bit forms) with the
/// colour-signal side-channel stamped.
struct JxlDecoder {
    codec_id: CodecId,
    pending: Option<Packet>,
    ready: VecDeque<VideoFrame>,
    /// `pacing = packet`: frames are released one per packet.
    paced: bool,
    /// Packets received but not yet matched with a frame (paced mode):
    /// each releases one frame and lends it its `pts`.
    credits: VecDeque<Option<i64>>,
    eof: bool,
}

impl Decoder for JxlDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> oxideav_core::Result<()> {
        if self.paced && packet.data.is_empty() {
            // A pacing packet releases the next decoded frame; with
            // nothing decoded or pending it has nothing to release and
            // is a no-op.
            if self.pending.is_some() || !self.ready.is_empty() {
                self.credits.push_back(packet.pts);
            }
            return Ok(());
        }
        if self.pending.is_some() || !self.ready.is_empty() {
            return Err(oxideav_core::Error::other(
                "jxl decoder: drain receive_frame before sending another packet",
            ));
        }
        if self.paced {
            self.credits.push_back(packet.pts);
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> oxideav_core::Result<Frame> {
        if let Some(pkt) = self.pending.take() {
            let frames = crate::decode_all(&pkt.data)?;
            // `pts` is applied to the first frame only; later frames of
            // an animation carry `None` (their spacing is the frame
            // header's tick duration, not a stream timebase) unless a
            // pacing packet lends them its own.
            let mut pts = pkt.pts;
            for f in frames {
                self.ready
                    .push_back(image_into_video_frame(f.image, pts.take()));
            }
        }
        if self.paced && !self.eof {
            // One frame per packet: hold the rest until the next packet
            // (or the flush) releases it.
            let Some(credit) = self.credits.pop_front() else {
                return Err(oxideav_core::Error::NeedMore);
            };
            return match self.ready.pop_front() {
                Some(mut vf) => {
                    if credit.is_some() {
                        vf.pts = credit;
                    }
                    Ok(Frame::Video(vf))
                }
                None => {
                    // The file decoded to no frame; nothing to release.
                    self.credits.clear();
                    Err(oxideav_core::Error::NeedMore)
                }
            };
        }
        match self.ready.pop_front() {
            Some(mut vf) => {
                if let Some(credit) = self.credits.pop_front() {
                    if credit.is_some() {
                        vf.pts = credit;
                    }
                }
                Ok(Frame::Video(vf))
            }
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
        assert!(ctx.containers.demuxer_names().any(|n| n == "jpegxl"));
        assert!(ctx.containers.muxer_names().any(|n| n == "jpegxl"));
        assert_eq!(
            ctx.containers.container_for_extension("jxl"),
            Some("jpegxl")
        );
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let dec = ctx
            .codecs
            .first_decoder(&params)
            .expect("expected live decoder");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
    }

    #[test]
    fn encoder_factory_declares_its_layouts_and_options() {
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        assert!(matches!(
            make_encoder(&params),
            Err(oxideav_core::Error::InvalidData(_))
        ));
        params.width = Some(2);
        params.height = Some(2);
        params.pixel_format = Some(PixelFormat::Yuv420P);
        assert!(matches!(
            make_encoder(&params),
            Err(oxideav_core::Error::Unsupported(_))
        ));
        params.pixel_format = Some(PixelFormat::Rgb24);
        params.options.insert("quality", "101");
        assert!(matches!(
            make_encoder(&params),
            Err(oxideav_core::Error::InvalidData(_))
        ));
        params.options = oxideav_core::CodecOptions::new().set("bogus", "1");
        assert!(matches!(
            make_encoder(&params),
            Err(oxideav_core::Error::InvalidData(_))
        ));
        params.options = oxideav_core::CodecOptions::new().set("quality", "90");
        let enc = make_encoder(&params).unwrap();
        assert_eq!(enc.output_params().pixel_format, Some(PixelFormat::Rgb24));
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let schema = ctx
            .codecs
            .encoder_options_schema(&CodecId::new(CODEC_ID_STR))
            .expect("schema");
        assert!(schema.iter().any(|f| f.name == "quality"));
        let caps = &ctx.codecs.implementations(&CodecId::new(CODEC_ID_STR))[0].caps;
        assert!(caps.encode && caps.decode && caps.lossy && caps.lossless);
        assert_eq!(caps.accepted_pixel_formats, ACCEPTED_PIXEL_FORMATS);
        assert!(ctx.containers.muxer_names().any(|n| n == "jpegxl"));
    }

    #[test]
    fn frame_blobs_bridge_both_ways() {
        let img = JxlImage::from_rgb8(1, 1, vec![1, 2, 3])
            .unwrap()
            .with_metadata(
                Metadata::new()
                    .with_icc(vec![1u8; 130])
                    .with_exif(b"II*\0".to_vec())
                    .with_xmp(b"<x/>".to_vec()),
            );
        let vf: VideoFrame = (&img).into();
        let blobs = vf.blobs();
        assert_eq!(blobs.len(), 3);
        assert_eq!(blobs[0].kind, BlobKind::ICC);
        assert_eq!(blobs[1].kind, BlobKind::EXIF);
        assert_eq!(blobs[2].kind, BlobKind::XMP);
        assert_eq!(blobs[2].mime.as_deref(), Some("application/rdf+xml"));
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(1);
        params.height = Some(1);
        params.pixel_format = Some(PixelFormat::Rgb24);
        // Stream-level Exif is overridden by the frame's; a stream XMP
        // fills in when the frame has none.
        params.blobs = vec![
            MetadataBlob::new(BlobKind::EXIF, b"MM\0*".to_vec()),
            MetadataBlob::new(BlobKind::XMP, b"<stream/>".to_vec()),
        ];
        let back = JxlImage::from_video_frame(&vf, &params).unwrap();
        assert_eq!(back.metadata, img.metadata);
        let mut bare: VideoFrame = (&img).into();
        bare.set_blobs(Vec::new());
        let back = JxlImage::from_video_frame(&bare, &params).unwrap();
        assert_eq!(back.metadata.exif.as_deref(), Some(&b"MM\0*"[..]));
        assert_eq!(back.metadata.xmp.as_deref(), Some(&b"<stream/>"[..]));
        assert!(back.metadata.icc.is_none());
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

        // Planar: three planes cross the bridge in Y / Cb / Cr order.
        let yuv = JxlImage::new(
            3,
            1,
            JxlPixelFormat::YuvJ420P,
            vec![
                Plane::new(3, vec![1, 2, 3]),
                Plane::new(2, vec![4, 5]),
                Plane::new(2, vec![6, 7]),
            ],
        )
        .unwrap()
        .with_color(ColorInfo::srgb().with_matrix(ColorInfo::MATRIX_BT601));
        let vf: VideoFrame = (&yuv).into();
        assert_eq!(vf.image_planes().len(), 3);
        assert_eq!(vf.image_planes()[1].data, vec![4, 5]);
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(3);
        params.height = Some(1);
        params.pixel_format = Some(PixelFormat::YuvJ420P);
        assert_eq!(JxlImage::from_video_frame(&vf, &params).unwrap(), yuv);
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
            JxlPixelFormat::YuvJ444P,
            JxlPixelFormat::YuvJ422P,
            JxlPixelFormat::YuvJ420P,
            JxlPixelFormat::Yuv440P,
        ] {
            let core: PixelFormat = pf.into();
            assert_eq!(format!("{core:?}"), format!("{pf:?}"));
            assert_eq!(JxlPixelFormat::try_from(core).unwrap(), pf);
        }
        assert!(JxlPixelFormat::try_from(PixelFormat::Yuv420P).is_err());
        assert!(JxlPixelFormat::try_from(PixelFormat::Yuv411P).is_err());
    }
}
