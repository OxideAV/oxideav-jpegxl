//! The image-crate contract surface: `probe` / `info` / `decode*` /
//! `encode*` over [`JxlImage`].
//!
//! Every function here is framework-free; the `registry` adapter calls
//! into these (one implementation).

use std::io::{Read, Write};
use std::time::Duration;

use crate::container::{self, MetadataKind, Signature};
use crate::error::{Error, Result};
use crate::image::{ColorInfo, ColorRange, JxlImage, Metadata, PixelFormat, Plane, RawFrame};
use crate::metadata_fdis::{
    ColourSpace, ExtraChannelType, ImageMetadataFdis, Primaries, TransferFunction,
};
use crate::options::{DecodeOptions, EncodeOptions};
use crate::{
    decode_sequence, frame_params, open, read_prelude_at_level, walk_frame_headers, Opened,
    Prelude, SequenceFrame, SequenceMode,
};

/// Cap on a Brotli-compressed (`brob`) metadata box's decompressed
/// size: Exif and XMP payloads larger than this are left out of
/// [`Metadata`] rather than decompressed.
const METADATA_BOX_CAP: usize = 64 << 20;

/// Animation parameters from the §A.6 `AnimationHeader`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct AnimationInfo {
    /// Ticks per second, as a `numerator / denominator` fraction
    /// (`tps_numerator`, `tps_denominator`).
    pub ticks_per_second: (u32, u32),
    /// Repeat count; `0` means loop forever.
    pub loops: u32,
}

impl AnimationInfo {
    /// Duration of `ticks` animation ticks.
    pub fn tick_duration(&self, ticks: u32) -> Duration {
        let (num, den) = self.ticks_per_second;
        if num == 0 {
            return Duration::ZERO;
        }
        // seconds = ticks × den / num
        let nanos = u128::from(ticks) * u128::from(den) * 1_000_000_000u128 / u128::from(num);
        Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64)
    }
}

/// Header-level description of a JPEG XL file, from `SizeHeader` /
/// `ImageMetadata` (plus the first `FrameHeader` for the sample layout)
/// — no pixels are decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImageInfo {
    /// Width in pixels after the Table A.17 orientation transform (what
    /// [`decode`] returns).
    pub width: u32,
    /// Height in pixels after the orientation transform.
    pub height: u32,
    /// Layout [`decode`] produces.
    pub format: PixelFormat,
    /// Presented frame count: `1` for a still image (layers compose
    /// into one picture), the number of presented frames for an
    /// animation.
    pub frames: u32,
    /// Whether an alpha extra channel is declared.
    pub has_alpha: bool,
    /// Colour signalling (see [`ColorInfo`]).
    pub color: ColorInfo,
    /// Whether the codestream embeds an ICC profile.
    pub has_icc: bool,
    /// Whether the container carries an `Exif` box.
    pub has_exif: bool,
    /// Whether the container carries an `xml ` box.
    pub has_xmp: bool,
    /// Declared `bits_per_sample` (integer depth; see `float_sample`).
    pub bits_per_sample: u32,
    /// Whether samples are declared floating point (not decodable by
    /// this crate; `decode` returns `Unsupported`).
    pub float_sample: bool,
    /// Whether the codestream is XYB-encoded.
    pub xyb_encoded: bool,
    /// Table A.17 orientation code (1 = upright).
    pub orientation: u8,
    /// Number of extra channels (alpha included).
    pub num_extra_channels: u32,
    /// Animation parameters when `have_animation`.
    pub animation: Option<AnimationInfo>,
    /// Wrapping: raw codestream or ISOBMFF box file.
    pub signature: Signature,
    /// Whether a `jbrd` JPEG-reconstruction box is present (see
    /// [`crate::jpeg_reconstruct`]).
    pub has_jpeg_reconstruction: bool,
}

/// One frame of [`decode_all`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Frame {
    /// The frame's pixels.
    pub image: JxlImage,
    /// Display duration from the frame header's `duration` ticks and
    /// the animation header's tick rate; `None` for non-animated files.
    pub delay: Option<Duration>,
    /// Position in the codestream's frame array.
    pub index: u32,
    /// Raw `duration` in animation ticks.
    pub ticks: u32,
    /// Crop offset in sample-grid space (non-zero only when
    /// [`DecodeOptions::coalesce`] is `false`).
    pub x: i32,
    /// See `x`.
    pub y: i32,
    /// Whether the frame is presented (`duration > 0 || is_last`);
    /// always `true` when coalescing.
    pub presented: bool,
}

/// Signature sniff: the `FF 0A` codestream signature or the 12-byte
/// ISOBMFF `JXL ` signature box. Allocation-free; `false` on short
/// input.
pub fn probe(bytes: &[u8]) -> bool {
    container::detect(bytes).is_some()
}

/// Header-only inspection (see [`ImageInfo`]).
pub fn info(bytes: &[u8]) -> Result<ImageInfo> {
    let opened = open(bytes)?;
    let prelude = read_prelude_at_level(&opened.codestream, opened.level())?;
    let md = &prelude.metadata;
    let (width, height) = oriented_size(&prelude);
    // The sample layout depends on the first frame's encoding: VarDCT
    // frames reconstruct to 8 bits per sample whatever the declared
    // depth; Modular frames keep the declared depth (1 or 2 bytes).
    let first_is_vardct = {
        let params = frame_params(&prelude);
        let slice = opened
            .codestream
            .get(prelude.frames_offset..)
            .unwrap_or(&[]);
        let mut br = crate::bitreader::BitReader::new(slice);
        match crate::read_frame_header_and_toc(&mut br, &params, slice) {
            Ok((fh, _)) => fh.encoding == crate::frame_header::Encoding::VarDct,
            Err(_) => false,
        }
    };
    let bytes_per_sample = if md.bit_depth.bits_per_sample > 8 && !first_is_vardct {
        2
    } else {
        1
    };
    let has_alpha = alpha_channel_index(md).is_some();
    let format = PixelFormat::for_layout(colour_channels(md), has_alpha, bytes_per_sample)?;
    let frames = if md.have_animation {
        walk_frame_headers(&opened.codestream, &prelude)?
            .presented
            .max(1)
    } else {
        1
    };
    let (has_exif, has_xmp, has_jbrd) = match &opened.file {
        Some(f) => (
            f.metadata.iter().any(|m| m.kind == MetadataKind::Exif),
            f.metadata.iter().any(|m| m.kind == MetadataKind::Xml),
            f.jbrd.is_some(),
        ),
        None => (false, false, false),
    };
    Ok(ImageInfo {
        width,
        height,
        format,
        frames,
        has_alpha,
        color: color_info(md),
        has_icc: md.colour_encoding.want_icc,
        has_exif,
        has_xmp,
        bits_per_sample: md.bit_depth.bits_per_sample,
        float_sample: md.bit_depth.float_sample,
        xyb_encoded: md.xyb_encoded,
        orientation: md.orientation,
        num_extra_channels: md.num_extra_channels,
        animation: animation_info(md),
        signature: opened.signature,
        has_jpeg_reconstruction: has_jbrd,
    })
}

/// Decode the primary image (the first presented frame, composed onto
/// the image canvas and oriented) with default options.
pub fn decode(bytes: &[u8]) -> Result<JxlImage> {
    decode_with(bytes, &DecodeOptions::default())
}

/// [`decode`] with limits / strictness.
pub fn decode_with(bytes: &[u8], opts: &DecodeOptions) -> Result<JxlImage> {
    let opened = open(bytes)?;
    let prelude = read_prelude_at_level(&opened.codestream, opened.level())?;
    check_limits(&prelude, opts)?;
    if opts.strict {
        check_trailing(&opened, &prelude)?;
    }
    let mut seq = decode_sequence(
        &opened.codestream,
        &prelude,
        None,
        SequenceMode::FirstPresented,
    )?;
    let frame = seq
        .frames
        .pop()
        .ok_or_else(|| Error::invalid("jxl decoder: no presentable frame"))?;
    pack_frame(
        frame.frame,
        frame.bytes_per_sample,
        &prelude,
        opened.file.as_ref(),
        true,
    )
}

/// Decode to tightly packed RGB8 (alpha dropped, grey replicated).
pub fn decode_rgb8(bytes: &[u8]) -> Result<crate::RgbImage> {
    let img = decode(bytes)?;
    Ok(crate::RgbImage::new(
        img.width,
        img.height,
        img.try_to_rgb8()?,
    ))
}

/// Decode to tightly packed RGBA8 (alpha 255 when the image has none).
pub fn decode_rgba8(bytes: &[u8]) -> Result<crate::RgbaImage> {
    let img = decode(bytes)?;
    Ok(crate::RgbaImage::new(
        img.width,
        img.height,
        img.try_to_rgba8()?,
    ))
}

/// Decode every presented frame (animations: in display order,
/// §C.2-composed) with default options. A still image yields one frame.
pub fn decode_all(bytes: &[u8]) -> Result<Vec<Frame>> {
    decode_all_with(bytes, &DecodeOptions::default())
}

/// [`decode_all`] with options. With [`DecodeOptions::coalesce`] off,
/// every frame of the array is returned un-composed at its own
/// frame-rectangle geometry (see [`Frame::x`]).
pub fn decode_all_with(bytes: &[u8], opts: &DecodeOptions) -> Result<Vec<Frame>> {
    let opened = open(bytes)?;
    let prelude = read_prelude_at_level(&opened.codestream, opened.level())?;
    check_limits(&prelude, opts)?;
    if opts.strict {
        check_trailing(&opened, &prelude)?;
    }
    let mode = if opts.coalesce {
        SequenceMode::Coalesced
    } else {
        SequenceMode::Layers
    };
    let seq = decode_sequence(&opened.codestream, &prelude, None, mode)?;
    let anim = animation_info(&prelude.metadata);
    let mut out = Vec::with_capacity(seq.frames.len());
    for (i, f) in seq.frames.into_iter().enumerate() {
        let SequenceFrame {
            frame,
            bytes_per_sample,
            duration,
            index,
            x0,
            y0,
            presented,
        } = f;
        // Metadata rides on the first frame only (one ICC / Exif / XMP
        // per file).
        let image = pack_frame(
            frame,
            bytes_per_sample,
            &prelude,
            opened.file.as_ref(),
            i == 0,
        )?;
        out.push(Frame {
            image,
            delay: anim.map(|a| a.tick_duration(duration)),
            index,
            ticks: duration,
            x: x0,
            y: y0,
            presented,
        });
    }
    Ok(out)
}

/// Read `r` to end and [`decode`] it.
pub fn decode_from<R: Read>(mut r: R) -> Result<JxlImage> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    decode(&buf)
}

const ENCODE_UNSUPPORTED: &str = "JPEG XL encoding is not implemented";

/// JPEG XL encoding is not implemented in this crate; always returns
/// [`Error::Unsupported`].
pub fn encode(_image: &JxlImage, _opts: &EncodeOptions) -> Result<Vec<u8>> {
    Err(Error::unsupported(ENCODE_UNSUPPORTED))
}

/// See [`encode`].
pub fn encode_rgb8(
    _width: u32,
    _height: u32,
    _rgb: &[u8],
    _opts: &EncodeOptions,
) -> Result<Vec<u8>> {
    Err(Error::unsupported(ENCODE_UNSUPPORTED))
}

/// See [`encode`].
pub fn encode_rgba8(
    _width: u32,
    _height: u32,
    _rgba: &[u8],
    _opts: &EncodeOptions,
) -> Result<Vec<u8>> {
    Err(Error::unsupported(ENCODE_UNSUPPORTED))
}

/// See [`encode`].
pub fn encode_to<W: Write>(_image: &JxlImage, _opts: &EncodeOptions, _w: W) -> Result<()> {
    Err(Error::unsupported(ENCODE_UNSUPPORTED))
}

/// Multi-frame mirror of [`decode_all`] (animations); see [`encode`] —
/// always [`Error::Unsupported`] in this decoder-only crate.
pub fn encode_all(_frames: &[Frame], _opts: &EncodeOptions) -> Result<Vec<u8>> {
    Err(Error::unsupported(ENCODE_UNSUPPORTED))
}

/// Parse the committee-draft (2019) `SizeHeader` + `ImageMetadata`
/// preamble. The 2024 bundle layout is [`crate::probe_fdis`]; the
/// contract-level header view is [`info`].
pub fn headers(input: &[u8]) -> Result<crate::Headers> {
    crate::metadata::parse_headers(input)
}

/// Signature detection returning the wrapping kind.
#[deprecated(note = "use oxideav_jpegxl::probe (IMAGE_CRATE_API); container::detect for the kind")]
pub fn detect(data: &[u8]) -> Option<Signature> {
    container::detect(data)
}

// ---- helpers -----------------------------------------------------------

fn colour_channels(md: &ImageMetadataFdis) -> usize {
    if md.colour_encoding.colour_space == ColourSpace::Grey {
        1
    } else {
        3
    }
}

/// Index (among the extra channels) of the first alpha channel.
fn alpha_channel_index(md: &ImageMetadataFdis) -> Option<usize> {
    md.extra_channel_info
        .iter()
        .position(|ec| ec.kind == ExtraChannelType::Alpha)
}

/// Image size after the orientation transform (5..=8 transpose).
fn oriented_size(prelude: &Prelude) -> (u32, u32) {
    let (w, h) = (prelude.size.width, prelude.size.height);
    if prelude.metadata.orientation >= 5 {
        (h, w)
    } else {
        (w, h)
    }
}

fn animation_info(md: &ImageMetadataFdis) -> Option<AnimationInfo> {
    if !md.have_animation {
        return None;
    }
    md.animation.map(|a| AnimationInfo {
        ticks_per_second: (a.tps_numerator, a.tps_denominator),
        loops: a.num_loops,
    })
}

/// [`ColorInfo`] from the §A.6 `ColourEncoding`. The enum values are
/// H.273 code points by construction (Tables A.9 – A.12); `Custom`
/// primaries map to unspecified (2), a `have_gamma` transfer to
/// unspecified with the exponent in [`Metadata::gamma`]. With an ICC
/// profile the enumerated fields are not meaningful and only the
/// (always full) range is reported.
fn color_info(md: &ImageMetadataFdis) -> ColorInfo {
    let ce = &md.colour_encoding;
    if ce.want_icc {
        return ColorInfo::unspecified().with_range(ColorRange::Full);
    }
    let primaries = match ce.primaries {
        Primaries::SRgb => ColorInfo::PRIMARIES_BT709,
        Primaries::P2100 => ColorInfo::PRIMARIES_BT2100,
        Primaries::P3 => ColorInfo::PRIMARIES_P3,
        Primaries::Custom => ColorInfo::UNSPECIFIED,
    };
    let transfer = if ce.tf.have_gamma {
        ColorInfo::UNSPECIFIED
    } else {
        match ce.tf.transfer_function {
            TransferFunction::Bt709 => 1,
            TransferFunction::Unknown => 2,
            TransferFunction::Linear => 8,
            TransferFunction::SRgb => 13,
            TransferFunction::Pq => 16,
            TransferFunction::Dci => 17,
            TransferFunction::Hlg => 18,
        }
    };
    ColorInfo::new(
        ColorRange::Full,
        primaries,
        transfer,
        ColorInfo::MATRIX_IDENTITY,
    )
}

fn gamma(md: &ImageMetadataFdis) -> Option<f32> {
    let ce = &md.colour_encoding;
    if ce.want_icc || !ce.tf.have_gamma {
        return None;
    }
    Some(ce.tf.gamma as f32 / 10_000_000.0)
}

/// Exif / XMP from the container boxes (first of each kind).
fn container_metadata(file: Option<&container::JxlFile<'_>>, meta: &mut Metadata) {
    let Some(file) = file else { return };
    for m in &file.metadata {
        match m.kind {
            MetadataKind::Exif if meta.exif.is_none() => {
                if let Ok(content) = m.content(METADATA_BOX_CAP) {
                    // 18181-2 Table 6: u32 `tiff header offset` then the
                    // Exif payload; expose the payload from its TIFF
                    // header onwards.
                    if content.len() >= 4 {
                        let off =
                            u32::from_be_bytes([content[0], content[1], content[2], content[3]])
                                as usize;
                        let start = 4usize.saturating_add(off).min(content.len());
                        meta.exif = Some(content[start..].to_vec());
                    }
                }
            }
            MetadataKind::Xml if meta.xmp.is_none() => {
                if let Ok(content) = m.content(METADATA_BOX_CAP) {
                    meta.xmp = Some(content.into_owned());
                }
            }
            _ => {}
        }
    }
}

/// Enforce [`DecodeOptions`] against the header geometry before any
/// pixel buffer exists. `max_bytes` is checked against the native
/// output size (`width × height × channels × bytes_per_sample`, alpha
/// included).
fn check_limits(prelude: &Prelude, opts: &DecodeOptions) -> Result<()> {
    let (w, h) = oriented_size(prelude);
    if let Some(mw) = opts.max_width {
        if w > mw {
            return Err(Error::limit(format!("width {w} exceeds max_width {mw}")));
        }
    }
    if let Some(mh) = opts.max_height {
        if h > mh {
            return Err(Error::limit(format!("height {h} exceeds max_height {mh}")));
        }
    }
    let pixels = u64::from(w) * u64::from(h);
    if let Some(mp) = opts.max_pixels {
        if pixels > mp {
            return Err(Error::limit(format!(
                "{pixels} pixels exceed max_pixels {mp}"
            )));
        }
    }
    let md = &prelude.metadata;
    let channels = colour_channels(md) as u64 + u64::from(alpha_channel_index(md).is_some());
    let bps: u64 = if md.bit_depth.bits_per_sample > 8 {
        2
    } else {
        1
    };
    let bytes = pixels.saturating_mul(channels).saturating_mul(bps);
    if let Some(mb) = opts.max_bytes {
        if bytes > mb {
            return Err(Error::limit(format!(
                "decoded image of {bytes} bytes exceeds max_bytes {mb}"
            )));
        }
    }
    if bytes > usize::MAX as u64 {
        return Err(Error::unsupported(format!(
            "decoded image of {bytes} bytes cannot be addressed on this platform"
        )));
    }
    Ok(())
}

/// `strict`: a raw codestream must end with its `is_last` frame.
fn check_trailing(opened: &Opened<'_>, prelude: &Prelude) -> Result<()> {
    if opened.signature != Signature::RawCodestream {
        return Ok(());
    }
    let walk = walk_frame_headers(&opened.codestream, prelude)?;
    let len = opened.codestream.len();
    if walk.end_offset < len {
        return Err(Error::invalid(format!(
            "jxl decoder: {} trailing byte(s) after the last frame",
            len - walk.end_offset
        )));
    }
    Ok(())
}

/// Interleave the per-channel planes of a decoded frame into the
/// contract's packed layout: colour channels (1 or 3) then the alpha
/// extra channel when the header declares one; other extra channels
/// (depth, spot colours, selection masks, …) are not carried.
pub(crate) fn pack_frame(
    frame: RawFrame,
    bytes: usize,
    prelude: &Prelude,
    file: Option<&container::JxlFile<'_>>,
    with_metadata: bool,
) -> Result<JxlImage> {
    let md = &prelude.metadata;
    let n_colour = colour_channels(md);
    if frame.planes.len() < n_colour {
        return Err(Error::invalid(format!(
            "jxl decoder: frame carries {} plane(s), {n_colour} colour channel(s) expected",
            frame.planes.len()
        )));
    }
    let p0 = &frame.planes[0];
    if p0.stride == 0 || p0.data.len() < p0.stride || p0.stride % bytes != 0 {
        return Err(Error::invalid("jxl decoder: degenerate plane geometry"));
    }
    let width = p0.stride / bytes;
    let height = p0.data.len() / p0.stride;
    if width == 0 || height == 0 || width > u32::MAX as usize || height > u32::MAX as usize {
        return Err(Error::invalid("jxl decoder: degenerate frame geometry"));
    }
    // A kGrey image may still arrive as three colour planes: the VarDCT
    // path and the §L.2.2 inverse XYB always produce (R, G, B). The
    // grey output is the GREEN plane — the luminance carrier of the
    // opsin matrix (R = G = B for genuinely grey content up to
    // quantisation; the same arbitration the Modular XYB path applies).
    let physical_colour = if n_colour == 1 && frame.planes.len() >= 3 {
        3
    } else {
        n_colour
    };
    let alpha_plane = alpha_channel_index(md)
        .map(|k| physical_colour + k)
        .filter(|&i| i < frame.planes.len());
    let mut chans: Vec<&Plane> = if physical_colour == 3 && n_colour == 1 {
        vec![&frame.planes[1]]
    } else {
        frame.planes[..n_colour].iter().collect()
    };
    if let Some(a) = alpha_plane {
        chans.push(&frame.planes[a]);
    }
    for (c, p) in chans.iter().enumerate() {
        if p.stride != p0.stride || p.data.len() < p0.stride * height {
            return Err(Error::unsupported(format!(
                "jxl decoder: channel {c} geometry ({} × {} bytes) differs from channel 0 \
                 ({} × {height}); sub-sampled extra channels are not packed",
                p.stride,
                p.data.len(),
                p0.stride
            )));
        }
    }
    let format = PixelFormat::for_layout(n_colour, alpha_plane.is_some(), bytes)?;
    let n = chans.len();
    let row = width * n * bytes;
    let mut data = vec![0u8; row * height];
    for y in 0..height {
        let dst = &mut data[y * row..(y + 1) * row];
        for (c, p) in chans.iter().enumerate() {
            let src = &p.data[y * p0.stride..y * p0.stride + width * bytes];
            if bytes == 1 {
                for x in 0..width {
                    dst[x * n + c] = src[x];
                }
            } else {
                for x in 0..width {
                    let o = (x * n + c) * 2;
                    dst[o] = src[2 * x];
                    dst[o + 1] = src[2 * x + 1];
                }
            }
        }
    }
    let declared = md.bit_depth.bits_per_sample.clamp(1, 16) as u8;
    let bits = if bytes == 1 {
        declared.min(8)
    } else {
        declared
    };
    let mut img = JxlImage::packed(width as u32, height as u32, format, data)?
        .with_color(color_info(md))
        .with_bits_per_sample(bits);
    if with_metadata {
        let mut meta = Metadata::new().with_gamma(gamma(md));
        meta.icc = prelude.icc.clone();
        container_metadata(file, &mut meta);
        img.metadata = meta;
    }
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_is_total() {
        assert!(!probe(&[]));
        assert!(!probe(&[0xFF]));
        assert!(probe(&[0xFF, 0x0A]));
        assert!(!probe(&[0x89, b'P', b'N', b'G']));
        assert!(probe(&[
            0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A
        ]));
    }

    #[test]
    fn tick_duration_uses_the_fraction() {
        let a = AnimationInfo {
            ticks_per_second: (1000, 1),
            loops: 0,
        };
        assert_eq!(a.tick_duration(250), Duration::from_millis(250));
        let b = AnimationInfo {
            ticks_per_second: (30000, 1001),
            loops: 0,
        };
        assert_eq!(b.tick_duration(1).as_nanos(), 33_366_666);
    }

    #[test]
    fn encode_is_unsupported() {
        let img = JxlImage::from_rgb8(1, 1, vec![0, 0, 0]).unwrap();
        assert!(matches!(
            encode(&img, &EncodeOptions::default()),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            encode_rgb8(1, 1, &[0, 0, 0], &EncodeOptions::default()),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            encode_rgba8(1, 1, &[0, 0, 0, 0], &EncodeOptions::default()),
            Err(Error::Unsupported(_))
        ));
        let mut sink = Vec::new();
        assert!(matches!(
            encode_to(&img, &EncodeOptions::default(), &mut sink),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn info_and_decode_fail_cleanly_on_a_bare_signature() {
        assert!(matches!(info(&[0xFF, 0x0A]), Err(Error::InvalidData(_))));
        assert!(matches!(decode(&[0xFF, 0x0A]), Err(Error::InvalidData(_))));
        assert!(matches!(info(b"PNG!"), Err(Error::InvalidData(_))));
    }
}
