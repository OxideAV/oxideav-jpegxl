//! The contract image records: [`JxlImage`], [`Plane`], [`ColorInfo`],
//! [`Metadata`], [`RgbImage`] / [`RgbaImage`] and the
//! [`JxlPixelFormat`] tag.
//!
//! Shapes follow `IMAGE_CRATE_API.md` so every OxideAV image crate reads
//! the same way; the names and fields are a copy, not a shared crate.
//! JPEG XL has no indexed output layout (the Modular palette transform
//! is undone inside the decoder), so `JxlImage` carries no `palette`
//! field.

use crate::error::{Error, Result};

/// Native pixel layouts this crate produces. Variant names mirror
/// `oxideav_core::PixelFormat`.
///
/// The packed layouts are one row-major plane. The `*16Le` / `48Le` /
/// `64Le` variants hold little-endian `u16` samples in the range
/// `0 ..= 2^bits_per_sample − 1` (see [`JxlImage::bits_per_sample`]);
/// the 8-bit variants hold `0 ..= 2^bits_per_sample − 1` for
/// `bits_per_sample ≤ 8`.
///
/// The planar `YuvJ444P` / `YuvJ422P` / `YuvJ420P` / `Yuv440P` layouts
/// are three 8-bit planes (`Y`, `Cb`, `Cr`) and are produced for a
/// losslessly recompressed JPEG (`do_YCbCr` VarDCT frame): the JPEG's
/// own full-range YCbCr samples on the JPEG's sampling lattice, chroma
/// planes `ceil(width / h) × ceil(height / v)`. They are what the
/// original JPEG decodes to (see the README's layout table).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum JxlPixelFormat {
    /// One grey sample per pixel, 1 byte.
    Gray8,
    /// Grey + alpha, 2 bytes per pixel.
    Ya8,
    /// R, G, B — 3 bytes per pixel.
    Rgb24,
    /// R, G, B, A — 4 bytes per pixel.
    Rgba,
    /// One grey sample per pixel, little-endian `u16`.
    Gray16Le,
    /// Grey + alpha, two little-endian `u16` per pixel.
    Ya16Le,
    /// R, G, B as little-endian `u16` (6 bytes per pixel).
    Rgb48Le,
    /// R, G, B, A as little-endian `u16` (8 bytes per pixel).
    Rgba64Le,
    /// Planar 8-bit full-range YCbCr 4:4:4 (JPEG transcode; three
    /// planes of `width × height`).
    YuvJ444P,
    /// Planar 8-bit full-range YCbCr 4:2:2 (chroma `ceil(width / 2) ×
    /// height`).
    YuvJ422P,
    /// Planar 8-bit full-range YCbCr 4:2:0 (chroma `ceil(width / 2) ×
    /// ceil(height / 2)`).
    YuvJ420P,
    /// Planar 8-bit YCbCr 4:4:0 (chroma `width × ceil(height / 2)`).
    /// Full range like every JPEG transcode — `oxideav_core` has no
    /// `YuvJ440P` label, so the range rides on [`ColorInfo::range`].
    Yuv440P,
}

/// Contract alias of [`JxlPixelFormat`].
pub type PixelFormat = JxlPixelFormat;

impl JxlPixelFormat {
    /// Bytes per packed pixel; for the planar YCbCr layouts, bytes per
    /// luma sample (1).
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray8 => 1,
            Self::Ya8 => 2,
            Self::Rgb24 => 3,
            Self::Rgba => 4,
            Self::Gray16Le => 2,
            Self::Ya16Le => 4,
            Self::Rgb48Le => 6,
            Self::Rgba64Le => 8,
            Self::YuvJ444P | Self::YuvJ422P | Self::YuvJ420P | Self::Yuv440P => 1,
        }
    }

    /// Number of channels (1, 2, 3 or 4): interleaved for the packed
    /// layouts, one plane each for the planar ones.
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray8 | Self::Gray16Le => 1,
            Self::Ya8 | Self::Ya16Le => 2,
            Self::Rgb24 | Self::Rgb48Le => 3,
            Self::Rgba | Self::Rgba64Le => 4,
            Self::YuvJ444P | Self::YuvJ422P | Self::YuvJ420P | Self::Yuv440P => 3,
        }
    }

    /// Bytes per sample (1 for the 8-bit family, 2 for the 16-bit one).
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Gray8 | Self::Ya8 | Self::Rgb24 | Self::Rgba => 1,
            Self::Gray16Le | Self::Ya16Le | Self::Rgb48Le | Self::Rgba64Le => 2,
            Self::YuvJ444P | Self::YuvJ422P | Self::YuvJ420P | Self::Yuv440P => 1,
        }
    }

    /// Whether the layout is planar YCbCr (three planes) rather than one
    /// packed plane.
    pub const fn is_planar(self) -> bool {
        matches!(
            self,
            Self::YuvJ444P | Self::YuvJ422P | Self::YuvJ420P | Self::Yuv440P
        )
    }

    /// Number of planes a [`JxlImage`] in this layout carries (1 or 3).
    pub const fn plane_count(self) -> usize {
        if self.is_planar() {
            3
        } else {
            1
        }
    }

    /// Chroma subsampling of the planar layouts as `(log2 horizontal,
    /// log2 vertical)` divisors; `(0, 0)` for 4:4:4 and every packed
    /// layout.
    pub const fn chroma_shift(self) -> (u32, u32) {
        match self {
            Self::YuvJ420P => (1, 1),
            Self::YuvJ422P => (1, 0),
            Self::Yuv440P => (0, 1),
            _ => (0, 0),
        }
    }

    /// Sample geometry `(samples per row, rows)` of plane `index` for a
    /// `width × height` image: the packed layouts have one plane of
    /// `width` pixels; planar chroma planes are `ceil(width / h) ×
    /// ceil(height / v)`. `None` when `index ≥ plane_count()`.
    pub fn plane_dims(self, width: u32, height: u32, index: usize) -> Option<(usize, usize)> {
        if index >= self.plane_count() {
            return None;
        }
        let (w, h) = (width as usize, height as usize);
        if index == 0 || !self.is_planar() {
            return Some((w, h));
        }
        let (hs, vs) = self.chroma_shift();
        Some((w.div_ceil(1 << hs), h.div_ceil(1 << vs)))
    }

    /// Whether the layout carries an alpha channel.
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::Ya8 | Self::Ya16Le | Self::Rgba | Self::Rgba64Le)
    }

    /// Whether the layout is grey (one colour channel).
    pub const fn is_gray(self) -> bool {
        matches!(
            self,
            Self::Gray8 | Self::Gray16Le | Self::Ya8 | Self::Ya16Le
        )
    }

    /// The layout for `colour_channels` (1 or 3) colour channels, an
    /// optional alpha channel and a storage width of 1 or 2 bytes.
    pub(crate) fn for_layout(
        colour_channels: usize,
        alpha: bool,
        bytes_per_sample: usize,
    ) -> Result<Self> {
        Ok(match (colour_channels, alpha, bytes_per_sample) {
            (1, false, 1) => Self::Gray8,
            (1, true, 1) => Self::Ya8,
            (3, false, 1) => Self::Rgb24,
            (3, true, 1) => Self::Rgba,
            (1, false, 2) => Self::Gray16Le,
            (1, true, 2) => Self::Ya16Le,
            (3, false, 2) => Self::Rgb48Le,
            (3, true, 2) => Self::Rgba64Le,
            _ => {
                return Err(Error::unsupported(format!(
                    "JPEG XL: no packed layout for {colour_channels} colour channel(s), \
                     alpha={alpha}, {bytes_per_sample} byte(s) per sample"
                )))
            }
        })
    }
}

/// One image plane: `stride` bytes per row, `data` row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Plane {
    /// Bytes per row.
    pub stride: usize,
    /// Row-major sample bytes, `stride × rows` long.
    pub data: Vec<u8>,
}

impl Plane {
    /// Build a plane from its stride and bytes.
    pub fn new(stride: usize, data: Vec<u8>) -> Self {
        Self { stride, data }
    }
}

/// Internal decoded-frame carrier: the per-channel planes a decode
/// stage hands to the next one (colour channels first, extra channels
/// after, Annex G.1.3 order), 1 or 2 bytes per sample little-endian.
///
/// Not part of the stable API; the contract surface is [`JxlImage`].
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawFrame {
    /// Caller-supplied presentation timestamp, threaded through
    /// unchanged.
    pub pts: Option<i64>,
    /// One plane per decoded channel.
    pub planes: Vec<Plane>,
}

/// Nominal sample range of [`ColorInfo`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ColorRange {
    /// Not signalled.
    #[default]
    Unspecified,
    /// Limited ("video") range.
    Limited,
    /// Full range — every JPEG XL sample is full range.
    Full,
}

/// Colour signalling as H.273 code points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ColorInfo {
    /// Sample range.
    pub range: ColorRange,
    /// H.273 `ColourPrimaries` (1 = BT.709/sRGB, 9 = BT.2100, 11 = P3,
    /// 2 = unspecified / custom).
    pub primaries: u8,
    /// H.273 `TransferCharacteristics` (13 = sRGB, 1 = BT.709,
    /// 8 = linear, 16 = PQ, 17 = DCI, 18 = HLG, 2 = unspecified /
    /// gamma — see [`Metadata::gamma`]).
    pub transfer: u8,
    /// H.273 `MatrixCoefficients`; always 0 (identity, RGB output).
    pub matrix: u8,
}

impl ColorInfo {
    /// H.273 "unspecified" code point.
    pub const UNSPECIFIED: u8 = 2;
    /// H.273 identity matrix (RGB).
    pub const MATRIX_IDENTITY: u8 = 0;
    /// H.273 matrix 5 (BT.601 / T.871 sYCC) — the YCbCr relationship of
    /// the planar JPEG-transcode layouts.
    pub const MATRIX_BT601: u8 = 5;
    /// H.273 BT.709 / sRGB primaries.
    pub const PRIMARIES_BT709: u8 = 1;
    /// H.273 BT.2100 primaries.
    pub const PRIMARIES_BT2100: u8 = 9;
    /// H.273 P3 (SMPTE RP 431-2) primaries.
    pub const PRIMARIES_P3: u8 = 11;
    /// H.273 sRGB transfer.
    pub const TRANSFER_SRGB: u8 = 13;
    /// H.273 BT.709 transfer.
    pub const TRANSFER_BT709: u8 = 1;
    /// H.273 linear transfer.
    pub const TRANSFER_LINEAR: u8 = 8;

    /// Build from explicit code points.
    pub const fn new(range: ColorRange, primaries: u8, transfer: u8, matrix: u8) -> Self {
        Self {
            range,
            primaries,
            transfer,
            matrix,
        }
    }

    /// Nothing signalled.
    pub const fn unspecified() -> Self {
        Self::new(
            ColorRange::Unspecified,
            Self::UNSPECIFIED,
            Self::UNSPECIFIED,
            Self::MATRIX_IDENTITY,
        )
    }

    /// Full-range sRGB — the JPEG XL `ColourEncoding` default
    /// (`all_default`: sRGB primaries, D65, sRGB transfer).
    pub const fn srgb() -> Self {
        Self::new(
            ColorRange::Full,
            Self::PRIMARIES_BT709,
            Self::TRANSFER_SRGB,
            Self::MATRIX_IDENTITY,
        )
    }

    /// Builder: replace `range`.
    pub fn with_range(mut self, range: ColorRange) -> Self {
        self.range = range;
        self
    }

    /// Builder: replace `primaries`.
    pub fn with_primaries(mut self, primaries: u8) -> Self {
        self.primaries = primaries;
        self
    }

    /// Builder: replace `transfer`.
    pub fn with_transfer(mut self, transfer: u8) -> Self {
        self.transfer = transfer;
        self
    }

    /// Builder: replace `matrix`.
    pub fn with_matrix(mut self, matrix: u8) -> Self {
        self.matrix = matrix;
        self
    }

    /// Whether any field differs from [`ColorInfo::unspecified`].
    pub fn is_specified(&self) -> bool {
        *self != Self::unspecified()
    }
}

impl Default for ColorInfo {
    fn default() -> Self {
        Self::unspecified()
    }
}

/// Embedded metadata.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct Metadata {
    /// ICC profile bytes (the codestream's Annex B encoded ICC
    /// stream, reconstructed).
    pub icc: Option<Vec<u8>>,
    /// Exif payload from the container's `Exif` box, starting at the
    /// TIFF header (the 4-byte `tiff header offset` prefix applied).
    pub exif: Option<Vec<u8>>,
    /// XMP document from the container's `xml ` box.
    pub xmp: Option<Vec<u8>>,
    /// Encoding gamma exponent when the colour encoding signals
    /// `have_gamma` (ISO/IEC 18181-1 `gamma / 10^7`, in `(0, 1]`, the
    /// same convention as PNG `gAMA`).
    pub gamma: Option<f32>,
}

impl Metadata {
    /// Empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set `icc`.
    pub fn with_icc(mut self, icc: impl Into<Option<Vec<u8>>>) -> Self {
        self.icc = icc.into();
        self
    }

    /// Builder: set `exif`.
    pub fn with_exif(mut self, exif: impl Into<Option<Vec<u8>>>) -> Self {
        self.exif = exif.into();
        self
    }

    /// Builder: set `xmp`.
    pub fn with_xmp(mut self, xmp: impl Into<Option<Vec<u8>>>) -> Self {
        self.xmp = xmp.into();
        self
    }

    /// Builder: set `gamma`.
    pub fn with_gamma(mut self, gamma: impl Into<Option<f32>>) -> Self {
        self.gamma = gamma.into();
        self
    }

    /// `true` when no field is set.
    pub fn is_empty(&self) -> bool {
        self.icc.is_none() && self.exif.is_none() && self.xmp.is_none() && self.gamma.is_none()
    }
}

/// Tightly packed RGB8, 3 bytes per pixel, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RgbImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `3 × width × height` bytes.
    pub data: Vec<u8>,
}

impl RgbImage {
    /// Wrap a tightly packed RGB8 buffer (not validated; the decode
    /// paths always hand over `3 × width × height` bytes).
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

    /// The pixel bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume into the pixel bytes.
    pub fn into_raw(self) -> Vec<u8> {
        self.data
    }
}

/// Tightly packed RGBA8, 4 bytes per pixel, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RgbaImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `4 × width × height` bytes.
    pub data: Vec<u8>,
}

impl RgbaImage {
    /// Wrap a tightly packed RGBA8 buffer (not validated; the decode
    /// paths always hand over `4 × width × height` bytes).
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

    /// The pixel bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume into the pixel bytes.
    pub fn into_raw(self) -> Vec<u8> {
        self.data
    }
}

/// A decoded JPEG XL image in its native packed layout.
///
/// The decoder produces exactly one plane (packed samples, stride
/// `width × bytes_per_pixel`). `bits_per_sample` is the codestream's
/// declared integer depth (1 ..= 16); samples never exceed
/// `2^bits_per_sample − 1`, and [`to_rgb8`](Self::to_rgb8) scales by
/// that maximum, not by the storage word.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct JxlImage {
    /// Width in pixels (after the Table A.17 orientation transform).
    pub width: u32,
    /// Height in pixels (after the orientation transform).
    pub height: u32,
    /// Native layout.
    pub format: PixelFormat,
    /// Exactly one packed plane, or the three `Y`, `Cb`, `Cr` planes of a
    /// planar layout ([`JxlPixelFormat::plane_count`]).
    pub planes: Vec<Plane>,
    /// Colour signalling from the codestream's `ColourEncoding`.
    pub color: ColorInfo,
    /// ICC / Exif / XMP / gamma.
    pub metadata: Metadata,
    /// Declared integer sample depth, 1 ..= 16. Equals 8 for every 8-bit
    /// layout produced by the VarDCT path and for 8-bit Modular images;
    /// deeper Modular images store `bits_per_sample`-bit values in the
    /// `u16` layouts.
    pub bits_per_sample: u8,
}

impl JxlImage {
    /// Build an image from its planes, validating the geometry: both
    /// dimensions `> 0`, [`JxlPixelFormat::plane_count`] planes, and for
    /// every plane `stride ≥ samples per row × bytes_per_sample` and
    /// `data.len() ≥ stride × rows` with the per-plane geometry of
    /// [`JxlPixelFormat::plane_dims`] ([`Error::InvalidData`]
    /// otherwise). `bits_per_sample` is set to the layout's storage
    /// width (8 or 16).
    pub fn new(width: u32, height: u32, format: PixelFormat, planes: Vec<Plane>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::invalid(format!(
                "JxlImage::new: {width}×{height} image (both dimensions must be > 0)"
            )));
        }
        let want = format.plane_count();
        if planes.len() != want {
            return Err(Error::invalid(format!(
                "JxlImage::new: {format:?} needs exactly {want} plane(s), got {}",
                planes.len()
            )));
        }
        for (i, p) in planes.iter().enumerate() {
            let (pw, ph) = format
                .plane_dims(width, height, i)
                .expect("index below plane_count");
            let row = pw
                .checked_mul(format.bytes_per_pixel())
                .ok_or_else(|| Error::invalid("JxlImage::new: row size overflows usize"))?;
            if p.stride < row {
                return Err(Error::invalid(format!(
                    "JxlImage::new: plane {i} stride {} shorter than the {row}-byte row",
                    p.stride
                )));
            }
            let need = p
                .stride
                .checked_mul(ph)
                .ok_or_else(|| Error::invalid("JxlImage::new: plane size overflows usize"))?;
            if p.data.len() < need {
                return Err(Error::invalid(format!(
                    "JxlImage::new: plane {i} holds {} bytes, {need} needed for {pw}×{ph}",
                    p.data.len()
                )));
            }
        }
        Ok(Self {
            width,
            height,
            format,
            planes,
            color: ColorInfo::unspecified(),
            metadata: Metadata::new(),
            bits_per_sample: (format.bytes_per_sample() * 8) as u8,
        })
    }

    /// Build from one tightly packed buffer in `format` (stride
    /// `width × bytes_per_pixel`).
    pub fn packed(width: u32, height: u32, format: PixelFormat, data: Vec<u8>) -> Result<Self> {
        let stride = (width as usize)
            .checked_mul(format.bytes_per_pixel())
            .ok_or_else(|| Error::invalid("JxlImage::packed: row size overflows usize"))?;
        Self::new(width, height, format, vec![Plane::new(stride, data)])
    }

    /// Packed `Rgb24` from `3 × width × height` bytes.
    pub fn from_rgb8(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        Self::packed(width, height, PixelFormat::Rgb24, data)
    }

    /// Packed `Rgba` from `4 × width × height` bytes.
    pub fn from_rgba8(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        Self::packed(width, height, PixelFormat::Rgba, data)
    }

    /// Builder: replace `color`.
    pub fn with_color(mut self, color: ColorInfo) -> Self {
        self.color = color;
        self
    }

    /// Builder: replace `metadata`.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Builder: declare the significant sample depth (1 ..= storage
    /// bits). Values outside that range are clamped into it.
    pub fn with_bits_per_sample(mut self, bits: u8) -> Self {
        let max = (self.format.bytes_per_sample() * 8) as u8;
        self.bits_per_sample = bits.clamp(1, max);
        self
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Native layout.
    pub fn format(&self) -> PixelFormat {
        self.format
    }

    /// Bytes per row of the packed plane (the luma plane of a planar
    /// layout).
    pub fn stride(&self) -> usize {
        self.planes.first().map(|p| p.stride).unwrap_or(0)
    }

    /// The packed sample bytes: `Some` for the packed layouts, `None`
    /// for the planar YCbCr ones (use [`into_raw`](Self::into_raw) or
    /// read `planes` directly).
    pub fn as_bytes(&self) -> Option<&[u8]> {
        if self.format.is_planar() {
            return None;
        }
        self.planes.first().map(|p| p.data.as_slice())
    }

    /// Consume into the packed sample bytes.
    pub fn into_raw(self) -> Vec<u8> {
        let mut planes = self.planes;
        match planes.len() {
            0 => Vec::new(),
            1 => planes.swap_remove(0).data,
            _ => planes.into_iter().flat_map(|p| p.data).collect(),
        }
    }

    /// Maximum sample value, `2^bits_per_sample − 1`.
    fn sample_max(&self) -> u32 {
        (1u32 << self.bits_per_sample.clamp(1, 16)) - 1
    }

    /// Read channel `c` of pixel `(x, y)` as a raw integer sample.
    #[inline]
    fn sample(&self, p: &Plane, x: usize, y: usize, c: usize) -> u32 {
        let bps = self.format.bytes_per_sample();
        let i = y * p.stride + (x * self.format.channels() + c) * bps;
        if bps == 1 {
            u32::from(p.data[i])
        } else {
            u32::from(u16::from_le_bytes([p.data[i], p.data[i + 1]]))
        }
    }

    /// Scale a sample in `0 ..= max` to `0 ..= 255`, rounding to nearest.
    #[inline]
    fn to8(v: u32, max: u32) -> u8 {
        if max == 255 {
            return v.min(255) as u8;
        }
        let v = v.min(max);
        ((v * 255 + max / 2) / max) as u8
    }

    /// Convert to tightly packed RGB8 (grey replicated to R = G = B,
    /// alpha dropped). Deeper samples are scaled by
    /// `255 / (2^bits_per_sample − 1)` with round-to-nearest. The planar
    /// YCbCr layouts are upsampled with the ISO/IEC 18181-1 J.2 triangle
    /// filter and converted with the §L.3 (T.871) full-range matrix.
    pub fn to_rgb8(&self) -> Vec<u8> {
        self.try_to_rgb8().unwrap_or_default()
    }

    /// Convert to tightly packed RGBA8 (alpha 255 when the layout has
    /// none; deeper samples scaled as in [`to_rgb8`](Self::to_rgb8)).
    pub fn to_rgba8(&self) -> Vec<u8> {
        self.try_to_rgba8().unwrap_or_default()
    }

    /// [`to_rgb8`](Self::to_rgb8) that reports a plane geometry
    /// mismatch instead of returning an empty buffer.
    pub fn try_to_rgb8(&self) -> Result<Vec<u8>> {
        self.convert(3)
    }

    /// [`to_rgba8`](Self::to_rgba8) that reports a plane geometry
    /// mismatch instead of returning an empty buffer.
    pub fn try_to_rgba8(&self) -> Result<Vec<u8>> {
        self.convert(4)
    }

    fn convert(&self, out_channels: usize) -> Result<Vec<u8>> {
        let (w, h) = (self.width as usize, self.height as usize);
        if self.format.is_planar() {
            let [y, cb, cr] = match self.planes.as_slice() {
                [y, cb, cr] => [y, cb, cr],
                _ => {
                    return Err(Error::invalid(format!(
                        "JxlImage: {:?} needs three planes, found {}",
                        self.format,
                        self.planes.len()
                    )))
                }
            };
            return crate::jpeg_pixels::planar_ycbcr_to_rgb(
                w,
                h,
                (&y.data, y.stride),
                (&cb.data, cb.stride),
                (&cr.data, cr.stride),
                self.format.chroma_shift(),
                out_channels,
            );
        }
        let p = self
            .planes
            .first()
            .ok_or_else(|| Error::invalid("JxlImage: no plane"))?;
        let row = w * self.format.bytes_per_pixel();
        if p.stride < row || p.data.len() < p.stride.saturating_mul(h) {
            return Err(Error::invalid(
                "JxlImage: plane geometry does not cover width × height",
            ));
        }
        let max = self.sample_max();
        let chans = self.format.channels();
        let gray = self.format.is_gray();
        let alpha = self.format.has_alpha();
        let mut out = Vec::with_capacity(w * h * out_channels);
        for y in 0..h {
            for x in 0..w {
                if gray {
                    let g = Self::to8(self.sample(p, x, y, 0), max);
                    out.extend_from_slice(&[g, g, g]);
                } else {
                    for c in 0..3 {
                        out.push(Self::to8(self.sample(p, x, y, c), max));
                    }
                }
                if out_channels == 4 {
                    let a = if alpha {
                        Self::to8(self.sample(p, x, y, chans - 1), max)
                    } else {
                        255
                    };
                    out.push(a);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_bad_geometry() {
        assert!(JxlImage::from_rgb8(2, 2, vec![0; 11]).is_err());
        assert!(JxlImage::from_rgb8(2, 2, vec![0; 12]).is_ok());
        assert!(JxlImage::new(
            1,
            1,
            PixelFormat::Rgb24,
            vec![Plane::new(3, vec![0; 3]), Plane::new(3, vec![0; 3])]
        )
        .is_err());
        assert!(JxlImage::new(2, 1, PixelFormat::Rgb24, vec![Plane::new(3, vec![0; 6])]).is_err());
    }

    #[test]
    fn to_rgba8_expands_gray_and_scales_deep() {
        let g = JxlImage::packed(2, 1, PixelFormat::Gray8, vec![0, 255]).unwrap();
        assert_eq!(g.to_rgba8(), vec![0, 0, 0, 255, 255, 255, 255, 255]);
        // 12-bit grey stored in Gray16Le: 4095 → 255, 2048 → 128.
        let d = JxlImage::packed(2, 1, PixelFormat::Gray16Le, vec![0xFF, 0x0F, 0x00, 0x08])
            .unwrap()
            .with_bits_per_sample(12);
        assert_eq!(d.to_rgb8(), vec![255, 255, 255, 128, 128, 128]);
        // 16-bit RGBA: 0x8000 → 128 (32768*255/65535 = 127.5 → rounds up).
        let r = JxlImage::packed(
            1,
            1,
            PixelFormat::Rgba64Le,
            vec![0, 0x80, 0xFF, 0xFF, 0, 0, 0, 0x80],
        )
        .unwrap();
        assert_eq!(r.to_rgba8(), vec![128, 255, 0, 128]);
        assert_eq!(r.to_rgb8(), vec![128, 255, 0]);
    }

    #[test]
    fn four_bit_in_gray8_scales_up() {
        let d = JxlImage::packed(2, 1, PixelFormat::Gray8, vec![15, 8])
            .unwrap()
            .with_bits_per_sample(4);
        assert_eq!(d.to_rgb8(), vec![255, 255, 255, 136, 136, 136]);
    }

    #[test]
    fn ya8_alpha_is_kept_and_dropped_as_appropriate() {
        let ya = JxlImage::packed(1, 1, PixelFormat::Ya8, vec![10, 20]).unwrap();
        assert_eq!(ya.to_rgb8(), vec![10, 10, 10]);
        assert_eq!(ya.to_rgba8(), vec![10, 10, 10, 20]);
    }

    #[test]
    fn planar_layouts_validate_per_plane_geometry() {
        let f = PixelFormat::YuvJ420P;
        assert_eq!(f.plane_count(), 3);
        assert_eq!(f.plane_dims(5, 3, 0), Some((5, 3)));
        assert_eq!(f.plane_dims(5, 3, 1), Some((3, 2)));
        assert_eq!(f.plane_dims(5, 3, 3), None);
        assert_eq!(PixelFormat::YuvJ422P.plane_dims(5, 3, 2), Some((3, 3)));
        assert_eq!(PixelFormat::Yuv440P.plane_dims(5, 3, 2), Some((5, 2)));
        assert_eq!(PixelFormat::YuvJ444P.plane_dims(5, 3, 1), Some((5, 3)));
        let ok = JxlImage::new(
            5,
            3,
            f,
            vec![
                Plane::new(5, vec![0; 15]),
                Plane::new(3, vec![128; 6]),
                Plane::new(3, vec![128; 6]),
            ],
        )
        .unwrap();
        assert_eq!(ok.as_bytes(), None);
        assert_eq!(ok.to_rgb8(), vec![0; 45]);
        assert_eq!(ok.to_rgba8().len(), 60);
        assert_eq!(ok.into_raw().len(), 27);
        // Short chroma plane.
        assert!(JxlImage::new(
            5,
            3,
            f,
            vec![
                Plane::new(5, vec![0; 15]),
                Plane::new(3, vec![128; 5]),
                Plane::new(3, vec![128; 6]),
            ],
        )
        .is_err());
        // Plane count.
        assert!(JxlImage::new(5, 3, f, vec![Plane::new(5, vec![0; 15])]).is_err());
    }

    #[test]
    fn into_raw_returns_the_plane() {
        let img = JxlImage::from_rgba8(1, 1, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(img.as_bytes(), Some(&[1u8, 2, 3, 4][..]));
        assert_eq!(img.into_raw(), vec![1, 2, 3, 4]);
    }
}
