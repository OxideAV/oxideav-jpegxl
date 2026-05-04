//! Crate-local uncompressed image representation.
//!
//! Defined here (rather than reusing `oxideav_core::VideoFrame`) so the
//! crate can be built with the default `registry` feature off — i.e.
//! without depending on `oxideav-core` at all. When the `registry`
//! feature is on the [`crate::registry`] module provides
//! `From<JxlImage> for oxideav_core::Frame` so the `Decoder` / `Encoder`
//! trait surface still interoperates cleanly.

/// Pixel layout used by [`JxlImage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JxlPixelFormat {
    /// 1-channel 8-bit greyscale, one plane.
    Gray8,
    /// 3-channel 8-bit packed RGB, one plane (3 bytes per pixel).
    Rgb24,
    /// 4-channel 8-bit packed RGBA, one plane (4 bytes per pixel).
    Rgba,
}

impl JxlPixelFormat {
    /// Number of bytes per pixel in the interleaved input buffer.
    pub fn channel_count(self) -> u32 {
        match self {
            JxlPixelFormat::Gray8 => 1,
            JxlPixelFormat::Rgb24 => 3,
            JxlPixelFormat::Rgba => 4,
        }
    }
}

/// One image plane: row-major bytes plus the row stride in bytes.
#[derive(Debug, Clone)]
pub struct JxlPlane {
    /// Bytes per row in `data` (may be larger than the logical row width).
    pub stride: usize,
    /// Raw plane bytes, packed `stride` × number of rows.
    pub data: Vec<u8>,
}

/// One decoded JPEG XL frame.
///
/// All-`std`, no `oxideav-core` types — the crate's standalone path
/// hands these out directly. The gated [`crate::registry`] module
/// provides a `From<JxlImage> for oxideav_core::Frame` conversion.
#[derive(Debug, Clone)]
pub struct JxlImage {
    /// Picture width in pixels.
    pub width: u32,
    /// Picture height in pixels.
    pub height: u32,
    /// Pixel layout. Currently only Gray8 is produced by the decoder
    /// path; the encoder accepts Gray8 / Rgb24 / Rgba.
    pub pixel_format: JxlPixelFormat,
    /// One entry per plane (always 1 today — interleaved layouts).
    pub planes: Vec<JxlPlane>,
    /// Optional presentation timestamp. The standalone decode path
    /// always leaves this `None`; the registry-backed `Decoder` impl
    /// fills it in from the `Packet` it consumed.
    pub pts: Option<i64>,
}
