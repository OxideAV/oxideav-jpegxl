//! Shared test helper: decode a reference PNG fixture via the
//! workspace's own pure-Rust `oxideav-png` crate.
//!
//! The output mirrors the raw (identity-transform) layout the tests
//! were written against: samples interleaved per pixel in PNG channel
//! order, 16-bit samples as big-endian byte pairs, no palette or
//! sub-byte expansion beyond what the fixtures need (8 / 16-bit grey,
//! grey + alpha, RGB and RGBA).

#![allow(dead_code)]

use oxideav_png::PixelFormat;

/// PNG colour type of a decoded reference image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorType {
    /// Colour type 0.
    Grayscale,
    /// Colour type 4.
    GrayscaleAlpha,
    /// Colour type 2.
    Rgb,
    /// Colour type 6.
    Rgba,
}

impl ColorType {
    /// Samples per pixel.
    pub fn channels(self) -> usize {
        match self {
            ColorType::Grayscale => 1,
            ColorType::GrayscaleAlpha => 2,
            ColorType::Rgb => 3,
            ColorType::Rgba => 4,
        }
    }
}

/// Sample bit depth of a decoded reference image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitDepth {
    /// One byte per sample.
    Eight,
    /// Two big-endian bytes per sample.
    Sixteen,
}

/// A decoded PNG: geometry, colour type, bit depth and the tightly
/// packed interleaved sample bytes (16-bit samples big-endian).
#[derive(Debug, Clone)]
pub struct DecodedPng {
    pub width: u32,
    pub height: u32,
    pub color_type: ColorType,
    pub bit_depth: BitDepth,
    pub data: Vec<u8>,
}

impl DecodedPng {
    /// Samples per pixel.
    pub fn channels(&self) -> usize {
        self.color_type.channels()
    }
}

/// Decode `bytes` as a non-palette 8- or 16-bit PNG. Panics on any
/// decode error or unsupported layout (test helper).
pub fn decode_png(bytes: &[u8]) -> DecodedPng {
    let info = oxideav_png::info(bytes).expect("png info");
    let img = oxideav_png::decode(bytes).expect("png decode");
    let (w, h) = (img.width as usize, img.height as usize);
    let bit_depth = match info.bit_depth {
        8 => BitDepth::Eight,
        16 => BitDepth::Sixteen,
        other => panic!("unsupported PNG bit depth {other}"),
    };
    let color_type = match info.colour_type {
        0 => ColorType::Grayscale,
        2 => ColorType::Rgb,
        4 => ColorType::GrayscaleAlpha,
        6 => ColorType::Rgba,
        other => panic!("unsupported PNG colour type {other}"),
    };
    let plane = &img.planes[0];
    let bpp = img.format.bytes_per_pixel();
    // Which native-layout channels to keep (16-bit grey + alpha is
    // delivered as replicated RGBA64).
    let keep: &[usize] = match (img.format, color_type) {
        (PixelFormat::Rgba64Le, ColorType::GrayscaleAlpha) => &[0, 3],
        _ => match color_type.channels() {
            1 => &[0],
            2 => &[0, 1],
            3 => &[0, 1, 2],
            _ => &[0, 1, 2, 3],
        },
    };
    let bytes_per_sample = match bit_depth {
        BitDepth::Eight => 1,
        BitDepth::Sixteen => 2,
    };
    assert_eq!(
        bpp,
        bytes_per_sample
            * match (img.format, color_type) {
                (PixelFormat::Rgba64Le, ColorType::GrayscaleAlpha) => 4,
                _ => color_type.channels(),
            },
        "unexpected native layout {:?} for {color_type:?}",
        img.format
    );
    let mut data = Vec::with_capacity(w * h * keep.len() * bytes_per_sample);
    for y in 0..h {
        let row = &plane.data[y * plane.stride..y * plane.stride + w * bpp];
        for px in row.chunks_exact(bpp) {
            for &c in keep {
                match bit_depth {
                    BitDepth::Eight => data.push(px[c]),
                    // Native layout is little-endian; re-emit as PNG's
                    // big-endian sample order.
                    BitDepth::Sixteen => data.extend_from_slice(&[px[2 * c + 1], px[2 * c]]),
                }
            }
        }
    }
    DecodedPng {
        width: img.width,
        height: img.height,
        color_type,
        bit_depth,
        data,
    }
}
