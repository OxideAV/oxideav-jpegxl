//! Round 487 — corpus measurement harness for the encoder.
//!
//! Not a pass/fail gate by itself: with `JXL_CORPUS_DIR` pointing at a
//! directory of 8- or 16-bit PNG files it encodes each one at every
//! effort, checks the lossless round trip through this crate's decoder
//! and prints `name effort bytes encode-ms decode-ms`. Without the
//! variable (the CI case) it returns immediately.

mod common;

use std::time::Instant;

use common::png::{BitDepth, ColorType, decode_png};
use oxideav_jpegxl::{Effort, EncodeOptions, JxlImage, PixelFormat, decode, encode};

fn load(path: &std::path::Path) -> JxlImage {
    let png = decode_png(&std::fs::read(path).unwrap());
    let (format, bits) = match (png.color_type, png.bit_depth) {
        (ColorType::Grayscale, BitDepth::Eight) => (PixelFormat::Gray8, 8),
        (ColorType::GrayscaleAlpha, BitDepth::Eight) => (PixelFormat::Ya8, 8),
        (ColorType::Rgb, BitDepth::Eight) => (PixelFormat::Rgb24, 8),
        (ColorType::Rgba, BitDepth::Eight) => (PixelFormat::Rgba, 8),
        (ColorType::Grayscale, BitDepth::Sixteen) => (PixelFormat::Gray16Le, 16),
        (ColorType::GrayscaleAlpha, BitDepth::Sixteen) => (PixelFormat::Ya16Le, 16),
        (ColorType::Rgb, BitDepth::Sixteen) => (PixelFormat::Rgb48Le, 16),
        (ColorType::Rgba, BitDepth::Sixteen) => (PixelFormat::Rgba64Le, 16),
    };
    let mut data = png.data;
    if bits == 16 {
        for pair in data.as_chunks_mut::<2>().0 {
            pair.swap(0, 1);
        }
    }
    JxlImage::packed(png.width, png.height, format, data)
        .unwrap()
        .with_bits_per_sample(bits)
}

#[test]
fn corpus_report() {
    let Ok(dir) = std::env::var("JXL_CORPUS_DIR") else {
        return;
    };
    let quality: u8 = std::env::var("JXL_CORPUS_QUALITY")
        .ok()
        .and_then(|q| q.parse().ok())
        .unwrap_or(100);
    let out_dir = std::env::var("JXL_CORPUS_OUT").ok();
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    paths.sort();
    for path in paths {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let img = load(&path);
        for (label, effort) in [
            ("fast", Effort::Fast),
            ("balanced", Effort::Balanced),
            ("thorough", Effort::Thorough),
        ] {
            let opts = EncodeOptions::new()
                .with_quality(quality)
                .with_effort(effort);
            let t = Instant::now();
            let bytes = encode(&img, &opts).unwrap();
            let enc_ms = t.elapsed().as_millis();
            let t = Instant::now();
            let back = decode(&bytes).unwrap();
            let dec_ms = t.elapsed().as_millis();
            if quality == 100 {
                assert_eq!(back.as_bytes().unwrap(), img.as_bytes().unwrap(), "{name}");
            }
            println!(
                "CORPUS {name} {label} {} bytes enc {enc_ms} ms dec {dec_ms} ms",
                bytes.len()
            );
            if let Some(out) = &out_dir {
                std::fs::write(format!("{out}/{name}.{label}.jxl"), &bytes).unwrap();
            }
        }
    }
}
