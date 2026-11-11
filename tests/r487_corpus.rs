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

fn psnr(a: &JxlImage, b: &JxlImage) -> f64 {
    let (a_bytes, b_bytes) = (a.as_bytes().unwrap(), b.as_bytes().unwrap());
    assert_eq!(a_bytes.len(), b_bytes.len(), "decoded size");
    let step = a.format.bytes_per_sample();
    let peak = if step == 1 { 255.0f64 } else { 65535.0 };
    let mut sse = 0.0f64;
    let mut n = 0u64;
    for i in (0..a_bytes.len()).step_by(step) {
        let (x, y) = if step == 1 {
            (f64::from(a_bytes[i]), f64::from(b_bytes[i]))
        } else {
            (
                f64::from(u16::from_le_bytes([a_bytes[i], a_bytes[i + 1]])),
                f64::from(u16::from_le_bytes([b_bytes[i], b_bytes[i + 1]])),
            )
        };
        sse += (x - y) * (x - y);
        n += 1;
    }
    10.0 * (peak * peak / (sse / n as f64).max(1e-12)).log10()
}

/// Rate–distortion sweep: `JXL_CORPUS_DISTANCES=0.5,1,2,4` encodes
/// every corpus picture at each distance and prints size and PSNR
/// (through this crate's decoder).
#[test]
fn corpus_rate_distortion() {
    let (Ok(dir), Ok(list)) = (
        std::env::var("JXL_CORPUS_DIR"),
        std::env::var("JXL_CORPUS_DISTANCES"),
    ) else {
        return;
    };
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
        for d in list.split(',').filter_map(|d| d.trim().parse::<f32>().ok()) {
            let opts = EncodeOptions::new().with_distance(d);
            let t = Instant::now();
            let bytes = encode(&img, &opts).unwrap();
            let enc_ms = t.elapsed().as_millis();
            if let Some(out) = &out_dir {
                std::fs::write(format!("{out}/{name}.d{d}.jxl"), &bytes).unwrap();
            }
            let t = Instant::now();
            let back = decode(&bytes).unwrap();
            let dec_ms = t.elapsed().as_millis();
            println!(
                "RD {name} d{d} {} bytes psnr {:.2} dB enc {enc_ms} ms dec {dec_ms} ms",
                bytes.len(),
                psnr(&img, &back)
            );
        }
    }
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
        let efforts: &[(&str, Effort)] = if quality == 100 {
            &[
                ("fast", Effort::Fast),
                ("balanced", Effort::Balanced),
                ("thorough", Effort::Thorough),
            ]
        } else {
            &[("balanced", Effort::Balanced)]
        };
        for &(label, effort) in efforts {
            let opts = EncodeOptions::new()
                .with_quality(quality)
                .with_effort(effort);
            let t = Instant::now();
            let bytes = encode(&img, &opts).unwrap();
            let enc_ms = t.elapsed().as_millis();
            if let Some(out) = &out_dir {
                std::fs::write(format!("{out}/{name}.{label}.q{quality}.jxl"), &bytes).unwrap();
            }
            let t = Instant::now();
            let back = decode(&bytes).unwrap();
            let dec_ms = t.elapsed().as_millis();
            let mut psnr = String::new();
            if quality == 100 {
                assert_eq!(back.as_bytes().unwrap(), img.as_bytes().unwrap(), "{name}");
            } else {
                psnr = format!(" psnr {:.2} dB", self::psnr(&img, &back));
            }
            println!(
                "CORPUS {name} {label} {} bytes enc {enc_ms} ms dec {dec_ms} ms{psnr}",
                bytes.len()
            );
        }
    }
}
