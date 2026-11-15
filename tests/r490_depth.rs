//! Sample depth beyond eight bits on the lossy path, and
//! floating-point output.
//!
//! * Streams of a black-box encoder (`cjxl` 0.12, from a 16-bit PPM, a
//!   binary32 PFM and a half-float EXR) decode to the declared storage
//!   and agree with the black-box decoder's output committed next to
//!   them.
//! * This crate's lossy encoder takes 10 / 12 / 16-bit input, with and
//!   without alpha, grey and colour; its streams decode here and — when
//!   `djxl` is installed — in the black-box decoder, to the same
//!   samples within the rounding of two independent float pipelines.

use std::path::{Path, PathBuf};
use std::process::Command;

use oxideav_jpegxl::{EncodeOptions, JxlImage, PixelFormat, decode, encode, info};

fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The next whitespace-delimited token of a Netpbm-style header.
fn token(data: &[u8], at: &mut usize) -> String {
    while data[*at].is_ascii_whitespace() {
        *at += 1;
    }
    let start = *at;
    while !data[*at].is_ascii_whitespace() {
        *at += 1;
    }
    String::from_utf8_lossy(&data[start..*at]).into_owned()
}

/// A PFM (`PF` colour / `Pf` grey) as `(width, height, channels,
/// samples top row first)`.
fn parse_pfm(data: &[u8]) -> (usize, usize, usize, Vec<f32>) {
    let mut at = 0;
    let magic = token(data, &mut at);
    let channels = match magic.as_str() {
        "PF" => 3,
        "Pf" => 1,
        other => panic!("not a PFM: {other}"),
    };
    let width: usize = token(data, &mut at).parse().unwrap();
    let height: usize = token(data, &mut at).parse().unwrap();
    let scale: f32 = token(data, &mut at).parse().unwrap();
    at += 1;
    let row = width * channels;
    let mut out = vec![0f32; row * height];
    // Rows are stored bottom to top; a negative scale is little-endian.
    for y in 0..height {
        let dst = &mut out[(height - 1 - y) * row..(height - y) * row];
        let src = data[at + y * row * 4..].as_chunks::<4>().0;
        for (slot, b) in dst.iter_mut().zip(src) {
            *slot = if scale < 0.0 {
                f32::from_le_bytes(*b)
            } else {
                f32::from_be_bytes(*b)
            };
        }
    }
    (width, height, channels, out)
}

/// A binary Netpbm (`P5` / `P6`, or `P7` with a `TUPLTYPE`) as
/// `(width, height, channels, maxval, samples)`.
fn parse_pnm(data: &[u8]) -> (usize, usize, usize, u32, Vec<u32>) {
    let mut at = 0;
    let magic = token(data, &mut at);
    let (width, height, channels, maxval);
    if magic == "P7" {
        let (mut w, mut h, mut d, mut m) = (0, 0, 0, 0);
        loop {
            let key = token(data, &mut at);
            match key.as_str() {
                "WIDTH" => w = token(data, &mut at).parse().unwrap(),
                "HEIGHT" => h = token(data, &mut at).parse().unwrap(),
                "DEPTH" => d = token(data, &mut at).parse().unwrap(),
                "MAXVAL" => m = token(data, &mut at).parse().unwrap(),
                "TUPLTYPE" => {
                    token(data, &mut at);
                }
                "ENDHDR" => break,
                other => panic!("PAM header key {other}"),
            }
        }
        (width, height, channels, maxval) = (w, h, d, m);
    } else {
        channels = match magic.as_str() {
            "P5" => 1,
            "P6" => 3,
            other => panic!("not a binary PNM: {other}"),
        };
        width = token(data, &mut at).parse().unwrap();
        height = token(data, &mut at).parse().unwrap();
        maxval = token(data, &mut at).parse().unwrap();
    }
    at += 1;
    let n = width * height * channels;
    let samples = if maxval > 255 {
        data[at..at + 2 * n]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u32::from(u16::from_be_bytes(*b)))
            .collect()
    } else {
        data[at..at + n].iter().map(|&b| u32::from(b)).collect()
    };
    (width, height, channels, maxval, samples)
}

fn floats_of(img: &JxlImage) -> Vec<f32> {
    img.as_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn words_of(img: &JxlImage) -> Vec<u32> {
    img.as_bytes()
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u32::from(u16::from_le_bytes(*b)))
        .collect()
}

#[test]
fn lossless_float_streams_decode_to_the_coded_values() {
    for (name, format, channels) in [
        ("float32_rgb_lossless", PixelFormat::RgbF32Le, 3),
        ("float32_grey_lossless", PixelFormat::GrayF32Le, 1),
        ("float16_rgb_lossless", PixelFormat::RgbF32Le, 3),
    ] {
        let bytes = fixture(&format!("{name}.jxl"));
        let (w, h, c, want) = parse_pfm(&fixture(&format!("{name}_expected.pfm")));
        assert_eq!(c, channels, "{name}");
        let meta = info(&bytes).unwrap();
        assert!(meta.float_sample, "{name}");
        assert_eq!(meta.format, format, "{name}");
        let img = decode(&bytes).unwrap();
        assert_eq!((img.width as usize, img.height as usize), (w, h), "{name}");
        assert_eq!(img.format, format, "{name}");
        assert_eq!(img.bits_per_sample, 32, "{name}");
        let got = floats_of(&img);
        assert_eq!(got.len(), want.len(), "{name}");
        // Lossless: the very bit patterns (a half float widened to
        // binary32 is exact).
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{name}: sample {i}: {g} vs {w}");
        }
        // The values leave the nominal range in these pictures, and
        // the float layout keeps them.
        if channels == 3 {
            assert!(got.iter().any(|&v| v < 0.0) && got.iter().any(|&v| v > 1.0));
        }
        // The 8-bit conversions clamp.
        let rgb = img.to_rgb8();
        assert_eq!(rgb.len(), w * h * 3);
        let first = (got[0].clamp(0.0, 1.0) * 255.0).round() as i32;
        assert!((i32::from(rgb[0]) - first).abs() <= 1, "{name}");
    }
}

#[test]
fn lossy_float_streams_decode_close_to_the_black_box_decode() {
    for name in ["float32_rgb_d1", "float16_rgb_d1"] {
        let bytes = fixture(&format!("{name}.jxl"));
        let (w, h, _, want) = parse_pfm(&fixture(&format!("{name}_expected.pfm")));
        let img = decode(&bytes).unwrap();
        assert_eq!((img.width as usize, img.height as usize), (w, h), "{name}");
        assert_eq!(img.format, PixelFormat::RgbF32Le, "{name}");
        let got = floats_of(&img);
        let mut worst = 0f32;
        let mut sum = 0f64;
        for (g, w) in got.iter().zip(&want) {
            let d = (g - w).abs();
            worst = worst.max(d);
            sum += f64::from(d);
        }
        let mean = sum / got.len() as f64;
        println!("{name}: worst {worst:.5}, mean {mean:.6}");
        assert!(
            worst < 0.02 && mean < 0.002,
            "{name}: worst {worst}, mean {mean}"
        );
        // Unclamped: the source runs below zero and above one.
        assert!(got.iter().any(|&v| v > 1.0), "{name}");
    }
}

#[test]
fn a_sixteen_bit_lossy_stream_decodes_at_sixteen_bits() {
    let bytes = fixture("depth16_rgb_d1.jxl");
    let (w, h, c, maxval, want) = parse_pnm(&fixture("depth16_rgb_d1_expected.ppm"));
    assert_eq!((c, maxval), (3, 65535));
    let meta = info(&bytes).unwrap();
    assert_eq!(
        (meta.format, meta.bits_per_sample),
        (PixelFormat::Rgb48Le, 16)
    );
    let img = decode(&bytes).unwrap();
    assert_eq!((img.width as usize, img.height as usize), (w, h));
    assert_eq!(
        (img.format, img.bits_per_sample),
        (PixelFormat::Rgb48Le, 16)
    );
    let got = words_of(&img);
    let mut worst = 0u32;
    let mut sum = 0u64;
    for (g, w) in got.iter().zip(&want) {
        let d = g.abs_diff(*w);
        worst = worst.max(d);
        sum += u64::from(d);
    }
    let mean = sum as f64 / got.len() as f64;
    println!("16-bit lossy: worst {worst}, mean {mean:.2} of 65535");
    // Two independent float pipelines: a few 8-bit codes at worst.
    assert!(worst <= 1024 && mean <= 96.0, "worst {worst}, mean {mean}");
    // Real 16-bit output, not 8-bit values widened.
    assert!(got.iter().any(|v| v % 257 != 0));
}

// ---- this crate's encoder --------------------------------------------------

/// A smooth picture with some structure, `channels` interleaved
/// samples of `bits` each (the last one a soft-edged alpha disc when
/// `alpha`).
fn picture(width: usize, height: usize, channels: usize, alpha: bool, bits: u32) -> Vec<u16> {
    let max = ((1u32 << bits) - 1) as f32;
    let colour = channels - usize::from(alpha);
    let mut out = Vec::with_capacity(width * height * channels);
    let mut s = 0x2468_ACE1u32;
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (x as f32 / width as f32, y as f32 / height as f32);
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let grain = ((s >> 24) as f32 / 255.0 - 0.5) * 0.02;
            let base = 0.5 + 0.35 * (fx * 9.0).sin() * (fy * 7.0).cos();
            let edge = if (x / 24 + y / 24) % 2 == 0 {
                0.12
            } else {
                -0.08
            };
            for c in 0..colour {
                let v = base + edge * (c as f32 * 0.5 + 0.5) + grain + 0.1 * c as f32 * fx;
                out.push((v.clamp(0.0, 1.0) * max).round() as u16);
            }
            if alpha {
                let d = ((fx - 0.5).powi(2) + (fy - 0.5).powi(2)).sqrt();
                let a = ((0.45 - d) * 12.0).clamp(0.0, 1.0);
                out.push((a * max).round() as u16);
            }
        }
    }
    out
}

fn image_of(
    width: usize,
    height: usize,
    format: PixelFormat,
    bits: u32,
    samples: &[u16],
) -> JxlImage {
    let data: Vec<u8> = if format.bytes_per_sample() == 1 {
        samples.iter().map(|&v| v as u8).collect()
    } else {
        samples.iter().flat_map(|v| v.to_le_bytes()).collect()
    };
    JxlImage::packed(width as u32, height as u32, format, data)
        .unwrap()
        .with_bits_per_sample(bits as u8)
}

fn samples_of(img: &JxlImage) -> Vec<u32> {
    if img.format.bytes_per_sample() == 1 {
        img.as_bytes()
            .unwrap()
            .iter()
            .map(|&b| u32::from(b))
            .collect()
    } else {
        words_of(img)
    }
}

fn psnr(a: &[u32], b: &[u32], max: u32, channels: usize, skip: Option<usize>) -> f64 {
    let mut sum = 0f64;
    let mut n = 0u64;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if skip == Some(i % channels) {
            continue;
        }
        let d = f64::from(*x) - f64::from(*y);
        sum += d * d;
        n += 1;
    }
    if sum == 0.0 {
        return 99.0;
    }
    10.0 * (f64::from(max) * f64::from(max) * n as f64 / sum).log10()
}

fn tool(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oxideav-jxl-r490-depth-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// Decode `jxl` with the black-box decoder into a Netpbm file of the
/// given extension; `None` when the tool is absent.
fn black_box_decode(
    jxl: &[u8],
    tag: &str,
    ext: &str,
) -> Option<(usize, usize, usize, u32, Vec<u32>)> {
    if !tool("djxl") {
        return None;
    }
    let src = scratch(&format!("{tag}.jxl"));
    let dst = scratch(&format!("{tag}.{ext}"));
    std::fs::write(&src, jxl).unwrap();
    let out = Command::new("djxl").arg(&src).arg(&dst).output().unwrap();
    assert!(
        out.status.success(),
        "{tag}: the black-box decoder refused the stream: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let data = std::fs::read(&dst).unwrap();
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&dst);
    Some(parse_pnm(&data))
}

#[test]
fn lossy_encoding_keeps_the_depth_of_deep_input() {
    let (w, h) = (200usize, 136usize);
    let cases: [(&str, PixelFormat, usize, bool, u32, &str); 8] = [
        ("rgb10", PixelFormat::Rgb48Le, 3, false, 10, "ppm"),
        ("rgb12", PixelFormat::Rgb48Le, 3, false, 12, "ppm"),
        ("rgb16", PixelFormat::Rgb48Le, 3, false, 16, "ppm"),
        ("grey10", PixelFormat::Gray16Le, 1, false, 10, "pgm"),
        ("grey16", PixelFormat::Gray16Le, 1, false, 16, "pgm"),
        ("rgba16", PixelFormat::Rgba64Le, 4, true, 16, "pam"),
        ("ya12", PixelFormat::Ya16Le, 2, true, 12, "pam"),
        ("rgba8", PixelFormat::Rgba, 4, true, 8, "pam"),
    ];
    for (tag, format, channels, alpha, bits, ext) in cases {
        let samples = picture(w, h, channels, alpha, bits);
        let img = image_of(w, h, format, bits, &samples);
        let source: Vec<u32> = samples.iter().map(|&v| u32::from(v)).collect();
        let max = (1u32 << bits) - 1;
        let alpha_at = alpha.then_some(channels - 1);
        let mut last = usize::MAX;
        for distance in [0.5f32, 1.0, 3.0] {
            let bytes = encode(&img, &EncodeOptions::new().with_distance(distance)).unwrap();
            assert!(
                bytes.len() < last,
                "{tag} d{distance}: {} bytes",
                bytes.len()
            );
            last = bytes.len();
            let meta = info(&bytes).unwrap();
            assert!(meta.xyb_encoded, "{tag}: lossy input goes to VarDCT in XYB");
            assert_eq!((meta.format, meta.bits_per_sample), (format, bits), "{tag}");
            let back = decode(&bytes).unwrap();
            assert_eq!(
                (back.format, u32::from(back.bits_per_sample)),
                (format, bits),
                "{tag}"
            );
            let got = samples_of(&back);
            assert_eq!(got.len(), source.len(), "{tag}");
            assert!(got.iter().all(|&v| v <= max), "{tag}");
            let quality = psnr(&got, &source, max, channels, alpha_at);
            let floor = if distance <= 1.0 { 38.0 } else { 32.0 };
            println!(
                "{tag} d{distance}: {} bytes, {quality:.2} dB ({:.3} bpp)",
                bytes.len(),
                bytes.len() as f64 * 8.0 / (w * h) as f64
            );
            assert!(quality > floor, "{tag} d{distance}: {quality:.2} dB");
            // Alpha rides losslessly next to the lossy colour.
            if let Some(a) = alpha_at {
                for (i, (g, s)) in got.iter().zip(&source).enumerate() {
                    if i % channels == a {
                        assert_eq!(g, s, "{tag}: alpha sample {i}");
                    }
                }
            }
            // Deep output carries real low-order bits.
            if bits > 8 {
                let step = 1u32 << (bits - 8);
                assert!(
                    got.iter().any(|v| v % step != 0),
                    "{tag}: no low-order bits"
                );
            }
            // The black-box decoder reads the same picture.
            if let Some((bw, bh, bc, bmax, theirs)) =
                black_box_decode(&bytes, &format!("{tag}-d{distance}"), ext)
            {
                assert_eq!((bw, bh, bc, bmax), (w, h, channels, max), "{tag}");
                let mut worst = 0u32;
                for (i, (g, t)) in got.iter().zip(&theirs).enumerate() {
                    let d = g.abs_diff(*t);
                    if alpha_at == Some(i % channels) {
                        assert_eq!(d, 0, "{tag}: alpha sample {i}");
                    }
                    worst = worst.max(d);
                }
                // Within two 8-bit codes of each other.
                let bound = (2 * max).div_ceil(255);
                println!("{tag} d{distance}: black-box decoder within {worst} of {max}");
                assert!(worst <= bound, "{tag} d{distance}: {worst} > {bound}");
            }
        }
    }
}

/// Size and quality at a matched distance against the black-box
/// encoder at its default effort, on 16-bit input (printed; the
/// distance scales differ, so only gross regressions are asserted).
#[test]
fn deep_lossy_size_against_a_black_box_encoder() {
    if !tool("cjxl") || !tool("djxl") {
        eprintln!("cjxl / djxl not installed: skipped");
        return;
    }
    let (w, h, bits) = (320usize, 240usize, 16u32);
    let samples = picture(w, h, 3, false, bits);
    let img = image_of(w, h, PixelFormat::Rgb48Le, bits, &samples);
    let source: Vec<u32> = samples.iter().map(|&v| u32::from(v)).collect();
    let ppm = scratch("deep.ppm");
    let mut file = format!("P6\n{w} {h}\n65535\n").into_bytes();
    file.extend(samples.iter().flat_map(|v| v.to_be_bytes()));
    std::fs::write(&ppm, &file).unwrap();
    for distance in [1.0f32, 2.0] {
        let ours = encode(&img, &EncodeOptions::new().with_distance(distance)).unwrap();
        let ours_q = psnr(
            &samples_of(&decode(&ours).unwrap()),
            &source,
            65535,
            3,
            None,
        );
        let theirs_path = scratch("deep-ref.jxl");
        let out = Command::new("cjxl")
            .arg(&ppm)
            .arg(&theirs_path)
            .args(["-d", &distance.to_string()])
            .output()
            .unwrap();
        assert!(out.status.success());
        let theirs = std::fs::read(&theirs_path).unwrap();
        let theirs_q = psnr(
            &samples_of(&decode(&theirs).unwrap()),
            &source,
            65535,
            3,
            None,
        );
        println!(
            "16-bit d{distance}: here {} bytes / {ours_q:.2} dB, black-box encoder {} bytes / \
             {theirs_q:.2} dB",
            ours.len(),
            theirs.len()
        );
        assert!(
            ours.len() < theirs.len() * 2,
            "d{distance}: {} bytes",
            ours.len()
        );
        assert!(ours_q > theirs_q - 6.0, "d{distance}: {ours_q:.2} dB");
    }
    let _ = std::fs::remove_file(&ppm);
}
