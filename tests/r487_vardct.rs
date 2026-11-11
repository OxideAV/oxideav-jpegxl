//! Round 487 — the VarDCT (lossy XYB) encoder: every lossy request on
//! 8-bit sRGB input writes `kVarDCT` frames that this crate's decoder
//! and — where installed — a black-box decoder reconstruct to the same
//! picture; distance / quality form a monotone rate–distortion ladder;
//! every geometry (1×1, single row, huge aspect, multi-group,
//! multi-LF-group) encodes without panicking; alpha rides along
//! losslessly; animations are coded frame by frame.

use std::process::Command;

use oxideav_jpegxl::{
    AnimationInfo, Effort, EncodeOptions, Frame, JxlImage, PixelFormat, decode, decode_all, encode,
    encode_all, info,
};

fn hash(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

/// Photo-like content: smooth shading, edges, a textured half.
fn picture(width: u32, height: u32, format: PixelFormat, seed: u32) -> JxlImage {
    let channels = format.channels();
    let mut s = seed.wrapping_mul(2654435761) | 1;
    let mut data = Vec::with_capacity((width * height) as usize * channels);
    for y in 0..height {
        for x in 0..width {
            let fx = f64::from(x) / f64::from(width.max(2));
            let fy = f64::from(y) / f64::from(height.max(2));
            let base = 0.5 + 0.3 * (fx * 9.0).sin() * (fy * 7.0).cos();
            let edge = if (x / 37 + y / 23) % 2 == 0 {
                0.12
            } else {
                -0.08
            };
            let rough = if fx > 0.5 { 0.06 } else { 0.0 };
            for c in 0..channels {
                let noise = f64::from(hash(&mut s) % 1000) / 1000.0 - 0.5;
                let v = if c == 3 || (channels == 2 && c == 1) {
                    let d = ((fx - 0.5).powi(2) + (fy - 0.5).powi(2)).sqrt();
                    (1.2 - d * 2.0).clamp(0.0, 1.0)
                } else {
                    base + edge + rough * noise + 0.07 * c as f64
                };
                data.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    JxlImage::packed(width, height, format, data).unwrap()
}

/// PSNR over the colour samples (alpha excluded), in dB.
fn psnr(a: &JxlImage, b: &JxlImage) -> f64 {
    assert_eq!((a.width, a.height, a.format), (b.width, b.height, b.format));
    let channels = a.format.channels();
    let colour = if channels == 2 || channels == 4 {
        channels - 1
    } else {
        channels
    };
    let (x, y) = (a.as_bytes().unwrap(), b.as_bytes().unwrap());
    let mut sse = 0.0f64;
    let mut n = 0u64;
    for (i, (&p, &q)) in x.iter().zip(y).enumerate() {
        if i % channels < colour {
            sse += (f64::from(p) - f64::from(q)).powi(2);
            n += 1;
        }
    }
    10.0 * (255.0f64 * 255.0 / (sse / n as f64).max(1e-9)).log10()
}

fn is_vardct(bytes: &[u8]) -> bool {
    // `xyb_encoded` is what a VarDCT file of this encoder announces.
    info(bytes).map(|i| i.xyb_encoded).unwrap_or(false)
}

#[test]
fn lossy_requests_write_vardct_and_reach_a_sane_quality() {
    for format in [PixelFormat::Rgb24, PixelFormat::Gray8] {
        let img = picture(320, 240, format, 7);
        let bytes = encode(&img, &EncodeOptions::new().with_quality(90)).unwrap();
        assert!(is_vardct(&bytes), "{format:?}: quality 90 is a VarDCT file");
        let back = decode(&bytes).unwrap();
        let q = psnr(&img, &back);
        assert!(q > 36.0, "{format:?}: PSNR {q:.2} dB at quality 90");
        // Lossless stays Modular and exact.
        let lossless = encode(&img, &EncodeOptions::new()).unwrap();
        assert!(!is_vardct(&lossless));
        assert_eq!(
            decode(&lossless).unwrap().as_bytes().unwrap(),
            img.as_bytes().unwrap()
        );
        assert!(bytes.len() < lossless.len() / 2);
        // `modular` keeps a lossy request off VarDCT.
        let modular = encode(
            &img,
            &EncodeOptions::new().with_quality(90).with_modular(true),
        )
        .unwrap();
        assert!(!is_vardct(&modular));
    }
}

#[test]
fn distance_is_a_monotone_rate_distortion_ladder() {
    let img = picture(384, 256, PixelFormat::Rgb24, 3);
    let mut prev_len = usize::MAX;
    let mut prev_psnr = f64::INFINITY;
    for d in [0.25f32, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0] {
        let bytes = encode(&img, &EncodeOptions::new().with_distance(d)).unwrap();
        let q = psnr(&img, &decode(&bytes).unwrap());
        assert!(
            bytes.len() < prev_len,
            "distance {d}: {} bytes after {prev_len}",
            bytes.len()
        );
        assert!(
            q < prev_psnr + 0.05,
            "distance {d}: PSNR {q:.2} after {prev_psnr:.2}"
        );
        prev_len = bytes.len();
        prev_psnr = q;
    }
    assert!(
        prev_psnr > 22.0,
        "distance 16 is still a picture: {prev_psnr:.2} dB"
    );
    // Quality maps onto the same ladder; distance 0 is lossless.
    let q90 = encode(&img, &EncodeOptions::new().with_quality(90)).unwrap();
    let d1 = encode(&img, &EncodeOptions::new().with_distance(1.0)).unwrap();
    assert_eq!(q90, d1);
    let d0 = encode(&img, &EncodeOptions::new().with_distance(0.0)).unwrap();
    assert_eq!(
        decode(&d0).unwrap().as_bytes().unwrap(),
        img.as_bytes().unwrap()
    );
}

#[test]
fn every_geometry_encodes_and_decodes() {
    for (w, h) in [
        (1u32, 1u32),
        (1, 9),
        (9, 1),
        (7, 7),
        (8, 8),
        (9, 17),
        (1, 600),
        (600, 1),
        (255, 257),
        (256, 256),
        (257, 255),
        (520, 40),
        (2100, 16),
        (16, 2100),
    ] {
        for format in [PixelFormat::Rgb24, PixelFormat::Gray8] {
            let img = picture(w, h, format, w * 131 + h);
            let bytes = encode(&img, &EncodeOptions::new().with_distance(1.5))
                .unwrap_or_else(|e| panic!("{w}x{h} {format:?}: encode: {e}"));
            let back = decode(&bytes).unwrap_or_else(|e| panic!("{w}x{h} {format:?}: decode: {e}"));
            assert_eq!((back.width, back.height, back.format), (w, h, format));
            let q = psnr(&img, &back);
            assert!(q > 28.0, "{w}x{h} {format:?}: PSNR {q:.2} dB");
        }
    }
}

#[test]
fn flat_pictures_are_tiny_and_exact_enough() {
    let img = JxlImage::packed(640, 480, PixelFormat::Rgb24, vec![0x7B; 640 * 480 * 3]).unwrap();
    let bytes = encode(&img, &EncodeOptions::new().with_distance(1.0)).unwrap();
    assert!(bytes.len() < 700, "flat picture: {} bytes", bytes.len());
    let back = decode(&bytes).unwrap();
    let worst = img
        .as_bytes()
        .unwrap()
        .iter()
        .zip(back.as_bytes().unwrap())
        .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
        .max()
        .unwrap();
    assert!(worst <= 1, "flat picture: worst error {worst}");
}

#[test]
fn alpha_rides_a_vardct_frame_losslessly() {
    for (w, h) in [(120u32, 90u32), (300, 280), (16, 16)] {
        for format in [PixelFormat::Rgba, PixelFormat::Ya8] {
            let img = picture(w, h, format, 11);
            let bytes = encode(&img, &EncodeOptions::new().with_distance(1.0))
                .unwrap_or_else(|e| panic!("{w}x{h} {format:?}: encode: {e}"));
            assert!(is_vardct(&bytes));
            let back = decode(&bytes).unwrap_or_else(|e| panic!("{w}x{h} {format:?}: decode: {e}"));
            assert_eq!(back.format, format);
            let channels = format.channels();
            let (a, b) = (img.as_bytes().unwrap(), back.as_bytes().unwrap());
            for (i, (&p, &q)) in a.iter().zip(b).enumerate() {
                if i % channels == channels - 1 {
                    assert_eq!(p, q, "{w}x{h} {format:?}: alpha sample {}", i / channels);
                }
            }
            let q = psnr(&img, &back);
            assert!(q > 34.0, "{w}x{h} {format:?}: colour PSNR {q:.2} dB");
        }
    }
}

#[test]
fn deep_and_non_srgb_input_falls_back_to_quantised_modular() {
    let img = picture(64, 64, PixelFormat::Rgb24, 5);
    let wide: Vec<u8> = img
        .as_bytes()
        .unwrap()
        .iter()
        .flat_map(|&v| (u16::from(v) * 257).to_le_bytes())
        .collect();
    let deep = JxlImage::packed(64, 64, PixelFormat::Rgb48Le, wide).unwrap();
    let bytes = encode(&deep, &EncodeOptions::new().with_quality(80)).unwrap();
    assert!(!is_vardct(&bytes), "16-bit input stays Modular");
    assert_eq!(decode(&bytes).unwrap().format, PixelFormat::Rgb48Le);
}

#[test]
fn lossy_animations_are_vardct_frames() {
    let frames: Vec<Frame> = (0..3)
        .map(|i| Frame::new(picture(96, 72, PixelFormat::Rgb24, 20 + i)).with_ticks(2 + i))
        .collect();
    let opts = EncodeOptions::new()
        .with_distance(2.0)
        .with_animation(AnimationInfo::new(25, 1, 0));
    let bytes = encode_all(&frames, &opts).unwrap();
    assert!(is_vardct(&bytes));
    let back = decode_all(&bytes).unwrap();
    assert_eq!(back.len(), 3);
    for (i, (src, got)) in frames.iter().zip(&back).enumerate() {
        assert_eq!(got.ticks, 2 + i as u32, "frame {i}: duration");
        let q = psnr(&src.image, &got.image);
        assert!(q > 30.0, "frame {i}: PSNR {q:.2} dB");
    }
}

#[test]
fn the_worker_budget_never_changes_the_bytes() {
    // Lossy (VarDCT) and lossless (Modular), with and without alpha,
    // on a multi-group picture: one byte stream at every budget.
    for format in [PixelFormat::Rgb24, PixelFormat::Rgba] {
        let img = picture(300, 270, format, 31); // 2×2 groups
        for opts in [
            EncodeOptions::new().with_distance(1.0),
            EncodeOptions::new().with_effort(Effort::Fast),
            EncodeOptions::new()
                .with_quality(70)
                .with_modular(true)
                .with_effort(Effort::Fast),
        ] {
            let serial = encode(&img, &opts).unwrap();
            for threads in [3usize, 64] {
                let parallel = encode(&img, &opts.clone().with_threads(threads)).unwrap();
                assert_eq!(parallel, serial, "{format:?} {opts:?}: {threads} workers");
            }
        }
    }
}

fn djxl_available() -> bool {
    Command::new("djxl").arg("--version").output().is_ok()
}

#[test]
fn black_box_decoder_reconstructs_the_same_picture() {
    if !djxl_available() {
        eprintln!("skip: no black-box decoder installed");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-jxl-r487v-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cases = [
        (picture(8, 8, PixelFormat::Rgb24, 1), 1.0f32),
        (picture(200, 136, PixelFormat::Rgb24, 2), 1.0),
        (picture(300, 200, PixelFormat::Gray8, 3), 2.0),
        (picture(600, 300, PixelFormat::Rgb24, 4), 0.5),
        (picture(300, 280, PixelFormat::Rgba, 5), 1.0),
        (picture(2100, 40, PixelFormat::Rgb24, 6), 3.0),
    ];
    for (i, (img, d)) in cases.iter().enumerate() {
        let bytes = encode(img, &EncodeOptions::new().with_distance(*d)).unwrap();
        let ours = decode(&bytes).unwrap();
        let jxl = dir.join(format!("c{i}.jxl"));
        let ext = match img.format {
            PixelFormat::Rgba => "pam",
            PixelFormat::Gray8 => "pgm",
            _ => "ppm",
        };
        let out = dir.join(format!("c{i}.{ext}"));
        std::fs::write(&jxl, &bytes).unwrap();
        let run = Command::new("djxl").arg(&jxl).arg(&out).output().unwrap();
        assert!(
            run.status.success(),
            "black-box decoder refused case {i}: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        let raw = std::fs::read(&out).unwrap();
        let want = ours.as_bytes().unwrap();
        assert!(raw.len() > want.len(), "case {i}: short output");
        let theirs = &raw[raw.len() - want.len()..];
        // Two float pipelines: the pictures agree to within rounding
        // (a sample here and there lands on the other side of .5).
        let mut worst = 0i32;
        let mut sum = 0u64;
        for (&a, &b) in want.iter().zip(theirs) {
            let e = (i32::from(a) - i32::from(b)).abs();
            worst = worst.max(e);
            sum += e as u64;
        }
        let mean = sum as f64 / want.len() as f64;
        assert!(
            worst <= 3 && mean < 0.5,
            "case {i}: decoders differ (worst {worst}, mean {mean:.3})"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
