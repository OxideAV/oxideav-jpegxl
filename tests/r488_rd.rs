//! Round 488 — rate–distortion measurement harness for the lossy
//! (VarDCT) encoder.
//!
//! Not a pass/fail gate: with `JXL_RD_DIR` pointing at a directory of
//! 8-bit PNG files and `JXL_RD_DISTANCES=0.5,1,2,4,8` it encodes each
//! picture at each distance and prints, per line,
//! `RD <name> d<distance> <bytes> psnr <dB> ssim <luma SSIM> enc <ms>
//! dec <ms>` measured through this crate's decoder. Options:
//!
//! * `JXL_RD_BLACK_BOX=1` — also decode every stream with the
//!   installed black-box decoder and print the largest sample
//!   difference to this crate's decode (`bb-diff`), and encode the
//!   picture with the installed black-box encoder at the same
//!   distance (`BB` lines, same metrics, through the black-box
//!   decoder).
//! * `JXL_RD_EFFORT=fast|balanced|thorough`, `JXL_RD_THREADS=n`.
//! * `JXL_RD_OUT=<dir>` keeps the encoded files; with `JXL_RD_DUMP`
//!   the two decodes of an RGB picture are written next to them as
//!   PPM files.
//! * `JXL_RD_BB16=1` prints the mean signed difference to the
//!   black-box decoder's 16-bit output by brightness quarter (a bias
//!   check on the output stage).
//!
//! Without `JXL_RD_DIR` (the CI case) it returns immediately.

mod common;

use std::process::Command;
use std::time::Instant;

use common::png::{BitDepth, ColorType, decode_png};
use oxideav_jpegxl::{Effort, EncodeOptions, JxlImage, PixelFormat, decode, encode};

fn load(path: &std::path::Path) -> Option<JxlImage> {
    let png = decode_png(&std::fs::read(path).unwrap());
    if png.bit_depth != BitDepth::Eight {
        return None;
    }
    let format = match png.color_type {
        ColorType::Grayscale => PixelFormat::Gray8,
        ColorType::GrayscaleAlpha => PixelFormat::Ya8,
        ColorType::Rgb => PixelFormat::Rgb24,
        ColorType::Rgba => PixelFormat::Rgba,
    };
    Some(JxlImage::packed(png.width, png.height, format, png.data).unwrap())
}

/// PSNR over every sample of two packed 8-bit pictures.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "decoded size");
    let sse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum();
    10.0 * (255.0 * 255.0 / (sse / a.len() as f64).max(1e-12)).log10()
}

/// Luma (Rec. 601 weights on the coded samples) of a packed picture.
fn luma(bytes: &[u8], channels: usize) -> Vec<f64> {
    bytes
        .chunks_exact(channels)
        .map(|p| {
            if channels >= 3 {
                0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2])
            } else {
                f64::from(p[0])
            }
        })
        .collect()
}

/// Mean structural similarity of two luma planes: 8×8 windows every
/// four samples, the usual stabilisers `(0.01 × 255)^2` and
/// `(0.03 × 255)^2`.
fn ssim(a: &[f64], b: &[f64], width: usize, height: usize) -> f64 {
    let (c1, c2) = (6.5025f64, 58.5225f64);
    let (mut total, mut n) = (0.0f64, 0u64);
    let mut y = 0;
    while y + 8 <= height {
        let mut x = 0;
        while x + 8 <= width {
            let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for r in 0..8 {
                let o = (y + r) * width + x;
                for c in 0..8 {
                    let (p, q) = (a[o + c], b[o + c]);
                    sa += p;
                    sb += q;
                    saa += p * p;
                    sbb += q * q;
                    sab += p * q;
                }
            }
            let (ma, mb) = (sa / 64.0, sb / 64.0);
            let (va, vb, cov) = (
                saa / 64.0 - ma * ma,
                sbb / 64.0 - mb * mb,
                sab / 64.0 - ma * mb,
            );
            total += (2.0 * ma * mb + c1) * (2.0 * cov + c2)
                / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            n += 1;
            x += 4;
        }
        y += 4;
    }
    total / n.max(1) as f64
}

fn metrics(orig: &JxlImage, got: &[u8]) -> (f64, f64) {
    let a = orig.as_bytes().unwrap();
    let ch = orig.format.bytes_per_pixel();
    let (w, h) = (orig.width as usize, orig.height as usize);
    (psnr(a, got), ssim(&luma(a, ch), &luma(got, ch), w, h))
}

/// Decode `jxl` with the black-box decoder into packed samples of the
/// layout of `like`.
fn black_box_decode(jxl: &std::path::Path, like: &JxlImage) -> Option<Vec<u8>> {
    let out = jxl.with_extension("bb.png");
    let run = Command::new("djxl").arg(jxl).arg(&out).output().ok()?;
    if !run.status.success() {
        eprintln!("djxl failed: {}", String::from_utf8_lossy(&run.stderr));
        return None;
    }
    let png = decode_png(&std::fs::read(&out).ok()?);
    let _ = std::fs::remove_file(&out);
    let want = like.format.bytes_per_pixel();
    if png.channels() == want {
        return Some(png.data);
    }
    // The black-box decoder may write grey as RGB or drop / add alpha.
    let have = png.channels();
    let mut data = Vec::with_capacity(png.data.len() / have * want);
    for p in png.data.chunks_exact(have) {
        match (have, want) {
            (3, 1) | (4, 1) => data.push(p[1]),
            (1, 3) => data.extend_from_slice(&[p[0], p[0], p[0]]),
            (4, 3) => data.extend_from_slice(&p[..3]),
            (3, 4) => data.extend_from_slice(&[p[0], p[1], p[2], 255]),
            (2, 1) => data.push(p[0]),
            (1, 2) => data.extend_from_slice(&[p[0], 255]),
            _ => return None,
        }
    }
    Some(data)
}

#[test]
fn rate_distortion_report() {
    let (Ok(dir), Ok(list)) = (
        std::env::var("JXL_RD_DIR"),
        std::env::var("JXL_RD_DISTANCES"),
    ) else {
        return;
    };
    let black_box = std::env::var("JXL_RD_BLACK_BOX").is_ok();
    let effort = match std::env::var("JXL_RD_EFFORT").as_deref() {
        Ok("fast") => Effort::Fast,
        Ok("thorough") => Effort::Thorough,
        _ => Effort::Balanced,
    };
    let threads: usize = std::env::var("JXL_RD_THREADS")
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or(1);
    let out_dir = std::env::var("JXL_RD_OUT").unwrap_or_else(|_| {
        let d = std::env::temp_dir().join(format!("oxideav-jxl-r488-rd-{}", std::process::id()));
        d.to_string_lossy().to_string()
    });
    std::fs::create_dir_all(&out_dir).unwrap();
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    paths.sort();
    for path in paths {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let Some(img) = load(&path) else { continue };
        for d in list.split(',').filter_map(|d| d.trim().parse::<f32>().ok()) {
            let opts = EncodeOptions::new()
                .with_distance(d)
                .with_effort(effort)
                .with_threads(threads);
            let t = Instant::now();
            let bytes = encode(&img, &opts).unwrap();
            let enc_ms = t.elapsed().as_millis();
            let jxl = std::path::PathBuf::from(format!("{out_dir}/{name}.d{d}.jxl"));
            std::fs::write(&jxl, &bytes).unwrap();
            let t = Instant::now();
            let back = decode(&bytes).unwrap();
            let dec_ms = t.elapsed().as_millis();
            let ours = back.as_bytes().unwrap();
            let (p, s) = metrics(&img, ours);
            let mut line = format!(
                "RD {name} d{d} {} psnr {p:.2} ssim {s:.4} enc {enc_ms} dec {dec_ms}",
                bytes.len()
            );
            if black_box {
                match black_box_decode(&jxl, &img) {
                    Some(bb) if bb.len() == ours.len() => {
                        let (at, worst) = bb
                            .iter()
                            .zip(ours)
                            .map(|(&a, &b)| a.abs_diff(b))
                            .enumerate()
                            .max_by_key(|&(_, d)| d)
                            .unwrap_or((0, 0));
                        let px = at / img.format.bytes_per_pixel();
                        let (wx, wy) = (px % img.width as usize, px / img.width as usize);
                        let mean = bb
                            .iter()
                            .zip(ours)
                            .map(|(&a, &b)| f64::from(a.abs_diff(b)))
                            .sum::<f64>()
                            / bb.len() as f64;
                        if std::env::var("JXL_RD_DUMP").is_ok() && img.format == PixelFormat::Rgb24
                        {
                            for (tag, data) in [("ours", ours), ("bb", bb.as_slice())] {
                                let mut ppm =
                                    format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
                                ppm.extend_from_slice(data);
                                std::fs::write(jxl.with_extension(format!("{tag}.ppm")), ppm)
                                    .unwrap();
                            }
                        }
                        let (bp, bs) = metrics(&img, &bb);
                        line += &format!(
                            " bb-diff max {worst} at {wx},{wy} mean {mean:.3} bb-psnr {bp:.2} bb-ssim {bs:.4}"
                        );
                    }
                    _ => line += " bb-decode FAILED",
                }
            }
            println!("{line}");
            if std::env::var("JXL_RD_BB16").is_ok() {
                // Signed difference to the black-box decoder's 16-bit
                // output, by brightness quarter (first channel).
                let out = jxl.with_extension("bb16.png");
                let run = Command::new("djxl")
                    .arg(&jxl)
                    .arg(&out)
                    .arg("--bits_per_sample=16")
                    .output()
                    .unwrap();
                assert!(run.status.success());
                let png = decode_png(&std::fs::read(&out).unwrap());
                let ch = png.channels();
                let step = img.format.bytes_per_pixel();
                let mut bins = [(0.0f64, 0u64); 4];
                for (i, px) in png.data.chunks_exact(2 * ch).enumerate() {
                    let theirs = f64::from(u16::from_be_bytes([px[0], px[1]])) / 257.0;
                    let mine = f64::from(ours[i * step]);
                    let bin = &mut bins[(ours[i * step] / 64) as usize];
                    bin.0 += mine - theirs;
                    bin.1 += 1;
                }
                let report: Vec<String> = bins
                    .iter()
                    .map(|b| format!("{:+.3}", b.0 / b.1.max(1) as f64))
                    .collect();
                println!(
                    "BB16 {name} d{d} mean(ours - theirs) by quarter: {}",
                    report.join(" ")
                );
            }
            if black_box {
                let bb_jxl = std::path::PathBuf::from(format!("{out_dir}/{name}.d{d}.ref.jxl"));
                let run = Command::new("cjxl")
                    .arg(&path)
                    .arg(&bb_jxl)
                    .arg("-d")
                    .arg(d.to_string())
                    .output();
                if let Ok(run) = run
                    && run.status.success()
                    && let Some(bb) = black_box_decode(&bb_jxl, &img)
                {
                    let (bp, bs) = metrics(&img, &bb);
                    let size = std::fs::metadata(&bb_jxl).map(|m| m.len()).unwrap_or(0);
                    // The same stream through this crate's decoder.
                    let own = std::fs::read(&bb_jxl)
                        .ok()
                        .and_then(|b| decode(&b).ok())
                        .filter(|i| i.format == img.format)
                        .map(|i| metrics(&img, i.as_bytes().unwrap()));
                    let own = match own {
                        Some((p, s)) => format!(" own-psnr {p:.2} own-ssim {s:.4}"),
                        None => String::new(),
                    };
                    println!("BB {name} d{d} {size} psnr {bp:.2} ssim {bs:.4}{own}");
                }
            }
        }
    }
}

#[test]
fn ssim_of_identical_planes_is_one() {
    let a: Vec<f64> = (0..64 * 64).map(|i| ((i * 37) % 251) as f64).collect();
    assert!((ssim(&a, &a, 64, 64) - 1.0).abs() < 1e-12);
    let b: Vec<f64> = a.iter().map(|v| 255.0 - v).collect();
    assert!(ssim(&a, &b, 64, 64) < 0.5);
    assert!(psnr(&[1, 2, 3], &[1, 2, 3]) > 100.0);
}
