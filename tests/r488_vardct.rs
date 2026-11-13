//! Round 488 — the VarDCT encoder's coding tools: every transform type
//! it can choose (DCT8×8 … DCT64×64, the rectangular DCTs, DCT4×4,
//! DCT4×8, DCT8×4, DCT2×2, IDENTITY, AFV0..3), the Gabor-like transform
//! with its encoder-side sharpening, the edge-preserving filter, the
//! quantisation field and the fitted chroma-from-luma factors. Each
//! stream decodes in this crate to the picture it was made from, and —
//! where a black-box decoder is installed — to the same samples there.

use std::process::Command;

use oxideav_jpegxl::dct_select::TransformType;
use oxideav_jpegxl::encoder::{ColourSpec, EncodeSpec, FrameInput, encode_codestream};
use oxideav_jpegxl::{Effort, EncodeOptions, JxlImage, PixelFormat, decode, encode};

/// Smooth shading, hard edges, saturated flats and a textured band: a
/// picture on which every transform size has something to do.
fn picture(width: usize, height: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2654435761) | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (x as f64 / width as f64, y as f64 / height as f64);
            let shade = 0.5 + 0.35 * (fx * 5.0 + fy * 3.0).sin();
            let flat = (x / 48 + y / 40) % 3 == 0;
            let texture = if fy > 0.6 { 0.08 } else { 0.02 };
            for c in 0..3 {
                let noise = f64::from(next() % 1000) / 1000.0 - 0.5;
                let v = if flat {
                    [0.95, 0.45, 0.05][c]
                } else {
                    shade * [1.0, 0.8, 0.6][c] + texture * noise
                };
                data.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    data
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sse: f64 = a
        .iter()
        .zip(b)
        .map(|(&p, &q)| (f64::from(p) - f64::from(q)).powi(2))
        .sum();
    10.0 * (255.0f64 * 255.0 / (sse / a.len() as f64).max(1e-9)).log10()
}

fn spec(width: usize, height: usize, distance: f32, effort: Effort) -> EncodeSpec {
    EncodeSpec {
        width: width as u32,
        height: height as u32,
        bits: 8,
        grey: false,
        alpha: false,
        colour: ColourSpec::Enumerated {
            primaries: 1,
            transfer: 13,
        },
        animation: None,
        quality: 90,
        group_size_shift: 1,
        effort,
        vardct_distance: Some(distance),
        threads: 1,
        vardct_transform: None,
    }
}

fn planes(rgb: &[u8]) -> FrameInput {
    FrameInput {
        channels: (0..3)
            .map(|c| {
                rgb.iter()
                    .skip(c)
                    .step_by(3)
                    .map(|&v| i32::from(v))
                    .collect()
            })
            .collect(),
        duration: 0,
    }
}

fn black_box_available() -> bool {
    Command::new("djxl").arg("--version").output().is_ok()
}

/// The black-box decode of `bytes` as packed RGB, or `None` when no
/// black-box decoder is installed.
fn black_box_decode(bytes: &[u8], tag: &str, len: usize) -> Option<Vec<u8>> {
    if !black_box_available() {
        return None;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-jxl-r488v-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (jxl, out) = (
        dir.join(format!("{tag}.jxl")),
        dir.join(format!("{tag}.ppm")),
    );
    std::fs::write(&jxl, bytes).unwrap();
    let run = Command::new("djxl").arg(&jxl).arg(&out).output().unwrap();
    assert!(
        run.status.success(),
        "{tag}: the black-box decoder refused the stream: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let raw = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_file(&jxl);
    let _ = std::fs::remove_file(&out);
    assert!(raw.len() > len, "{tag}: short black-box output");
    Some(raw[raw.len() - len..].to_vec())
}

/// Worst and mean absolute sample difference.
fn difference(a: &[u8], b: &[u8]) -> (u8, f64) {
    let mut worst = 0u8;
    let mut sum = 0u64;
    for (&p, &q) in a.iter().zip(b) {
        let d = p.abs_diff(q);
        worst = worst.max(d);
        sum += u64::from(d);
    }
    (worst, sum as f64 / a.len() as f64)
}

const ALL_TRANSFORMS: [TransformType; 21] = [
    TransformType::Dct8x8,
    TransformType::Hornuss,
    TransformType::Dct2x2,
    TransformType::Dct4x4,
    TransformType::Dct16x16,
    TransformType::Dct32x32,
    TransformType::Dct16x8,
    TransformType::Dct8x16,
    TransformType::Dct32x8,
    TransformType::Dct8x32,
    TransformType::Dct32x16,
    TransformType::Dct16x32,
    TransformType::Dct4x8,
    TransformType::Dct8x4,
    TransformType::Afv0,
    TransformType::Afv1,
    TransformType::Afv2,
    TransformType::Afv3,
    TransformType::Dct64x64,
    TransformType::Dct64x32,
    TransformType::Dct32x64,
];

/// A frame coded with one transform type wherever it fits (the edge
/// blocks of the 200×136 picture fall back to DCT8×8) decodes to the
/// source, here and in the black-box decoder.
#[test]
fn every_transform_type_round_trips() {
    let (w, h) = (200usize, 136usize);
    let rgb = picture(w, h, 3);
    for (i, &t) in ALL_TRANSFORMS.iter().enumerate() {
        let mut spec = spec(w, h, 1.0, Effort::Thorough);
        spec.vardct_transform = Some(t);
        let bytes = encode_codestream(&spec, &[planes(&rgb)]).unwrap();
        let ours = decode(&bytes).unwrap();
        assert_eq!(ours.format, PixelFormat::Rgb24);
        let ours = ours.as_bytes().unwrap();
        let q = psnr(&rgb, ours);
        // The coarse default matrices of IDENTITY and AFV sit lower.
        let floor = match t {
            TransformType::Hornuss => 24.0,
            TransformType::Afv0
            | TransformType::Afv1
            | TransformType::Afv2
            | TransformType::Afv3
            | TransformType::Dct2x2 => 27.0,
            _ => 30.0,
        };
        assert!(q > floor, "{t:?}: PSNR {q:.2} dB");
        if let Some(theirs) = black_box_decode(&bytes, &format!("t{i}"), ours.len()) {
            let (worst, mean) = difference(ours, &theirs);
            assert!(
                worst <= 2 && mean < 0.4,
                "{t:?}: decoders differ (worst {worst}, mean {mean:.3})"
            );
        }
    }
}

/// Every effort and the whole distance range — which covers no
/// edge-preserving filter, one and two iterations of it — decode to
/// the same picture here and in the black-box decoder; the size falls
/// and the error grows with the distance.
#[test]
fn efforts_and_distances_agree_with_the_black_box_decoder() {
    let (w, h) = (300usize, 280usize);
    let rgb = picture(w, h, 11);
    let img = JxlImage::packed(w as u32, h as u32, PixelFormat::Rgb24, rgb.clone()).unwrap();
    for effort in [Effort::Fast, Effort::Balanced, Effort::Thorough] {
        let mut last = (usize::MAX, f64::INFINITY);
        for (k, d) in [0.4f32, 1.0, 3.0, 7.0, 14.0].into_iter().enumerate() {
            let opts = EncodeOptions::new().with_distance(d).with_effort(effort);
            let bytes = encode(&img, &opts).unwrap();
            let ours = decode(&bytes).unwrap();
            let ours = ours.as_bytes().unwrap();
            let q = psnr(&rgb, ours);
            assert!(
                bytes.len() < last.0 && q < last.1 + 0.3,
                "{effort:?} d{d}: {} bytes / {q:.2} dB after {} / {:.2}",
                bytes.len(),
                last.0,
                last.1
            );
            last = (bytes.len(), q);
            let tag = format!("e{}d{k}", effort as u8);
            if let Some(theirs) = black_box_decode(&bytes, &tag, ours.len()) {
                let (worst, mean) = difference(ours, &theirs);
                assert!(
                    worst <= 2 && mean < 0.4,
                    "{effort:?} d{d}: decoders differ (worst {worst}, mean {mean:.3})"
                );
            }
        }
        assert!(
            last.1 > 22.0,
            "{effort:?}: PSNR {:.2} dB at distance 14",
            last.1
        );
    }
}

/// The transform search does not lose to DCT8×8 alone: the default and
/// the largest effort land on or above the fastest effort's
/// rate–distortion curve (its points at neighbouring distances,
/// interpolated in the logarithm of the size), within half a decibel —
/// the search moves the operating point, it must not move it off the
/// curve. Measured on the green plane as a grey picture: PSNR over RGB
/// is not a fair judge on saturated colour, where a chroma error far
/// below the step lands on a near-black primary and is magnified by
/// the transfer curve.
#[test]
fn the_transform_search_stays_on_the_rate_distortion_curve() {
    let (w, h) = (512usize, 384usize);
    let grey: Vec<u8> = picture(w, h, 5)
        .iter()
        .skip(1)
        .step_by(3)
        .copied()
        .collect();
    let img = JxlImage::packed(w as u32, h as u32, PixelFormat::Gray8, grey.clone()).unwrap();
    let run = |effort, d: f32| {
        let bytes = encode(
            &img,
            &EncodeOptions::new().with_distance(d).with_effort(effort),
        )
        .unwrap();
        let q = psnr(&grey, decode(&bytes).unwrap().as_bytes().unwrap());
        ((bytes.len() as f64).log2(), q)
    };
    // The fastest effort's curve, from small files to large.
    let curve: Vec<(f64, f64)> = [6.0f32, 4.0, 2.5, 1.6, 1.0, 0.6, 0.35]
        .iter()
        .map(|&d| run(Effort::Fast, d))
        .collect();
    let on_curve = |size: f64| -> f64 {
        let k = curve
            .windows(2)
            .position(|p| size <= p[1].0)
            .unwrap_or(curve.len() - 2);
        let (a, b) = (curve[k], curve[k + 1]);
        a.1 + (b.1 - a.1) * (size - a.0) / (b.0 - a.0)
    };
    for d in [1.0f32, 3.0] {
        for effort in [Effort::Balanced, Effort::Thorough] {
            let got = run(effort, d);
            let expected = on_curve(got.0);
            assert!(
                got.1 > expected - 0.5,
                "d{d} {effort:?}: {:.2} dB at 2^{:.2} bytes, the fastest effort has {expected:.2} dB there",
                got.1,
                got.0
            );
        }
    }
}

/// Chroma from luma: a picture whose chroma follows its luma (one hue
/// at varying brightness, textured) is coded smaller with the fitted
/// factors than a grey picture of the same luma plus the cost of a
/// flat chroma would suggest — the X and B planes cost a small
/// fraction of the luma plane.
#[test]
fn correlated_chroma_costs_little() {
    let (w, h) = (256usize, 256usize);
    let mut s = 99u32;
    let mut grey = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let v = 110.0
                + 60.0 * ((x as f64) * 0.11).sin() * ((y as f64) * 0.07).cos()
                + f64::from(s % 60);
            grey.push(v.clamp(0.0, 255.0) as u8);
        }
    }
    // The same luma pattern in one hue.
    let tinted: Vec<u8> = grey
        .iter()
        .flat_map(|&g| {
            let v = f64::from(g);
            [v, v * 0.62, v * 0.35].map(|c| c.round() as u8)
        })
        .collect();
    let opts = EncodeOptions::new().with_distance(1.5);
    let grey_img = JxlImage::packed(w as u32, h as u32, PixelFormat::Gray8, grey).unwrap();
    let tinted_img = JxlImage::packed(w as u32, h as u32, PixelFormat::Rgb24, tinted).unwrap();
    let grey_bytes = encode(&grey_img, &opts).unwrap().len();
    let tinted_bytes = encode(&tinted_img, &opts).unwrap().len();
    assert!(
        (tinted_bytes as f64) < grey_bytes as f64 * 1.6,
        "tinted {tinted_bytes} bytes against grey {grey_bytes}"
    );
}
