//! Round 488 — the VarDCT decoder under a worker budget: PassGroup
//! sections decode in parallel (each worker on its own copy of the
//! entropy state), the edge-preserving filter runs in row bands, the
//! Gabor-like transform per channel and the colour conversion in row
//! bands. The pixels never depend on the budget, and the stream
//! diagnostics the workers count are handed back to the caller.

use oxideav_jpegxl::{
    DecodeOptions, EncodeOptions, JxlImage, PixelFormat, decode_all_with, decode_with, encode,
};

fn picture(width: u32, height: u32, format: PixelFormat, seed: u32) -> JxlImage {
    let channels = format.channels();
    let mut s = seed.wrapping_mul(2654435761) | 1;
    let mut data = Vec::with_capacity((width * height) as usize * channels);
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (f64::from(x) / 97.0, f64::from(y) / 61.0);
            for c in 0..channels {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let noise = f64::from(s % 1000) / 1000.0 - 0.5;
                let v = 0.5
                    + 0.3 * (fx + c as f64).sin() * (fy * 1.3).cos()
                    + if (x / 53 + y / 41) % 2 == 0 {
                        0.1
                    } else {
                        -0.1
                    }
                    + 0.05 * noise;
                data.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    JxlImage::packed(width, height, format, data).unwrap()
}

fn same_at_every_budget(name: &str, bytes: &[u8], budgets: &[usize]) {
    let serial = decode_with(bytes, &DecodeOptions::default()).unwrap();
    for &threads in budgets {
        let got = decode_with(bytes, &DecodeOptions::default().with_threads(threads)).unwrap();
        assert_eq!(
            (got.width, got.height, got.format),
            (serial.width, serial.height, serial.format),
            "{name}, {threads} workers"
        );
        assert!(
            got.as_bytes() == serial.as_bytes(),
            "{name}: {threads} workers decode different samples"
        );
    }
}

/// This crate's own streams: multi-group pictures at distances that
/// signal the Gabor-like transform alone, one and two iterations of
/// the edge-preserving filter, with and without alpha (whose Modular
/// data follows each group's coefficients).
#[test]
fn own_streams_decode_identically_at_every_budget() {
    for (i, (w, h, format, d)) in [
        (700u32, 520u32, PixelFormat::Rgb24, 1.0f32),
        (700, 520, PixelFormat::Rgb24, 3.0),
        (530, 300, PixelFormat::Gray8, 8.0),
        (600, 280, PixelFormat::Rgba, 2.0),
        (300, 200, PixelFormat::Rgb24, 4.0),
    ]
    .into_iter()
    .enumerate()
    {
        let img = picture(w, h, format, 20 + i as u32);
        let bytes = encode(&img, &EncodeOptions::new().with_distance(d)).unwrap();
        same_at_every_budget(&format!("{w}x{h} {format:?} d{d}"), &bytes, &[2, 3, 8, 64]);
    }
}

/// Black-box-encoder streams: 12 and 96 groups with a permuted TOC,
/// three passes over two presets, patches, noise, custom orders, and
/// the committed filter fixtures.
#[test]
fn staged_streams_decode_identically_at_every_budget() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    for name in [
        "large_1024x768_d2",
        "large_3072x2048_multigroup",
        "progressive_ac_multipass",
        "patches_vardct_256x256",
        "noise_feature_256x256",
        "custom_orders_t256_e1",
        "vardct_256x256_d1",
        "r488_transforms",
        "r488_epf2_gab",
        "r488_epf3",
    ] {
        let bytes = std::fs::read(format!("{dir}/{name}.jxl")).unwrap();
        // Two budgets keep the 6 MP streams affordable in an
        // unoptimised test build.
        same_at_every_budget(name, &bytes, &[3, 8]);
    }
}

/// Every frame of an animation, and the stream diagnostics: a decode
/// under a budget reports the same closure / underrun counts as the
/// serial one (zero on these streams).
#[test]
fn animations_and_diagnostics_do_not_depend_on_the_budget() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let bytes = std::fs::read(format!("{dir}/large_3072x2048_multigroup.jxl")).unwrap();
    for threads in [1usize, 8] {
        oxideav_jpegxl::hf_coefficient_histograms::reset_section_closure_failures();
        oxideav_jpegxl::pass_group_hf::reset_walk_underruns();
        decode_with(&bytes, &DecodeOptions::default().with_threads(threads)).unwrap();
        assert_eq!(
            (
                oxideav_jpegxl::hf_coefficient_histograms::section_closure_failures(),
                oxideav_jpegxl::pass_group_hf::walk_underruns(),
            ),
            (0, 0),
            "{threads} workers"
        );
    }
    let frames: Vec<oxideav_jpegxl::Frame> = (0..3)
        .map(|i| {
            oxideav_jpegxl::Frame::new(picture(280, 300, PixelFormat::Rgb24, 70 + i)).with_ticks(2)
        })
        .collect();
    let opts = EncodeOptions::new()
        .with_distance(2.0)
        .with_animation(oxideav_jpegxl::AnimationInfo::new(10, 1, 0));
    let anim = oxideav_jpegxl::encode_all(&frames, &opts).unwrap();
    let serial = decode_all_with(&anim, &DecodeOptions::default()).unwrap();
    let parallel = decode_all_with(&anim, &DecodeOptions::default().with_threads(8)).unwrap();
    assert_eq!(serial.len(), parallel.len());
    for (a, b) in serial.iter().zip(&parallel) {
        assert!(a.image.as_bytes() == b.image.as_bytes());
    }
}
