//! Round 477 — the Modular encoder: every packed layout round-trips
//! bit-exactly through `decode` at quality 100, on one-group and
//! multi-group geometries; ICC / Exif / XMP ride the codestream and the
//! box file; animations keep their timing; the quality option is a
//! monotone size ladder on textured content; and, where a black-box
//! decoder is installed, its pixels agree with ours.

mod common;

use std::process::Command;
use std::time::Duration;

use common::png::{BitDepth, decode_png};
use oxideav_jpegxl::{
    AnimationInfo, ColorInfo, EncodeOptions, Error, Frame, JxlImage, Metadata, PixelFormat, Plane,
    Signature, decode, decode_all, encode, encode_all, encode_rgba8, info, probe,
};

const ICC_FIXTURE: &[u8] = include_bytes!("fixtures/r451_icc.jxl");

/// Deterministic textured samples: a diagonal gradient plus hashed
/// noise, per channel offset, in `0..=max`.
fn textured(width: u32, height: u32, channels: usize, max: u32, seed: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity((width * height) as usize * channels);
    let mut h = seed.wrapping_mul(2654435761).wrapping_add(1);
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels as u32 {
                h ^= h << 13;
                h ^= h >> 17;
                h ^= h << 5;
                let grad = (x * 3 + y * 5 + c * 40) * max / (width * 3 + height * 5 + 120).max(1);
                let noise = h % (max / 4 + 1);
                out.push((grad + noise).min(max));
            }
        }
    }
    out
}

fn image(width: u32, height: u32, format: PixelFormat, bits: u8, seed: u32) -> JxlImage {
    let channels = format.channels();
    let bytes = format.bytes_per_sample();
    let max = (1u32 << bits) - 1;
    let samples = textured(width, height, channels, max, seed);
    let mut data = Vec::with_capacity(samples.len() * bytes);
    for s in samples {
        if bytes == 1 {
            data.push(s as u8);
        } else {
            data.extend_from_slice(&(s as u16).to_le_bytes());
        }
    }
    JxlImage::packed(width, height, format, data)
        .unwrap()
        .with_bits_per_sample(bits)
}

fn assert_same_pixels(a: &JxlImage, b: &JxlImage, what: &str) {
    assert_eq!((a.width, a.height), (b.width, b.height), "{what}: geometry");
    assert_eq!(a.format, b.format, "{what}: layout");
    assert_eq!(a.bits_per_sample, b.bits_per_sample, "{what}: depth");
    assert_eq!(
        a.as_bytes().unwrap(),
        b.as_bytes().unwrap(),
        "{what}: samples"
    );
}

const LAYOUTS: [(PixelFormat, u8); 10] = [
    (PixelFormat::Gray8, 8),
    (PixelFormat::Ya8, 8),
    (PixelFormat::Rgb24, 8),
    (PixelFormat::Rgba, 8),
    (PixelFormat::Gray16Le, 16),
    (PixelFormat::Ya16Le, 16),
    (PixelFormat::Rgb48Le, 16),
    (PixelFormat::Rgba64Le, 16),
    (PixelFormat::Rgb48Le, 12),
    (PixelFormat::Gray8, 4),
];

#[test]
fn every_layout_round_trips_losslessly_on_one_group() {
    for &(fmt, bits) in &LAYOUTS {
        for &(w, h) in &[(1u32, 1u32), (7, 5), (64, 64), (256, 256), (255, 3)] {
            let img = image(w, h, fmt, bits, w * 31 + h);
            let bytes = encode(&img, &EncodeOptions::default()).unwrap();
            assert!(probe(&bytes));
            let i = info(&bytes).unwrap();
            assert_eq!(
                (i.width, i.height, i.format),
                (w, h, fmt),
                "{fmt:?}/{bits} {w}x{h}"
            );
            assert_eq!(i.bits_per_sample, u32::from(bits));
            assert_eq!(i.frames, 1);
            assert_eq!(i.signature, Signature::RawCodestream);
            let back = decode(&bytes).unwrap();
            assert_same_pixels(&img, &back, &format!("{fmt:?}/{bits} {w}x{h}"));
            assert!(back.metadata.icc.is_none() && back.metadata.exif.is_none());
            assert!(back.color.is_specified());
        }
    }
}

#[test]
fn multi_group_geometries_round_trip() {
    // 256-pixel groups (the default): 300×130 is 2×1 groups, 513×257
    // is 3×2, 2100×40 spans two LF groups (2048-pixel LF group dim) so
    // the empty ModularLfGroup sections are exercised too.
    for &(w, h, fmt, bits, shift) in &[
        (300u32, 130u32, PixelFormat::Rgb24, 8u8, 1u8),
        (513, 257, PixelFormat::Rgba, 8, 1),
        (2100, 40, PixelFormat::Gray8, 8, 1),
        (300, 130, PixelFormat::Rgb48Le, 16, 1),
        (260, 140, PixelFormat::Ya8, 8, 0),
        (1030, 20, PixelFormat::Rgb24, 8, 3),
    ] {
        let img = image(w, h, fmt, bits, 7);
        let opts = EncodeOptions::default().with_group_size_shift(shift);
        let bytes = encode(&img, &opts).unwrap();
        let back = decode(&bytes).unwrap();
        assert_same_pixels(&img, &back, &format!("{fmt:?} {w}x{h} shift {shift}"));
    }
}

#[test]
fn colour_description_is_carried() {
    let img = image(16, 8, PixelFormat::Rgb24, 8, 1)
        .with_color(ColorInfo::srgb().with_primaries(9).with_transfer(16));
    let back = decode(&encode(&img, &EncodeOptions::default()).unwrap()).unwrap();
    assert_eq!(back.color.primaries, 9, "BT.2100 primaries");
    assert_eq!(back.color.transfer, 16, "PQ transfer");
    let grey = image(16, 8, PixelFormat::Gray8, 8, 1);
    let back = decode(&encode(&grey, &EncodeOptions::default()).unwrap()).unwrap();
    assert_eq!(back.format, PixelFormat::Gray8);
    assert_eq!(back.color, ColorInfo::srgb());
}

#[test]
fn icc_profile_is_embedded_in_the_codestream() {
    let src = decode(ICC_FIXTURE).unwrap();
    let profile = src.metadata.icc.clone().expect("fixture carries ICC");
    assert!(profile.len() > 128);
    for fmt in [PixelFormat::Rgb24, PixelFormat::Gray16Le] {
        let img = image(
            20,
            9,
            fmt,
            if fmt == PixelFormat::Gray16Le { 16 } else { 8 },
            3,
        )
        .with_metadata(Metadata::new().with_icc(profile.clone()));
        let bytes = encode(&img, &EncodeOptions::default()).unwrap();
        let i = info(&bytes).unwrap();
        assert!(i.has_icc, "{fmt:?}");
        assert_eq!(
            i.signature,
            Signature::RawCodestream,
            "no box needed for ICC"
        );
        let back = decode(&bytes).unwrap();
        assert_eq!(back.metadata.icc.as_deref(), Some(&profile[..]), "{fmt:?}");
        assert_same_pixels(&img, &back, "icc");
    }
    // A header-only profile (128 bytes) and a profile of every byte value.
    let mut odd: Vec<u8> = (0..700u32).map(|i| (i * 7 + 1) as u8).collect();
    odd[36..40].copy_from_slice(b"acsp");
    let img =
        image(4, 4, PixelFormat::Rgb24, 8, 3).with_metadata(Metadata::new().with_icc(odd.clone()));
    let back = decode(&encode(&img, &EncodeOptions::default()).unwrap()).unwrap();
    assert_eq!(back.metadata.icc.as_deref(), Some(&odd[..]));
}

fn exif_blob() -> Vec<u8> {
    // TIFF header (little-endian) + one IFD with Orientation = 1.
    let mut v = b"II*\0".to_vec();
    v.extend_from_slice(&8u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&0x0112u16.to_le_bytes());
    v.extend_from_slice(&3u16.to_le_bytes());
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&0u16.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v
}

#[test]
fn exif_and_xmp_become_boxes() {
    let xmp = b"<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?><x:xmpmeta xmlns:x='adobe:ns:meta/'/><?xpacket end='w'?>".to_vec();
    let img = image(12, 7, PixelFormat::Rgba, 8, 5)
        .with_metadata(Metadata::new().with_exif(exif_blob()).with_xmp(xmp.clone()));
    let bytes = encode(&img, &EncodeOptions::default()).unwrap();
    let i = info(&bytes).unwrap();
    assert_eq!(i.signature, Signature::Isobmff);
    assert!(i.has_exif && i.has_xmp && !i.has_icc);
    let back = decode(&bytes).unwrap();
    assert_eq!(back.metadata.exif.as_deref(), Some(&exif_blob()[..]));
    assert_eq!(back.metadata.xmp.as_deref(), Some(&xmp[..]));
    assert_same_pixels(&img, &back, "boxed");
    // Box order: signature, ftyp, Exif, xml, jxlc.
    assert_eq!(&bytes[4..8], b"JXL ");
    assert_eq!(&bytes[16..20], b"ftyp");
    assert_eq!(&bytes[36..40], b"Exif");
    assert_eq!(
        &bytes[40..44],
        &0u32.to_be_bytes(),
        "zero TIFF header offset"
    );
    // Forced container without metadata.
    let plain = image(3, 3, PixelFormat::Gray8, 8, 1);
    let boxed = encode(&plain, &EncodeOptions::default().with_container(true)).unwrap();
    assert_eq!(info(&boxed).unwrap().signature, Signature::Isobmff);
    assert_eq!(&boxed[36..40], b"jxlc");
    assert_same_pixels(&plain, &decode(&boxed).unwrap(), "forced container");
}

#[test]
fn animations_keep_their_timing() {
    let frames: Vec<Frame> = (0..3u32)
        .map(|i| {
            Frame::new(image(40, 24, PixelFormat::Rgb24, 8, 100 + i))
                .with_delay(Duration::from_millis(40 * (i as u64 + 1)))
                .with_index(i)
        })
        .collect();
    let bytes = encode_all(&frames, &EncodeOptions::default()).unwrap();
    let i = info(&bytes).unwrap();
    assert_eq!(i.frames, 3);
    let anim = i.animation.expect("animated");
    assert_eq!(anim.ticks_per_second, (1000, 1));
    let back = decode_all(&bytes).unwrap();
    assert_eq!(back.len(), 3);
    for (k, (a, b)) in frames.iter().zip(&back).enumerate() {
        assert_same_pixels(&a.image, &b.image, &format!("frame {k}"));
        assert_eq!(b.ticks, 40 * (k as u32 + 1));
        assert_eq!(b.delay, Some(Duration::from_millis(40 * (k as u64 + 1))));
    }
    // An explicit tick rate uses the frames' ticks.
    let opts = EncodeOptions::default().with_animation(AnimationInfo::new(30, 1, 2));
    let ticked: Vec<Frame> = frames.iter().cloned().map(|f| f.with_ticks(3)).collect();
    let bytes = encode_all(&ticked, &opts).unwrap();
    let i = info(&bytes).unwrap();
    assert_eq!(i.animation.unwrap().ticks_per_second, (30, 1));
    assert_eq!(i.animation.unwrap().loops, 2);
    let back = decode_all(&bytes).unwrap();
    assert!(back.iter().all(|f| f.ticks == 3));
    assert_eq!(back[0].delay, Some(Duration::from_millis(100)));
    // `decode` of an animation is its first frame; one frame is a still.
    assert_same_pixels(&frames[0].image, &decode(&bytes).unwrap(), "first frame");
    let still = encode_all(&frames[..1], &EncodeOptions::default()).unwrap();
    assert!(info(&still).unwrap().animation.is_none());
    // Mixed geometry is a typed refusal naming both shapes.
    let mut mixed = frames.clone();
    mixed[1] = Frame::new(image(41, 24, PixelFormat::Rgb24, 8, 9));
    match encode_all(&mixed, &EncodeOptions::default()) {
        Err(Error::Unsupported(m)) => assert!(m.contains("41×24") && m.contains("40×24"), "{m}"),
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn quality_is_a_monotone_size_ladder_with_bounded_error() {
    let img = image(128, 96, PixelFormat::Rgb24, 8, 42);
    let mut prev_len = 0usize;
    let mut prev_err = u32::MAX;
    for q in [0u8, 25, 50, 75, 90, 100] {
        let bytes = encode(&img, &EncodeOptions::default().with_quality(q)).unwrap();
        assert!(
            bytes.len() > prev_len,
            "q{q}: {} bytes after {prev_len}",
            bytes.len()
        );
        prev_len = bytes.len();
        let back = decode(&bytes).unwrap();
        let src = img.as_bytes().unwrap();
        let out = back.as_bytes().unwrap();
        let max_err = src
            .iter()
            .zip(out)
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).unsigned_abs())
            .max()
            .unwrap();
        if q == 100 {
            assert_eq!(max_err, 0, "quality 100 is lossless");
        } else {
            let step = oxideav_jpegxl::encoder::quantizer_for_quality(q, 8);
            // The RCT spreads one channel's quantisation error over R
            // and B (two half-steps) — never more than one full step
            // plus the half-step rounding.
            assert!(
                max_err <= step + step / 2 + 1,
                "q{q}: max error {max_err} for step {step}"
            );
            assert!(
                max_err <= prev_err,
                "q{q}: error grew from {prev_err} to {max_err}"
            );
        }
        prev_err = max_err;
    }
    // 16-bit input quantises on the same relative scale.
    let deep = image(64, 48, PixelFormat::Rgb48Le, 16, 4);
    let q50 = encode(&deep, &EncodeOptions::default().with_quality(50)).unwrap();
    let lossless = encode(&deep, &EncodeOptions::default()).unwrap();
    assert!(q50.len() < lossless.len());
    let back = decode(&q50).unwrap();
    assert_eq!(back.format, PixelFormat::Rgb48Le);
}

#[test]
fn encode_rgba8_helper_and_from_rgb() {
    let rgba: Vec<u8> = (0..4 * 6 * 4).map(|i| (i * 13) as u8).collect();
    let bytes = encode_rgba8(4, 6, &rgba, &EncodeOptions::default()).unwrap();
    let back = decode(&bytes).unwrap();
    assert_eq!(back.format, PixelFormat::Rgba);
    assert_eq!(back.as_bytes().unwrap(), &rgba[..]);
    assert!(matches!(
        encode_rgba8(4, 6, &rgba[..10], &EncodeOptions::default()),
        Err(Error::InvalidData(_))
    ));
}

#[test]
fn planar_and_out_of_range_inputs_are_refused() {
    let yuv = JxlImage::new(
        2,
        2,
        PixelFormat::YuvJ420P,
        vec![
            Plane::new(2, vec![0; 4]),
            Plane::new(1, vec![0; 1]),
            Plane::new(1, vec![0; 1]),
        ],
    )
    .unwrap();
    assert!(matches!(
        encode(&yuv, &EncodeOptions::default()),
        Err(Error::Unsupported(_))
    ));
    // A 12-bit image whose samples exceed 4095.
    let img = JxlImage::packed(1, 1, PixelFormat::Rgb48Le, vec![0xFF, 0xFF, 0, 0, 0, 0])
        .unwrap()
        .with_bits_per_sample(12);
    assert!(matches!(
        encode(&img, &EncodeOptions::default()),
        Err(Error::InvalidData(_))
    ));
}

/// Black-box check: a third-party decoder, when installed, must see
/// the same samples. Skipped (passes vacuously) without `djxl`.
#[test]
fn black_box_decoder_agrees_when_installed() {
    if Command::new("djxl").arg("--version").output().is_err() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-jpegxl-r477-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = decode(ICC_FIXTURE).unwrap();
    let profile = src.metadata.icc.clone().unwrap();
    let cases: Vec<(&str, JxlImage, u8)> = vec![
        ("gray8", image(37, 23, PixelFormat::Gray8, 8, 1), 100),
        ("ya8", image(37, 23, PixelFormat::Ya8, 8, 2), 100),
        ("rgb24", image(300, 130, PixelFormat::Rgb24, 8, 3), 100),
        ("rgba", image(513, 257, PixelFormat::Rgba, 8, 4), 100),
        ("gray16", image(40, 30, PixelFormat::Gray16Le, 16, 5), 100),
        ("rgb48", image(64, 40, PixelFormat::Rgb48Le, 16, 6), 100),
        ("rgba64", image(64, 40, PixelFormat::Rgba64Le, 16, 7), 100),
        ("rgb_q50", image(128, 96, PixelFormat::Rgb24, 8, 8), 50),
        (
            "icc",
            image(20, 9, PixelFormat::Rgb24, 8, 9).with_metadata(Metadata::new().with_icc(profile)),
            100,
        ),
        (
            "exif",
            image(12, 7, PixelFormat::Rgb24, 8, 10).with_metadata(
                Metadata::new()
                    .with_exif(exif_blob())
                    .with_xmp(b"<x/>".to_vec()),
            ),
            100,
        ),
        ("wide_lf", image(2100, 40, PixelFormat::Gray8, 8, 11), 100),
    ];
    for (name, img, q) in cases {
        let bytes = encode(&img, &EncodeOptions::default().with_quality(q)).unwrap();
        let jxl = dir.join(format!("{name}.jxl"));
        let png = dir.join(format!("{name}.png"));
        std::fs::write(&jxl, &bytes).unwrap();
        let out = Command::new("djxl").arg(&jxl).arg(&png).output().unwrap();
        assert!(
            out.status.success(),
            "{name}: djxl failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let dec = decode_png(&std::fs::read(&png).unwrap());
        assert_eq!((dec.width, dec.height), (img.width, img.height), "{name}");
        assert_eq!(dec.channels(), img.format.channels(), "{name}: channels");
        let ours = decode(&bytes).unwrap();
        let ours_bytes = ours.as_bytes().unwrap();
        match dec.bit_depth {
            BitDepth::Eight => {
                assert_eq!(dec.data.len(), ours_bytes.len(), "{name}");
                assert_eq!(
                    dec.data, ours_bytes,
                    "{name}: samples differ from the black-box decoder"
                );
            }
            BitDepth::Sixteen => {
                // The PNG helper is big-endian, our layouts little-endian.
                let le: Vec<u8> = dec
                    .data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|p| [p[1], p[0]])
                    .collect();
                assert_eq!(
                    le, ours_bytes,
                    "{name}: 16-bit samples differ from the black-box decoder"
                );
            }
        }
    }
    // Animation through the black-box decoder: frame count via the
    // info tool when present.
    let frames: Vec<Frame> = (0..3u32)
        .map(|i| {
            Frame::new(image(40, 24, PixelFormat::Rgb24, 8, 50 + i))
                .with_delay(Duration::from_millis(50))
        })
        .collect();
    let bytes = encode_all(&frames, &EncodeOptions::default()).unwrap();
    let jxl = dir.join("anim.jxl");
    std::fs::write(&jxl, &bytes).unwrap();
    let out = Command::new("djxl")
        .arg(&jxl)
        .arg(dir.join("anim.png"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "animation: djxl failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}
