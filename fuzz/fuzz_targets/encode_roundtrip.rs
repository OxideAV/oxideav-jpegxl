#![no_main]

//! Encoder robustness + correctness target: a fuzz-chosen geometry,
//! layout, depth, quality, group size and metadata set drive the
//! contract `encode` / `encode_all`; the output must parse (`probe` /
//! `info`), decode, and — at quality 100 — reproduce the input samples
//! and metadata bit for bit. Lossy output must decode to the same
//! geometry and layout. Nothing may panic.

use libfuzzer_sys::fuzz_target;
use oxideav_jpegxl::{
    decode, decode_all, encode, encode_all, info, probe, AnimationInfo, ColorInfo, EncodeOptions,
    Frame, JxlImage, Metadata, PixelFormat,
};

const FORMATS: [(PixelFormat, u8); 10] = [
    (PixelFormat::Gray8, 8),
    (PixelFormat::Ya8, 8),
    (PixelFormat::Rgb24, 8),
    (PixelFormat::Rgba, 8),
    (PixelFormat::Gray16Le, 16),
    (PixelFormat::Ya16Le, 16),
    (PixelFormat::Rgb48Le, 16),
    (PixelFormat::Rgba64Le, 16),
    (PixelFormat::Rgb48Le, 12),
    (PixelFormat::Gray8, 1),
];

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let (fmt, bits) = FORMATS[usize::from(data[0]) % FORMATS.len()];
    // Up to 300 × 70 so multi-group paths (256-pixel groups) are hit
    // while staying cheap.
    let width = u32::from(data[1]) % 300 + 1;
    let height = u32::from(data[2]) % 70 + 1;
    let quality = data[3] % 101;
    let shift = data[4] % 4;
    let flags = data[5];
    let frames_n = usize::from(data[6] % 3) + 1;
    let payload = &data[7..];
    let bytes = fmt.bytes_per_sample();
    let n = width as usize * height as usize * fmt.channels();
    let max = (1u32 << bits) - 1;
    let mut img_data = Vec::with_capacity(n * bytes);
    let mut h = u32::from(flags) | 1;
    for i in 0..n {
        h ^= h << 13;
        h ^= h >> 17;
        h ^= h << 5;
        let src = payload.get(i % payload.len().max(1)).copied().unwrap_or(0);
        let v = (u32::from(src) * 257 ^ h) & max;
        if bytes == 1 {
            img_data.push(v as u8);
        } else {
            img_data.extend_from_slice(&(v as u16).to_le_bytes());
        }
    }
    let mut meta = Metadata::new();
    if flags & 1 != 0 {
        let mut icc = vec![0u8; 128 + usize::from(data[7 % data.len()]) * 3];
        for (i, b) in icc.iter_mut().enumerate() {
            *b = payload.get(i % payload.len().max(1)).copied().unwrap_or(0);
        }
        icc[36..40].copy_from_slice(b"acsp");
        meta = meta.with_icc(icc);
    }
    if flags & 2 != 0 {
        meta = meta.with_exif(payload[..payload.len().min(40)].to_vec());
    }
    if flags & 4 != 0 {
        meta = meta.with_xmp(payload[..payload.len().min(60)].to_vec());
    }
    let image = JxlImage::packed(width, height, fmt, img_data)
        .expect("valid geometry")
        .with_bits_per_sample(bits)
        .with_metadata(meta)
        .with_color(ColorInfo::srgb().with_primaries(if flags & 8 != 0 { 9 } else { 1 }));
    let opts = EncodeOptions::new()
        .with_quality(quality)
        .with_group_size_shift(shift)
        .with_container(flags & 16 != 0);
    let out = encode(&image, &opts).expect("encode accepts every valid packed image");
    assert!(probe(&out));
    let i = info(&out).expect("own output parses");
    assert_eq!((i.width, i.height, i.format), (width, height, fmt));
    let back = decode(&out).expect("own output decodes");
    assert_eq!((back.width, back.height, back.format), (width, height, fmt));
    if quality == 100 {
        assert_eq!(back.as_bytes(), image.as_bytes(), "lossless round trip");
        assert_eq!(back.metadata.icc, image.metadata.icc);
        assert_eq!(back.metadata.exif, image.metadata.exif);
        assert_eq!(back.metadata.xmp, image.metadata.xmp);
    }
    if frames_n > 1 {
        let frames: Vec<Frame> = (0..frames_n as u32)
            .map(|k| Frame::new(image.clone()).with_ticks(k + 1))
            .collect();
        let anim_opts = opts.clone().with_animation(AnimationInfo::new(10, 1, 0));
        let out = encode_all(&frames, &anim_opts).expect("encode_all");
        let back = decode_all(&out).expect("animation decodes");
        assert_eq!(back.len(), frames_n);
        for (k, f) in back.iter().enumerate() {
            assert_eq!(f.ticks, k as u32 + 1);
            if quality == 100 {
                assert_eq!(f.image.as_bytes(), image.as_bytes());
            }
        }
    }
});
