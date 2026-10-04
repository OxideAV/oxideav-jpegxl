//! Round 469 — the image-crate contract surface (IMAGE_CRATE_API):
//! `probe` / `info` / `decode` / `decode_with` / `decode_rgb8` /
//! `decode_rgba8` / `decode_all` / `decode_from` / `encode*` over
//! `JxlImage`, pinned against the planar drivers the fixture suite has
//! always exercised (byte-identical samples, interleaved) and against
//! the committed `expected.png` references.

mod common;
use common::png::{decode_png, ColorType};
use std::io::Cursor;

use oxideav_jpegxl::{
    decode, decode_all, decode_all_planar, decode_all_with, decode_from, decode_planar,
    decode_rgb8, decode_rgba8, decode_with, encode, encode_rgb8, encode_rgba8, encode_to, info,
    probe, ColorInfo, ColorRange, DecodeOptions, EncodeOptions, Error, JxlImage, PixelFormat,
    RawFrame, Signature,
};

const PIXEL_1X1: &[u8] = include_bytes!("fixtures/pixel_1x1.jxl");
const GRAY_64: &[u8] = include_bytes!("fixtures/gray_64x64_lossless.jxl");
const GRADIENT: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
const ALPHA_64: &[u8] = include_bytes!("fixtures/alpha_64x64.jxl");
const ALPHA_64_PNG: &[u8] = include_bytes!("fixtures/alpha_64x64_expected.png");
const BIT_DEPTH_16: &[u8] = include_bytes!("fixtures/bit_depth_16.jxl");
const ANIMATION: &[u8] = include_bytes!("fixtures/animation_3frame.jxl");
const ICC_DIGITS: &[u8] = include_bytes!("fixtures/r408_icc_digits.jxl");
const CUSTOM_ICC: &[u8] = include_bytes!("fixtures/r408_custom.icc");
const MODULAR_XYB: &[u8] = include_bytes!("fixtures/modular_xyb_256x256.jxl");
const VARDCT_D3: &[u8] = include_bytes!("fixtures/vardct_256x256_d3.jxl");
const GREY_XYB: &[u8] = include_bytes!("fixtures/r451_grey_xyb.jxl");
const BLENDMODES: &[u8] = include_bytes!("fixtures/conformance_blendmodes.jxl");
const JPEG_TRANSCODE: &[u8] = include_bytes!("fixtures/jpeg_transcode.jxl");

/// Interleave the first `n` planes of a planar frame (`bytes` per
/// sample) into the contract's packed row-major layout.
fn interleave(f: &RawFrame, n: usize, bytes: usize) -> Vec<u8> {
    let p0 = &f.planes[0];
    let w = p0.stride / bytes;
    let h = p0.data.len() / p0.stride;
    let mut out = Vec::with_capacity(w * h * n * bytes);
    for y in 0..h {
        for x in 0..w {
            for p in &f.planes[..n] {
                out.extend_from_slice(&p.data[y * p.stride + x * bytes..][..bytes]);
            }
        }
    }
    out
}

fn assert_matches_planar(name: &str, data: &[u8]) -> JxlImage {
    let img = decode(data).unwrap_or_else(|e| panic!("{name}: decode: {e}"));
    let planar = decode_planar(data, None).unwrap_or_else(|e| panic!("{name}: planar: {e}"));
    let bytes = img.format.bytes_per_sample();
    let chans = img.format.channels();
    assert_eq!(img.planes.len(), 1, "{name}: one packed plane");
    assert_eq!(
        img.stride(),
        img.width as usize * img.format.bytes_per_pixel()
    );
    assert_eq!(
        img.planes[0].data,
        interleave(&planar, chans, bytes),
        "{name}: packed samples differ from the planar drivers"
    );
    let i = info(data).unwrap();
    assert_eq!(
        (i.width, i.height),
        (img.width, img.height),
        "{name}: info geometry"
    );
    assert_eq!(i.format, img.format, "{name}: info layout");
    assert_eq!(i.color, img.color, "{name}: info colour");
    img
}

#[test]
fn probe_is_a_total_signature_sniff() {
    assert!(probe(PIXEL_1X1));
    assert!(probe(JPEG_TRANSCODE));
    assert!(!probe(&[]));
    assert!(!probe(&[0xFF]));
    assert!(!probe(ALPHA_64_PNG));
    assert!(!probe(&JPEG_TRANSCODE[..11]));
}

#[test]
fn decode_matches_the_planar_drivers_byte_for_byte() {
    let img = assert_matches_planar("pixel_1x1", PIXEL_1X1);
    assert_eq!(img.format, PixelFormat::Rgb24);
    assert_eq!(img.as_bytes(), Some(&[255u8, 0, 0][..]));
    assert_eq!(img.color, ColorInfo::srgb());
    assert_eq!(img.bits_per_sample, 8);

    let g = assert_matches_planar("gray_64x64", GRAY_64);
    assert_eq!(g.format, PixelFormat::Gray8);
    assert_eq!(
        assert_matches_planar("gradient", GRADIENT).format,
        PixelFormat::Rgb24
    );
    assert_eq!(
        assert_matches_planar("modular_xyb", MODULAR_XYB).format,
        PixelFormat::Rgb24
    );
    assert_eq!(
        assert_matches_planar("vardct_d3", VARDCT_D3).format,
        PixelFormat::Rgb24
    );
    assert_eq!(
        assert_matches_planar("grey_xyb", GREY_XYB).format,
        PixelFormat::Gray8
    );

    let a = assert_matches_planar("alpha_64x64", ALPHA_64);
    assert_eq!(a.format, PixelFormat::Rgba);
    assert!(info(ALPHA_64).unwrap().has_alpha);

    let d = assert_matches_planar("bit_depth_16", BIT_DEPTH_16);
    assert_eq!(d.format, PixelFormat::Rgb48Le);
    assert_eq!(d.bits_per_sample, 16);
}

#[test]
fn decode_is_the_composed_primary_image() {
    // Multi-layer conformance stream: the primary image is the §C.2
    // composition of its layers (what `decode_all_frames` pinned against
    // `conformance_blendmodes_expected.png` since round 406), not the
    // bare first layer.
    let img = decode(BLENDMODES).unwrap();
    let frames = decode_all_planar(BLENDMODES, None).unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(img.format, PixelFormat::Rgba64Le);
    assert_eq!(img.bits_per_sample, 12);
    assert_eq!(img.planes[0].data, interleave(&frames[0], 4, 2));
    let i = info(BLENDMODES).unwrap();
    assert_eq!((i.width, i.height, i.frames), (1024, 1024, 1));
}

#[test]
fn decode_rgb8_and_rgba8_match_expected_png() {
    let pinfo = decode_png(ALPHA_64_PNG);
    assert_eq!(pinfo.color_type, ColorType::Rgba);
    let expected = &pinfo.data[..];

    let rgba = decode_rgba8(ALPHA_64).unwrap();
    assert_eq!((rgba.width, rgba.height), (64, 64));
    assert_eq!(rgba.data, expected);

    let rgb = decode_rgb8(ALPHA_64).unwrap();
    assert_eq!(rgb.data.len(), 64 * 64 * 3);
    let dropped: Vec<u8> = expected
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    assert_eq!(rgb.data, dropped);

    // Grey replicates to R = G = B; alpha opaque.
    let g = decode_rgba8(GRAY_64).unwrap();
    assert!(g
        .data
        .chunks_exact(4)
        .all(|p| p[0] == p[1] && p[1] == p[2] && p[3] == 255));
}

#[test]
fn to_rgb8_scales_sixteen_bit_by_the_declared_maximum() {
    let img = decode(BIT_DEPTH_16).unwrap();
    let rgb = img.to_rgb8();
    assert_eq!(rgb.len(), 64 * 64 * 3);
    let raw = img.as_bytes().unwrap();
    for (i, v) in rgb.iter().enumerate() {
        let s = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]) as u32;
        assert_eq!(u32::from(*v), (s * 255 + 32767) / 65535);
    }
}

#[test]
fn decode_all_returns_animation_frames_with_delays() {
    let i = info(ANIMATION).unwrap();
    assert_eq!(i.frames, 3);
    let anim = i.animation.expect("animated");
    assert!(anim.ticks_per_second.0 > 0);

    let frames = decode_all(ANIMATION).unwrap();
    assert_eq!(frames.len(), 3);
    let planar = decode_all_planar(ANIMATION, None).unwrap();
    for (f, p) in frames.iter().zip(&planar) {
        assert_eq!(f.image.format, PixelFormat::Rgb24);
        assert_eq!(f.image.planes[0].data, interleave(p, 3, 1));
        assert!(f.presented);
        let d = f.delay.expect("animated frames carry a delay");
        assert_eq!(d, anim.tick_duration(f.ticks));
        assert!(f.ticks > 0 || f.index == 2);
    }
    // The first frame of `decode_all` is `decode`.
    assert_eq!(frames[0].image, decode(ANIMATION).unwrap());

    // Layers mode: every frame of the array, un-composed.
    let layers =
        decode_all_with(ANIMATION, &DecodeOptions::default().with_coalesce(false)).unwrap();
    assert!(layers.len() >= 3);
    assert_eq!(
        layers[0].image.planes[0].data,
        frames[0].image.planes[0].data
    );
}

#[test]
fn still_image_yields_one_frame() {
    let frames = decode_all(GRADIENT).unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].delay, None);
    assert_eq!(frames[0].image, decode(GRADIENT).unwrap());
}

#[test]
fn icc_profile_lands_in_metadata() {
    let i = info(ICC_DIGITS).unwrap();
    assert!(i.has_icc);
    assert_eq!(i.color.range, ColorRange::Full);
    assert_eq!(i.color.primaries, ColorInfo::UNSPECIFIED);
    let img = decode(ICC_DIGITS).unwrap();
    assert_eq!(img.metadata.icc.as_deref(), Some(CUSTOM_ICC));
    assert_eq!(img.metadata.gamma, None);
    // Files without an ICC stream carry none.
    assert_eq!(decode(PIXEL_1X1).unwrap().metadata.icc, None);
}

#[test]
fn info_reports_container_facts() {
    let i = info(JPEG_TRANSCODE).unwrap();
    assert_eq!(i.signature, Signature::Isobmff);
    assert!(i.has_jpeg_reconstruction);
    assert_eq!(i.frames, 1);
    let r = info(PIXEL_1X1).unwrap();
    assert_eq!(r.signature, Signature::RawCodestream);
    assert!(!r.has_exif && !r.has_xmp && !r.has_icc);
    assert_eq!(r.bits_per_sample, 8);
    assert!(!r.float_sample);
}

#[test]
fn limits_are_enforced_before_decoding() {
    let small = DecodeOptions::default().with_max_width(32);
    assert!(matches!(
        decode_with(GRADIENT, &small),
        Err(Error::LimitExceeded(_))
    ));
    let small = DecodeOptions::default().with_max_height(63);
    assert!(matches!(
        decode_with(GRADIENT, &small),
        Err(Error::LimitExceeded(_))
    ));
    let small = DecodeOptions::default().with_max_pixels(4095u64);
    assert!(matches!(
        decode_with(GRADIENT, &small),
        Err(Error::LimitExceeded(_))
    ));
    let small = DecodeOptions::default().with_max_bytes(64 * 64 * 3 - 1);
    assert!(matches!(
        decode_with(GRADIENT, &small),
        Err(Error::LimitExceeded(_))
    ));
    let exact = DecodeOptions::default().with_max_bytes(64 * 64 * 3);
    assert!(decode_with(GRADIENT, &exact).is_ok());
    assert!(decode_with(GRADIENT, &DecodeOptions::default().unlimited()).is_ok());
    assert!(matches!(
        decode_all_with(ANIMATION, &DecodeOptions::default().with_max_pixels(1)),
        Err(Error::LimitExceeded(_))
    ));
}

#[test]
fn strict_rejects_trailing_bytes_on_a_raw_codestream() {
    assert_eq!(info(GRADIENT).unwrap().signature, Signature::RawCodestream);
    let mut padded = GRADIENT.to_vec();
    padded.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    let lenient = decode(&padded).unwrap();
    assert_eq!(lenient, decode(GRADIENT).unwrap());
    let strict = DecodeOptions::default().with_strict(true);
    assert!(matches!(
        decode_with(&padded, &strict),
        Err(Error::InvalidData(_))
    ));
    assert!(decode_with(GRADIENT, &strict).is_ok());
    assert!(decode_all_with(GRADIENT, &strict).is_ok());
}

#[test]
fn decode_from_reads_to_end() {
    let a = decode_from(Cursor::new(GRADIENT.to_vec())).unwrap();
    assert_eq!(a, decode(GRADIENT).unwrap());
}

#[test]
fn hostile_inputs_fail_cleanly() {
    assert!(matches!(decode(&[]), Err(Error::InvalidData(_))));
    assert!(matches!(info(&[0xFF, 0x0A]), Err(Error::InvalidData(_))));
    assert!(decode(&GRADIENT[..GRADIENT.len() / 2]).is_err());
    assert!(decode_all(&ANIMATION[..ANIMATION.len() - 3]).is_err());
}

#[test]
fn encoding_is_unsupported() {
    let img = decode(PIXEL_1X1).unwrap();
    let opts = EncodeOptions::default();
    assert!(matches!(encode(&img, &opts), Err(Error::Unsupported(_))));
    assert!(matches!(
        encode_rgb8(1, 1, &[1, 2, 3], &opts),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        encode_rgba8(1, 1, &[1, 2, 3, 4], &opts),
        Err(Error::Unsupported(_))
    ));
    let mut sink = Vec::new();
    assert!(matches!(
        encode_to(&img, &opts, &mut sink),
        Err(Error::Unsupported(_))
    ));
    assert!(sink.is_empty());
}

#[test]
fn image_constructors_validate_geometry() {
    assert!(JxlImage::from_rgb8(2, 2, vec![0; 12]).is_ok());
    assert!(matches!(
        JxlImage::from_rgb8(2, 2, vec![0; 11]),
        Err(Error::InvalidData(_))
    ));
    assert!(matches!(
        JxlImage::from_rgba8(2, 2, vec![0; 15]),
        Err(Error::InvalidData(_))
    ));
    let img = JxlImage::from_rgba8(1, 2, vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    assert_eq!(img.to_rgb8(), vec![1, 2, 3, 5, 6, 7]);
    assert_eq!(img.into_raw(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
}

#[cfg(feature = "registry")]
mod registry {
    use super::*;
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, VideoFrame};

    #[test]
    fn framework_decoder_emits_the_contract_layout() {
        let mut ctx = oxideav_core::RuntimeContext::new();
        oxideav_jpegxl::register(&mut ctx);
        let params = CodecParameters::video(CodecId::new(oxideav_jpegxl::CODEC_ID_STR));
        let mut dec = ctx.codecs.first_decoder(&params).expect("registered");
        let mut pkt = Packet::new(0, oxideav_core::TimeBase::new(1, 1000), ALPHA_64.to_vec());
        pkt.pts = Some(7);
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(vf) = dec.receive_frame().unwrap() else {
            panic!("video frame expected");
        };
        let img = decode(ALPHA_64).unwrap();
        assert_eq!(vf.pts, Some(7));
        assert_eq!(vf.image_planes().len(), 1);
        assert_eq!(vf.image_planes()[0].data, img.planes[0].data);
        assert_eq!(vf.image_planes()[0].stride, img.stride());
        let sig = vf.color_signal().expect("JPEG XL always signals colour");
        assert_eq!(sig.primaries.0, img.color.primaries);
        assert_eq!(sig.transfer.0, img.color.transfer);
        dec.flush().unwrap();
        assert!(matches!(dec.receive_frame(), Err(oxideav_core::Error::Eof)));
    }

    #[test]
    fn framework_decoder_drains_animation_frames() {
        let params = CodecParameters::video(CodecId::new(oxideav_jpegxl::CODEC_ID_STR));
        let mut dec = oxideav_jpegxl::make_decoder(&params).unwrap();
        dec.send_packet(&Packet::new(
            0,
            oxideav_core::TimeBase::new(1, 1000),
            ANIMATION.to_vec(),
        ))
        .unwrap();
        let mut n = 0;
        while let Ok(Frame::Video(_)) = dec.receive_frame() {
            n += 1;
        }
        assert_eq!(n, 3);
    }

    #[test]
    fn frame_bridge_round_trips_a_decoded_image() {
        let img = decode(BIT_DEPTH_16).unwrap();
        let vf: VideoFrame = img.clone().into();
        let mut params = CodecParameters::video(CodecId::new(oxideav_jpegxl::CODEC_ID_STR));
        params.width = Some(img.width);
        params.height = Some(img.height);
        params.pixel_format = Some(img.format.into());
        let back = JxlImage::from_video_frame(&vf, &params).unwrap();
        assert_eq!(back.planes, img.planes);
        assert_eq!(back.color, img.color);
        assert_eq!(back.format, img.format);
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_planar_wrappers_still_decode() {
        let legacy = oxideav_jpegxl::decode_one_frame(GRADIENT, Some(1)).unwrap();
        assert_eq!(legacy.pts, Some(1));
        assert_eq!(legacy.planes.len(), 3);
        let all = oxideav_jpegxl::decode_all_frames(ANIMATION, None).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(
            oxideav_jpegxl::make_encoder(&CodecParameters::video(CodecId::new("jpegxl")))
                .err()
                .map(|e| matches!(e, oxideav_core::Error::Unsupported(_))),
            Some(true)
        );
    }
}
