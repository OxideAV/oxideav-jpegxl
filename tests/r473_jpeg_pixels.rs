//! Round 473 — pixel decode of JPEG-recompressed files on every
//! sampling lattice `jbrd` can describe (4:4:4, 4:2:0, 4:2:2, 4:4:0).
//!
//! Until this round the contract `decode` ran every VarDCT frame through
//! the XYB pipeline: a `do_YCbCr` transcode frame came out as garbage
//! RGB (4:4:4) or as `Unsupported` (any chroma subsampling) on both the
//! standalone and the registry path — the gateway-suite finding behind
//! this round. A transcode now decodes to the JPEG's own planar
//! full-range YCbCr (`YuvJ444P` / `YuvJ420P` / `YuvJ422P` / `Yuv440P`,
//! chroma planes `ceil(w / h) × ceil(h / v)`) through the exact
//! coefficient decoder Annex A reconstruction already pins byte-exact,
//! plus the 10918-1 A.3.3 IDCT per channel; `to_rgb8` applies the
//! ISO/IEC 18181-1 J.2 triangle upsampling and the §L.3 matrix.
//!
//! Oracle: `djpeg` (accurate integer IDCT, fancy
//! upsampling — the same triangle filter J.2 specifies) decoding the
//! ORIGINAL JPEG, committed as `*_djpeg.png`. Tolerance held by every
//! fixture here: **≤ 3 per 8-bit sample**, with ≥ 96 % of samples exact
//! on 4:4:4 / greyscale (IDCT rounding only) and ≥ 60 % exact with a
//! mean error ≤ 0.36 on the subsampled layouts (the upsampled chroma
//! stays float here where `djpeg` rounds it to integers before the
//! colour matrix). The reference JPEG XL decoder itself sits ≤ 3 from
//! `djpeg` on smooth content and up to ±15 on noisy 4:4:4 content (it
//! does not clamp YCbCr before the colour transform; this crate does,
//! like every JPEG decoder, which is what the planar `YuvJ*` layouts
//! require).

mod common;

use common::png::decode_png;
use oxideav_jpegxl::{decode, decode_all, decode_rgb8, decode_rgba8, info, PixelFormat};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap_or_else(|e| panic!("{name}: {e}"))
}

struct Tolerance {
    max: u8,
    min_exact: f64,
    max_mean: f64,
}

const TIGHT: Tolerance = Tolerance {
    max: 3,
    min_exact: 0.96,
    max_mean: 0.06,
};
const SUBSAMPLED: Tolerance = Tolerance {
    max: 3,
    min_exact: 0.60,
    max_mean: 0.36,
};

/// Compare packed RGB8 against the `djpeg` reference PNG (RGB or grey).
fn assert_close(name: &str, w: u32, h: u32, rgb: &[u8], reference_png: &str, tol: &Tolerance) {
    let png = decode_png(&fixture(reference_png));
    assert_eq!(
        (png.width, png.height),
        (w, h),
        "{name}: reference geometry"
    );
    let expected: Vec<u8> = match png.channels() {
        3 => png.data.clone(),
        1 => png.data.iter().flat_map(|&g| [g, g, g]).collect(),
        c => panic!("{name}: reference PNG has {c} channels"),
    };
    assert_eq!(rgb.len(), expected.len(), "{name}: RGB length");
    let mut max = 0u8;
    let mut exact = 0usize;
    let mut sum = 0u64;
    for (&a, &b) in rgb.iter().zip(&expected) {
        let d = a.abs_diff(b);
        max = max.max(d);
        if d == 0 {
            exact += 1;
        }
        sum += u64::from(d);
    }
    let n = rgb.len() as f64;
    let exact_ratio = exact as f64 / n;
    let mean = sum as f64 / n;
    assert!(
        max <= tol.max && exact_ratio >= tol.min_exact && mean <= tol.max_mean,
        "{name}: vs djpeg max {max} (≤ {}), exact {exact_ratio:.4} (≥ {}), mean {mean:.4} \
         (≤ {})",
        tol.max,
        tol.min_exact,
        tol.max_mean
    );
}

/// Full check of one transcode fixture: native layout, plane geometry,
/// `info` agreement, `decode_all` parity, RGB conversions vs `djpeg`.
fn check(jxl_name: &str, reference_png: &str, format: PixelFormat, tol: &Tolerance) {
    let bytes = fixture(jxl_name);
    let img = decode(&bytes).unwrap_or_else(|e| panic!("{jxl_name}: {e}"));
    assert_eq!(img.format, format, "{jxl_name}: native layout");
    assert_eq!(img.bits_per_sample, 8);
    assert_eq!(img.planes.len(), format.plane_count(), "{jxl_name}: planes");
    for (i, p) in img.planes.iter().enumerate() {
        let (pw, ph) = format.plane_dims(img.width, img.height, i).unwrap();
        assert_eq!(p.stride, pw, "{jxl_name}: plane {i} stride");
        assert_eq!(p.data.len(), pw * ph, "{jxl_name}: plane {i} size");
    }
    if format.is_planar() {
        assert_eq!(
            img.as_bytes(),
            None,
            "{jxl_name}: planar has no packed bytes"
        );
        assert_eq!(
            img.color.matrix, 5,
            "{jxl_name}: sYCC matrix on planar output"
        );
        assert_eq!(
            img.color.range,
            oxideav_jpegxl::ColorRange::Full,
            "{jxl_name}: JPEG = full range"
        );
    }
    let i = info(&bytes).unwrap();
    assert_eq!(i.format, format, "{jxl_name}: info layout");
    assert_eq!((i.width, i.height), (img.width, img.height));

    let all = decode_all(&bytes).unwrap();
    assert_eq!(all.len(), 1, "{jxl_name}: one presented frame");
    assert_eq!(all[0].image, img, "{jxl_name}: decode_all == decode");
    assert!(all[0].presented && all[0].delay.is_none());

    let rgb = img.try_to_rgb8().unwrap();
    assert_eq!(rgb.len(), (img.width * img.height * 3) as usize);
    assert_eq!(
        decode_rgb8(&bytes).unwrap().data,
        rgb,
        "{jxl_name}: decode_rgb8"
    );
    let rgba = decode_rgba8(&bytes).unwrap().data;
    assert_eq!(rgba.len(), (img.width * img.height * 4) as usize);
    for (px, pa) in rgb.chunks_exact(3).zip(rgba.chunks_exact(4)) {
        assert_eq!(&pa[..3], px);
        assert_eq!(pa[3], 255);
    }
    assert_close(jxl_name, img.width, img.height, &rgb, reference_png, tol);
}

// ---- the round-448 / 451 transcode fixtures ----------------------------

#[test]
fn transcode_444_gradient() {
    check(
        "r448_grad444.jxl",
        "r448_grad444_djpeg.png",
        PixelFormat::YuvJ444P,
        &TIGHT,
    );
}

#[test]
fn transcode_444_edge_is_exact() {
    // Hard-edge 16×16 content: IDCT rounding never crosses a half.
    check(
        "r448_edge444.jxl",
        "r448_edge444_djpeg.png",
        PixelFormat::YuvJ444P,
        &Tolerance {
            max: 0,
            min_exact: 1.0,
            max_mean: 0.0,
        },
    );
}

#[test]
fn transcode_444_noisy() {
    check(
        "r448_noise444.jxl",
        "r448_noise444_djpeg.png",
        PixelFormat::YuvJ444P,
        &TIGHT,
    );
}

#[test]
fn transcode_444_progressive() {
    check(
        "r451_prog.jxl",
        "r451_prog_djpeg.png",
        PixelFormat::YuvJ444P,
        &TIGHT,
    );
}

#[test]
fn transcode_444_icc_carries_the_profile() {
    check(
        "r451_icc.jxl",
        "r451_icc_djpeg.png",
        PixelFormat::YuvJ444P,
        &TIGHT,
    );
    let img = decode(&fixture("r451_icc.jxl")).unwrap();
    let icc = img.metadata.icc.expect("ICC profile on the planar image");
    assert_eq!(&icc[36..40], b"acsp");
    // With an ICC profile the enumerated colour is unspecified, the
    // YCbCr relationship is still the T.871 matrix.
    assert_eq!(img.color.primaries, 2);
    assert_eq!(img.color.matrix, 5);
}

#[test]
fn transcode_greyscale_is_the_luma_plane() {
    check(
        "r451_grey.jxl",
        "r451_grey_djpeg.png",
        PixelFormat::Gray8,
        &Tolerance {
            max: 1,
            min_exact: 0.96,
            max_mean: 0.03,
        },
    );
}

#[test]
fn transcode_420_gradient() {
    check(
        "r448_grad420.jxl",
        "r448_grad420_djpeg.png",
        PixelFormat::YuvJ420P,
        &SUBSAMPLED,
    );
}

#[test]
fn transcode_420_edge() {
    check(
        "r448_edge420.jxl",
        "r448_edge420_djpeg.png",
        PixelFormat::YuvJ420P,
        &SUBSAMPLED,
    );
}

#[test]
fn transcode_420_big_multi_group() {
    // 512×320: four groups, two LfGroups' worth of CfL tiles, MCU-aligned.
    check(
        "r448_big420.jxl",
        "r448_big420_djpeg.png",
        PixelFormat::YuvJ420P,
        &SUBSAMPLED,
    );
}

#[test]
fn transcode_420_odd_dimensions_mcu_padded() {
    // 100×60: luma grid padded to 14×8 blocks, chroma 50×30 samples.
    check(
        "r451_odd420.jxl",
        "r451_odd420_djpeg.png",
        PixelFormat::YuvJ420P,
        &SUBSAMPLED,
    );
    let img = decode(&fixture("r451_odd420.jxl")).unwrap();
    assert_eq!((img.width, img.height), (100, 60));
    assert_eq!(img.planes[1].stride, 50);
    assert_eq!(img.planes[1].data.len(), 50 * 30);
}

#[test]
fn transcode_420_progressive_odd() {
    check(
        "r451_prog420.jxl",
        "r451_prog420_djpeg.png",
        PixelFormat::YuvJ420P,
        &SUBSAMPLED,
    );
}

#[test]
fn transcode_420_restart_intervals() {
    check(
        "r451_seq420_rst.jxl",
        "r451_seq420_rst_djpeg.png",
        PixelFormat::YuvJ420P,
        &SUBSAMPLED,
    );
}

#[test]
fn transcode_422_gradient() {
    check(
        "r448_grad422.jxl",
        "r448_grad422_djpeg.png",
        PixelFormat::YuvJ422P,
        &SUBSAMPLED,
    );
}

#[test]
fn gateway_fixture_jpeg_transcode_256x256_420() {
    // The 256×256 4:2:0 transcode the contract / registry suites carry.
    check(
        "jpeg_transcode.jxl",
        "jpeg_transcode_djpeg.png",
        PixelFormat::YuvJ420P,
        &SUBSAMPLED,
    );
}

// ---- the round-473 matrix: sampling × baseline/progressive × DRI ------

/// All variants of one sampling class carry the same coefficients
/// (`cjpeg -quality 80` from one source), so they decode to byte-
/// identical planes and share one `djpeg` reference.
fn check_matrix(sampling: &str, format: PixelFormat, variants: &[&str]) {
    let reference = format!("r473_{sampling}_djpeg.png");
    let mut first: Option<oxideav_jpegxl::JxlImage> = None;
    for v in variants {
        let name = format!("r473_{sampling}_{v}.jxl");
        let tol = if sampling == "444" {
            &TIGHT
        } else {
            &SUBSAMPLED
        };
        check(&name, &reference, format, tol);
        let img = decode(&fixture(&name)).unwrap();
        assert_eq!((img.width, img.height), (100, 60));
        match &first {
            None => first = Some(img),
            Some(f) => assert_eq!(
                f.planes, img.planes,
                "{name}: variants of one sampling decode to identical planes"
            ),
        }
    }
}

#[test]
fn matrix_444() {
    check_matrix(
        "444",
        PixelFormat::YuvJ444P,
        &["base_nodri", "base_dri", "prog_nodri", "prog_dri"],
    );
}

#[test]
fn matrix_420() {
    // `cjxl` refuses the progressive + DRI 4:2:0 source (see the notes
    // next to the fixtures), so that cell has no `.jxl`.
    check_matrix(
        "420",
        PixelFormat::YuvJ420P,
        &["base_nodri", "base_dri", "prog_nodri"],
    );
}

#[test]
fn matrix_422() {
    check_matrix(
        "422",
        PixelFormat::YuvJ422P,
        &["base_nodri", "base_dri", "prog_nodri"],
    );
}

#[test]
fn matrix_440() {
    check_matrix(
        "440",
        PixelFormat::Yuv440P,
        &["base_nodri", "base_dri", "prog_nodri", "prog_dri"],
    );
    let img = decode(&fixture("r473_440_base_nodri.jxl")).unwrap();
    // 4:4:0: full-width chroma, half height.
    assert_eq!(img.planes[1].stride, 100);
    assert_eq!(img.planes[1].data.len(), 100 * 30);
}

// ---- limits on the planar output -----------------------------------------

#[test]
fn max_bytes_counts_the_planar_planes() {
    use oxideav_jpegxl::{decode_with, DecodeOptions, JxlError};
    let bytes = fixture("r451_odd420.jxl");
    // 100×60 luma + 2 × 50×30 chroma = 9 000 bytes.
    let ok = DecodeOptions::default().with_max_bytes(9_000u64);
    assert!(decode_with(&bytes, &ok).is_ok());
    let too_small = DecodeOptions::default().with_max_bytes(8_999u64);
    assert!(matches!(
        decode_with(&bytes, &too_small),
        Err(JxlError::LimitExceeded(_))
    ));
}

// ---- registry parity -----------------------------------------------------

#[cfg(feature = "registry")]
mod registry {
    use super::*;
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, RuntimeContext};
    use oxideav_jpegxl::{register, JxlImage, CODEC_ID_STR};

    /// Layer 1 and the framework decoder hand out the same planes with
    /// the same label, for every sampling class.
    #[test]
    fn framework_decoder_matches_layer_1_for_every_sampling() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        for (name, pf) in [
            (
                "r473_444_base_nodri.jxl",
                oxideav_core::PixelFormat::YuvJ444P,
            ),
            (
                "r473_420_base_nodri.jxl",
                oxideav_core::PixelFormat::YuvJ420P,
            ),
            (
                "r473_422_base_nodri.jxl",
                oxideav_core::PixelFormat::YuvJ422P,
            ),
            (
                "r473_440_base_nodri.jxl",
                oxideav_core::PixelFormat::Yuv440P,
            ),
            ("r451_grey.jxl", oxideav_core::PixelFormat::Gray8),
        ] {
            let bytes = fixture(name);
            let img = decode(&bytes).unwrap();
            assert_eq!(oxideav_core::PixelFormat::from(img.format), pf, "{name}");
            let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
            let mut dec = ctx.codecs.first_decoder(&params).unwrap();
            dec.send_packet(&Packet::new(
                0,
                oxideav_core::TimeBase::new(1, 1000),
                bytes.clone(),
            ))
            .unwrap();
            let Frame::Video(vf) = dec.receive_frame().unwrap() else {
                panic!("{name}: video frame expected");
            };
            assert_eq!(vf.image_planes().len(), img.planes.len(), "{name}");
            for (c, p) in vf.image_planes().iter().enumerate() {
                assert_eq!(p.stride, img.planes[c].stride, "{name}: plane {c}");
                assert_eq!(p.data, img.planes[c].data, "{name}: plane {c}");
            }
            let mut back_params = params.clone();
            back_params.width = Some(img.width);
            back_params.height = Some(img.height);
            back_params.pixel_format = Some(pf);
            let back = JxlImage::from_video_frame(&vf, &back_params).unwrap();
            assert_eq!(back.planes, img.planes, "{name}: bridge round trip");
            assert_eq!(back.format, img.format);
        }
    }
}
