//! Round 385 — XYB-domain accuracy of the integrated VarDCT decode on
//! `vardct-256x256-d1`, measured against the reference decode inverted
//! through the spec **forward** XYB transform (no 8-bit RGB round-trip
//! on our side: the pre-§L.2.2 planes are read via the
//! `VARDCT_XYB_CAPTURE` diagnostic hook).
//!
//! These are the domain-pure pins of the three round-385 fixes:
//!
//! 1. Listing C.1 corrected LF-multiplier reading
//!    (`LfMultipliers::compute`),
//! 2. the Annex G / Figure 2 CfL branch split (LF factors on the LF
//!    planes, per-tile HF factors on the dequantised HF coefficients),
//! 3. the Annex G LF-factor `-128` bias (`chroma_from_luma::kx_kb_lf`).
//!
//! Measured at this baseline: internal XYB frame-means X 0.00014 /
//! Y 0.45788 / B 0.47213 against reference forward-XYB means 0.00016 /
//! 0.45807 / 0.47229; per-channel LF-part (8×8 block-mean) LSQ scales
//! 1.14 (X, noise-limited) / 1.003 (Y) / 1.002 (B). The HF part is NOT
//! pinned here — the reference applies the §J restoration filters,
//! which the integrated path does not yet run.
//!
//! Clean-room: the reference values are the `djxl` validator's opaque
//! output PNG inverted through the ISO/IEC 18181-1 forward XYB math
//! (Annex L.2 + the default OpsinInverseMatrix). No external
//! implementation source is consulted.

use std::io::Cursor;
use std::sync::atomic::Ordering;

use oxideav_jpegxl::metadata_fdis::{OpsinInverseMatrix, ToneMapping};
use oxideav_jpegxl::{VARDCT_XYB_CAPTURE, VARDCT_XYB_CAPTURE_ARMED};

const JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");
const REF_PNG: &[u8] = include_bytes!("fixtures/vardct_256x256_d1_expected.png");

/// Run the integrated decode with the XYB capture hook armed and return
/// the cropped pre-§L.2.2 `[X, Y, B]` planes.
fn decode_internal_xyb() -> [Vec<f32>; 3] {
    VARDCT_XYB_CAPTURE.with(|s| *s.borrow_mut() = None);
    VARDCT_XYB_CAPTURE_ARMED.store(true, Ordering::Relaxed);
    let r = oxideav_jpegxl::decode_vardct_frame_from_codestream(JXL, None);
    VARDCT_XYB_CAPTURE_ARMED.store(false, Ordering::Relaxed);
    r.expect("integrated VarDCT decode");
    VARDCT_XYB_CAPTURE
        .with(|s| s.borrow_mut().take())
        .expect("XYB capture populated by finish_vardct_decode")
}

/// Invert the reference PNG through the spec forward XYB transform.
/// Returns per-pixel `[X, Y, B]` planes (256×256 row-major).
fn reference_xyb() -> [Vec<f64>; 3] {
    let oim = OpsinInverseMatrix::default();
    let tm = ToneMapping::default();
    let a = oim.inv_mat;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let fwd = [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det,
        ],
    ];
    let itscale = 255.0 / tm.intensity_target;
    let srgb_to_linear = |c: f32| -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let dec = png::Decoder::new(Cursor::new(REF_PNG));
    let mut reader = dec.read_info().expect("png read_info");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf).expect("png next_frame");
    let ch = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => panic!("unexpected reference colour type {other:?}"),
    };
    assert_eq!((info.width, info.height), (256, 256));
    let n = 256usize * 256;
    let mut planes = [vec![0f64; n], vec![0f64; n], vec![0f64; n]];
    for i in 0..n {
        let c = &buf[i * ch..i * ch + 3];
        let rl = srgb_to_linear(c[0] as f32 / 255.0) / itscale;
        let gl = srgb_to_linear(c[1] as f32 / 255.0) / itscale;
        let bl = srgb_to_linear(c[2] as f32 / 255.0) / itscale;
        let lm = fwd[0][0] * rl + fwd[0][1] * gl + fwd[0][2] * bl;
        let mm = fwd[1][0] * rl + fwd[1][1] * gl + fwd[1][2] * bl;
        let sm = fwd[2][0] * rl + fwd[2][1] * gl + fwd[2][2] * bl;
        let gl_ = (lm - oim.opsin_bias[0]).cbrt() + oim.opsin_bias[0].cbrt();
        let gm_ = (mm - oim.opsin_bias[1]).cbrt() + oim.opsin_bias[1].cbrt();
        let gs_ = (sm - oim.opsin_bias[2]).cbrt() + oim.opsin_bias[2].cbrt();
        planes[0][i] = ((gl_ - gm_) * 0.5) as f64;
        planes[1][i] = ((gl_ + gm_) * 0.5) as f64;
        planes[2][i] = gs_ as f64;
    }
    planes
}

/// The internal XYB frame-means match the reference's forward-XYB means
/// on all three channels — the DC (LF) path end to end: modular LfQuant
/// decode → corrected Listing C.1 multipliers → Listing F.1 dequant →
/// F.2 smoothing → Annex G LF CfL (corrected bias) → Listing I.16 LLF →
/// IDCT → placement → crop.
#[test]
fn vardct_d1_internal_xyb_means_match_reference() {
    let ours = decode_internal_xyb();
    let refp = reference_xyb();
    let n = 256 * 256;
    // (channel, name, absolute mean tolerance). Y/B are ~0.46/0.47-scale
    // quantities; X is a ~1e-4-scale quantity so its tolerance is
    // absolute-tight but relatively loose.
    for (c, name, tol) in [(0usize, "X", 5e-4), (1, "Y", 2e-3), (2, "B", 2e-3)] {
        let our_mean = ours[c].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let ref_mean = refp[c].iter().sum::<f64>() / n as f64;
        assert!(
            (our_mean - ref_mean).abs() < tol,
            "XYB {name}: internal frame-mean {our_mean:.5} should match reference \
             {ref_mean:.5} within {tol}"
        );
    }
}

/// The per-8×8-block means (the LF part) of the internal XYB planes
/// least-squares-fit the reference at unit scale on Y and B, and within
/// the 8-bit-quantisation noise floor on X. Pins the multiplier chain +
/// the CfL branch split; the HF part (intra-block detail) is excluded —
/// it awaits the §J restoration filters.
#[test]
fn vardct_d1_lf_block_means_fit_reference_at_unit_scale() {
    let ours = decode_internal_xyb();
    let refp = reference_xyb();
    let block_mean_f32 = |v: &[f32]| -> Vec<f64> {
        let mut m = vec![0f64; 1024];
        for py in 0..256 {
            for px in 0..256 {
                m[(py / 8) * 32 + px / 8] += v[py * 256 + px] as f64 / 64.0;
            }
        }
        m
    };
    let block_mean_f64 = |v: &[f64]| -> Vec<f64> {
        let mut m = vec![0f64; 1024];
        for py in 0..256 {
            for px in 0..256 {
                m[(py / 8) * 32 + px / 8] += v[py * 256 + px] / 64.0;
            }
        }
        m
    };
    for (c, name, tol) in [(0usize, "X", 0.25), (1, "Y", 0.02), (2, "B", 0.02)] {
        let om = block_mean_f32(&ours[c]);
        let rm = block_mean_f64(&refp[c]);
        let mut num = 0f64;
        let mut den = 0f64;
        for i in 0..1024 {
            num += rm[i] * om[i];
            den += om[i] * om[i];
        }
        let scale = num / den;
        assert!(
            (scale - 1.0).abs() < tol,
            "XYB {name}: LF block-mean LSQ scale (ref/ours) = {scale:.4}, expected \
             ~1.0 ± {tol}"
        );
    }
}
