//! Round 372 pinned the VarDCT `vardct-256x256-d1` LF-magnitude
//! divergence as an exact measured ratio (our dequantised LF Y-mean was
//! **4.0×** the reference's forward-XYB Y-mean under the literal FDIS
//! Listing C.1 formula). Round 385 root-caused that divergence and this
//! suite now pins the **fix**.
//!
//! ## The round-385 root cause (corrected Listing C.1 reading)
//!
//! The FDIS Listing C.1 reads `mXDC = m_x_lf_unscaled / (global_scale ×
//! quant_lf)`. Regressing our per-channel dequantised LF against the
//! reference decode's forward-XYB LF (box-averaged to the 32×32 LF
//! grid, LF chroma-from-luma removed per Annex G's `x_factor_lf /
//! b_factor_lf` terms) measured per-channel least-squares scales of
//! X ≈ 1/256, Y = 1/4 (exact) and B ≈ 1 — i.e. the literal formula is
//! wrong by exactly `m² / 65536` per channel. The unique reading
//! consistent with all three channels at once is
//!
//! ```text
//! mXDC = 65536 / (m_x_lf_unscaled × global_scale × quant_lf)
//! ```
//!
//! (`global_scale` is 16.16 fixed-point; the `m_*_lf_unscaled` values
//! are per-channel quantisation divisors). The default B divisor is the
//! self-reciprocal point (`65536 / 256 = 256`), which is why the B
//! channel appeared correct under the literal reading while Y was 4×
//! and X 256× too large. `LfMultipliers::compute` implements the
//! corrected reading; these tests pin it against the fixture.
//!
//! Clean-room: our LF values come from the in-crate decoder driven on
//! the committed `.jxl` fixture; the reference XYB values are derived
//! from the `djxl` validator's **opaque output PNG** inverted through
//! the ISO/IEC 18181-1 forward XYB math (Annex L.2 + the default
//! OpsinInverseMatrix). No external implementation source is consulted.

use std::io::Cursor;

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::frame_header::{FrameDecodeParams, FrameHeader};
use oxideav_jpegxl::lf_dequant::{dequant_lf, LfMultipliers};
use oxideav_jpegxl::lf_global::LfGlobal;
use oxideav_jpegxl::lf_group::LfGroup;
use oxideav_jpegxl::metadata_fdis::{
    ImageMetadataFdis, OpsinInverseMatrix, SizeHeaderFdis, ToneMapping,
};
use oxideav_jpegxl::toc::Toc;

const JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");
const REF_PNG: &[u8] = include_bytes!("fixtures/vardct_256x256_d1_expected.png");

/// Decode the LfGroup of `vardct-256x256-d1` and return the dequantised
/// LF image as `[X, Y, B]` f32 planes (pre-CfL, pre-smoothing) plus the
/// LF chroma-from-luma factors `(kX, kB)` from the frame's
/// LfChannelCorrelation bundle, and the plane `width × height`.
#[allow(clippy::type_complexity)]
fn decode_dequant_lf() -> ([Vec<f32>; 3], (f32, f32), usize, usize) {
    assert_eq!(&JXL[..2], &[0xFF, 0x0A], "raw codestream signature");
    let cs = &JXL[2..];
    let mut br = BitReader::new(cs);
    let size = SizeHeaderFdis::read(&mut br).expect("size header");
    let metadata = ImageMetadataFdis::read(&mut br).expect("image metadata");
    assert!(metadata.xyb_encoded, "fixture is XYB-encoded VarDCT");
    if metadata.colour_encoding.want_icc {
        let enc = oxideav_jpegxl::icc::decode_encoded_icc_stream(&mut br).expect("icc stream");
        let _ = oxideav_jpegxl::icc::reconstruct_icc_profile(&enc).expect("icc profile");
    }
    br.pu0().expect("byte align before frame data");
    let fhp = FrameDecodeParams {
        xyb_encoded: metadata.xyb_encoded,
        num_extra_channels: metadata.num_extra_channels,
        have_animation: metadata.have_animation,
        have_animation_timecodes: metadata
            .animation
            .map(|a| a.have_timecodes)
            .unwrap_or(false),
        image_width: size.width,
        image_height: size.height,
    };
    let fh = FrameHeader::read_with_edition(
        &mut br,
        &fhp,
        oxideav_jpegxl::frame_header::RfEdition::V2024,
    )
    .expect("frame header");
    let toc = Toc::read(&mut br, &fh).expect("toc");
    assert_eq!(toc.entries.len(), 1, "single-TOC single-group frame");

    let frame_start = br.bytes_consumed();
    let frame_bytes = &cs[frame_start..];
    // Single-TOC layout: LfGlobal then LfGroup share one bit cursor.
    let mut shared = BitReader::new_section(frame_bytes);
    let lf_global = LfGlobal::read(&mut shared, &fh, &metadata).expect("lf global");
    let quantizer = lf_global.quantizer.expect("quantizer present (VarDCT)");
    let cfl = lf_global
        .lf_channel_correlation
        .expect("LfChannelCorrelation present (VarDCT)");
    let lf_group = LfGroup::read(&mut shared, &fh, &lf_global, &metadata, 0).expect("lf group");
    let lf_coeff = lf_group.lf_coeff.expect("lf coefficients present (VarDCT)");

    // Modular sub-bitstream channel order is (Y, X, B); dequant_lf wants
    // [X, Y, B] (Listing F.1 applies m_x_dc to channel 0). Same reindex as
    // the integrated decode path.
    let lf_quant: [Vec<i32>; 3] = [
        lf_coeff.lf_quant[1].clone(),
        lf_coeff.lf_quant[0].clone(),
        lf_coeff.lf_quant[2].clone(),
    ];
    let widths = [
        lf_coeff.lf_quant_widths[1],
        lf_coeff.lf_quant_widths[0],
        lf_coeff.lf_quant_widths[2],
    ];
    let heights = [
        lf_coeff.lf_quant_heights[1],
        lf_coeff.lf_quant_heights[0],
        lf_coeff.lf_quant_heights[2],
    ];
    let mult = LfMultipliers::compute(&lf_global.lf_dequant, &quantizer);
    let out = dequant_lf(&lf_quant, widths, heights, lf_coeff.extra_precision, &mult);
    let w = widths[1] as usize;
    let h = heights[1] as usize;
    // Annex G LF factors: kX = base_correlation_x + (x_factor_lf - 127)
    // / colour_factor (and B likewise).
    let k_x = cfl.base_correlation_x + (cfl.x_factor_lf as f32 - 127.0) / cfl.colour_factor as f32;
    let k_b = cfl.base_correlation_b + (cfl.b_factor_lf as f32 - 127.0) / cfl.colour_factor as f32;
    (out.samples, (k_x, k_b), w, h)
}

/// Invert the reference PNG through the spec forward XYB transform and
/// box-average to the 32×32 LF grid. Returns `[X, Y, B]` planes.
fn reference_lf_xyb() -> [Vec<f64>; 3] {
    let oim = OpsinInverseMatrix::default();
    let tm = ToneMapping::default();
    // Forward opsin matrix = inverse of the (published) inverse matrix.
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
    let w = info.width as usize;
    let h = info.height as usize;
    assert_eq!((w, h), (256, 256), "reference frame extent");
    let mut planes = [vec![0f64; 1024], vec![0f64; 1024], vec![0f64; 1024]];
    for py in 0..h {
        for px in 0..w {
            let c = &buf[(py * w + px) * ch..(py * w + px) * ch + 3];
            let rl = srgb_to_linear(c[0] as f32 / 255.0) / itscale;
            let gl = srgb_to_linear(c[1] as f32 / 255.0) / itscale;
            let bl = srgb_to_linear(c[2] as f32 / 255.0) / itscale;
            let lm = fwd[0][0] * rl + fwd[0][1] * gl + fwd[0][2] * bl;
            let mm = fwd[1][0] * rl + fwd[1][1] * gl + fwd[1][2] * bl;
            let sm = fwd[2][0] * rl + fwd[2][1] * gl + fwd[2][2] * bl;
            // gamma = cbrt(mix - bias) + cbrt(bias) (inverse of the
            // `(gamma - cbrt(bias))^3 + bias` mix used in
            // inverse_xyb_to_rgb).
            let gl_ = (lm - oim.opsin_bias[0]).cbrt() + oim.opsin_bias[0].cbrt();
            let gm_ = (mm - oim.opsin_bias[1]).cbrt() + oim.opsin_bias[1].cbrt();
            let gs_ = (sm - oim.opsin_bias[2]).cbrt() + oim.opsin_bias[2].cbrt();
            let lf = (py / 8) * 32 + (px / 8);
            planes[0][lf] += ((gl_ - gm_) * 0.5) as f64 / 64.0;
            planes[1][lf] += ((gl_ + gm_) * 0.5) as f64 / 64.0;
            planes[2][lf] += gs_ as f64 / 64.0;
        }
    }
    planes
}

/// With the corrected Listing C.1 reading, our dequantised LF Y-mean
/// matches the reference's forward-XYB Y-mean (ratio 1.0) — the
/// round-372 measured 4.0× divergence is fixed.
#[test]
fn vardct_d1_lf_y_magnitude_matches_reference() {
    let (lf, _cfl, _w, _h) = decode_dequant_lf();
    let y = &lf[1];
    let our_y_mean = y.iter().map(|&v| v as f64).sum::<f64>() / y.len() as f64;

    let refp = reference_lf_xyb();
    let ref_y = refp[1].iter().sum::<f64>() / refp[1].len() as f64;
    assert!(
        ref_y > 0.1,
        "reference Y-mean {ref_y:.4} should be a normal mid-tone luma (sanity)"
    );

    let ratio = our_y_mean / ref_y;
    assert!(
        (ratio - 1.0).abs() < 0.02,
        "vardct-d1 LF Y magnitude ratio (ours/ref) = {ratio:.4}, expected ~1.0 under the \
         corrected Listing C.1 reading (our Y-mean {our_y_mean:.4}, ref Y-mean {ref_y:.4})"
    );
}

/// Per-channel least-squares scale between our dequantised LF and the
/// reference LF (CfL removed per Annex G's LF factors) is ≈ 1.0 on all
/// three channels. This is the three-channel consistency measurement
/// that uniquely pins the `65536 / (m × global_scale × quant_lf)`
/// reading: under the literal FDIS formula these scales measured
/// X ≈ 1/256, Y = 1/4, B ≈ 1.
#[test]
fn vardct_d1_lf_per_channel_scale_is_unity() {
    let (lf, (k_x, k_b), _w, _h) = decode_dequant_lf();
    let refp = reference_lf_xyb();

    // (channel index, name, tolerance). X is noise-limited: the
    // reference X residual after CfL removal is tiny (|X| ~ 0.01)
    // against 8-bit sRGB quantisation of the reference PNG, so its
    // scale estimate carries more measurement noise than Y / B.
    for (c, name, tol) in [(0usize, "X", 0.10), (1, "Y", 0.02), (2, "B", 0.05)] {
        let ours = &lf[c];
        let mut num = 0f64;
        let mut den = 0f64;
        for i in 0..1024 {
            let ref_d = match c {
                0 => refp[0][i] - k_x as f64 * refp[1][i],
                1 => refp[1][i],
                _ => refp[2][i] - k_b as f64 * refp[1][i],
            };
            let o = ours[i] as f64;
            num += ref_d * o;
            den += o * o;
        }
        let scale = num / den;
        assert!(
            (scale - 1.0).abs() < tol,
            "channel {name}: LSQ scale (ref/ours) = {scale:.4}, expected ~1.0 ± {tol} under \
             the corrected Listing C.1 reading"
        );
    }
}

/// The Y plane is shape-correct: a smooth low-frequency field (small
/// local gradients), not entropy garbage. Unchanged from round 372
/// except the /4 rescale is gone — the decoded magnitudes are now
/// reference-scale.
#[test]
fn vardct_d1_lf_y_is_smooth() {
    let (lf, _cfl, w, h) = decode_dequant_lf();
    let y = &lf[1];
    assert_eq!(y.len(), w * h, "Y plane is w×h");
    assert!(
        w >= 4 && h >= 4,
        "LF grid large enough to measure smoothness"
    );

    // Mean absolute horizontal first-difference of the Y plane, as a
    // fraction of the plane mean. A smooth luma DC field has tiny
    // relative neighbour-to-neighbour deltas; entropy garbage would
    // have large ones.
    let vals: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    assert!(mean > 0.0, "Y mean positive");

    let mut sum_abs_dx = 0f64;
    let mut count = 0u64;
    for row in 0..h {
        for col in 1..w {
            sum_abs_dx += (vals[row * w + col] - vals[row * w + col - 1]).abs();
            count += 1;
        }
    }
    let mad_dx = sum_abs_dx / count as f64;
    let rel = mad_dx / mean;
    assert!(
        rel < 0.10,
        "Y horizontal first-difference is {rel:.3} of the mean — too large for a smooth \
         luma DC field; a structural mis-decode would produce this."
    );
}
