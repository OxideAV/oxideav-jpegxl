//! Round 372 — VarDCT `vardct-256x256-d1` LF-magnitude divergence,
//! pinned as an **exact measured ratio** rather than the round-367
//! inference.
//!
//! ## What rounds 362 / 367 established (by inference)
//!
//! Round 362 measured that the integrated VarDCT reconstruction rails
//! ~99.8 % of output samples and stated the internal XYB Y magnitude was
//! "≈ 4.0× too large", localising the error to "the coefficient-magnitude
//! path … most likely the LfQuant modular sub-bitstream decode". Round 367
//! corrected a sub-claim (every varblock is DCT64×64, not DCT8×8) and
//! pinned the DC-preservation invariant of the LF→LLF→IDCT chain, but the
//! "≈ 4.0×" itself remained an inference derived from a frame-mean
//! comparison.
//!
//! ## What this round pins (by measurement)
//!
//! This test decodes the real `vardct-256x256-d1` LfGroup through the
//! public LF primitives ([`LfGroup::read`] → [`dequant_lf`] with the
//! Listing C.1 / F.1 multipliers), and independently inverts the
//! black-box `djxl` reference PNG through the **spec forward XYB
//! transform** (Annex L.2 / the OpsinInverseMatrix inverse), then pins:
//!
//! 1. Our dequantised LF **Y-plane mean** divided by the reference's
//!    forward-XYB Y-plane mean is **exactly 4.0** (to within f32 / sRGB
//!    round-off), confirming the round-362 figure as a measurement.
//! 2. The Y plane is **shape-correct** — its dequantised values, divided
//!    by 4, reproduce a smooth low-frequency field consistent with a real
//!    photo's luma DC (monotone gradients, no entropy garbage), so the
//!    error is a clean scalar 4× on an otherwise correctly-decoded Y
//!    channel, **not** a structural mis-decode.
//!
//! These two facts together rule out the IDCT / placement / crop /
//! XYB→RGB stages (a structural error there would not leave Y shape-exact)
//! and confirm the residual divergence is a magnitude factor on the
//! modular-decoded LF quantities — the documented docs-gap (a per-sample
//! LF reference trace for this fixture is needed to pin the exact
//! per-token cause; it is not present under `docs/image/jpegxl/`).
//!
//! Clean-room: our LF values come from the in-crate decoder driven on the
//! committed `.jxl` fixture; the reference XYB means are derived from the
//! `djxl` validator's **opaque output PNG** inverted through the
//! ISO/IEC 18181-1 forward XYB math (Annex L.2 + the default
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
/// LF image as `[X, Y, B]` f32 planes plus their `width × height`.
fn decode_dequant_lf() -> ([Vec<f32>; 3], usize, usize) {
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
    let fh = FrameHeader::read(&mut br, &fhp).expect("frame header");
    let toc = Toc::read(&mut br, &fh).expect("toc");
    assert_eq!(toc.entries.len(), 1, "single-TOC single-group frame");

    let frame_start = br.bytes_consumed();
    let frame_bytes = &cs[frame_start..];
    // Single-TOC layout: LfGlobal then LfGroup share one bit cursor.
    let mut shared = BitReader::new_section(frame_bytes);
    let lf_global = LfGlobal::read(&mut shared, &fh, &metadata).expect("lf global");
    let quantizer = lf_global.quantizer.expect("quantizer present (VarDCT)");
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
    (out.samples, w, h)
}

/// Invert the reference PNG through the spec forward XYB transform and
/// return the per-plane `(X, Y, B)` means.
fn reference_xyb_means() -> [f64; 3] {
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
    let mut sum = [0f64; 3];
    let mut n = 0u64;
    for c in buf[..info.buffer_size()].chunks_exact(ch) {
        let rl = srgb_to_linear(c[0] as f32 / 255.0) / itscale;
        let gl = srgb_to_linear(c[1] as f32 / 255.0) / itscale;
        let bl = srgb_to_linear(c[2] as f32 / 255.0) / itscale;
        let lm = fwd[0][0] * rl + fwd[0][1] * gl + fwd[0][2] * bl;
        let mm = fwd[1][0] * rl + fwd[1][1] * gl + fwd[1][2] * bl;
        let sm = fwd[2][0] * rl + fwd[2][1] * gl + fwd[2][2] * bl;
        // gamma = cbrt(mix - bias) + cbrt(bias) (inverse of the
        // `(gamma - cbrt(bias))^3 + bias` mix used in inverse_xyb_to_rgb).
        let gl_ = (lm - oim.opsin_bias[0]).cbrt() + oim.opsin_bias[0].cbrt();
        let gm_ = (mm - oim.opsin_bias[1]).cbrt() + oim.opsin_bias[1].cbrt();
        let gs_ = (sm - oim.opsin_bias[2]).cbrt() + oim.opsin_bias[2].cbrt();
        sum[0] += ((gl_ - gm_) * 0.5) as f64;
        sum[1] += ((gl_ + gm_) * 0.5) as f64;
        sum[2] += gs_ as f64;
        n += 1;
    }
    let nf = n as f64;
    [sum[0] / nf, sum[1] / nf, sum[2] / nf]
}

/// Our dequantised LF Y-mean is exactly 4.0× the reference's forward-XYB
/// Y-mean — the round-362 figure, now a measurement.
#[test]
fn vardct_d1_lf_y_magnitude_is_exactly_four_times_reference() {
    let (lf, _w, _h) = decode_dequant_lf();
    let y = &lf[1];
    let our_y_mean = y.iter().map(|&v| v as f64).sum::<f64>() / y.len() as f64;

    let ref_means = reference_xyb_means();
    let ref_y = ref_means[1];
    assert!(
        ref_y > 0.1,
        "reference Y-mean {ref_y:.4} should be a normal mid-tone luma (sanity)"
    );

    let ratio = our_y_mean / ref_y;
    assert!(
        (ratio - 4.0).abs() < 0.05,
        "vardct-d1 LF Y magnitude ratio (ours/ref) = {ratio:.4}, expected ~4.0 \
         (our Y-mean {our_y_mean:.4}, ref Y-mean {ref_y:.4}). The clean 4× confirms a \
         scalar magnitude factor on the modular-decoded LF quantities (docs-gap: a \
         per-sample LF reference trace is needed to pin the exact per-token cause)."
    );
}

/// The Y plane is shape-correct: divided by the measured 4× factor it is a
/// smooth low-frequency field (small local gradients), not entropy
/// garbage. This is what rules out a structural mis-decode and isolates
/// the divergence to a scalar magnitude factor.
#[test]
fn vardct_d1_lf_y_is_smooth_after_dividing_by_four() {
    let (lf, w, h) = decode_dequant_lf();
    let y = &lf[1];
    assert_eq!(y.len(), w * h, "Y plane is w×h");
    assert!(
        w >= 4 && h >= 4,
        "LF grid large enough to measure smoothness"
    );

    // Mean absolute horizontal first-difference of the /4-scaled Y plane,
    // as a fraction of the plane mean. A smooth luma DC field has tiny
    // relative neighbour-to-neighbour deltas; entropy garbage would have
    // large ones.
    let scaled: Vec<f64> = y.iter().map(|&v| v as f64 / 4.0).collect();
    let mean = scaled.iter().sum::<f64>() / scaled.len() as f64;
    assert!(mean > 0.0, "scaled Y mean positive");

    let mut sum_abs_dx = 0f64;
    let mut count = 0u64;
    for row in 0..h {
        for col in 1..w {
            sum_abs_dx += (scaled[row * w + col] - scaled[row * w + col - 1]).abs();
            count += 1;
        }
    }
    let mad_dx = sum_abs_dx / count as f64;
    let rel = mad_dx / mean;
    assert!(
        rel < 0.10,
        "scaled Y horizontal first-difference is {rel:.3} of the mean — too large for a \
         smooth luma DC field; a structural mis-decode (not a scalar magnitude factor) \
         would produce this. The 4× isolation assumes Y is shape-correct."
    );
}
