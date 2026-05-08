//! Round-12 integration tests: LF dequantisation (FDIS Listing F.1) +
//! adaptive LF smoothing (FDIS F.2) + HfMetadata (FDIS C.5.4).
//!
//! Round 12 extends round 11 by:
//!
//! * Implementing Listing F.1 LF dequantisation
//!   `dX = mXDC × qX / (1 << extra_precision)` (and Y, B) where
//!   `mXDC = m_x_lf_unscaled / (global_scale × quant_lf)` is the
//!   per-channel LF multiplier from C.4.3.
//! * Implementing the F.2 adaptive LF smoothing pass over interior
//!   samples of a non-subsampled LF channel.
//! * Wiring `HfMetadata` (G.2.4 / FDIS C.5.4) — the 4-channel modular
//!   sub-bitstream (XFromY, BFromY, BlockInfo, Sharpness).
//!
//! The pixel-decode path for VarDCT is still gated by the round-13+
//! IDCT / Chroma-from-Luma / restoration filters; this round only
//! advances the per-LfGroup decode further down Table G.3.

use oxideav_jpegxl::decode_one_frame;
use oxideav_jpegxl::frame_header::flags;
use oxideav_jpegxl::lf_dequant::{
    apply_adaptive_lf_smoothing, dequant_lf, should_apply_adaptive_lf_smoothing, LfDequantOutput,
    LfMultipliers,
};
use oxideav_jpegxl::lf_global::{LfChannelDequantization, Quantizer};

const PIXEL_1X1_JXL: &[u8] = include_bytes!("fixtures/pixel_1x1.jxl");
const GRAY_64X64_JXL: &[u8] = include_bytes!("fixtures/gray_64x64_lossless.jxl");
const GRADIENT_JXL: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
const PALETTE_JXL: &[u8] = include_bytes!("fixtures/palette_32x32.jxl");
const GREY_8X8_JXL: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

/// Round-12 sentinel: adding LF dequant + adaptive smoothing + HfMetadata
/// must not regress the five small modular fixtures. They go through the
/// Modular decode path which doesn't touch any of the new VarDCT-only
/// code, but a small refactor of `LfGroup::read` to plumb HfMetadata
/// changed the function signature so the regression sweep is non-trivial.
#[test]
fn five_small_lossless_fixtures_still_decode_round_12() {
    for (name, bytes) in [
        ("pixel_1x1", PIXEL_1X1_JXL),
        ("gray_64x64", GRAY_64X64_JXL),
        ("gradient_64x64", GRADIENT_JXL),
        ("palette_32x32", PALETTE_JXL),
        ("grey_8x8", GREY_8X8_JXL),
    ] {
        let vf = decode_one_frame(bytes, None);
        assert!(
            vf.is_ok(),
            "round-12 regression: {name} should still decode (round-10 baseline); got {:?}",
            vf.err()
        );
    }
}

/// Listing F.1 closed-form check: with default LfChannelDequantization
/// (mX=4096, mY=512, mB=256), default Quantizer (global_scale=1,
/// quant_lf=16), and extra_precision=0, the per-sample dequant is
/// equivalent to multiplying qX by 256, qY by 32, qB by 16.
#[test]
fn lf_dequant_default_quantizer_listing_f1() {
    let lfd = LfChannelDequantization::default();
    let q = Quantizer {
        global_scale: 1,
        quant_lf: 16,
    };
    let m = LfMultipliers::compute(&lfd, &q);
    // Single 1×1 sample per channel: qX=2, qY=3, qB=5.
    let lf_quant = [vec![2i32], vec![3i32], vec![5i32]];
    let widths = [1, 1, 1];
    let heights = [1, 1, 1];
    let out = dequant_lf(&lf_quant, widths, heights, 0, &m);
    // dX = 256 * 2 = 512, dY = 32 * 3 = 96, dB = 16 * 5 = 80.
    assert_eq!(out.samples[0][0], 512.0);
    assert_eq!(out.samples[1][0], 96.0);
    assert_eq!(out.samples[2][0], 80.0);
}

/// Listing F.1 with `extra_precision != 0` shifts the result by a
/// reciprocal power of two: `dX = mXDC × qX / (1 << extra_precision)`.
#[test]
fn lf_dequant_extra_precision_shifts() {
    let lfd = LfChannelDequantization::default();
    let q = Quantizer {
        global_scale: 1,
        quant_lf: 16,
    };
    let m = LfMultipliers::compute(&lfd, &q);
    let lf_quant = [vec![16i32], vec![16i32], vec![16i32]];
    let widths = [1, 1, 1];
    let heights = [1, 1, 1];
    // extra_precision=2 → divide by 4.
    let out = dequant_lf(&lf_quant, widths, heights, 2, &m);
    assert_eq!(out.samples[0][0], 256.0 * 16.0 / 4.0); // 1024
    assert_eq!(out.samples[1][0], 32.0 * 16.0 / 4.0); // 128
    assert_eq!(out.samples[2][0], 16.0 * 16.0 / 4.0); // 64
}

/// FDIS F.2 smoothing must NOT run when `kSkipAdaptiveLFSmoothing` is
/// set, even with otherwise-good inputs.
#[test]
fn adaptive_smoothing_gated_by_skip_flag() {
    use oxideav_jpegxl::frame_header::{Encoding, FrameDecodeParams, FrameHeader};

    // Build a minimal FrameHeader (all_default=1).
    let bytes = [0b1u8];
    let mut br = oxideav_jpegxl::bitreader::BitReader::new(&bytes);
    let params = FrameDecodeParams {
        xyb_encoded: false,
        num_extra_channels: 0,
        have_animation: false,
        have_animation_timecodes: false,
        image_width: 8,
        image_height: 8,
    };
    let mut fh = FrameHeader::read(&mut br, &params).unwrap();
    fh.encoding = Encoding::VarDct;
    fh.flags = flags::SKIP_ADAPTIVE_LF_SMOOTHING;
    fh.jpeg_upsampling = [0, 0, 0];
    assert!(!should_apply_adaptive_lf_smoothing(&fh));

    fh.flags = 0;
    assert!(should_apply_adaptive_lf_smoothing(&fh));

    // Subsampled chroma also disables smoothing.
    fh.jpeg_upsampling = [0, 1, 1];
    assert!(!should_apply_adaptive_lf_smoothing(&fh));
}

/// FDIS F.2 closed-form on a flat 5×5 LF field — every interior sample
/// has a weighted average equal to itself, so smoothing is a no-op
/// across all interior samples.
#[test]
fn adaptive_smoothing_flat_field_is_noop() {
    let m = LfMultipliers {
        m_x_dc: 1.0,
        m_y_dc: 1.0,
        m_b_dc: 1.0,
    };
    let mut out = LfDequantOutput {
        samples: [vec![7.0; 25], vec![11.0; 25], vec![13.0; 25]],
        widths: [5, 5, 5],
        heights: [5, 5, 5],
    };
    let copy = out.samples.clone();
    apply_adaptive_lf_smoothing(&mut out, &m);
    for (c, ch) in out.samples.iter().enumerate() {
        for (i, &v) in ch.iter().enumerate() {
            assert!(
                (v - copy[c][i]).abs() < 1e-5,
                "channel {c} index {i}: got {v} expected {}",
                copy[c][i]
            );
        }
    }
}
