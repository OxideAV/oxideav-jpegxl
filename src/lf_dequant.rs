//! LF dequantization + adaptive LF smoothing — ISO/IEC 18181-1:2024
//! Annex F.1 (Listing F.1) and Annex F.2 (the prose immediately
//! following Listing F.1).
//!
//! ## Round 12 scope
//!
//! Round 12 lands the two pieces of Annex F that operate purely on a
//! per-LF-group scale of decoded LF coefficients:
//!
//! 1. **LF dequantization** — converts the per-channel quantized
//!    coefficients `qX, qY, qB` (decoded by [`crate::lf_group`]) into
//!    real-valued LF samples `dX, dY, dB` per Listing F.1:
//!    ```text
//!    mXDC = m_x_lf_unscaled / (global_scale × quant_lf);   // C.4.3
//!    mYDC = m_y_lf_unscaled / (global_scale × quant_lf);
//!    mBDC = m_b_lf_unscaled / (global_scale × quant_lf);
//!    dX = mXDC × qX / (1 << extra_precision);
//!    dY = mYDC × qY / (1 << extra_precision);
//!    dB = mBDC × qB / (1 << extra_precision);
//!    ```
//! 2. **Adaptive LF smoothing** — when
//!    `flags & kSkipAdaptiveLFSmoothing == 0` AND no LF channel is
//!    subsampled (`jpeg_upsampling[c] == 0` for `c ∈ 0..3`), the spec
//!    applies a weighted-average smoothing pass to interior samples
//!    (skipping the first and last row + column).
//!
//! Round-12 deliberately **does not** apply Chroma-from-Luma (Annex G),
//! the Listing I.5 LLF-from-downsampled step, nor IDCT — those land
//! round 13+. This module is the bridge between LfCoefficients (Annex
//! G.2.2 / FDIS C.5.3) and the round-13 IDCT / CfL composition step.
//!
//! ## Algorithmic detail (FDIS 4601-4609, normative)
//!
//! For each LF sample `s` of the image not in the first or last row /
//! column (per channel — but the caller passes the same shape across
//! all three), the decoder:
//!
//! * Computes a weighted average `wa` of the 9 samples in the 3×3
//!   neighbourhood. Weights are:
//!   - center sample weight: 0.05226273532324128
//!   - horizontally / vertically adjacent (4 samples): 0.20345139757231578
//!   - diagonally adjacent (4 samples): 0.0334829185968739
//! * Computes the channel-shared
//!   `gap = max(0.5, abs(waX-sX)/mXDC, abs(waY-sY)/mYDC,
//!   abs(waB-sB)/mBDC)`.
//! * Replaces each per-channel sample with
//!   `(s - wa) × max(0, 3 - 4 × gap) + wa`.
//!
//! Edge samples (first / last row, first / last column) are left
//! unchanged. The smoothing is in-place over the dequantised LF
//! samples.
//!
//! ## Allocation bound
//!
//! No per-call allocation: the smoothing pass clones into a scratch
//! buffer once per channel (3 buffers per LF group, total) and writes
//! results back. Total memory is `3 × width × height × 4 bytes` for the
//! LfGroup's LF channels, which is bounded by `3 × group_dim^2 × 4`
//! bytes (for the default group_dim of 256 pixels in a frame's
//! coefficient grid that's 32x32 LF samples = ~12 KB).

use crate::frame_header::{flags, FrameHeader};
use crate::lf_global::{LfChannelDequantization, Quantizer};

/// Per-channel LF multipliers `mXDC, mYDC, mBDC` per FDIS C.4.3 +
/// Listing F.1, indexed `[X=0, Y=1, B=2]`.
#[derive(Debug, Clone, Copy)]
pub struct LfMultipliers {
    pub m_x_dc: f32,
    pub m_y_dc: f32,
    pub m_b_dc: f32,
}

impl LfMultipliers {
    /// Compute the three per-channel LF multipliers per Listing F.1's
    /// preamble. These are shared across every LfGroup of the frame
    /// (they depend only on LfGlobal contents).
    ///
    /// Spec: `mXDC = m_x_lf_unscaled / (global_scale × quant_lf)` and
    /// similarly for Y / B.
    pub fn compute(lf_dequant: &LfChannelDequantization, quantizer: &Quantizer) -> Self {
        let denom = (quantizer.global_scale as f32) * (quantizer.quant_lf as f32);
        // The spec asserts both global_scale and quant_lf are at least 1
        // (their U32 distributions either start at 1 or have an offset
        // of 1 — see C.4.3). But guard anyway in the floating-point
        // domain.
        let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
        Self {
            m_x_dc: lf_dequant.m_x_lf_unscaled * inv,
            m_y_dc: lf_dequant.m_y_lf_unscaled * inv,
            m_b_dc: lf_dequant.m_b_lf_unscaled * inv,
        }
    }
}

/// Per-LfGroup dequantised LF samples, three channels indexed [X, Y, B]
/// row-major.
#[derive(Debug, Clone)]
pub struct LfDequantOutput {
    /// Three channels, each `width[c] * height[c]` samples row-major.
    /// Channel order is `[X, Y, B]` matching FDIS Listing F.1.
    pub samples: [Vec<f32>; 3],
    /// Per-channel widths.
    pub widths: [u32; 3],
    /// Per-channel heights.
    pub heights: [u32; 3],
}

/// Apply Listing F.1 LF dequantisation to the per-channel quantized LF
/// coefficients `lf_quant` decoded by [`crate::lf_group::LfCoefficients`].
///
/// `lf_quant[c]` holds `widths[c] * heights[c]` samples row-major; the
/// caller has already validated channel count == 3 and the dimensions
/// match what came out of the modular sub-bitstream.
///
/// `extra_precision` is the `u(2)` field read at the start of the
/// LfCoefficients sub-bitstream (FDIS C.5.3).
pub fn dequant_lf(
    lf_quant: &[Vec<i32>; 3],
    widths: [u32; 3],
    heights: [u32; 3],
    extra_precision: u32,
    multipliers: &LfMultipliers,
) -> LfDequantOutput {
    // Listing F.1 — divide by (1 << extra_precision). For
    // extra_precision in [0, 3] that's a small left shift constant.
    let inv_extra = 1.0 / ((1u32 << extra_precision) as f32);
    let m = [
        multipliers.m_x_dc * inv_extra,
        multipliers.m_y_dc * inv_extra,
        multipliers.m_b_dc * inv_extra,
    ];
    let mut samples: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for c in 0..3 {
        let n = (widths[c] as usize) * (heights[c] as usize);
        let mut out = Vec::with_capacity(n);
        for &q in lf_quant[c].iter().take(n) {
            out.push(m[c] * (q as f32));
        }
        samples[c] = out;
    }
    LfDequantOutput {
        samples,
        widths,
        heights,
    }
}

/// FDIS Annex F.2 weights for the 3×3 weighted-average kernel used in
/// adaptive LF smoothing. Order: center, horizontal/vertical adjacent,
/// diagonal adjacent — verbatim from the spec text.
/// Spec value: `0.05226273532324128` (truncated to f32 precision).
pub const ADAPTIVE_LF_WEIGHT_CENTER: f32 = 0.052_262_735;
/// Spec value: `0.20345139757231578` (truncated to f32 precision).
pub const ADAPTIVE_LF_WEIGHT_HV: f32 = 0.203_451_4;
/// Spec value: `0.0334829185968739` (truncated to f32 precision).
pub const ADAPTIVE_LF_WEIGHT_DIAG: f32 = 0.033_482_92;

/// Should adaptive LF smoothing be applied for this LfGroup? Per the
/// FDIS prose: "...the decoder applies the following adaptive smoothing
/// algorithm, unless the kSkipAdaptiveLFSmoothing flag is set in
/// frame_header. If this adaptive smoothing procedure is applied, no
/// channel is subsampled."
///
/// In other words: smoothing happens IFF
/// `(flags & kSkipAdaptiveLFSmoothing) == 0` AND every channel has
/// `jpeg_upsampling[c] == 0` (zero-shift = no subsampling).
pub fn should_apply_adaptive_lf_smoothing(fh: &FrameHeader) -> bool {
    if (fh.flags & flags::SKIP_ADAPTIVE_LF_SMOOTHING) != 0 {
        return false;
    }
    fh.jpeg_upsampling.iter().all(|&u| u == 0)
}

/// Apply FDIS F.2 adaptive LF smoothing to the dequantised LF samples
/// in place.
///
/// Edge samples (first / last row, first / last column) are left
/// unchanged. Interior samples are replaced with
/// `(s - wa) × max(0, 3 - 4 × gap) + wa` per the FDIS prose.
///
/// Caller must verify [`should_apply_adaptive_lf_smoothing`] returned
/// `true` (the spec requires no channel be subsampled — i.e. all three
/// channels share the same dimensions; the routine asserts that).
pub fn apply_adaptive_lf_smoothing(out: &mut LfDequantOutput, multipliers: &LfMultipliers) {
    let w = out.widths[0] as usize;
    let h = out.heights[0] as usize;
    // F.2 requires no channel subsampled — all three must share dims.
    debug_assert!(
        out.widths[1] == out.widths[0]
            && out.widths[2] == out.widths[0]
            && out.heights[1] == out.heights[0]
            && out.heights[2] == out.heights[0],
        "adaptive LF smoothing called with subsampled channels — caller must \
         have verified `should_apply_adaptive_lf_smoothing`"
    );
    if w < 3 || h < 3 {
        // No interior samples — nothing to smooth.
        return;
    }

    // Snapshot the three channels so we can read 3×3 neighbourhoods
    // from the original samples while writing into the in-place buffer.
    let snap_x = out.samples[0].clone();
    let snap_y = out.samples[1].clone();
    let snap_b = out.samples[2].clone();

    // Reciprocals so the smoothing inner loop avoids division on the
    // hot path. Per spec the multipliers are positive (`m_x/y/b_lf`
    // are positive F16 values in the O(10^2)..O(10^3) range, and
    // `global_scale + quant_lf` are at least 1). Defensively guard
    // against zero.
    let inv_m = [
        if multipliers.m_x_dc != 0.0 {
            1.0 / multipliers.m_x_dc
        } else {
            0.0
        },
        if multipliers.m_y_dc != 0.0 {
            1.0 / multipliers.m_y_dc
        } else {
            0.0
        },
        if multipliers.m_b_dc != 0.0 {
            1.0 / multipliers.m_b_dc
        } else {
            0.0
        },
    ];

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            // 3×3 weighted average per channel.
            let wa_x = compute_weighted_average(&snap_x, w, x, y);
            let wa_y = compute_weighted_average(&snap_y, w, x, y);
            let wa_b = compute_weighted_average(&snap_b, w, x, y);

            let s_x = snap_x[y * w + x];
            let s_y = snap_y[y * w + x];
            let s_b = snap_b[y * w + x];

            // FDIS: gap = max(0.5, |waX-sX|/mXDC, |waY-sY|/mYDC,
            // |waB-sB|/mBDC).
            let gap = 0.5f32
                .max((wa_x - s_x).abs() * inv_m[0])
                .max((wa_y - s_y).abs() * inv_m[1])
                .max((wa_b - s_b).abs() * inv_m[2]);
            // FDIS: smoothed = (s - wa) × max(0, 3 - 4 × gap) + wa.
            let factor = (3.0 - 4.0 * gap).max(0.0);

            let out_x = (s_x - wa_x) * factor + wa_x;
            let out_y = (s_y - wa_y) * factor + wa_y;
            let out_b = (s_b - wa_b) * factor + wa_b;
            out.samples[0][y * w + x] = out_x;
            out.samples[1][y * w + x] = out_y;
            out.samples[2][y * w + x] = out_b;
        }
    }
}

/// 3×3 weighted average per F.2: center (5,5)/100..., HV (4 cells),
/// diag (4 cells). Caller has guaranteed `1 <= x <= w-2` and the row
/// equivalent for `y`.
#[inline]
fn compute_weighted_average(channel: &[f32], w: usize, x: usize, y: usize) -> f32 {
    let center = channel[y * w + x];
    let h_left = channel[y * w + (x - 1)];
    let h_right = channel[y * w + (x + 1)];
    let v_up = channel[(y - 1) * w + x];
    let v_down = channel[(y + 1) * w + x];
    let d_ul = channel[(y - 1) * w + (x - 1)];
    let d_ur = channel[(y - 1) * w + (x + 1)];
    let d_dl = channel[(y + 1) * w + (x - 1)];
    let d_dr = channel[(y + 1) * w + (x + 1)];
    ADAPTIVE_LF_WEIGHT_CENTER * center
        + ADAPTIVE_LF_WEIGHT_HV * (h_left + h_right + v_up + v_down)
        + ADAPTIVE_LF_WEIGHT_DIAG * (d_ul + d_ur + d_dl + d_dr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;
    use crate::bitreader::BitReader;
    use crate::frame_header::FrameDecodeParams;

    fn build_fh(w: u32, h: u32) -> FrameHeader {
        let params = FrameDecodeParams {
            xyb_encoded: false,
            num_extra_channels: 0,
            have_animation: false,
            have_animation_timecodes: false,
            image_width: w,
            image_height: h,
        };
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let mut fh = FrameHeader::read(&mut br, &params).unwrap();
        fh.width = w;
        fh.height = h;
        fh
    }

    #[test]
    fn multipliers_with_default_quantizer_compute_per_spec() {
        // Default lf_dequant: 4096 / 512 / 256.
        // Default quantizer: global_scale = 1, quant_lf = 16.
        // mXDC = 4096 / (1 * 16) = 256. Y: 32. B: 16.
        let lfd = LfChannelDequantization::default();
        let q = Quantizer {
            global_scale: 1,
            quant_lf: 16,
        };
        let m = LfMultipliers::compute(&lfd, &q);
        assert_eq!(m.m_x_dc, 256.0);
        assert_eq!(m.m_y_dc, 32.0);
        assert_eq!(m.m_b_dc, 16.0);
    }

    #[test]
    fn dequant_zero_lf_quant_yields_zero_samples() {
        // qX = qY = qB = 0 → dX = dY = dB = 0 regardless of multipliers.
        let m = LfMultipliers {
            m_x_dc: 1.0,
            m_y_dc: 2.0,
            m_b_dc: 4.0,
        };
        let lf_quant = [vec![0i32; 4], vec![0i32; 4], vec![0i32; 4]];
        let widths = [2, 2, 2];
        let heights = [2, 2, 2];
        let out = dequant_lf(&lf_quant, widths, heights, 0, &m);
        for c in 0..3 {
            for &v in &out.samples[c] {
                assert_eq!(v, 0.0);
            }
        }
    }

    #[test]
    fn dequant_listing_f1_extra_precision_divides() {
        // qX = 4, mXDC = 8, extra_precision = 2 →
        // dX = 8 * 4 / (1 << 2) = 32 / 4 = 8.
        let m = LfMultipliers {
            m_x_dc: 8.0,
            m_y_dc: 0.0,
            m_b_dc: 0.0,
        };
        let lf_quant = [vec![4i32], vec![0i32], vec![0i32]];
        let widths = [1, 1, 1];
        let heights = [1, 1, 1];
        let out = dequant_lf(&lf_quant, widths, heights, 2, &m);
        assert_eq!(out.samples[0][0], 8.0);
    }

    #[test]
    fn dequant_listing_f1_no_extra_precision() {
        let m = LfMultipliers {
            m_x_dc: 0.5,
            m_y_dc: 1.0,
            m_b_dc: 2.0,
        };
        // 3 channels, 1 sample each.
        let lf_quant = [vec![10], vec![20], vec![5]];
        let widths = [1, 1, 1];
        let heights = [1, 1, 1];
        let out = dequant_lf(&lf_quant, widths, heights, 0, &m);
        assert_eq!(out.samples[0][0], 5.0);
        assert_eq!(out.samples[1][0], 20.0);
        assert_eq!(out.samples[2][0], 10.0);
    }

    #[test]
    fn smoothing_skipped_when_flag_set() {
        let mut fh = build_fh(8, 8);
        fh.flags = flags::SKIP_ADAPTIVE_LF_SMOOTHING;
        fh.jpeg_upsampling = [0, 0, 0];
        assert!(!should_apply_adaptive_lf_smoothing(&fh));
    }

    #[test]
    fn smoothing_skipped_when_chroma_subsampled() {
        let mut fh = build_fh(8, 8);
        fh.flags = 0;
        fh.jpeg_upsampling = [0, 1, 1];
        assert!(!should_apply_adaptive_lf_smoothing(&fh));
    }

    #[test]
    fn smoothing_applied_when_flag_clear_and_no_subsampling() {
        let mut fh = build_fh(8, 8);
        fh.flags = 0;
        fh.jpeg_upsampling = [0, 0, 0];
        assert!(should_apply_adaptive_lf_smoothing(&fh));
    }

    #[test]
    fn smoothing_constant_field_preserves_values() {
        // A flat (constant) field has wa = s everywhere → smoothed
        // value = (s - wa) * factor + wa = wa = s. Smoothing is a
        // no-op on a flat field.
        let m = LfMultipliers {
            m_x_dc: 1.0,
            m_y_dc: 1.0,
            m_b_dc: 1.0,
        };
        let mut out = LfDequantOutput {
            samples: [vec![3.0; 9], vec![5.0; 9], vec![7.0; 9]],
            widths: [3, 3, 3],
            heights: [3, 3, 3],
        };
        apply_adaptive_lf_smoothing(&mut out, &m);
        for c_idx in 0..3 {
            let expected = match c_idx {
                0 => 3.0,
                1 => 5.0,
                _ => 7.0,
            };
            for &v in &out.samples[c_idx] {
                assert!(
                    (v - expected).abs() < 1e-5,
                    "channel {c_idx}: got {v} expected {expected}"
                );
            }
        }
    }

    #[test]
    fn smoothing_skips_edges_no_op_for_2x2() {
        // 2x2 has no interior — smoothing is a strict no-op.
        let m = LfMultipliers {
            m_x_dc: 1.0,
            m_y_dc: 1.0,
            m_b_dc: 1.0,
        };
        let before = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = LfDequantOutput {
            samples: [before.clone(), before.clone(), before.clone()],
            widths: [2, 2, 2],
            heights: [2, 2, 2],
        };
        apply_adaptive_lf_smoothing(&mut out, &m);
        for c in 0..3 {
            assert_eq!(out.samples[c], before);
        }
    }

    #[test]
    fn smoothing_modifies_only_interior_for_3x3() {
        // 3x3 has exactly one interior sample at (1, 1). Set up a
        // small spike: center = 100, surrounding samples = 0. The
        // weighted average of the 9 samples (center=100, 8 zeros) is:
        // wa = 100 * 0.05226... = 5.226...
        // mXDC = 1 → gap = max(0.5, |5.226 - 100|/1, ...) = 94.77
        // factor = max(0, 3 - 4 * 94.77) = 0 → smoothed = wa = 5.226.
        let m = LfMultipliers {
            m_x_dc: 1.0,
            m_y_dc: 1.0,
            m_b_dc: 1.0,
        };
        let mut grid = vec![0.0f32; 9];
        grid[4] = 100.0; // center
        let mut out = LfDequantOutput {
            samples: [grid.clone(), grid.clone(), grid.clone()],
            widths: [3, 3, 3],
            heights: [3, 3, 3],
        };
        apply_adaptive_lf_smoothing(&mut out, &m);
        // Edges unchanged.
        for &i in [0usize, 1, 2, 3, 5, 6, 7, 8].iter() {
            assert_eq!(out.samples[0][i], 0.0);
        }
        let expected = 100.0 * ADAPTIVE_LF_WEIGHT_CENTER;
        assert!(
            (out.samples[0][4] - expected).abs() < 1e-5,
            "center got {} expected {}",
            out.samples[0][4],
            expected
        );
    }

    #[test]
    fn smoothing_flat_with_one_off_passes_low_gap_branch() {
        // Verify the low-gap branch (factor > 0) by computing wa and
        // factor by hand for a deliberately small perturbation. Use
        // mXDC = 100 so gap is dominated by 0.5 → factor = 1.
        // 3x3 channel: surround = 10, center = 11 → wa = 10 * (4 *
        // weight_hv + 4 * weight_diag) + 11 * weight_center =
        // 10 * (4 * 0.20345... + 4 * 0.03348...) + 11 * 0.05226...
        //   = 10 * 0.94774... + 0.57489...
        //   = 9.4774 + 0.5749 = 10.0523... ≈ 10.0523.
        // gap = max(0.5, |10.0523 - 11|/100, ...) = 0.5
        // factor = 3 - 4 * 0.5 = 1.
        // smoothed = (11 - 10.0523) * 1 + 10.0523 = 11.
        // Confirms: when gap saturates to 0.5, factor=1 and the
        // smoothing pass is a no-op on the channel sample.
        let m = LfMultipliers {
            m_x_dc: 100.0,
            m_y_dc: 100.0,
            m_b_dc: 100.0,
        };
        let mut grid = vec![10.0f32; 9];
        grid[4] = 11.0;
        let mut out = LfDequantOutput {
            samples: [grid.clone(), grid.clone(), grid.clone()],
            widths: [3, 3, 3],
            heights: [3, 3, 3],
        };
        apply_adaptive_lf_smoothing(&mut out, &m);
        // factor = 1 on every channel → output[4] = original = 11.
        for c in 0..3 {
            assert!(
                (out.samples[c][4] - 11.0).abs() < 1e-5,
                "channel {c}: got {}",
                out.samples[c][4]
            );
        }
    }
}
