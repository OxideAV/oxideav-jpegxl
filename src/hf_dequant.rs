//! HF coefficient dequantisation — ISO/IEC FDIS 18181-1:2021 Annex
//! F.3 (page 72 of the FDIS PDF) and ISO/IEC 18181-1:2024 Annex F.3.
//!
//! ## Scope (round 95)
//!
//! Round 95 lands the **per-sample HF dequantisation formula** that
//! converts a single quantised HF coefficient `quant` (the integer
//! ANS output of the §C.8.3 per-block coefficient decode) into its
//! final floating-point `d` value.
//!
//! This is the glue that the round-90 follow-up list named: it
//! consumes
//!
//! 1. The per-channel **dequantisation matrix** from
//!    [`crate::dct_quant_weights`] (the round-89 §I.2.4 / Table I.6
//!    materialisation).
//! 2. The [`crate::metadata_fdis::OpsinInverseMatrix`] `quant_bias`
//!    and `quant_bias_numerator` (round-2 FDIS A.8 parser).
//! 3. The per-frame `x_qm_scale` / `b_qm_scale` from
//!    [`crate::frame_header::FrameHeader`] (round-2 C.2 parser).
//! 4. The per-block `HfMul` value (the round-19 LfGroup HF-metadata
//!    parser exposes this; see [`crate::lf_group`]).
//!
//! and produces the final dequantised `d` (a single `f32`).
//!
//! ## Spec listing (FDIS page 72 — Annex F.3, normative)
//!
//! > Every quantized HF coefficient `quant` is first bias-adjusted as
//! > specified by Listing F.2 depending on its `channel` (0 for X, 1
//! > for Y or 2 for B).
//!
//! ```text
//! Listing F.2 — HF dequantization
//! oim = metadata.opsin_inverse_matrix;
//! if (abs(quant) <= 1) quant *= oim.quant_bias[[channel]];
//! else quant -= oim.quant_bias_numerator / quant;
//! ```
//!
//! > The resulting `quant` is then multiplied by a per-block
//! > multiplier, the value of `HfMul` at the coordinates of the 8 × 8
//! > rectangle containing the current sample, and, for the X and B
//! > channels, by `0.8^(frame_header.x_qm_scale - 2)` and
//! > `0.8^(frame_header.b_qm_scale - 2)`, respectively.
//! >
//! > The final dequantized value is obtained by multiplying the
//! > result by a multiplier defined by the channel, the transform
//! > type and the coefficient index inside the varblock, as specified
//! > in C.6.2.
//!
//! ## Implementation notes
//!
//! * **Operand widths.** The spec listing uses an implementation-
//!   defined `quant` type. After bias adjustment it is the product of
//!   an integer coefficient and a small floating-point bias (or
//!   integer minus a small fraction), so it is naturally a float. We
//!   carry it as `f32` end-to-end, matching the storage type of
//!   `OpsinInverseMatrix::quant_bias` (per the FDIS F16 wire format).
//! * **`abs(quant) <= 1` branch.** The pre-bias `quant` is an
//!   integer (the ANS output). The spec writes `abs(quant) <= 1`,
//!   which covers `{-1, 0, 1}`. We accept any input that satisfies
//!   the same predicate even if the caller has already converted it
//!   to `f32` (which is safe — the comparison is exact for integer
//!   values in `f32`'s mantissa range, and the round-95 caller surface
//!   accepts an `i32` to make this unambiguous).
//! * **`0.8^(scale - 2)` formula.** `x_qm_scale` and `b_qm_scale` are
//!   3-bit fields per FrameHeader (defaults 3 and 2 respectively).
//!   `0.8^(scale - 2)` is therefore `0.8^k` for `k ∈ {-2, -1, 0, 1,
//!   2, 3, 4, 5}`. Computing this once per frame and reusing is the
//!   efficient form; we expose a [`QmScaleFactors`] type for the
//!   per-frame precompute.
//! * **Y channel.** Listing F.2's listing applies to all three
//!   channels; only the post-listing `0.8^(qm_scale - 2)` factor is
//!   restricted to X and B. The Y channel skips that step entirely
//!   (effective multiplier 1.0).
//! * **Dequant-matrix sourcing.** Round 95 takes the dequant-matrix
//!   entry as a single `f32` argument so the function is decoupled
//!   from how the caller indexed into the
//!   [`crate::dct_quant_weights::DequantMatrixSet`]. The caller is
//!   responsible for indexing: given the varblock's `(transform_type,
//!   coeff_index)` it picks the right slot (via
//!   [`crate::dct_quant_weights::slot_for_transform`]), the right
//!   `(x, y)` cell of the slot's dequant matrix, and the right
//!   channel.
//!
//! ## What this module does NOT do
//!
//! * It does not decode the §C.8.3 per-block ANS coefficient stream —
//!   that is the next round's structural work (the shared 8-cluster
//!   ANS stream + §C.7.2 histograms scheduled for round 91+).
//! * It does not apply Chroma-from-Luma (Annex G) — that runs **after**
//!   F.3, on the dequantised dX / dY / dB values produced here.
//! * It does not apply the IDCT — same; that runs after CfL.

use crate::frame_header::FrameHeader;
use crate::metadata_fdis::OpsinInverseMatrix;

/// Bias-adjust a single quantised HF coefficient per FDIS Listing F.2
/// (the first stage of F.3).
///
/// `quant` is the raw integer output of the §C.8.3 per-block coefficient
/// decode for the current sample. `channel` is `0 = X`, `1 = Y`, `2 = B`.
/// `oim` carries the parsed-or-default OpsinInverseMatrix.
///
/// Returns the bias-adjusted `quant` as `f32`.
///
/// Panics: never. Out-of-range `channel` (>=3) wraps via array index
/// → caller must satisfy `channel < 3`. We assert in debug builds.
#[inline]
pub fn bias_adjust(quant: i32, channel: usize, oim: &OpsinInverseMatrix) -> f32 {
    debug_assert!(channel < 3, "JXL F.3: channel {channel} must be < 3");
    if quant.abs() <= 1 {
        // `if (abs(quant) <= 1) quant *= oim.quant_bias[[channel]];`
        (quant as f32) * oim.quant_bias[channel]
    } else {
        // `else quant -= oim.quant_bias_numerator / quant;`
        // The spec writes the subtraction in the integer-quant domain;
        // we lift to f32 immediately (the result is a float regardless).
        let q = quant as f32;
        q - oim.quant_bias_numerator / q
    }
}

/// Per-frame `0.8^(x_qm_scale - 2)` and `0.8^(b_qm_scale - 2)`
/// factors. Computed once per frame from
/// [`FrameHeader::x_qm_scale`] / [`FrameHeader::b_qm_scale`].
///
/// The Y channel's effective factor is implicitly 1.0 per FDIS F.3
/// "for the X and B channels" wording.
#[derive(Debug, Clone, Copy)]
pub struct QmScaleFactors {
    pub x_factor: f32,
    pub b_factor: f32,
}

impl QmScaleFactors {
    /// Compute the two `0.8^(scale - 2)` factors for the given frame.
    ///
    /// The `scale - 2` exponent can be negative (the FDIS field is a
    /// `u(3)` so the absolute exponent is in `-2..=5`); we use
    /// `f32::powi(i32)` which handles the full integer range exactly.
    pub fn for_frame(frame_header: &FrameHeader) -> Self {
        // `0.8^(scale - 2)` per FDIS F.3.
        let x_exp = frame_header.x_qm_scale as i32 - 2;
        let b_exp = frame_header.b_qm_scale as i32 - 2;
        Self {
            x_factor: 0.8_f32.powi(x_exp),
            b_factor: 0.8_f32.powi(b_exp),
        }
    }

    /// Per-channel factor: X uses `x_factor`, B uses `b_factor`, Y is
    /// 1.0 (the spec's "for the X and B channels" exclusion).
    #[inline]
    pub fn for_channel(&self, channel: usize) -> f32 {
        debug_assert!(channel < 3, "JXL F.3: channel {channel} must be < 3");
        match channel {
            0 => self.x_factor,
            1 => 1.0,
            _ => self.b_factor,
        }
    }
}

/// Full per-sample F.3 HF dequantisation.
///
/// Pipeline order (FDIS p. 72, line-by-line):
///
/// 1. `bias_adjust` (Listing F.2 on the raw integer `quant`).
/// 2. Multiply by `hf_mul` (the per-block `HfMul` value).
/// 3. For X and B channels only, multiply by
///    `0.8^(frame_header.x_qm_scale - 2)` or
///    `0.8^(frame_header.b_qm_scale - 2)`.
/// 4. Multiply by the dequant-matrix entry (the per-channel,
///    per-(transform_type, coeff_index) `1 / weights[i]` value from
///    [`crate::dct_quant_weights::DequantMatrixSet`]).
///
/// `dequant_matrix_entry` is the `f32`-cast of the corresponding
/// `DequantMatrixSet::matrices[slot][channel][y * x_dim + x]` value.
/// `hf_mul` is the integer multiplier from the per-block HfMul
/// metadata (the round-19 LfGroup parser exposes it as an `i32` per
/// FDIS §I.2.7).
///
/// Returns the final dequantised HF coefficient (the FDIS `dX`, `dY`,
/// or `dB` for the current sample).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dequant_hf_coefficient(
    quant: i32,
    channel: usize,
    hf_mul: i32,
    dequant_matrix_entry: f32,
    oim: &OpsinInverseMatrix,
    qm: &QmScaleFactors,
) -> f32 {
    let mut q = bias_adjust(quant, channel, oim);
    // Per-block multiplier (HfMul) — the integer-valued field from
    // I.2.7 / C.5.4. The product is taken in f32 to match the
    // spec's mixed-type expression.
    q *= hf_mul as f32;
    // X and B channels carry the `0.8^(qm_scale - 2)` factor; Y is
    // exempt.
    q *= qm.for_channel(channel);
    // Final per-coefficient dequant matrix multiplier (C.6.2).
    q * dequant_matrix_entry
}

/// Convenience helper: bias-adjust *and* multiply by the per-block
/// HfMul + per-channel qm-scale factor, but **do not** apply the
/// dequant-matrix entry. Useful for callers that want to apply the
/// matrix multiplication in a separate vectorised pass over the
/// 64+ coefficients of a varblock.
#[inline]
pub fn dequant_hf_pre_matrix(
    quant: i32,
    channel: usize,
    hf_mul: i32,
    oim: &OpsinInverseMatrix,
    qm: &QmScaleFactors,
) -> f32 {
    let q = bias_adjust(quant, channel, oim);
    q * (hf_mul as f32) * qm.for_channel(channel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_header::{FrameDecodeParams, FrameHeader};
    use crate::metadata_fdis::OpsinInverseMatrix;

    fn default_frame_decode_params() -> FrameDecodeParams {
        FrameDecodeParams {
            xyb_encoded: true,
            num_extra_channels: 0,
            have_animation: false,
            have_animation_timecodes: false,
            image_width: 64,
            image_height: 64,
        }
    }

    /// Build a minimal FrameHeader by overriding `x_qm_scale` and
    /// `b_qm_scale` on a default-constructed value. We thread through
    /// the existing `FrameHeader::default_with(&params)` constructor
    /// so the structure stays in sync with the parser.
    fn default_frame_header() -> FrameHeader {
        FrameHeader::default_with(&default_frame_decode_params())
    }

    fn frame_header_with_qm_scales(x: u32, b: u32) -> FrameHeader {
        let mut fh = default_frame_header();
        fh.x_qm_scale = x;
        fh.b_qm_scale = b;
        fh
    }

    #[test]
    fn bias_adjust_abs_le_one_uses_quant_bias() {
        let oim = OpsinInverseMatrix::default();
        // quant = 0 → 0 × quant_bias = 0.
        assert_eq!(bias_adjust(0, 0, &oim), 0.0);
        // quant = 1 → quant_bias[c].
        assert_eq!(bias_adjust(1, 0, &oim), oim.quant_bias[0]);
        assert_eq!(bias_adjust(1, 1, &oim), oim.quant_bias[1]);
        assert_eq!(bias_adjust(1, 2, &oim), oim.quant_bias[2]);
        // quant = -1 → -quant_bias[c].
        assert_eq!(bias_adjust(-1, 0, &oim), -oim.quant_bias[0]);
    }

    #[test]
    fn bias_adjust_abs_gt_one_subtracts_numerator_over_quant() {
        let oim = OpsinInverseMatrix::default();
        // quant = 2 → 2 - num / 2.
        let expected = 2.0_f32 - oim.quant_bias_numerator / 2.0;
        assert_eq!(bias_adjust(2, 0, &oim), expected);
        // quant = -2 → -2 - num / -2 = -2 + num / 2.
        let expected_neg = -2.0_f32 - oim.quant_bias_numerator / -2.0;
        assert_eq!(bias_adjust(-2, 0, &oim), expected_neg);
        // quant = 1000 → 1000 - 0.145 / 1000.
        let expected_big = 1000.0_f32 - oim.quant_bias_numerator / 1000.0;
        assert!((bias_adjust(1000, 1, &oim) - expected_big).abs() < 1e-6);
    }

    #[test]
    fn bias_adjust_branches_at_boundary() {
        // The spec writes `if (abs(quant) <= 1)`, so quant=1 takes the
        // bias branch and quant=2 takes the subtraction branch.
        let oim = OpsinInverseMatrix::default();
        let v1 = bias_adjust(1, 0, &oim);
        let v2 = bias_adjust(2, 0, &oim);
        // quant=1 (multiplicative bias) gives ≈ 0.945; quant=2
        // (subtractive bias) gives ≈ 1.9275 — they shouldn't be close.
        assert!((v1 - 1.0).abs() > 0.01);
        assert!((v2 - 2.0).abs() < 0.1);
    }

    #[test]
    fn qm_scale_factors_default_frame_header() {
        // Default FrameHeader: x_qm_scale = 3, b_qm_scale = 2.
        let fh = default_frame_header();
        let qm = QmScaleFactors::for_frame(&fh);
        // x_factor = 0.8^(3-2) = 0.8.
        assert!((qm.x_factor - 0.8).abs() < 1e-7);
        // b_factor = 0.8^(2-2) = 1.0.
        assert!((qm.b_factor - 1.0).abs() < 1e-7);
        // Y channel always 1.0.
        assert_eq!(qm.for_channel(1), 1.0);
        // X channel matches x_factor.
        assert_eq!(qm.for_channel(0), qm.x_factor);
        // B channel matches b_factor.
        assert_eq!(qm.for_channel(2), qm.b_factor);
    }

    #[test]
    fn qm_scale_factors_negative_exponent() {
        // x_qm_scale = 0 → exponent -2 → 0.8^-2 = 1.5625.
        let fh = frame_header_with_qm_scales(0, 0);
        let qm = QmScaleFactors::for_frame(&fh);
        assert!((qm.x_factor - 1.5625).abs() < 1e-6, "got {}", qm.x_factor);
        assert!((qm.b_factor - 1.5625).abs() < 1e-6);
    }

    #[test]
    fn qm_scale_factors_max_exponent() {
        // x_qm_scale = 7 → exponent 5 → 0.8^5 = 0.32768.
        let fh = frame_header_with_qm_scales(7, 7);
        let qm = QmScaleFactors::for_frame(&fh);
        assert!((qm.x_factor - 0.32768).abs() < 1e-6);
        assert!((qm.b_factor - 0.32768).abs() < 1e-6);
    }

    #[test]
    fn dequant_hf_coefficient_zero_quant_is_zero() {
        // quant = 0 → bias step gives 0 → final result is 0
        // (multiplying by anything finite stays at 0).
        let oim = OpsinInverseMatrix::default();
        let fh = default_frame_header();
        let qm = QmScaleFactors::for_frame(&fh);
        let d = dequant_hf_coefficient(0, 1, 100, 0.0012, &oim, &qm);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn dequant_hf_coefficient_y_channel_skips_qm_factor() {
        // For Y channel the qm-scale factor is 1.0 by spec; the
        // result should equal bias_adjust(quant) × hf_mul × matrix.
        let oim = OpsinInverseMatrix::default();
        let fh = frame_header_with_qm_scales(7, 7);
        let qm = QmScaleFactors::for_frame(&fh);
        let d = dequant_hf_coefficient(5, 1, 100, 0.001, &oim, &qm);
        let pre = bias_adjust(5, 1, &oim);
        let expected = pre * 100.0 * 1.0 * 0.001;
        assert!((d - expected).abs() < 1e-6, "got {d}, expected {expected}");
    }

    #[test]
    fn dequant_hf_coefficient_x_channel_applies_qm_factor() {
        // For X channel with x_qm_scale = 4 → 0.8^2 = 0.64.
        let oim = OpsinInverseMatrix::default();
        let fh = frame_header_with_qm_scales(4, 2);
        let qm = QmScaleFactors::for_frame(&fh);
        let d = dequant_hf_coefficient(5, 0, 100, 0.001, &oim, &qm);
        let pre = bias_adjust(5, 0, &oim);
        let expected = pre * 100.0 * 0.64 * 0.001;
        assert!((d - expected).abs() < 1e-6, "got {d}, expected {expected}");
    }

    #[test]
    fn dequant_hf_coefficient_b_channel_applies_qm_factor() {
        // For B channel with b_qm_scale = 3 → 0.8^1 = 0.8.
        let oim = OpsinInverseMatrix::default();
        let fh = frame_header_with_qm_scales(2, 3);
        let qm = QmScaleFactors::for_frame(&fh);
        let d = dequant_hf_coefficient(5, 2, 100, 0.001, &oim, &qm);
        let pre = bias_adjust(5, 2, &oim);
        let expected = pre * 100.0 * 0.8 * 0.001;
        assert!((d - expected).abs() < 1e-6, "got {d}, expected {expected}");
    }

    #[test]
    fn dequant_hf_pre_matrix_omits_matrix_factor() {
        // The "pre-matrix" helper should match
        // dequant_hf_coefficient(...) / dequant_matrix_entry as long
        // as matrix entry is non-zero.
        let oim = OpsinInverseMatrix::default();
        let fh = default_frame_header();
        let qm = QmScaleFactors::for_frame(&fh);
        let entry = 0.001_f32;
        let full = dequant_hf_coefficient(7, 0, 50, entry, &oim, &qm);
        let pre = dequant_hf_pre_matrix(7, 0, 50, &oim, &qm);
        assert!((full - pre * entry).abs() < 1e-7);
    }

    #[test]
    fn bias_adjust_sign_preservation() {
        // For |quant| > 1 the subtractive bias must preserve the
        // sign of quant (because |num/quant| < |quant| for all
        // |quant| > sqrt(num) ≈ 0.38 with num=0.145 — and we only
        // hit this branch when |quant| > 1).
        let oim = OpsinInverseMatrix::default();
        for q in (-10..=-2).chain(2..=10) {
            let v = bias_adjust(q, 1, &oim);
            assert_eq!(
                v.is_sign_negative(),
                (q as f32).is_sign_negative(),
                "sign mismatch at quant={q}: got {v}"
            );
        }
    }

    #[test]
    fn full_pipeline_neutral_state() {
        // Sanity: quant = 1, channel = 1 (Y), hf_mul = 1, matrix =
        // 1.0. Then dequant_hf = quant_bias[1] × 1 × 1 × 1.0 =
        // quant_bias[1].
        let oim = OpsinInverseMatrix::default();
        let fh = default_frame_header();
        let qm = QmScaleFactors::for_frame(&fh);
        let d = dequant_hf_coefficient(1, 1, 1, 1.0, &oim, &qm);
        assert_eq!(d, oim.quant_bias[1]);
    }
}
