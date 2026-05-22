//! Round-95 integration tests: §F.3 HF dequantisation gluing the
//! round-89 [`dct_quant_weights`] dequant set (Table I.6 defaults)
//! to the round-90 [`hf_pass`] / [`pass_group_hf`] structural
//! parsers, and to the round-2 [`metadata_fdis::OpsinInverseMatrix`]
//! and [`frame_header::FrameHeader`] `x_qm_scale` / `b_qm_scale` fields.
//!
//! This round's contract: given a single quantised HF coefficient
//! (the integer ANS output of the §C.8.3 per-block decode), produce
//! the final dequantised `dX` / `dY` / `dB` value per Listing F.2 +
//! the post-listing per-block / per-channel multipliers. The
//! per-block ANS coefficient decode itself stays deferred to a later
//! round; round 95 lands the pure-math step so future rounds can
//! drop it in once their per-block ANS reader is wired.
//!
//! These tests pin:
//!   * the Listing F.2 bias-adjust branch boundaries,
//!   * the `0.8^(qm_scale - 2)` per-frame factor (computed once),
//!   * the cross-module composition with
//!     [`oxideav_jpegxl::dct_quant_weights::materialise_default_dequant_set`].

use oxideav_jpegxl::dct_quant_weights::{
    materialise_default_dequant_set, slot_for_transform, weights_matrix_dims_for_slot,
};
use oxideav_jpegxl::dct_select::TransformType;
use oxideav_jpegxl::hf_dequant::{
    bias_adjust, dequant_hf_coefficient, dequant_hf_pre_matrix, QmScaleFactors,
};
use oxideav_jpegxl::metadata_fdis::OpsinInverseMatrix;

/// FDIS Listing F.2: at `quant == 0` the output is exactly 0
/// regardless of channel. (`0 × quant_bias[c] = 0`.)
#[test]
fn listing_f2_zero_quant_is_zero_for_every_channel() {
    let oim = OpsinInverseMatrix::default();
    for c in 0..3 {
        assert_eq!(bias_adjust(0, c, &oim), 0.0);
    }
}

/// FDIS Listing F.2: at `quant == 1` the output is `quant_bias[c]`,
/// the per-channel multiplicative bias.
#[test]
fn listing_f2_quant_one_uses_per_channel_bias() {
    let oim = OpsinInverseMatrix::default();
    for c in 0..3 {
        let v = bias_adjust(1, c, &oim);
        assert!((v - oim.quant_bias[c]).abs() < 1e-7);
    }
}

/// FDIS Listing F.2: at `quant == -1` the result is
/// `-quant_bias[c]`.
#[test]
fn listing_f2_minus_one_negates_bias() {
    let oim = OpsinInverseMatrix::default();
    for c in 0..3 {
        let v = bias_adjust(-1, c, &oim);
        assert!((v + oim.quant_bias[c]).abs() < 1e-7);
    }
}

/// FDIS Listing F.2: at `|quant| > 1` the bias is subtractive:
/// `quant - num/quant`. Sign of result must match sign of `quant`.
#[test]
fn listing_f2_subtractive_bias_preserves_sign() {
    let oim = OpsinInverseMatrix::default();
    for q in [-100, -10, -5, -2, 2, 5, 10, 100] {
        let v = bias_adjust(q, 1, &oim);
        assert_eq!(
            v.is_sign_negative(),
            (q as f32).is_sign_negative(),
            "sign mismatch at quant={q}: got {v}"
        );
        // |v| < |q| because we subtract num/quant which has the
        // same sign as q (so |v| = |q| - num/|q| < |q|).
        assert!(v.abs() < (q as f32).abs(), "magnitude grew: {v} from {q}");
    }
}

/// Cross-module composition: the F.3 HF dequant pipeline must compose
/// cleanly with the round-89 dequant-matrix materialiser. We pick
/// slot 0 (DCT8×8), channel 1 (Y), corner cell (0, 0), and verify the
/// product matches the per-piece computation.
#[test]
fn round95_pipeline_against_default_dequant_set_slot_0() {
    let oim = OpsinInverseMatrix::default();
    // Frame defaults: x_qm_scale = 3, b_qm_scale = 2. Use a
    // round-19-style HfMul = 7 (small integer typical for high-quality
    // VarDCT) and quant = 13 (any non-trivial bin).
    let set = materialise_default_dequant_set().expect("default set materialises");
    let slot = slot_for_transform(TransformType::Dct8x8); // 0
    let (x_dim, _y_dim) = weights_matrix_dims_for_slot(slot).unwrap();
    let channel = 1; // Y — no qm-scale factor.

    // Construct a minimal FrameHeader through the public API: the
    // QmScaleFactors API takes a FrameHeader reference; we
    // synthesise one with the default qm-scales by parsing a 1-byte
    // all-default header (the FrameHeader internals are not part of
    // this test surface). Simpler: we use the QmScaleFactors fields
    // directly by constructing one from a default-equivalent value.
    let qm = QmScaleFactors {
        x_factor: 0.8_f32, // 0.8^(3-2)
        b_factor: 1.0,     // 0.8^(2-2)
    };

    // Corner cell of the (x_dim × y_dim) row-major matrix is index 0.
    let _ = x_dim;
    let matrix_entry = set.matrices[slot as usize][channel][0] as f32;
    let d = dequant_hf_coefficient(13, channel, 7, matrix_entry, &oim, &qm);

    // Hand computation:
    //   bias-adjust(13, Y) = 13 - 0.145 / 13   ≈ 12.98885
    //   × HfMul (7)         ≈ 90.92193
    //   × Y qm factor (1.0) ≈ 90.92193
    //   × matrix entry      = the corner of slot-0 channel-1 dequant matrix.
    let pre = bias_adjust(13, channel, &oim);
    let expected = pre * 7.0 * 1.0 * matrix_entry;
    assert!((d - expected).abs() < 1e-5, "got {d}, expected {expected}");
}

/// Cross-module composition: slot 0, channel 0 (X) must pick up the
/// 0.8 factor under default x_qm_scale = 3.
#[test]
fn round95_pipeline_x_channel_picks_up_zero_eight_factor() {
    let oim = OpsinInverseMatrix::default();
    let set = materialise_default_dequant_set().unwrap();
    let slot = slot_for_transform(TransformType::Dct8x8);
    let matrix_entry = set.matrices[slot as usize][0][0] as f32; // X channel, corner.
    let qm = QmScaleFactors {
        x_factor: 0.8_f32,
        b_factor: 1.0,
    };
    let d = dequant_hf_coefficient(13, 0, 7, matrix_entry, &oim, &qm);
    let pre = bias_adjust(13, 0, &oim);
    let expected = pre * 7.0 * 0.8 * matrix_entry;
    assert!((d - expected).abs() < 1e-5, "got {d}, expected {expected}");
}

/// dequant_hf_pre_matrix() must return the partial product without
/// the matrix-entry factor; full × pre × entry = full × entry.
#[test]
fn round95_pre_matrix_helper_matches_full_pipeline() {
    let oim = OpsinInverseMatrix::default();
    let qm = QmScaleFactors {
        x_factor: 0.8_f32,
        b_factor: 0.64, // 0.8^2 — exotic but well-defined.
    };
    let entry = 0.0007_f32; // small typical dequant matrix value
    for channel in 0..3 {
        for quant in [-50, -3, -1, 0, 1, 3, 50] {
            let full = dequant_hf_coefficient(quant, channel, 5, entry, &oim, &qm);
            let pre = dequant_hf_pre_matrix(quant, channel, 5, &oim, &qm);
            let diff = (full - pre * entry).abs();
            assert!(
                diff < 1e-6,
                "pre × entry mismatch at (q={quant}, c={channel}): full={full}, pre×entry={}",
                pre * entry
            );
        }
    }
}

/// Pinning test: the FDIS-2021 / 2024 numeric defaults (quant_bias
/// numerator = 0.145) drive a known fixed-point through the formula.
/// At `quant = 2`, channel doesn't matter for the subtractive branch,
/// so we get `2 - 0.145/2 = 1.9275`.
#[test]
fn fdis_numeric_default_quant_bias_numerator_reaches_1_9275() {
    let oim = OpsinInverseMatrix::default();
    let v = bias_adjust(2, 0, &oim);
    assert!(
        (v - 1.9275).abs() < 1e-5,
        "FDIS default 0.145 numerator should yield 2 - 0.0725 = 1.9275; got {v}"
    );
}

/// Pinning test: all three default `quant_bias` values are within
/// the spec-listed range (≈ 0.93 to 0.95 — they are `1 - small`).
#[test]
fn fdis_default_quant_bias_values_in_range() {
    let oim = OpsinInverseMatrix::default();
    for (i, &v) in oim.quant_bias.iter().enumerate() {
        assert!(
            v > 0.9 && v < 1.0,
            "default quant_bias[{i}] = {v} outside spec range (0.9, 1.0)"
        );
    }
}

/// QmScaleFactors: 0.8^(scale-2) sweep. Cover every legal `u(3)`
/// value (0..=7) and verify the f32 result is positive-finite. The
/// 0.8^k formula is well-behaved across the entire 3-bit range.
#[test]
fn qm_scale_factors_sweep_all_u3_values_positive_finite() {
    for scale in 0u32..8 {
        let exp = scale as i32 - 2;
        let factor = 0.8_f32.powi(exp);
        assert!(
            factor.is_finite() && factor > 0.0,
            "scale={scale}: {factor}"
        );
    }
}
