//! Round-89 integration tests: `GetDCTQuantWeights` + Table I.6
//! default dequantization-matrix materialisation (ISO/IEC
//! 18181-1:2024 §I.2.4 / §I.2.5 + Table I.4 + Table I.6).
//!
//! This round lands the **matrix-materialisation step** that bridges
//! HfGlobal's `u(1) == 1` default-encoding fast path to actual
//! per-channel dequantization matrices. Pre-round-89 the HfGlobal
//! bundle was parsed (round 14) but the matrices were never
//! materialised; round 89 transcribes the spec listing verbatim and
//! exposes the 17-slot × 3-channel default set through
//! [`oxideav_jpegxl::dct_quant_weights`].
//!
//! These integration tests pin the public API surface + the
//! cross-slot invariants the spec promises (positive-finite values,
//! correct Table I.4 dimensions, per-slot first-cell sanity).

use oxideav_jpegxl::dct_quant_weights::{
    compute_dct_weights, interpolate, materialise_default_dequant_set,
    materialise_default_weights_for_dct_select, mult, slot_for_transform,
    weights_matrix_dims_for_slot,
};
use oxideav_jpegxl::dct_select::TransformType;

/// The full 17-slot default dequantization set must materialise
/// without error and every cell must be positive-finite (spec
/// §I.2.4 last paragraph invariant: "None of the resulting values
/// are non-positive or infinity").
#[test]
fn default_dequant_set_is_positive_finite_everywhere() {
    let set = materialise_default_dequant_set().expect("default set must materialise");
    assert_eq!(set.matrices.len(), 17);
    let mut total_cells = 0usize;
    for (slot_index, slot) in set.matrices.iter().enumerate() {
        let (x_dim, y_dim) = weights_matrix_dims_for_slot(slot_index as u32).unwrap();
        let expected = (x_dim as usize) * (y_dim as usize);
        for (channel, mat) in slot.iter().enumerate() {
            assert_eq!(
                mat.len(),
                expected,
                "slot {slot_index} channel {channel}: got len {}, expected {expected}",
                mat.len()
            );
            for (i, &v) in mat.iter().enumerate() {
                assert!(
                    v > 0.0 && v.is_finite(),
                    "slot {slot_index} channel {channel} cell {i}: {v} is not positive-finite"
                );
            }
            total_cells += mat.len();
        }
    }
    // Sanity: sum of per-slot cells × 3 channels — 8×8 (slots 0..3 + 9 + 10) + 16×16 + 32×32 + 16×8 + 32×8 + 32×16 + 64×64 + 64×32 + 128×128 + 128×64 + 256×256 + 256×128
    // = (8*8)*6 + 16*16 + 32*32 + 16*8 + 32*8 + 32*16 + 64*64 + 64*32 + 128*128 + 128*64 + 256*256 + 256*128
    // per channel × 3 channels.
    let expected_cells = ((8 * 8) * 6
        + 16 * 16
        + 32 * 32
        + 16 * 8
        + 32 * 8
        + 32 * 16
        + 64 * 64
        + 64 * 32
        + 128 * 128
        + 128 * 64
        + 256 * 256
        + 256 * 128)
        * 3;
    assert_eq!(
        total_cells, expected_cells,
        "total cells over all 17 slots × 3 channels"
    );
}

/// `weights_matrix_dims_for_slot` must agree with Table I.4 page 57.
#[test]
fn table_i_4_dimensions_match_spec() {
    let cases: &[(u32, (u32, u32))] = &[
        (0, (8, 8)),      // DCT8×8
        (1, (8, 8)),      // Hornuss
        (2, (8, 8)),      // DCT2×2
        (3, (8, 8)),      // DCT4×4
        (4, (16, 16)),    // DCT16×16
        (5, (32, 32)),    // DCT32×32
        (6, (16, 8)),     // DCT16×8/DCT8×16
        (7, (32, 8)),     // DCT32×8/DCT8×32
        (8, (32, 16)),    // DCT16×32/DCT32×16
        (9, (8, 8)),      // DCT4×8/DCT8×4
        (10, (8, 8)),     // AFV
        (11, (64, 64)),   // DCT64×64
        (12, (64, 32)),   // DCT32×64/DCT64×32
        (13, (128, 128)), // DCT128×128
        (14, (128, 64)),  // DCT64×128/DCT128×64
        (15, (256, 256)), // DCT256×256
        (16, (256, 128)), // DCT128×256/DCT256×128
    ];
    for (slot, expected) in cases {
        let dims = weights_matrix_dims_for_slot(*slot).expect("valid slot");
        assert_eq!(dims, *expected, "slot {slot}");
    }
}

/// Every `TransformType` (Table C.16 0..=26) must map to a valid
/// Table I.4 slot (0..=16).
#[test]
fn all_transform_types_map_to_valid_slot() {
    let all = [
        TransformType::Dct8x8,
        TransformType::Hornuss,
        TransformType::Dct2x2,
        TransformType::Dct4x4,
        TransformType::Dct16x16,
        TransformType::Dct32x32,
        TransformType::Dct16x8,
        TransformType::Dct8x16,
        TransformType::Dct32x8,
        TransformType::Dct8x32,
        TransformType::Dct32x16,
        TransformType::Dct16x32,
        TransformType::Dct4x8,
        TransformType::Dct8x4,
        TransformType::Afv0,
        TransformType::Afv1,
        TransformType::Afv2,
        TransformType::Afv3,
        TransformType::Dct64x64,
        TransformType::Dct64x32,
        TransformType::Dct32x64,
        TransformType::Dct128x128,
        TransformType::Dct128x64,
        TransformType::Dct64x128,
        TransformType::Dct256x256,
        TransformType::Dct256x128,
        TransformType::Dct128x256,
    ];
    for t in all {
        let slot = slot_for_transform(t);
        assert!(
            slot < 17,
            "{:?} maps to slot {slot} which is out of range",
            t
        );
        // The slot must have valid dims.
        assert!(weights_matrix_dims_for_slot(slot).is_ok());
    }
}

/// `Mult` must produce the spec piecewise function exactly.
#[test]
fn mult_spec_branch_at_zero() {
    // Spec: `if (v > 0) return 1 + v; else return 1 / (1 - v);`.
    // Note the strict `>`: v == 0.0 falls through to the negative
    // branch.
    assert_eq!(mult(0.0), 1.0);
    assert_eq!(mult(0.5), 1.5);
    assert_eq!(mult(-1.0), 0.5);
    // Negative branch for v very negative.
    let v = mult(-9.0);
    assert!((v - 0.1).abs() < 1e-12, "Mult(-9) = {v}");
}

/// `Interpolate` with single-band must short-circuit per spec.
#[test]
fn interpolate_single_band_short_circuits() {
    let v = interpolate(99.0, 0.001, &[7.5]).unwrap();
    assert_eq!(v, 7.5);
}

/// `compute_dct_weights` must hit `bands[0]` at the (0, 0) corner
/// (distance = 0).
#[test]
fn corner_takes_first_band() {
    let w = compute_dct_weights(&[100.0, -0.5, -0.5], 8, 8).unwrap();
    assert!((w[0] - 100.0).abs() < 1e-9, "(0,0) = {}", w[0]);
}

/// Default weights for slot 0 (DCT8×8), channel 0 must produce a
/// matrix where the corner equals `bands[0] = params[0] = 3150.0`.
#[test]
fn default_dct8x8_weights_corner_is_3150() {
    let w = materialise_default_weights_for_dct_select(0, 0).unwrap();
    assert_eq!(w.len(), 64);
    assert!(
        (w[0] - 3150.0).abs() < 1e-6,
        "DCT8×8 (0,0) weight = {}",
        w[0]
    );
}

/// Default weights for Hornuss slot must have weights(0,0) = 1.
#[test]
fn default_hornuss_corner_is_unity() {
    let w = materialise_default_weights_for_dct_select(1, 0).unwrap();
    assert!((w[0] - 1.0).abs() < 1e-12, "Hornuss (0,0) = {}", w[0]);
}

/// Default weights for AFV slot must be 8×8 with no NaN/zero cells.
#[test]
fn default_afv_weights_8x8_fully_populated() {
    for channel in 0..3 {
        let w = materialise_default_weights_for_dct_select(10, channel).unwrap();
        assert_eq!(w.len(), 64);
        for (i, &v) in w.iter().enumerate() {
            assert!(
                v > 0.0 && v.is_finite(),
                "AFV channel {channel} cell {i}: {v} is bad"
            );
        }
    }
}

/// Out-of-range slot index must fail cleanly (not panic).
#[test]
fn out_of_range_slot_is_invalid_data() {
    let r = weights_matrix_dims_for_slot(17);
    assert!(r.is_err(), "slot 17 should be out of range");
    let r = materialise_default_weights_for_dct_select(17, 0);
    assert!(r.is_err(), "default weights for slot 17 should error");
}

/// Out-of-range channel must fail cleanly.
#[test]
fn out_of_range_channel_is_invalid_data() {
    let r = materialise_default_weights_for_dct_select(0, 3);
    assert!(r.is_err(), "channel 3 should be out of range");
}
