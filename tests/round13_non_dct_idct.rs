//! Round-13 non-DCT IDCT integration tests (ISO/IEC 18181-1:2024
//! Annex I.9.3..I.9.7).
//!
//! Round 13 lands the non-DCT IDCT helpers in
//! `oxideav_jpegxl::idct`:
//!
//! * `aux_idct_2x2(block, S)` — Annex I.9.3 Hadamard-style butterfly.
//! * `idct_dct2x2(coefficients)` — Annex I.9.3 closing recipe (chained
//!   `aux_idct_2x2` at S=2, 4, 8).
//! * `idct_dct4x4(coefficients)` — Annex I.9.4.
//! * `idct_hornuss(coefficients)` — Annex I.9.5.
//! * `idct_dct8x4(coefficients)` — Annex I.9.6.
//! * `idct_dct4x8(coefficients)` — Annex I.9.7.
//!
//! These tests cross-check the public API surface and confirm that
//! dispatching through `idct_for_transform` for the corresponding
//! `TransformType` reaches the right helper. AFV0..AFV3 routing
//! through `idct_afv` (Listing I.13) landed r150 — see the
//! corresponding round-12 regression test for the four-AFV-dispatch
//! assertion + `round147_afv_idct` for `AFV_IDCT` primitive coverage.

use oxideav_jpegxl::dct_select::TransformType;
use oxideav_jpegxl::idct::{
    aux_idct_2x2, idct_dct2x2, idct_dct4x4, idct_dct4x8, idct_dct8x4, idct_for_transform,
    idct_hornuss, non_dct_pixel_dims,
};

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

#[test]
fn non_dct_pixel_dims_for_each_non_dct_variant_is_8x8() {
    for t in [
        TransformType::Hornuss,
        TransformType::Dct2x2,
        TransformType::Dct4x4,
        TransformType::Dct4x8,
        TransformType::Dct8x4,
        TransformType::Afv0,
        TransformType::Afv1,
        TransformType::Afv2,
        TransformType::Afv3,
    ] {
        assert_eq!(non_dct_pixel_dims(t), Some((8, 8)), "{t:?}");
    }
}

#[test]
fn idct_for_transform_non_dct_paths_match_helper_output() {
    // For each newly-supported non-DCT TransformType, dispatch through
    // `idct_for_transform` and verify the byte-for-byte output equals
    // the dedicated helper's output.
    let mut coeffs = vec![0.0f32; 64];
    // Sprinkle a few non-zero coefficients so the helpers actually do
    // work (DC + a couple AC).
    coeffs[0] = 5.0;
    coeffs[1] = 1.0;
    coeffs[8] = -1.0;
    coeffs[10] = 2.0;
    coeffs[20] = -0.5;

    type Helper = fn(&[f32]) -> oxideav_core::Result<Vec<f32>>;
    let cases: &[(TransformType, Helper)] = &[
        (TransformType::Dct2x2, idct_dct2x2_wrap),
        (TransformType::Dct4x4, idct_dct4x4_wrap),
        (TransformType::Hornuss, idct_hornuss_wrap),
        (TransformType::Dct8x4, idct_dct8x4_wrap),
        (TransformType::Dct4x8, idct_dct4x8_wrap),
    ];
    for &(t, helper) in cases {
        let dispatch = idct_for_transform(t, &coeffs).unwrap();
        let direct = helper(&coeffs).unwrap();
        assert_eq!(dispatch.len(), 64, "{t:?}");
        assert_eq!(direct.len(), 64, "{t:?}");
        for i in 0..64 {
            assert!(
                approx_eq(dispatch[i], direct[i], 1e-6),
                "{t:?} pos {i}: dispatch={} direct={}",
                dispatch[i],
                direct[i]
            );
        }
    }
}

fn idct_dct2x2_wrap(c: &[f32]) -> oxideav_core::Result<Vec<f32>> {
    idct_dct2x2(c)
}
fn idct_dct4x4_wrap(c: &[f32]) -> oxideav_core::Result<Vec<f32>> {
    idct_dct4x4(c)
}
fn idct_hornuss_wrap(c: &[f32]) -> oxideav_core::Result<Vec<f32>> {
    idct_hornuss(c)
}
fn idct_dct8x4_wrap(c: &[f32]) -> oxideav_core::Result<Vec<f32>> {
    idct_dct8x4(c)
}
fn idct_dct4x8_wrap(c: &[f32]) -> oxideav_core::Result<Vec<f32>> {
    idct_dct4x8(c)
}

#[test]
fn aux_idct_2x2_size_8_full_block() {
    // Run AuxIDCT2x2 at S=8 on an 8×8 input where the top-left 4×4
    // happens to be the only filled region. Verifies the Hadamard
    // butterfly fans out across the full block while leaving cells
    // outside the top-left 4×4 of the *input* (which is the entire
    // block at S=8) untouched relative to the algorithm's intent.
    let mut block = vec![0.0f32; 64];
    // Set c00 in each of the (S/2)² = 16 quadrants to a constant.
    // Without loss of generality, set every cell to 1.0 — this is a
    // "constant input" and the butterfly should yield non-trivial
    // structure that we can sanity-check.
    for v in block.iter_mut() {
        *v = 1.0;
    }
    let out = aux_idct_2x2(&block, 8).unwrap();
    // For each 2×2 quadrant of the constant-1 block:
    //   c00 = c01 = c10 = c11 = 1
    //   r00 = 4, r01 = 0, r10 = 0, r11 = 0
    // Output positions (x*2 + dx, y*2 + dy) for dx,dy ∈ {0, 1}:
    //   (even, even) = 4
    //   (odd , even) = 0
    //   (even, odd ) = 0
    //   (odd , odd ) = 0
    for y in 0..8 {
        for x in 0..8 {
            let expected = if x % 2 == 0 && y % 2 == 0 { 4.0 } else { 0.0 };
            assert!(
                approx_eq(out[y * 8 + x], expected, 1e-6),
                "({x},{y}): got {} expected {}",
                out[y * 8 + x],
                expected
            );
        }
    }
}

#[test]
fn idct_dct2x2_three_step_chain_equivalence() {
    // Verify the closing-recipe chain: DCT2x2 == three nested
    // AuxIDCT2x2 calls at S=2, 4, 8.
    let mut coeffs = vec![0.0f32; 64];
    for (i, slot) in coeffs.iter_mut().enumerate() {
        *slot = ((i as f32) - 32.0) * 0.1;
    }
    let chain_b1 = aux_idct_2x2(&coeffs, 2).unwrap();
    let chain_b2 = aux_idct_2x2(&chain_b1, 4).unwrap();
    let chain_b3 = aux_idct_2x2(&chain_b2, 8).unwrap();
    let direct = idct_dct2x2(&coeffs).unwrap();
    assert_eq!(direct.len(), 64);
    for i in 0..64 {
        assert!(
            approx_eq(direct[i], chain_b3[i], 1e-6),
            "pos {i}: direct={} chain={}",
            direct[i],
            chain_b3[i]
        );
    }
}

#[test]
fn aux_idct_2x2_zero_input_returns_zero() {
    let coeffs = vec![0.0f32; 64];
    for s in [1usize, 2, 4, 8] {
        let out = aux_idct_2x2(&coeffs, s).unwrap();
        assert_eq!(out.len(), 64);
        for &v in &out {
            assert!(approx_eq(v, 0.0, 1e-9), "size {s}: got {v}");
        }
    }
}
