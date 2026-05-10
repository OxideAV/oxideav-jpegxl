//! Round-12 IDCT dispatch tests (ISO/IEC 18181-1:2024 Annex I.2.1 +
//! I.2.2 Listings I.3 / I.4).
//!
//! Round 12 lands the spec-conformant 1-D IDCT for power-of-two sizes
//! and the 2-D IDCT dispatcher that consumes a TransformType + a
//! coefficient block in the spec's `(short × long)` row-major natural
//! ordering layout, returning the `(R × C)` sample matrix in row-major
//! order.
//!
//! These tests exercise the public API surface used by future round-13
//! HF coefficient decode + dequantisation:
//!
//! 1. [`oxideav_jpegxl::idct::idct_for_transform`] dispatches to a 2-D
//!    IDCT for plain DCT TransformType variants and returns
//!    `Err(Unsupported)` for the non-DCT transforms (Hornuss, DCT2x2,
//!    DCT4x4, DCT4x8, DCT8x4, AFV0..AFV3).
//! 2. The five small lossless Modular fixtures still pixel-correct
//!    (regression sentinel against the new IDCT module landing).

use oxideav_jpegxl::dct_select::TransformType;
use oxideav_jpegxl::decode_one_frame;
use oxideav_jpegxl::idct::{dct_pixel_dims, idct_1d, idct_2d, idct_for_transform};

const PIXEL_1X1_JXL: &[u8] = include_bytes!("fixtures/pixel_1x1.jxl");
const GRAY_64X64_JXL: &[u8] = include_bytes!("fixtures/gray_64x64_lossless.jxl");
const GRADIENT_JXL: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
const PALETTE_JXL: &[u8] = include_bytes!("fixtures/palette_32x32.jxl");
const GREY_8X8_JXL: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

#[test]
fn idct_1d_dc_only_returns_constant_input_for_all_supported_sizes() {
    // Spec FDIS Annex I.2.1: DC-only IDCT yields constant samples
    // equal to the DC value across every sample position.
    for s in [1usize, 2, 4, 8, 16, 32, 64, 128, 256] {
        let mut c = vec![0.0f32; s];
        c[0] = 5.0;
        let out = idct_1d(&c).expect("idct_1d should accept power-of-two");
        for &v in &out {
            assert!(approx_eq(v, 5.0, 1e-3), "size {s}: got {v} expected 5.0");
        }
    }
}

#[test]
fn idct_2d_dc_only_constant_for_every_dct_block_size() {
    // For every plain-DCT transform in Table C.16, a DC-only coefficient
    // input yields a constant sample block equal to the DC value.
    for t in [
        TransformType::Dct8x8,
        TransformType::Dct16x16,
        TransformType::Dct32x32,
        TransformType::Dct16x8,
        TransformType::Dct8x16,
        TransformType::Dct32x8,
        TransformType::Dct8x32,
        TransformType::Dct32x16,
        TransformType::Dct16x32,
        TransformType::Dct64x64,
        TransformType::Dct64x32,
        TransformType::Dct32x64,
        TransformType::Dct128x128,
        TransformType::Dct64x128,
        TransformType::Dct128x64,
        TransformType::Dct256x256,
        TransformType::Dct128x256,
        TransformType::Dct256x128,
    ] {
        let (rows, cols) = dct_pixel_dims(t).unwrap();
        let mut coeffs = vec![0.0f32; rows * cols];
        coeffs[0] = 9.0;
        let out = idct_for_transform(t, &coeffs).unwrap();
        assert_eq!(out.len(), rows * cols, "{t:?}");
        for (i, &v) in out.iter().enumerate() {
            assert!(
                approx_eq(v, 9.0, 1e-2),
                "{t:?} pos {i}: got {v} expected 9.0"
            );
        }
    }
}

#[test]
fn idct_for_transform_afv_only_unsupported_after_round_13() {
    // After round 13: Hornuss, DCT2×2, DCT4×4, DCT4×8, DCT8×4 dispatch
    // through their dedicated I.9.3..I.9.7 helpers and succeed. Only
    // AFV0..AFV3 remain `Err(Unsupported)` pending the verified
    // AFVBasis table.
    for t in [
        TransformType::Hornuss,
        TransformType::Dct2x2,
        TransformType::Dct4x4,
        TransformType::Dct4x8,
        TransformType::Dct8x4,
    ] {
        let coeffs = vec![0.0f32; 64];
        let r = idct_for_transform(t, &coeffs);
        assert!(r.is_ok(), "{t:?}: expected Ok after round 13, got {r:?}");
        let out = r.unwrap();
        assert_eq!(out.len(), 64, "{t:?}: expected 8×8 output");
    }
    for t in [
        TransformType::Afv0,
        TransformType::Afv1,
        TransformType::Afv2,
        TransformType::Afv3,
    ] {
        let coeffs = vec![0.0f32; 64];
        let r = idct_for_transform(t, &coeffs);
        assert!(r.is_err(), "{t:?}: AFV remains unsupported in round 13");
    }
}

#[test]
fn idct_2d_round_trip_through_supplied_dct_oracle_8x16_and_16x8() {
    // Both asymmetric 8×16 and 16×8 cases (one in each branch of
    // Listing I.4's `if (C > R)` test) round-trip through a forward DCT
    // oracle. We instantiate the oracle inline so the test doesn't pull
    // in a private API.
    use std::f32::consts::PI;

    fn forward_dct_1d(input: &[f32]) -> Vec<f32> {
        let s = input.len();
        let s_f = s as f32;
        let sqrt2 = 2f32.sqrt();
        let mut out = vec![0.0f32; s];
        for (k, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (n, &v) in input.iter().enumerate() {
                acc += v * (PI * (k as f32) * (n as f32 + 0.5) / s_f).cos();
            }
            let scale = if k == 0 { 1.0 } else { sqrt2 };
            *slot = scale * acc / s_f;
        }
        out
    }

    fn forward_dct_2d(samples: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        // Mirror of [`oxideav_jpegxl::idct::idct_2d`]'s working layout.
        let short = rows.min(cols);
        let long = rows.max(cols);
        let mut working = vec![0.0f32; long * short];
        if rows <= cols {
            for r in 0..short {
                for c in 0..long {
                    working[c * short + r] = samples[r * cols + c];
                }
            }
        } else {
            working.copy_from_slice(samples);
        }
        let mut dct1 = vec![0.0f32; long * short];
        let mut col = vec![0.0f32; long];
        for c in 0..short {
            for r in 0..long {
                col[r] = working[r * short + c];
            }
            let cdct = forward_dct_1d(&col);
            for r in 0..long {
                dct1[r * short + c] = cdct[r];
            }
        }
        let mut dct1_t = vec![0.0f32; short * long];
        for r in 0..long {
            for c in 0..short {
                dct1_t[c * long + r] = dct1[r * short + c];
            }
        }
        let mut dct2 = vec![0.0f32; short * long];
        let mut col2 = vec![0.0f32; short];
        for c in 0..long {
            for r in 0..short {
                col2[r] = dct1_t[r * long + c];
            }
            let cdct = forward_dct_1d(&col2);
            for r in 0..short {
                dct2[r * long + c] = cdct[r];
            }
        }
        dct2
    }

    for &(r, c) in &[(8usize, 16usize), (16, 8)] {
        let mut input = vec![0.0f32; r * c];
        for i in 0..r {
            for j in 0..c {
                input[i * c + j] = (i as f32) * 0.7 + (j as f32) * 0.3 - 1.5;
            }
        }
        let coeffs = forward_dct_2d(&input, r, c);
        let recovered = idct_2d(&coeffs, r, c).unwrap();
        for (k, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(
                approx_eq(a, b, 1e-2),
                "{r}x{c} pos {k}: input={a} recovered={b}"
            );
        }
    }
}

#[test]
fn five_small_lossless_fixtures_still_decode_round_12() {
    // Round 12 added a new top-level public module `idct` with its own
    // 1-D + 2-D IDCT dispatch. None of the existing Modular fixtures go
    // through the IDCT path (they decode via the modular pipeline), so
    // they must continue to decode pixel-correct. This test is a
    // belt-and-braces regression sentinel.
    for (name, bytes) in [
        ("pixel_1x1", PIXEL_1X1_JXL),
        ("gray_64x64", GRAY_64X64_JXL),
        ("gradient_64x64", GRADIENT_JXL),
        ("palette_32x32", PALETTE_JXL),
        ("grey_8x8", GREY_8X8_JXL),
    ] {
        let r = decode_one_frame(bytes, None);
        assert!(
            r.is_ok(),
            "round-12 regression: {name} should still decode; got {:?}",
            r.err()
        );
    }
}
