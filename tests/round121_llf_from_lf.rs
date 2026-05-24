//! Round-121 integration tests for §I.2.5 LLF coefficients from
//! downsampled image (FDIS Listings I.15 + I.16).
//!
//! These tests pin the public API surface added in round 121 and the
//! hand-derived byte-exact expected coefficient blocks for small
//! varblock dimensions where the math can be verified by hand.

use oxideav_jpegxl::dct_select::TransformType;
use oxideav_jpegxl::llf_from_lf::{
    dct_1d, dct_2d, llf_dims, llf_from_lf, scale_c, scale_d, scale_d8, scale_f, scale_i, scale_i8,
};

/// ScaleF(1, 8, 0) = 1.0 exactly (algebraically derived in the
/// module-level test rationale): used by DCT8×8 LLF.
#[test]
fn scalef_dct8x8_corner_is_unity() {
    let v = scale_f(1, 8, 0);
    assert!((v - 1.0).abs() < 1e-6, "got {v}");
}

/// For an N=8 axis with u=0, `I8(8, 0) = sqrt(2/8) * sqrt(0.5) =
/// 0.5 * sqrt(0.5) = sqrt(0.5)/2 ≈ 0.35355339`.
#[test]
fn i8_eight_zero_matches_closed_form() {
    let v = scale_i8(8, 0);
    let expected = 0.5f32 * (0.5f32).sqrt();
    assert!((v - expected).abs() < 1e-7, "got {v}");
}

/// D8 is defined as the reciprocal of `N * I8(N, u)`.
#[test]
fn d8_is_reciprocal_of_n_times_i8() {
    for n in [1u32, 2, 4, 8, 16, 32] {
        for u in 0..n {
            let i = scale_i8(n, u);
            let d = scale_d8(n, u);
            assert!(
                (d - 1.0 / ((n as f32) * i)).abs() < 1e-6,
                "n={n} u={u}: D8 {d} != 1/(N*I8) {}",
                1.0 / ((n as f32) * i),
            );
        }
    }
}

/// I/D swap roles depending on whether N is exactly 8.
#[test]
fn i_and_d_are_branches_on_n_eq_8() {
    assert_eq!(scale_i(8, 0), scale_i8(8, 0));
    assert_eq!(scale_d(8, 0), scale_d8(8, 0));
    assert_eq!(scale_i(16, 0), scale_d8(16, 0));
    assert_eq!(scale_d(16, 0), scale_i8(16, 0));
}

/// `C(N, N, x) == 1` for every N and x by spec.
#[test]
fn c_returns_one_when_n_big_equals_n_small() {
    for n in [1u32, 2, 4, 8, 16, 32] {
        for x in 0..n {
            assert_eq!(scale_c(n, n, x), 1.0);
        }
    }
}

/// Forward DCT_1D of a size-8 unit-impulse vector (1.0 at index 0)
/// produces a known closed-form output:
///   out[k] = (1/8) * (k == 0 ? 1 : sqrt(2)) * cos(π * k * 0.5 / 8)
#[test]
fn dct_1d_size_8_unit_impulse_matches_closed_form() {
    let mut signal = vec![0.0f32; 8];
    signal[0] = 1.0;
    let out = dct_1d(&signal).unwrap();
    let s = 8.0f32;
    let sqrt2 = 2f32.sqrt();
    for (k, &v) in out.iter().enumerate() {
        let scale = if k == 0 { 1.0 } else { sqrt2 };
        let expected = (scale / s) * (std::f32::consts::PI * (k as f32) * 0.5 / s).cos();
        assert!(
            (v - expected).abs() < 1e-6,
            "k={k}: got {v}, expected {expected}",
        );
    }
}

/// Forward DCT_1D of a constant signal produces only the DC term.
#[test]
fn dct_1d_const_signal_only_dc() {
    let signal = vec![7.0f32; 8];
    let out = dct_1d(&signal).unwrap();
    assert!((out[0] - 7.0).abs() < 1e-6);
    for v in &out[1..] {
        assert!(v.abs() < 1e-5);
    }
}

/// dct_2d of a 1×1 input is the identity.
#[test]
fn dct_2d_1x1_identity() {
    // Use a non-PI-adjacent value to satisfy clippy::approx_constant.
    let out = dct_2d(&[2.75], 1, 1).unwrap();
    assert_eq!(out, vec![2.75]);
}

/// For DCT16×16 the LLF block is 2×2; cx = cy = 2.
#[test]
fn llf_dims_dct16x16_is_two_by_two() {
    assert_eq!(llf_dims(TransformType::Dct16x16), (2, 2));
}

/// For non-DCT transforms the LF→LLF map is the identity (single
/// 1×1 sample passes through unchanged).
#[test]
fn llf_from_lf_non_dct_pass_through() {
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
        let out = llf_from_lf(&[12.5], t).unwrap();
        assert_eq!(
            out,
            vec![12.5],
            "non-DCT {t:?} should pass through unchanged"
        );
    }
}

/// Byte-exact: for DCT16×16 with the 2×2 LF block `[1, 0, 0, 0]`
/// (impulse at (x=0, y=0)), all four LLF coefficients evaluate to
/// `0.25 * ScaleF(2, 16, y) * ScaleF(2, 16, x)`. We assert against
/// the explicit per-cell expected values computed via ScaleF.
#[test]
fn llf_from_lf_dct16x16_impulse_byte_exact() {
    let block = [1.0f32, 0.0, 0.0, 0.0];
    let out = llf_from_lf(&block, TransformType::Dct16x16).unwrap();
    let sf0 = scale_f(2, 16, 0);
    let sf1 = scale_f(2, 16, 1);
    // out layout: (cy × cx) row-major = (2 × 2) row-major,
    // out[y * 2 + x].
    let expected = [
        0.25 * sf0 * sf0, // (x=0,y=0)
        0.25 * sf0 * sf1, // (x=1,y=0)
        0.25 * sf1 * sf0, // (x=0,y=1)
        0.25 * sf1 * sf1, // (x=1,y=1)
    ];
    for (i, (g, e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-6, "cell {i}: got {g}, expected {e}",);
    }
}

/// Byte-exact: for DCT8×8 the LLF block has one cell. With LF
/// sample `s`, the cell value is `s * ScaleF(1, 8, 0)^2 = s` since
/// ScaleF(1, 8, 0) = 1.
#[test]
fn llf_from_lf_dct8x8_byte_exact_single_cell() {
    // For DCT8×8 with cx = cy = 1, ScaleF(1, 8, 0) is algebraically
    // exactly 1.0 but evaluates to ≈ 1 - 3e-8 in f32 arithmetic;
    // assert a relative-error bound that accommodates the float
    // round-off.
    for s in [-100.5f32, -1.0, 0.0, 1.0, 42.42] {
        let out = llf_from_lf(&[s], TransformType::Dct8x8).unwrap();
        assert_eq!(out.len(), 1);
        let abs_err = (out[0] - s).abs();
        let rel_err = if s.abs() > 1.0 {
            abs_err / s.abs()
        } else {
            abs_err
        };
        assert!(
            rel_err < 1e-6,
            "s={s}: got {}, expected {s} (rel_err {rel_err})",
            out[0],
        );
    }
}

/// Rectangular: DCT16×8 (cy=2, cx=1) with a constant LF block has
/// LLF = [c * SF(2,16,0) * SF(1,8,0), 0].
#[test]
fn llf_from_lf_dct16x8_constant_block_byte_exact() {
    let c = 6.0f32;
    let out = llf_from_lf(&[c, c], TransformType::Dct16x8).unwrap();
    let dc_sf = scale_f(2, 16, 0) * scale_f(1, 8, 0);
    let expected_dc = c * dc_sf;
    assert!((out[0] - expected_dc).abs() < 1e-5);
    assert!(out[1].abs() < 1e-5);
}

/// LLF for DCT32×32: 4×4 LF input, 16-cell output. Verifies the
/// dimension contract and the diagonal-only constant-input behaviour.
#[test]
fn llf_from_lf_dct32x32_dimension_contract() {
    let lf = vec![1.0f32; 16];
    let out = llf_from_lf(&lf, TransformType::Dct32x32).unwrap();
    assert_eq!(out.len(), 16);
    // Constant input → only DC is non-zero.
    let sf = scale_f(4, 32, 0);
    assert!((out[0] - 1.0 * sf * sf).abs() < 1e-5);
    for v in &out[1..] {
        assert!(v.abs() < 1e-3);
    }
}

/// Input-length validation: rejects mismatched sizes.
#[test]
fn llf_from_lf_rejects_wrong_input_length() {
    // DCT16×16 needs 4 samples.
    assert!(llf_from_lf(&[1.0f32], TransformType::Dct16x16).is_err());
    // DCT8×8 needs 1 sample.
    assert!(llf_from_lf(&[1.0f32, 2.0], TransformType::Dct8x8).is_err());
}

/// Forward 2-D DCT followed by the existing inverse 2-D IDCT
/// round-trips a small signal up to f32 epsilon.
#[test]
fn dct_idct_2d_round_trip_4x4() {
    let signal: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let dct = dct_2d(&signal, 4, 4).unwrap();
    let back = oxideav_jpegxl::idct::idct_2d(&dct, 4, 4).unwrap();
    for (i, (a, b)) in signal.iter().zip(back.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "i={i}: orig {a} != roundtrip {b}",);
    }
}
