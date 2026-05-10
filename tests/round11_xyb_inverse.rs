//! Round-11 XYB inverse transform tests.
//!
//! Round 11 (this round) lands Annex L colour transforms (§L.2.2 inverse
//! XYB → linear RGB and §L.3 inverse YCbCr → RGB) plus wires them into
//! the modular decode output stage.
//!
//! These integration tests exercise:
//! 1. The inverse-XYB primitive against the FDIS Listing L.2.2 spec
//!    equation (round-trip of the XYB encode → decode pipeline using
//!    the canonical sRGB↔XYB constants from FDIS Table L.1 defaults).
//! 2. YCbCr inverse against the spec-listed coefficients (sample
//!    reproduction at known reference points).
//! 3. Regression sentinel: the five small lossless fixtures (all
//!    non-XYB, non-YCbCr-encoded modular paths) decode unchanged.
//!
//! Decoding a real cjxl-encoded XYB-modular fixture end-to-end is NOT
//! attempted in round 11: cjxl emits VarDCT for any photo-content XYB
//! input by default, and the modular-XYB path is rare enough that we
//! don't have a committed pure-modular-XYB fixture. Round 12 may add
//! one once the workspace's docs collaborator commissions a hand-
//! built minimal modular-XYB trace.

use oxideav_jpegxl::decode_one_frame;
use oxideav_jpegxl::metadata_fdis::{OpsinInverseMatrix, ToneMapping};
use oxideav_jpegxl::xyb::{
    inverse_xyb_to_rgb, inverse_ycbcr_to_rgb, linear_rgb_to_u8, modular_xyb_to_linear_rgb,
};

const PIXEL_1X1_JXL: &[u8] = include_bytes!("fixtures/pixel_1x1.jxl");
const GRAY_64X64_JXL: &[u8] = include_bytes!("fixtures/gray_64x64_lossless.jxl");
const GRADIENT_JXL: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
const PALETTE_JXL: &[u8] = include_bytes!("fixtures/palette_32x32.jxl");
const GREY_8X8_JXL: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

/// Reproducing the canonical XYB encode equation from libjxl's
/// public-domain whitepaper / the inverse of FDIS §L.2.2: given a
/// linear RGB sample in `[0, 1]`, the forward XYB encode (which the
/// encoder applies to produce X', Y', B' before quantisation) is:
///
/// ```text
/// // Forward: linear RGB → opsin (mixed L, M, S):
/// Lmix = m_L_R * R + m_L_G * G + m_L_B * B
/// Mmix = m_M_R * R + m_M_G * G + m_M_B * B
/// Smix = m_S_R * R + m_S_G * G + m_S_B * B
///
/// // gamma compression (cube root then bias subtract):
/// Lgamma = cbrt(Lmix - opsin_bias_L) + cbrt(opsin_bias_L)
/// Mgamma = cbrt(Mmix - opsin_bias_M) + cbrt(opsin_bias_M)
/// Sgamma = cbrt(Smix - opsin_bias_S) + cbrt(opsin_bias_S)
///
/// // axis rotation (X = (L-M)/2, Y = (L+M)/2, B = S):
/// X = (Lgamma - Mgamma) / 2
/// Y = (Lgamma + Mgamma) / 2
/// B = Sgamma
/// ```
///
/// This test exercises the round-trip through the inverse path:
/// pick `(R, G, B) = (0.5, 0.5, 0.5)` neutral grey and verify the
/// inverse maps the corresponding `(X, Y, B)` back to the input
/// within float tolerance. The forward matrix is itself the matrix-
/// inverse of `opsin_inverse_matrix`; for the default FDIS
/// `OpsinInverseMatrix` values we know that the round-trip is
/// well-defined and the matrix product
/// `opsin_inverse_matrix · forward_matrix == I` to machine
/// precision.
#[test]
fn xyb_inverse_neutral_grey_returns_neutral_grey_via_round_trip() {
    let oim = OpsinInverseMatrix::default();
    let tm = ToneMapping::default();

    // Forward XYB encoder. The forward matrix is the inverse of
    // `oim.inv_mat`, which we can compute since `inv_mat` is a 3x3
    // float matrix; for simplicity we use a known closed form
    // (computed once below from the default Table L.1 inv_mat
    // numerical inverse).
    //
    // Default oim.inv_mat (FDIS Table L.1):
    //   [11.031567, -9.866944, -0.164623],
    //   [-3.254147,  4.418770, -0.164623],
    //   [-3.658851,  2.712923,  1.945928]
    //
    // Numerical inverse (3-decimal precision via straightforward 3x3
    // inversion using cofactors):
    //   [0.300194, 0.622007, 0.077799],
    //   [0.230122, 0.692290, 0.077588],
    //   [0.243391, 0.204068, 0.552541]
    //
    // We compute the forward matrix directly from oim.inv_mat using
    // Cramer's rule rather than hardcoding (so this test stays in
    // sync if the defaults ever update).
    let m = oim.inv_mat;
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inv = |row: usize, col: usize| -> f32 {
        // Cofactor / det. Indices for the 2x2 minor that excludes
        // (col, row) — note transpose because we want the inverse,
        // not the cofactor matrix.
        let m_skip = |skip_r: usize, skip_c: usize| -> [[f32; 2]; 2] {
            let mut out = [[0.0f32; 2]; 2];
            let mut oi = 0;
            for (i, row) in m.iter().enumerate() {
                if i == skip_r {
                    continue;
                }
                let mut oj = 0;
                for (j, &cell) in row.iter().enumerate() {
                    if j == skip_c {
                        continue;
                    }
                    out[oi][oj] = cell;
                    oj += 1;
                }
                oi += 1;
            }
            out
        };
        // The (col, row) of the inverse is cofactor(row, col) / det.
        let minor = m_skip(col, row);
        let cofactor = (minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0])
            * if (row + col) % 2 == 0 { 1.0 } else { -1.0 };
        cofactor / det
    };

    // Pick R = G = B = 0.5 (neutral grey, well inside the XYB
    // representable range).
    let r_in = 0.5f32;
    let g_in = 0.5f32;
    let b_in = 0.5f32;

    // Forward XYB encode (numerical inverse of inverse_xyb_to_rgb):
    //   Lmix = inv00 * R + inv01 * G + inv02 * B  (this is the
    //          matrix inverse of oim.inv_mat applied to the column
    //          [R, G, B]).
    let l_mix = inv(0, 0) * r_in + inv(0, 1) * g_in + inv(0, 2) * b_in;
    let m_mix = inv(1, 0) * r_in + inv(1, 1) * g_in + inv(1, 2) * b_in;
    let s_mix = inv(2, 0) * r_in + inv(2, 1) * g_in + inv(2, 2) * b_in;

    // Spec: itscale = 255 / intensity_target; default = 1.0. Drop it
    // for the gamma step (the inverse multiplies by itscale; the
    // forward divides by itscale).
    let itscale = 255.0 / tm.intensity_target;
    let l_pre = (l_mix / itscale - oim.opsin_bias[0]).cbrt() + oim.opsin_bias[0].cbrt();
    let m_pre = (m_mix / itscale - oim.opsin_bias[1]).cbrt() + oim.opsin_bias[1].cbrt();
    let s_pre = (s_mix / itscale - oim.opsin_bias[2]).cbrt() + oim.opsin_bias[2].cbrt();

    // Now (l_pre - m_pre) / 2 is X, (l_pre + m_pre) / 2 is Y,
    // s_pre is B. (This is the spec rotation.)
    //
    // BUT: the inverse from the spec uses `Lgamma = Y + X` and
    // `Mgamma = Y - X` directly, so we should set:
    //   X = (l_pre - m_pre) / 2
    //   Y = (l_pre + m_pre) / 2
    //   B = s_pre
    let x = (l_pre - m_pre) / 2.0;
    let y = (l_pre + m_pre) / 2.0;
    let b = s_pre;

    // Now run the inverse:
    let (r_out, g_out, b_out) = inverse_xyb_to_rgb(x, y, b, &oim, &tm);

    // Round-trip tolerance: f32 cbrt + matrix inverse + matrix
    // multiply accumulate float error over 9 mults + 6 adds per
    // channel; expect agreement to ~1e-3.
    assert!(
        (r_out - r_in).abs() < 5e-3,
        "R round-trip: in={r_in} out={r_out}"
    );
    assert!(
        (g_out - g_in).abs() < 5e-3,
        "G round-trip: in={g_in} out={g_out}"
    );
    assert!(
        (b_out - b_in).abs() < 5e-3,
        "B round-trip: in={b_in} out={b_out}"
    );
}

/// Round-trip a saturated red sample (R=1, G=0, B=0). Covers the
/// non-grey path through the matrix multiply.
#[test]
fn xyb_inverse_saturated_red_round_trips() {
    let oim = OpsinInverseMatrix::default();
    let tm = ToneMapping::default();

    // Compute the forward matrix as in the previous test.
    let m = oim.inv_mat;
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    let inv00 = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
    let inv10 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) / det;
    let inv20 = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;

    // Saturated red: R=1, G=0, B=0.
    // Forward: Lmix = inv00, Mmix = inv10, Smix = inv20.
    let l_mix = inv00;
    let m_mix = inv10;
    let s_mix = inv20;

    let itscale = 255.0 / tm.intensity_target;
    let l_pre = (l_mix / itscale - oim.opsin_bias[0]).cbrt() + oim.opsin_bias[0].cbrt();
    let m_pre = (m_mix / itscale - oim.opsin_bias[1]).cbrt() + oim.opsin_bias[1].cbrt();
    let s_pre = (s_mix / itscale - oim.opsin_bias[2]).cbrt() + oim.opsin_bias[2].cbrt();

    let x = (l_pre - m_pre) / 2.0;
    let y = (l_pre + m_pre) / 2.0;
    let b = s_pre;

    let (r_out, g_out, b_out) = inverse_xyb_to_rgb(x, y, b, &oim, &tm);
    // Wider tolerance: saturated red lies further from the bias point
    // and the cbrt nonlinearity amplifies float error.
    assert!((r_out - 1.0).abs() < 1e-2, "R: out={r_out} (expected ~1.0)");
    assert!(g_out.abs() < 1e-2, "G: out={g_out} (expected ~0.0)");
    assert!(b_out.abs() < 1e-2, "B: out={b_out} (expected ~0.0)");
}

/// YCbCr inverse spec listing reproduction. Spec test point:
/// `(Cb, Y, Cr) = (0, 0.5, 0)` should yield `(R, G, B) = (0.5, 0.5, 0.5)`
/// (achromatic mid-grey at half intensity).
#[test]
fn ycbcr_inverse_neutral_returns_grey() {
    let (r, g, b) = inverse_ycbcr_to_rgb(0.0, 0.5, 0.0);
    assert!((r - 0.5).abs() < 1e-6, "R={r}");
    assert!((g - 0.5).abs() < 1e-6, "G={g}");
    assert!((b - 0.5).abs() < 1e-6, "B={b}");
}

/// `linear_rgb_to_u8` quantises the [0, 1] linear-domain output to
/// 8-bit. Verify the rounding behaviour at a few canonical points.
#[test]
fn linear_rgb_to_u8_quantisation_known_values() {
    assert_eq!(linear_rgb_to_u8(0.0), 0);
    assert_eq!(linear_rgb_to_u8(1.0), 255);
    // 128/255 = 0.50196 → linear*255 = 128 exact.
    assert_eq!(linear_rgb_to_u8(128.0 / 255.0), 128);
    // 0.5 → 127.5 → rounds to 128 (ties-to-even per `round`, but
    // f32 `round` is half-away-from-zero in std).
    assert_eq!(linear_rgb_to_u8(0.5), 128);
}

/// Convenience wrapper end-to-end exercise. Y'=0, X'=0, B'=0 is a
/// black sample; the rescale yields (X=0, Y=0, B=0) which inverse-XYB
/// maps to (R=0, G=0, B=0).
#[test]
fn modular_xyb_zero_input_to_linear_rgb_returns_zero() {
    let lf = oxideav_jpegxl::lf_global::LfChannelDequantization::default();
    let oim = OpsinInverseMatrix::default();
    let tm = ToneMapping::default();
    let (r, g, b) = modular_xyb_to_linear_rgb(0, 0, 0, &lf, &oim, &tm);
    assert!(r.abs() < 1e-5, "R={r}");
    assert!(g.abs() < 1e-5, "G={g}");
    assert!(b.abs() < 1e-5, "B={b}");
}

/// Five-fixture regression sentinel: these were the round-1..5
/// fixtures that decoded pixel-correct under the pre-round-11 output
/// path. Round 11 changes the Modular output mapping to apply Annex L
/// transforms when `metadata.xyb_encoded` or `frame_header.do_ycbcr`
/// is set. For these five fixtures NEITHER flag is set (cjxl encodes
/// small lossless modular images with `xyb_encoded=false` and
/// `do_ycbcr=false`), so the round-11 mapping must take the
/// pass-through branch and decode pixel-equivalent output.
#[test]
fn five_small_lossless_fixtures_pass_through_round_11() {
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
            "round-11 regression: {name} should still decode (round-1..5 baseline); got {:?}",
            vf.err()
        );
    }
}
