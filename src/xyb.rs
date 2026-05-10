//! Annex L colour transforms — ISO/IEC 18181-1:2024.
//!
//! Round 11 (2024-spec) wires the inverse XYB transform (§L.2.2) and
//! the inverse YCbCr transform (§L.3) end-to-end.
//!
//! ## §L.2 XYB
//!
//! For frames with `metadata.xyb_encoded == true` the decoded sample
//! values are XYB-domain (X', Y', B' for `frame_header.encoding ==
//! kModular`; X, Y, B post-IDCT for `kVarDCT`). The inverse transform
//! produces linear sRGB-primary, D65-white-point samples in the range
//! `[0, 1]` (display-referred, with `1.0 == intensity_target cd/m^2`).
//!
//! Spec listing (§L.2.2 verbatim):
//!
//! ```text
//! Lgamma = Y + X;
//! Mgamma = Y - X;
//! Sgamma = B;
//! itscale = 255 / metadata.tone_mapping.intensity_target;
//! Lmix = (pow(Lgamma - cbrt(oim.opsin_bias0), 3) + oim.opsin_bias0) * itscale;
//! Mmix = (pow(Mgamma - cbrt(oim.opsin_bias1), 3) + oim.opsin_bias1) * itscale;
//! Smix = (pow(Sgamma - cbrt(oim.opsin_bias2), 3) + oim.opsin_bias2) * itscale;
//! R = oim.inv_mat00 * Lmix + oim.inv_mat01 * Mmix + oim.inv_mat02 * Smix;
//! G = oim.inv_mat10 * Lmix + oim.inv_mat11 * Mmix + oim.inv_mat12 * Smix;
//! B = oim.inv_mat20 * Lmix + oim.inv_mat21 * Mmix + oim.inv_mat22 * Smix;
//! ```
//!
//! For `frame_header.encoding == kModular` the spec's preamble first
//! rescales the integer `Y' / X' / B'` channels:
//!
//! ```text
//! X = X' * m_x_lf_unscaled
//! Y = Y' * m_y_lf_unscaled
//! B = (B' + Y') * m_b_lf_unscaled
//! ```
//!
//! Where `m_x_lf_unscaled, m_y_lf_unscaled, m_b_lf_unscaled` come from
//! the LfChannelDequantization bundle (FDIS Table C.11).
//!
//! ## §L.3 YCbCr
//!
//! For `frame_header.do_YCbCr == true` (which can only be set when
//! `metadata.xyb_encoded == false`), the first three channels are
//! interpreted as `(Cb, Y, Cr)` and converted to RGB:
//!
//! ```text
//! R = Y + 128/255 * 1.402 * Cr;
//! G = Y + 128/255 * (-0.344136 * Cb - 0.714136 * Cr);
//! B = Y + 128/255 * 1.772 * Cb;
//! ```
//!
//! Note: the spec uses the cb/cr-centered convention where chroma
//! samples have the offset `128/255` applied during conversion (the
//! samples themselves carry the raw `(stored_value - 128)` signed
//! offset implicitly when input is unsigned). Following the spec
//! verbatim, we apply the formula as written.
//!
//! ## §L.4 Extra-channel rendering
//!
//! Out of round-11 scope (alpha blending, kSpotColour overlay). Round
//! 11 only wires §L.2 + §L.3.
//!
//! ## Wall enumeration
//!
//! Implemented strictly from §L.2.2 / §L.3 verbatim plus the §L.2.2
//! preamble for the `kModular` rescale step. No external library
//! source consulted; OpsinInverseMatrix defaults match the
//! `metadata_fdis::OpsinInverseMatrix::default()` constants
//! independently transcribed from FDIS Table L.1.

use crate::lf_global::LfChannelDequantization;
use crate::metadata_fdis::{OpsinInverseMatrix, ToneMapping};

/// Inverse XYB → linear RGB per §L.2.2. `(x, y, b)` are the
/// post-rescale XYB samples (i.e. for `kModular` callers, the
/// caller has already applied the §L.2.2 preamble's `m_*_lf_unscaled`
/// multipliers; for `kVarDCT` callers, the values come straight out
/// of the IDCT). Returns `(R, G, B)` linear samples in display-
/// referred units (1.0 == `intensity_target cd/m^2`).
///
/// `oim` carries the inv_mat / opsin_bias / quant_bias_numerator
/// values from the metadata (defaults from FDIS Table L.1 if the
/// encoder didn't override).
///
/// `tone_mapping.intensity_target` parameterises the `itscale`
/// rescale; defaults to 255.0 (so itscale = 1.0).
pub fn inverse_xyb_to_rgb(
    x: f32,
    y: f32,
    b: f32,
    oim: &OpsinInverseMatrix,
    tone_mapping: &ToneMapping,
) -> (f32, f32, f32) {
    let l_gamma = y + x;
    let m_gamma = y - x;
    let s_gamma = b;
    // Spec: itscale = 255 / metadata.tone_mapping.intensity_target.
    // Defaults: intensity_target = 255 → itscale = 1.0.
    let itscale = 255.0 / tone_mapping.intensity_target;
    let l_mix = pow3_minus_cbrt_bias(l_gamma, oim.opsin_bias[0], itscale);
    let m_mix = pow3_minus_cbrt_bias(m_gamma, oim.opsin_bias[1], itscale);
    let s_mix = pow3_minus_cbrt_bias(s_gamma, oim.opsin_bias[2], itscale);
    let r = oim.inv_mat[0][0] * l_mix + oim.inv_mat[0][1] * m_mix + oim.inv_mat[0][2] * s_mix;
    let g = oim.inv_mat[1][0] * l_mix + oim.inv_mat[1][1] * m_mix + oim.inv_mat[1][2] * s_mix;
    let b_out = oim.inv_mat[2][0] * l_mix + oim.inv_mat[2][1] * m_mix + oim.inv_mat[2][2] * s_mix;
    (r, g, b_out)
}

/// Helper: spec inner term
/// `(pow(gamma - cbrt(opsin_bias), 3) + opsin_bias) * itscale`.
///
/// Pulled out so the three channel computations are unmistakably
/// identical (avoids a copy-paste bias bug).
#[inline]
fn pow3_minus_cbrt_bias(gamma: f32, opsin_bias: f32, itscale: f32) -> f32 {
    let inner = gamma - opsin_bias.cbrt();
    let cubed = inner * inner * inner;
    (cubed + opsin_bias) * itscale
}

/// Apply the §L.2.2 preamble for `frame_header.encoding == kModular`:
/// rescale the integer Y'/X'/B' channel samples into XYB-domain
/// floats. Returns `(X, Y, B)` ready for [`inverse_xyb_to_rgb`].
///
/// Channel order on input: `(y_prime, x_prime, b_prime)` matches the
/// JXL Modular convention "first three channels are Y', X', B'" per
/// FDIS §L.2.2 first paragraph.
pub fn modular_xyb_rescale(
    y_prime: i32,
    x_prime: i32,
    b_prime: i32,
    lf_dequant: &LfChannelDequantization,
) -> (f32, f32, f32) {
    let x = (x_prime as f32) * lf_dequant.m_x_lf_unscaled;
    let y = (y_prime as f32) * lf_dequant.m_y_lf_unscaled;
    let b = ((b_prime + y_prime) as f32) * lf_dequant.m_b_lf_unscaled;
    (x, y, b)
}

/// Convenience wrapper combining [`modular_xyb_rescale`] +
/// [`inverse_xyb_to_rgb`]. Returns linear RGB ready for clamping +
/// quantisation to the output bit depth.
pub fn modular_xyb_to_linear_rgb(
    y_prime: i32,
    x_prime: i32,
    b_prime: i32,
    lf_dequant: &LfChannelDequantization,
    oim: &OpsinInverseMatrix,
    tone_mapping: &ToneMapping,
) -> (f32, f32, f32) {
    let (x, y, b) = modular_xyb_rescale(y_prime, x_prime, b_prime, lf_dequant);
    inverse_xyb_to_rgb(x, y, b, oim, tone_mapping)
}

/// Inverse YCbCr → RGB per §L.3. Spec listing verbatim:
///
/// ```text
/// R = Y + 128.0 / 255 * 1.402 * Cr;
/// G = Y + 128.0 / 255 * (-0.344136 * Cb - 0.714136 * Cr);
/// B = Y + 128.0 / 255 * 1.772 * Cb;
/// ```
///
/// `(cb, y, cr)` parameter order matches the FDIS first-paragraph
/// statement "the values (Cb, Y, Cr) are replaced by (R, G, B)".
pub fn inverse_ycbcr_to_rgb(cb: f32, y: f32, cr: f32) -> (f32, f32, f32) {
    let scale = 128.0 / 255.0;
    let r = y + scale * 1.402 * cr;
    let g = y + scale * (-0.344_136 * cb - 0.714_136 * cr);
    let b = y + scale * 1.772 * cb;
    (r, g, b)
}

/// Linear-domain RGB → unsigned 8-bit clamped. The output of
/// [`inverse_xyb_to_rgb`] is display-referred linear in `[0, 1]`
/// (under default `intensity_target == 255`); we clamp + round +
/// scale to `0..=255` for 8-bit output.
///
/// Round-11 ships this helper for the modular-XYB output path. Per
/// §L.2.2 NOTE the spec output is "linear" — strict conformance
/// would require gamma encoding before display; we accept the
/// linear-output simplification and document it as an XYB output-
/// gamma SPECGAP (cleanest conformance handoff is to a downstream
/// colour-management consumer; this crate's job is decode, not
/// display).
pub fn linear_rgb_to_u8(linear: f32) -> u8 {
    let scaled = (linear.clamp(0.0, 1.0) * 255.0).round();
    scaled as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// XYB at (X=0, Y=0, B=0) under default OpsinInverseMatrix:
    /// L_gamma = M_gamma = 0; cbrt(opsin_bias) for the three biases is
    /// the same negative number (since all three biases are equal at
    /// default), so L_mix = M_mix = S_mix = (pow(-cbrt(bias), 3) +
    /// bias) * itscale = (-bias + bias) * itscale = 0.
    /// Therefore R = G = B = 0.
    #[test]
    fn xyb_zero_input_yields_zero_rgb() {
        let oim = OpsinInverseMatrix::default();
        let tm = ToneMapping::default();
        let (r, g, b) = inverse_xyb_to_rgb(0.0, 0.0, 0.0, &oim, &tm);
        assert!(approx_eq(r, 0.0, 1e-5), "R={r}");
        assert!(approx_eq(g, 0.0, 1e-5), "G={g}");
        assert!(approx_eq(b, 0.0, 1e-5), "B={b}");
    }

    /// Verify the §L.2.2 listing transcription is faithful: pick a
    /// non-trivial XYB triple and reproduce the spec computation
    /// step-by-step inline. This is the spec-equation oracle: any
    /// future drift in `inverse_xyb_to_rgb` against the listing
    /// trips this test.
    #[test]
    fn xyb_spec_listing_matches_handcomputed() {
        let oim = OpsinInverseMatrix::default();
        let tm = ToneMapping::default(); // intensity_target = 255 → itscale = 1.0

        // Pick: X=0.05, Y=0.5, B=0.4
        let x = 0.05f32;
        let y = 0.5f32;
        let b = 0.4f32;

        // Spec listing inline reproduction:
        let l_gamma = y + x; // 0.55
        let m_gamma = y - x; // 0.45
        let s_gamma = b; // 0.4
        let itscale = 255.0 / tm.intensity_target; // 1.0
        let cb0 = oim.opsin_bias[0].cbrt();
        let cb1 = oim.opsin_bias[1].cbrt();
        let cb2 = oim.opsin_bias[2].cbrt();
        let l_mix = ((l_gamma - cb0).powi(3) + oim.opsin_bias[0]) * itscale;
        let m_mix = ((m_gamma - cb1).powi(3) + oim.opsin_bias[1]) * itscale;
        let s_mix = ((s_gamma - cb2).powi(3) + oim.opsin_bias[2]) * itscale;
        let r_exp =
            oim.inv_mat[0][0] * l_mix + oim.inv_mat[0][1] * m_mix + oim.inv_mat[0][2] * s_mix;
        let g_exp =
            oim.inv_mat[1][0] * l_mix + oim.inv_mat[1][1] * m_mix + oim.inv_mat[1][2] * s_mix;
        let b_exp =
            oim.inv_mat[2][0] * l_mix + oim.inv_mat[2][1] * m_mix + oim.inv_mat[2][2] * s_mix;

        let (r, g, b_out) = inverse_xyb_to_rgb(x, y, b, &oim, &tm);
        assert!(approx_eq(r, r_exp, 1e-5), "R: got {r} exp {r_exp}");
        assert!(approx_eq(g, g_exp, 1e-5), "G: got {g} exp {g_exp}");
        assert!(approx_eq(b_out, b_exp, 1e-5), "B: got {b_out} exp {b_exp}");
    }

    /// `intensity_target` directly scales the L/M/S mix samples: doubling
    /// it should HALVE the resulting RGB amplitudes (linearly through
    /// the mix → matrix multiply chain).
    #[test]
    fn xyb_intensity_target_scales_output_linearly() {
        let oim = OpsinInverseMatrix::default();
        let tm_lo = ToneMapping {
            intensity_target: 255.0,
            ..ToneMapping::default()
        };
        let tm_hi = ToneMapping {
            intensity_target: 510.0,
            ..ToneMapping::default()
        };

        let (r1, g1, b1) = inverse_xyb_to_rgb(0.1, 0.4, 0.3, &oim, &tm_lo);
        let (r2, g2, b2) = inverse_xyb_to_rgb(0.1, 0.4, 0.3, &oim, &tm_hi);

        // r2 should equal r1 / 2 (since itscale halves: 255/510 = 0.5).
        assert!(approx_eq(r2, r1 * 0.5, 1e-5), "R: r1={r1} r2={r2}");
        assert!(approx_eq(g2, g1 * 0.5, 1e-5), "G: g1={g1} g2={g2}");
        assert!(approx_eq(b2, b1 * 0.5, 1e-5), "B: b1={b1} b2={b2}");
    }

    /// kModular preamble: rescale `(Y', X', B')` integer samples by
    /// `m_*_lf_unscaled`. The B channel adds Y' before scaling per
    /// §L.2.2 first paragraph.
    #[test]
    fn modular_xyb_rescale_applies_lf_multipliers() {
        let lf = LfChannelDequantization::default();
        // Y'=10, X'=2, B'=3. With defaults m_x=4096, m_y=512, m_b=256:
        //   X = 2 * 4096 = 8192
        //   Y = 10 * 512 = 5120
        //   B = (3 + 10) * 256 = 3328
        let (x, y, b) = modular_xyb_rescale(10, 2, 3, &lf);
        assert!(approx_eq(x, 8192.0, 1e-3), "X={x}");
        assert!(approx_eq(y, 5120.0, 1e-3), "Y={y}");
        assert!(approx_eq(b, 3328.0, 1e-3), "B={b}");
    }

    /// Modular path produces a sane RGB triple from a small +ve XYB
    /// integer triple. The actual RGB values are governed by the
    /// listing; this test just exercises the convenience wrapper end-
    /// to-end and verifies `(0, 0, 0)` round-trips to `(0, 0, 0)`.
    #[test]
    fn modular_xyb_zero_input_yields_zero_rgb() {
        let lf = LfChannelDequantization::default();
        let oim = OpsinInverseMatrix::default();
        let tm = ToneMapping::default();
        let (r, g, b) = modular_xyb_to_linear_rgb(0, 0, 0, &lf, &oim, &tm);
        assert!(approx_eq(r, 0.0, 1e-5), "R={r}");
        assert!(approx_eq(g, 0.0, 1e-5), "G={g}");
        assert!(approx_eq(b, 0.0, 1e-5), "B={b}");
    }

    /// YCbCr inverse: spec listing verbatim with chosen sample.
    /// (Cb=0, Y=0.5, Cr=0) → R = G = B = 0.5 (no chroma offset).
    #[test]
    fn ycbcr_zero_chroma_yields_grey() {
        let (r, g, b) = inverse_ycbcr_to_rgb(0.0, 0.5, 0.0);
        assert!(approx_eq(r, 0.5, 1e-6), "R={r}");
        assert!(approx_eq(g, 0.5, 1e-6), "G={g}");
        assert!(approx_eq(b, 0.5, 1e-6), "B={b}");
    }

    /// YCbCr inverse with positive Cr increases R, decreases G:
    /// R = Y + 128/255 * 1.402 * Cr  → +ve delta
    /// G = Y + 128/255 * (-0.714136 * Cr) → -ve delta
    /// B = Y (Cb=0)
    #[test]
    fn ycbcr_positive_cr_red_dominant() {
        let (r, g, b) = inverse_ycbcr_to_rgb(0.0, 0.5, 0.1);
        // R = 0.5 + (128/255) * 1.402 * 0.1 = 0.5 + 0.07037...
        let scale = 128.0 / 255.0;
        let r_exp = 0.5 + scale * 1.402 * 0.1;
        let g_exp = 0.5 + scale * (-0.714_136 * 0.1);
        let b_exp = 0.5;
        assert!(approx_eq(r, r_exp, 1e-5), "R={r} exp {r_exp}");
        assert!(approx_eq(g, g_exp, 1e-5), "G={g} exp {g_exp}");
        assert!(approx_eq(b, b_exp, 1e-5), "B={b} exp {b_exp}");
    }

    /// `linear_rgb_to_u8` clamps below 0 and above 1.
    #[test]
    fn linear_rgb_to_u8_clamps() {
        assert_eq!(linear_rgb_to_u8(-0.5), 0);
        assert_eq!(linear_rgb_to_u8(0.0), 0);
        assert_eq!(linear_rgb_to_u8(1.0), 255);
        assert_eq!(linear_rgb_to_u8(2.0), 255);
        assert_eq!(linear_rgb_to_u8(0.5), 128);
    }

    /// The spec gives Lgamma/Mgamma a symmetric structure: swapping
    /// the sign of X swaps L and M roles, so feeding `(-X, Y, B)`
    /// should be equivalent to running the equation with L and M
    /// expressions swapped (i.e., reading inv_mat columns 0 and 1
    /// swapped).
    #[test]
    fn xyb_x_sign_flip_swaps_lgamma_mgamma() {
        let oim = OpsinInverseMatrix::default();
        let tm = ToneMapping::default();
        let (rp, gp, bp) = inverse_xyb_to_rgb(0.1, 0.5, 0.3, &oim, &tm);
        let (rm, gm, bm) = inverse_xyb_to_rgb(-0.1, 0.5, 0.3, &oim, &tm);
        // With default opsin_bias (all three equal), L_mix and M_mix
        // are identical functions f(L_gamma) and f(M_gamma); flipping
        // the X sign swaps L_gamma and M_gamma, so RGB at -X is
        // computed with l_mix and m_mix swapped — i.e. inv_mat
        // columns 0 and 1 swap roles. Verify by recomputing.
        let l_gamma_p = 0.5 + 0.1; // 0.6
        let m_gamma_p = 0.5 - 0.1; // 0.4
        let l_gamma_m = 0.5 + (-0.1); // 0.4
        let m_gamma_m = 0.5 - (-0.1); // 0.6
                                      // l_gamma_m == m_gamma_p, m_gamma_m == l_gamma_p. So with all
                                      // three biases equal, swapping X sign should produce the same
                                      // RGB output as keeping X and swapping inv_mat columns 0/1
                                      // — but more directly, since L_mix(L_gamma_p) == M_mix(M_gamma_m)
                                      // due to identical biases, we get:
                                      //   r_p = im00 * f(0.6) + im01 * f(0.4) + im02 * f(0.3)
                                      //   r_m = im00 * f(0.4) + im01 * f(0.6) + im02 * f(0.3)
                                      // i.e. r_p - r_m = (im00 - im01) * (f(0.6) - f(0.4)).
        let cb0 = oim.opsin_bias[0].cbrt();
        let f = |gamma: f32| ((gamma - cb0).powi(3) + oim.opsin_bias[0]) * 1.0_f32;
        let f06 = f(l_gamma_p);
        let f04 = f(m_gamma_p);
        let _ = (l_gamma_m, m_gamma_m); // silence the temp bindings
        let dr_exp = (oim.inv_mat[0][0] - oim.inv_mat[0][1]) * (f06 - f04);
        let dg_exp = (oim.inv_mat[1][0] - oim.inv_mat[1][1]) * (f06 - f04);
        let db_exp = (oim.inv_mat[2][0] - oim.inv_mat[2][1]) * (f06 - f04);
        assert!(
            approx_eq(rp - rm, dr_exp, 1e-4),
            "ΔR: {} exp {}",
            rp - rm,
            dr_exp
        );
        assert!(
            approx_eq(gp - gm, dg_exp, 1e-4),
            "ΔG: {} exp {}",
            gp - gm,
            dg_exp
        );
        assert!(
            approx_eq(bp - bm, db_exp, 1e-4),
            "ΔB: {} exp {}",
            bp - bm,
            db_exp
        );
    }
}
