//! Chroma-from-Luma (CfL) — ISO/IEC FDIS 18181-1:2021 Annex G
//! ("Chroma from luma", page 73) and ISO/IEC 18181-1:2024 §I.2.6.
//!
//! ## Scope (round 138)
//!
//! Round 138 lands the **per-sample Chroma-from-Luma reconstruction
//! formula** specified by Listing G.1: given dequantised
//! `(dX, dY, dB)` samples (the output of §F.2 / §F.3) and the colour-
//! correlation parameters from §C.4.4 (`LfChannelCorrelation`) plus,
//! for HF coefficients, the per-64×64-tile factor samples decoded
//! into [`crate::lf_group::HfMetadata`]'s `XFromY` / `BFromY`
//! channels, this module produces the final `(X, Y, B)` plane samples
//! that go on to Annex L colour transforms (and ultimately the
//! [`crate::xyb`] inverse).
//!
//! This is a pure-math primitive — the same shape as the round-89
//! [`crate::dct_quant_weights`], round-95 [`crate::hf_dequant`], and
//! round-121 [`crate::llf_from_lf`] steps already landed. The
//! VarDCT pipeline glue that drives this from a per-LfGroup loop is
//! deferred to a follow-up round; this round lands the bit-exact
//! arithmetic + the per-coefficient and per-plane application
//! helpers so a future round can wire it in without re-deriving any
//! G.1 formulae.
//!
//! ## Spec listing (FDIS page 73 — Annex G, normative)
//!
//! > This Annex only applies to var-DCT mode. This annex is skipped
//! > if any channel is subsampled.
//! >
//! > Each X, Y and B sample is reconstructed from the dequantized
//! > samples `dX`, `dY` and `dB` using a linear chroma from luma
//! > model. The reconstruction uses the colour correlation
//! > coefficient multipliers `kX` and `kB` (computed from constants
//! > defined in C.4.4) to restore the correlation between the X/B
//! > and the Y channel, as specified by Listing G.1.
//!
//! ```text
//! Listing G.1 — Chroma from luma
//! kX = base_correlation_x + x_factor / colour_factor;
//! kB = base_correlation_b + b_factor / colour_factor;
//! Y  = dY;
//! X  = dX + kX × Y;
//! B  = dB + kB × Y;
//! ```
//!
//! > For LF coefficients, `x_factor` and `b_factor` correspond to
//! > `x_factor_lf - 127` and `b_factor_lf - 127`, respectively
//! > (C.4.4). For HF coefficients, `x_factor` and `b_factor` are
//! > values from `XFromY` and `BFromY` (C.5.4), respectively, at the
//! > coordinates of the 64 × 64 rectangle containing the current
//! > sample.
//!
//! ## Implementation notes
//!
//! * **Operand widths.** `colour_factor` is `u32` (defaults to 84
//!   per §C.4.4 `all_default`). `base_correlation_x` and
//!   `base_correlation_b` are FDIS F16 values (defaults `0.0` and
//!   `1.0`). The per-sample `dX` / `dY` / `dB` values are f32 (the
//!   output of §F.2 LF dequantisation and §F.3 HF dequantisation,
//!   both of which already produce f32 in our pipeline). We carry
//!   `kX` and `kB` as f32 throughout to match downstream Annex L.
//! * **Per-block `kX` / `kB` precompute.** For HF samples the
//!   per-64×64-tile `(x_factor, b_factor)` come from the integer
//!   `i32` channels of [`crate::lf_group::HfMetadata`]. A typical
//!   driver caches `(kX, kB)` per tile and reuses it for every
//!   sample in that tile; we expose [`kx_kb_hf`] for the per-tile
//!   compute and [`apply_hf_sample`] / [`apply_hf_plane_tiled`] for
//!   the per-sample / per-plane application.
//! * **Subsampling guard.** The spec says CfL is skipped when "any
//!   channel is subsampled." This module does not know about the
//!   frame's chroma subsampling state — that lives in
//!   [`crate::frame_header::FrameHeader`] and the modular channel
//!   list. The caller is responsible for the skip; this module
//!   applies the formula unconditionally on the inputs it is given.
//! * **No bit reading.** Like the round-89 / 95 / 121 primitives,
//!   this module is a pure-math helper: it consumes already-decoded
//!   structures and produces float outputs. No `BitReader` use, no
//!   error paths beyond dim-mismatch on the plane helpers.
//!
//! ## What this module does NOT do
//!
//! * It does not decode the §C.5.4 HfMetadata sub-bitstream — that
//!   lives in [`crate::lf_group::HfMetadata::read`] (round 12+16).
//! * It does not parse the §C.4.4 `LfChannelCorrelation` bundle —
//!   that lives in
//!   [`crate::lf_global::LfChannelCorrelation::read`] (round 11).
//! * It does not drive the per-LfGroup loop or compose with the
//!   §I.2.5 LLF-from-LF step — the per-LfGroup wiring is a
//!   follow-up round's responsibility.
//! * It does not handle subsampled chroma (the spec excludes that
//!   case from CfL entirely).

use crate::lf_global::LfChannelCorrelation;
use oxideav_core::{Error, Result};

/// Compute the per-sample colour-correlation multipliers `(kX, kB)`
/// from a raw `(x_factor, b_factor)` pair and the bundle's
/// `base_correlation_*` / `colour_factor` parameters.
///
/// This is the first two lines of Listing G.1:
///
/// ```text
/// kX = base_correlation_x + x_factor / colour_factor;
/// kB = base_correlation_b + b_factor / colour_factor;
/// ```
///
/// Per the listing the division is float division (not integer
/// truncation) — `base_correlation_*` are float defaults `0.0` /
/// `1.0` and `kX` / `kB` are float multipliers consumed by the
/// `X = dX + kX × Y` / `B = dB + kB × Y` lines.
///
/// `colour_factor` is documented as a denominator and per §C.4.4
/// must be ≥ 1 (default 84; valid range covers `{84, 256, 2+u(8),
/// 258+u(16)}`). A pathological caller-supplied `colour_factor == 0`
/// is not produced by [`LfChannelCorrelation::read`] (the spec
/// `U32Dist` lower bound is 84 / 256 / 2 / 258); we still treat 0 as
/// `Error::InvalidData` defensively in [`kx_kb_lf`] / [`kx_kb_hf`].
///
/// The standalone `kx_kb_raw` helper exists so callers that already
/// know `colour_factor` is non-zero (e.g. driving from a parsed
/// [`LfChannelCorrelation`]) can stay on the infallible path.
#[inline]
pub fn kx_kb_raw(
    base_correlation_x: f32,
    base_correlation_b: f32,
    colour_factor: u32,
    x_factor: i32,
    b_factor: i32,
) -> (f32, f32) {
    let cf = colour_factor as f32;
    let kx = base_correlation_x + (x_factor as f32) / cf;
    let kb = base_correlation_b + (b_factor as f32) / cf;
    (kx, kb)
}

/// Compute `(kX, kB)` for **LF coefficients** per the second-to-last
/// paragraph of Annex G:
///
/// > For LF coefficients, `x_factor` and `b_factor` correspond to
/// > `x_factor_lf - 127` and `b_factor_lf - 127`, respectively.
///
/// The `x_factor_lf` / `b_factor_lf` u(8) fields default to `128`
/// (so the default LF `x_factor = b_factor = 1`).
pub fn kx_kb_lf(cfl: &LfChannelCorrelation) -> Result<(f32, f32)> {
    if cfl.colour_factor == 0 {
        return Err(Error::InvalidData(
            "JXL CfL: colour_factor == 0 (LfChannelCorrelation invariant violated)".into(),
        ));
    }
    let x_factor = cfl.x_factor_lf as i32 - 127;
    let b_factor = cfl.b_factor_lf as i32 - 127;
    Ok(kx_kb_raw(
        cfl.base_correlation_x,
        cfl.base_correlation_b,
        cfl.colour_factor,
        x_factor,
        b_factor,
    ))
}

/// Compute `(kX, kB)` for **HF coefficients** per the last paragraph
/// of Annex G:
///
/// > For HF coefficients, `x_factor` and `b_factor` are values from
/// > `XFromY` and `BFromY` (C.5.4), respectively, at the coordinates
/// > of the 64 × 64 rectangle containing the current sample.
///
/// The `x_factor_hf` / `b_factor_hf` arguments are the per-64×64-tile
/// samples (i32, as decoded into [`crate::lf_group::HfMetadata`]'s
/// `x_from_y` / `b_from_y` channels).
pub fn kx_kb_hf(
    cfl: &LfChannelCorrelation,
    x_factor_hf: i32,
    b_factor_hf: i32,
) -> Result<(f32, f32)> {
    if cfl.colour_factor == 0 {
        return Err(Error::InvalidData(
            "JXL CfL: colour_factor == 0 (LfChannelCorrelation invariant violated)".into(),
        ));
    }
    Ok(kx_kb_raw(
        cfl.base_correlation_x,
        cfl.base_correlation_b,
        cfl.colour_factor,
        x_factor_hf,
        b_factor_hf,
    ))
}

/// Apply the last three lines of Listing G.1 to a single `(dX, dY,
/// dB)` triple, given pre-computed `(kX, kB)`.
///
/// ```text
/// Y = dY;
/// X = dX + kX × Y;
/// B = dB + kB × Y;
/// ```
///
/// Returns `(X, Y, B)`. The Y output is just dY by definition (CfL
/// adjusts only X and B); we return it from the same function so a
/// per-pixel caller can take the three-channel result in one call.
#[inline]
pub fn apply_sample(dx: f32, dy: f32, db: f32, kx: f32, kb: f32) -> (f32, f32, f32) {
    let y = dy;
    let x = dx + kx * y;
    let b = db + kb * y;
    (x, y, b)
}

/// Apply Listing G.1 to a single LF sample triple. Combines
/// [`kx_kb_lf`] + [`apply_sample`] in one call for callers that
/// have not pre-computed the per-LfGroup `(kX, kB)` constants.
pub fn apply_lf_sample(
    dx: f32,
    dy: f32,
    db: f32,
    cfl: &LfChannelCorrelation,
) -> Result<(f32, f32, f32)> {
    let (kx, kb) = kx_kb_lf(cfl)?;
    Ok(apply_sample(dx, dy, db, kx, kb))
}

/// Apply Listing G.1 to a single HF sample triple, given the
/// `(x_factor_hf, b_factor_hf)` sampled from the 64×64-tile
/// containing the current sample (per §C.5.4 + Annex G last
/// paragraph).
pub fn apply_hf_sample(
    dx: f32,
    dy: f32,
    db: f32,
    cfl: &LfChannelCorrelation,
    x_factor_hf: i32,
    b_factor_hf: i32,
) -> Result<(f32, f32, f32)> {
    let (kx, kb) = kx_kb_hf(cfl, x_factor_hf, b_factor_hf)?;
    Ok(apply_sample(dx, dy, db, kx, kb))
}

/// Apply Listing G.1 to three equal-length LF planes in-place.
///
/// For LF the per-plane `kX` / `kB` are constants for the whole
/// frame (they come from the `LfChannelCorrelation` bundle in
/// [`crate::lf_global::LfGlobal`]), so this is a flat per-element
/// pass with no per-sample lookup.
///
/// On success `dx_plane` holds the final `X` plane, `db_plane`
/// holds the final `B` plane, and `dy_plane` is unchanged (Y output
/// equals dY).
///
/// Returns `Error::InvalidData` on dim-mismatch between the three
/// planes or on `colour_factor == 0`.
pub fn apply_lf_plane_inplace(
    dx_plane: &mut [f32],
    dy_plane: &[f32],
    db_plane: &mut [f32],
    cfl: &LfChannelCorrelation,
) -> Result<()> {
    if dx_plane.len() != dy_plane.len() || db_plane.len() != dy_plane.len() {
        return Err(Error::InvalidData(format!(
            "JXL CfL: LF plane length mismatch (X={}, Y={}, B={})",
            dx_plane.len(),
            dy_plane.len(),
            db_plane.len()
        )));
    }
    let (kx, kb) = kx_kb_lf(cfl)?;
    for i in 0..dy_plane.len() {
        let y = dy_plane[i];
        dx_plane[i] += kx * y;
        db_plane[i] += kb * y;
    }
    Ok(())
}

/// Apply Listing G.1 to three equal-length HF planes in-place,
/// looking up `(x_factor, b_factor)` from the per-64×64-tile sample
/// channels per the last paragraph of Annex G.
///
/// `width` and `height` are the per-plane dimensions of `dx_plane` /
/// `dy_plane` / `db_plane` in **pixels** (each plane has
/// `width * height` f32 samples in row-major order).
///
/// `x_from_y` / `b_from_y` are the per-64×64-tile factor channels
/// from [`crate::lf_group::HfMetadata`], each of dimensions
/// `ceil(width / 64) × ceil(height / 64)` (row-major). Per Annex G
/// the lookup is "at the coordinates of the 64 × 64 rectangle
/// containing the current sample," i.e. `tile_x = x / 64`,
/// `tile_y = y / 64`.
///
/// On success `dx_plane` holds the final `X` plane and `db_plane`
/// holds the final `B` plane; `dy_plane` is unchanged.
///
/// Returns `Error::InvalidData` on:
/// * plane length not equal to `width * height`;
/// * `dx_plane` / `dy_plane` / `db_plane` length mismatch;
/// * tile-plane length not equal to
///   `ceil(width / 64) * ceil(height / 64)`;
/// * `x_from_y` / `b_from_y` length mismatch;
/// * `colour_factor == 0`.
#[allow(clippy::too_many_arguments)]
pub fn apply_hf_plane_inplace(
    dx_plane: &mut [f32],
    dy_plane: &[f32],
    db_plane: &mut [f32],
    width: u32,
    height: u32,
    x_from_y: &[i32],
    b_from_y: &[i32],
    cfl: &LfChannelCorrelation,
) -> Result<()> {
    let w = width as usize;
    let h = height as usize;
    let expected_plane = w
        .checked_mul(h)
        .ok_or_else(|| Error::InvalidData("JXL CfL: width * height overflows usize".into()))?;
    if dx_plane.len() != expected_plane
        || dy_plane.len() != expected_plane
        || db_plane.len() != expected_plane
    {
        return Err(Error::InvalidData(format!(
            "JXL CfL: HF plane size != width * height = {} (got X={}, Y={}, B={})",
            expected_plane,
            dx_plane.len(),
            dy_plane.len(),
            db_plane.len()
        )));
    }
    let tw = width.div_ceil(64).max(1) as usize;
    let th = height.div_ceil(64).max(1) as usize;
    let expected_tile = tw
        .checked_mul(th)
        .ok_or_else(|| Error::InvalidData("JXL CfL: tile w * h overflows usize".into()))?;
    if x_from_y.len() != expected_tile || b_from_y.len() != expected_tile {
        return Err(Error::InvalidData(format!(
            "JXL CfL: HF factor-plane size != ceil(w/64)*ceil(h/64) = {} (got XFromY={}, \
             BFromY={})",
            expected_tile,
            x_from_y.len(),
            b_from_y.len()
        )));
    }
    if cfl.colour_factor == 0 {
        return Err(Error::InvalidData(
            "JXL CfL: colour_factor == 0 (LfChannelCorrelation invariant violated)".into(),
        ));
    }

    // Per-tile cache of (kX, kB): walk the tile grid once and
    // precompute the multipliers, then apply per-sample.
    let mut kx_kb: Vec<(f32, f32)> = Vec::with_capacity(expected_tile);
    for i in 0..expected_tile {
        let (kx, kb) = kx_kb_raw(
            cfl.base_correlation_x,
            cfl.base_correlation_b,
            cfl.colour_factor,
            x_from_y[i],
            b_from_y[i],
        );
        kx_kb.push((kx, kb));
    }

    for y in 0..h {
        let ty = y / 64;
        let row = y * w;
        for x in 0..w {
            let tx = x / 64;
            let (kx, kb) = kx_kb[ty * tw + tx];
            let yv = dy_plane[row + x];
            dx_plane[row + x] += kx * yv;
            db_plane[row + x] += kb * yv;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lf_global::LfChannelCorrelation;

    /// Default `LfChannelCorrelation` per §C.4.4 (the `all_default`
    /// branch): `colour_factor = 84`, `base_correlation_x = 0.0`,
    /// `base_correlation_b = 1.0`, `x_factor_lf = b_factor_lf = 128`.
    /// With the LF derivation `x_factor = b_factor = 1`, so
    /// `kX = 0 + 1/84 ≈ 0.01190`, `kB = 1 + 1/84 ≈ 1.01190`.
    #[test]
    fn lf_default_kx_kb() {
        let cfl = LfChannelCorrelation::default();
        let (kx, kb) = kx_kb_lf(&cfl).unwrap();
        let expected_kx = 1.0_f32 / 84.0;
        let expected_kb = 1.0_f32 + 1.0_f32 / 84.0;
        assert!((kx - expected_kx).abs() < 1e-7, "kX = {kx}");
        assert!((kb - expected_kb).abs() < 1e-7, "kB = {kb}");
    }

    /// Default HF factors (`x_factor_hf = b_factor_hf = 0`) collapse
    /// to the bundle's `base_correlation_*` constants directly:
    /// `kX = 0.0`, `kB = 1.0`.
    #[test]
    fn hf_default_factors_kx_kb() {
        let cfl = LfChannelCorrelation::default();
        let (kx, kb) = kx_kb_hf(&cfl, 0, 0).unwrap();
        assert_eq!(kx, 0.0);
        assert_eq!(kb, 1.0);
    }

    /// `kx_kb_raw` is linear in `x_factor` / `b_factor`: doubling
    /// the factor doubles the offset over `base_correlation_*`.
    ///
    /// Tolerance is relaxed for kB because subtracting `1.0` after
    /// adding `1/84` to it loses precision: `1.0 + 1/84` rounds to a
    /// nearby f32 representable value, and the cancellation in
    /// `kb1 - 1.0` exposes that rounding (~1 ulp at the f32 scale of
    /// `1/84 ≈ 0.0119`, i.e. roughly `2^-25 ≈ 3e-8`). Comparing the
    /// kX path (base 0.0, no cancellation) at the tighter tolerance
    /// pins the linearity property to f32 epsilon.
    #[test]
    fn raw_linear_in_factor() {
        let (kx1, kb1) = kx_kb_raw(0.0, 1.0, 84, 1, 1);
        let (kx2, kb2) = kx_kb_raw(0.0, 1.0, 84, 2, 2);
        assert!(((kx2 - 0.0) - 2.0 * (kx1 - 0.0)).abs() < 1e-7);
        assert!(((kb2 - 1.0) - 2.0 * (kb1 - 1.0)).abs() < 1e-6);
    }

    /// `kx_kb_raw` is inverse in `colour_factor`: at fixed factor,
    /// doubling `colour_factor` halves the offset.
    #[test]
    fn raw_inverse_in_colour_factor() {
        let (kx_84, _) = kx_kb_raw(0.0, 1.0, 84, 1, 0);
        let (kx_168, _) = kx_kb_raw(0.0, 1.0, 168, 1, 0);
        assert!((kx_84 - 2.0 * kx_168).abs() < 1e-7);
    }

    /// `apply_sample` is the identity on Y; Y output equals dY
    /// regardless of `(kX, kB)` or `(dX, dB)`.
    #[test]
    fn apply_sample_y_identity() {
        let (_, y, _) = apply_sample(123.0, 4.5, -6.7, 0.5, 1.5);
        assert_eq!(y, 4.5);
    }

    /// `apply_sample` Listing G.1 line 4: `X = dX + kX × Y`.
    /// At `kX = 0.5`, `Y = 10.0`, `dX = -3.0` we get
    /// `X = -3.0 + 0.5 * 10.0 = 2.0`.
    #[test]
    fn apply_sample_x_listing_g1_line_4() {
        let (x, _, _) = apply_sample(-3.0, 10.0, 0.0, 0.5, 0.0);
        assert!((x - 2.0).abs() < 1e-7);
    }

    /// `apply_sample` Listing G.1 line 5: `B = dB + kB × Y`.
    /// At `kB = 1.5`, `Y = 4.0`, `dB = -1.0` we get
    /// `B = -1.0 + 1.5 * 4.0 = 5.0`.
    #[test]
    fn apply_sample_b_listing_g1_line_5() {
        let (_, _, b) = apply_sample(0.0, 4.0, -1.0, 0.0, 1.5);
        assert!((b - 5.0).abs() < 1e-7);
    }

    /// At dY == 0 the X and B reconstructions degenerate to dX / dB
    /// (the `+ kX × Y` and `+ kB × Y` terms vanish).
    #[test]
    fn apply_sample_zero_y_passthrough() {
        let (x, _, b) = apply_sample(2.5, 0.0, -7.5, 0.5, 1.5);
        assert_eq!(x, 2.5);
        assert_eq!(b, -7.5);
    }

    /// CfL is invertible on (X, Y, B) ↔ (dX, dY, dB) at the formula
    /// level: applying the encoder-side decorrelation
    /// `dX = X - kX × Y`, `dB = B - kB × Y` and then `apply_sample`
    /// recovers (X, Y, B) exactly.
    #[test]
    fn apply_sample_round_trip_against_forward_decorrelation() {
        let kx = 0.25;
        let kb = 1.125;
        for &(x, y, b) in &[
            (1.0, 2.0, 3.0),
            (-1.0, 0.5, 0.0),
            (100.0, -50.0, 25.0),
            (0.0, 0.0, 0.0),
        ] {
            let dx = x - kx * y;
            let db = b - kb * y;
            let (xr, yr, br) = apply_sample(dx, y, db, kx, kb);
            assert!((xr - x).abs() < 1e-5, "x: got {xr} want {x}");
            assert_eq!(yr, y);
            assert!((br - b).abs() < 1e-5, "b: got {br} want {b}");
        }
    }

    /// `apply_lf_sample` matches `apply_sample(.., kx_kb_lf(cfl))`.
    #[test]
    fn apply_lf_sample_matches_precomputed() {
        let cfl = LfChannelCorrelation::default();
        let (kx, kb) = kx_kb_lf(&cfl).unwrap();
        let (xa, ya, ba) = apply_lf_sample(1.0, 2.0, 3.0, &cfl).unwrap();
        let (xb, yb, bb) = apply_sample(1.0, 2.0, 3.0, kx, kb);
        assert_eq!((xa, ya, ba), (xb, yb, bb));
    }

    /// `apply_hf_sample` matches `apply_sample(.., kx_kb_hf(..))`.
    #[test]
    fn apply_hf_sample_matches_precomputed() {
        let cfl = LfChannelCorrelation::default();
        let (kx, kb) = kx_kb_hf(&cfl, 3, -5).unwrap();
        let (xa, ya, ba) = apply_hf_sample(1.0, 2.0, 3.0, &cfl, 3, -5).unwrap();
        let (xb, yb, bb) = apply_sample(1.0, 2.0, 3.0, kx, kb);
        assert_eq!((xa, ya, ba), (xb, yb, bb));
    }

    /// `kx_kb_lf` rejects pathological `colour_factor == 0` even
    /// though a spec-conformant bundle never produces it.
    #[test]
    fn kx_kb_lf_rejects_zero_colour_factor() {
        let cfl = LfChannelCorrelation {
            all_default: false,
            colour_factor: 0,
            base_correlation_x: 0.0,
            base_correlation_b: 1.0,
            x_factor_lf: 128,
            b_factor_lf: 128,
        };
        assert!(kx_kb_lf(&cfl).is_err());
    }

    /// `kx_kb_hf` rejects pathological `colour_factor == 0`.
    #[test]
    fn kx_kb_hf_rejects_zero_colour_factor() {
        let cfl = LfChannelCorrelation {
            all_default: false,
            colour_factor: 0,
            base_correlation_x: 0.0,
            base_correlation_b: 1.0,
            x_factor_lf: 128,
            b_factor_lf: 128,
        };
        assert!(kx_kb_hf(&cfl, 0, 0).is_err());
    }

    /// `apply_lf_plane_inplace` matches a per-sample loop calling
    /// `apply_lf_sample` for every element.
    #[test]
    fn apply_lf_plane_inplace_matches_per_sample() {
        let cfl = LfChannelCorrelation::default();
        let dy = vec![0.0_f32, 1.0, 2.0, 3.0, -1.0, -2.0];
        let mut dx = vec![0.5_f32; 6];
        let mut db = vec![-0.5_f32; 6];
        let mut dx_ref = dx.clone();
        let mut db_ref = db.clone();
        for i in 0..6 {
            let (x, _, b) = apply_lf_sample(dx_ref[i], dy[i], db_ref[i], &cfl).unwrap();
            dx_ref[i] = x;
            db_ref[i] = b;
        }
        apply_lf_plane_inplace(&mut dx, &dy, &mut db, &cfl).unwrap();
        for i in 0..6 {
            assert!((dx[i] - dx_ref[i]).abs() < 1e-7);
            assert!((db[i] - db_ref[i]).abs() < 1e-7);
        }
    }

    /// `apply_lf_plane_inplace` returns `InvalidData` on dim mismatch.
    #[test]
    fn apply_lf_plane_inplace_rejects_dim_mismatch() {
        let cfl = LfChannelCorrelation::default();
        let mut dx = vec![0.0_f32; 3];
        let dy = vec![0.0_f32; 4];
        let mut db = vec![0.0_f32; 3];
        assert!(apply_lf_plane_inplace(&mut dx, &dy, &mut db, &cfl).is_err());
    }

    /// `apply_hf_plane_inplace` on a single-tile 64×64 plane (one
    /// tile per dim, single `(x_from_y, b_from_y)` entry) is
    /// equivalent to `apply_lf_plane_inplace` with the corresponding
    /// constant `(kX, kB)` (modulo the LF / HF factor derivation,
    /// which the test pins by hand).
    #[test]
    fn apply_hf_plane_inplace_single_tile() {
        let cfl = LfChannelCorrelation::default();
        let w = 64u32;
        let h = 64u32;
        let n = (w * h) as usize;
        let dy: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.5).collect();
        let mut dx = vec![0.25_f32; n];
        let mut db = vec![-0.25_f32; n];
        let x_from_y = vec![2_i32];
        let b_from_y = vec![-3_i32];

        let mut dx_ref = dx.clone();
        let mut db_ref = db.clone();
        let (kx, kb) = kx_kb_hf(&cfl, 2, -3).unwrap();
        for i in 0..n {
            let y = dy[i];
            dx_ref[i] += kx * y;
            db_ref[i] += kb * y;
        }

        apply_hf_plane_inplace(&mut dx, &dy, &mut db, w, h, &x_from_y, &b_from_y, &cfl).unwrap();
        for i in 0..n {
            assert!((dx[i] - dx_ref[i]).abs() < 1e-6);
            assert!((db[i] - db_ref[i]).abs() < 1e-6);
        }
    }

    /// On a 128×64 plane there are two 64×64 tiles laid out
    /// horizontally. Each half-plane gets a different `(kX, kB)`;
    /// `apply_hf_plane_inplace` must apply the correct tile to
    /// the correct sample range.
    #[test]
    fn apply_hf_plane_inplace_two_tiles_horizontal() {
        let cfl = LfChannelCorrelation::default();
        let w = 128u32;
        let h = 64u32;
        let n = (w * h) as usize;
        let dy = vec![1.0_f32; n];
        let mut dx = vec![0.0_f32; n];
        let mut db = vec![0.0_f32; n];
        // Tile (0,0) gets factor 4; tile (1,0) gets factor -4.
        let x_from_y = vec![4_i32, -4];
        let b_from_y = vec![0_i32, 0];

        apply_hf_plane_inplace(&mut dx, &dy, &mut db, w, h, &x_from_y, &b_from_y, &cfl).unwrap();

        let (kx_left, _) = kx_kb_hf(&cfl, 4, 0).unwrap();
        let (kx_right, _) = kx_kb_hf(&cfl, -4, 0).unwrap();
        // Sample at (x=0, y=0): tile (0,0) → kx_left × 1.0.
        assert!((dx[0] - kx_left).abs() < 1e-6);
        // Sample at (x=64, y=0): tile (1,0) → kx_right × 1.0.
        assert!((dx[64] - kx_right).abs() < 1e-6);
        // Sample at (x=63, y=63): still tile (0,0).
        assert!((dx[63 * 128 + 63] - kx_left).abs() < 1e-6);
        // Sample at (x=127, y=63): tile (1,0).
        assert!((dx[63 * 128 + 127] - kx_right).abs() < 1e-6);
    }

    /// `apply_hf_plane_inplace` returns `InvalidData` on wrong
    /// plane size.
    #[test]
    fn apply_hf_plane_inplace_rejects_plane_size_mismatch() {
        let cfl = LfChannelCorrelation::default();
        let mut dx = vec![0.0_f32; 10];
        let dy = vec![0.0_f32; 10];
        let mut db = vec![0.0_f32; 10];
        let x_from_y = vec![0_i32];
        let b_from_y = vec![0_i32];
        // width * height = 4 * 4 = 16, but planes are length 10.
        assert!(
            apply_hf_plane_inplace(&mut dx, &dy, &mut db, 4, 4, &x_from_y, &b_from_y, &cfl)
                .is_err()
        );
    }

    /// `apply_hf_plane_inplace` returns `InvalidData` on wrong
    /// tile-plane size.
    #[test]
    fn apply_hf_plane_inplace_rejects_tile_size_mismatch() {
        let cfl = LfChannelCorrelation::default();
        let n = (64 * 64) as usize;
        let mut dx = vec![0.0_f32; n];
        let dy = vec![0.0_f32; n];
        let mut db = vec![0.0_f32; n];
        // 64×64 plane → 1 tile expected; give 2.
        let x_from_y = vec![0_i32; 2];
        let b_from_y = vec![0_i32; 2];
        assert!(
            apply_hf_plane_inplace(&mut dx, &dy, &mut db, 64, 64, &x_from_y, &b_from_y, &cfl)
                .is_err()
        );
    }

    /// Non-default `LfChannelCorrelation` with `x_factor_lf = 130`
    /// (so derived `x_factor = 3`) and `colour_factor = 256`
    /// gives `kX = 0.0 + 3/256 = 0.01171875` exactly (representable
    /// in f32).
    #[test]
    fn kx_kb_lf_non_default_bundle() {
        let cfl = LfChannelCorrelation {
            all_default: false,
            colour_factor: 256,
            base_correlation_x: 0.0,
            base_correlation_b: 1.0,
            x_factor_lf: 130,
            b_factor_lf: 125, // → b_factor = -2 → kB = 1.0 + (-2)/256 = 0.9921875
        };
        let (kx, kb) = kx_kb_lf(&cfl).unwrap();
        assert_eq!(kx, 3.0_f32 / 256.0);
        assert_eq!(kb, 1.0 + (-2.0_f32) / 256.0);
    }
}
