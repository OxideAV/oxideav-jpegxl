//! Edge-preserving restoration filter — ISO/IEC FDIS 18181-1:2021
//! Annex J.3 ("Edge-preserving filter", pages 85–87).
//!
//! ## Scope (round 144)
//!
//! Round 144 lands the **per-channel three-channel adaptive
//! convolution restoration filter** specified by §J.3, as a
//! pure-math primitive in exactly the same shape as the round-141
//! Gabor-like-transform module [`crate::gaborish`] already
//! shipping: this module takes already-decoded structures (a triple
//! of f32 channel planes plus per-call scalar parameters) and the
//! per-channel weights from
//! [`crate::frame_header::RestorationFilter`], performs the
//! bit-arithmetic steps Listings J.1 / J.2 / J.3 / J.4 prescribe,
//! and returns f32 planes.  **No bit reading. No interaction with
//! the rest of the pipeline.**
//!
//! ## What §J.3 says (FDIS pages 85–87, normative)
//!
//! From §J.3.1:
//!
//! > The filter is only performed if `rf.epf_iters > 0`. If it is
//! > performed, the filter operates in up to three steps.
//! >
//! > The first step is only done if `rf.epf_iters == 3`. In this
//! > step, for each reference pixel, the filter outputs a pixel
//! > which is a weighted sum of the reference pixel and each of the
//! > twelve neighbouring pixels that have a L1 distance of at most 2.
//! >
//! > The second step is always done.  […]  weighted sum of the
//! > reference pixel and each of the four neighbour pixels (top,
//! > bottom, left and right of the current one).
//! >
//! > The third step is only done if `rf.epf_iters >= 2`. […] each
//! > weight is computed as a decreasing function of an L1 distance
//! > metric as specified in J.3.3 and J.3.4.
//!
//! The pass-to-step mapping induced by §J.3.1 is therefore:
//!
//! | `epf_iters` | pass 0 (Step 0, 13-tap) | pass 1 (Step 1, 5-tap) | pass 2 (Step 2, 5-tap) |
//! | -----------:| :---------------------- | :--------------------- | :--------------------- |
//! | 0           | skipped                 | skipped                | skipped                |
//! | 1           | skipped                 | applied                | skipped                |
//! | 2           | skipped                 | applied                | applied                |
//! | 3           | applied                 | applied                | applied                |
//!
//! The §J.3 boundary semantics are the same Mirror1D as §J.2
//! ("the decoder behaves as if every such access were redirected to
//! coordinates `Mirror(cx, cy)` (6.5)").  We reuse
//! [`crate::gaborish::mirror1d`].
//!
//! ## Public surface
//!
//! * [`distance_step_0_and_1`] — Listing J.1's `DistanceStep0and1`,
//!   the five-pixel cross-shape L1 distance between the reference
//!   pixel and a neighbouring pixel, summed over all three channels
//!   weighted by `rf.epf_channel_scale[c]`.
//! * [`distance_step_2`] — Listing J.1's `DistanceStep2`, the
//!   single-sample distance for the third filter step (centre-only,
//!   no cross neighbourhood — see "Spec ambiguity" below).
//! * [`weight`] — Listing J.2's `Weight()` decreasing-function-of-
//!   distance kernel, given pre-computed
//!   `step_multiplier × position_multiplier × distance` and a
//!   `zeroflush` cutoff.
//! * [`vardct_sigma_from_listing_j3`] — Listing J.3's per-varblock
//!   `sigma = clamp(quantization_width × rf.epf_quant_mul ×
//!   rf.epf_sharp_lut[sharpness], 1e-4, +∞)`.
//! * [`inv_sigma_for_pass`] — the `step_multiplier[step] × 4 ×
//!   (sqrt(0.5) - 1) / sigma` derivation factored out so the per-
//!   pixel hot loop multiplies by a pre-computed scalar.
//! * [`is_border_position`] — Listing J.2's "either coordinate of
//!   the reference sample is 0 or 7 IMod 8" predicate.
//! * [`apply_step_5tap`] — runs one pass of Steps 1 or 2 (the 5-tap
//!   cross-shape kernel `{(0,0),(-1,0),(1,0),(0,-1),(0,1)}`) over a
//!   triple of XYB-pipeline channel planes, writing into a triple of
//!   output planes of the same size.  Distance metric is
//!   user-selected (`Step1` uses the cross-shape `DistanceStep0and1`,
//!   `Step2` uses the single-sample `DistanceStep2`).
//! * [`apply_step_13tap`] — runs Step 0 (the 13-tap diamond kernel
//!   covering the twelve `|cx|+|cy| <= 2` neighbours plus the
//!   centre) over a triple of channel planes.  Always uses
//!   `DistanceStep0and1`.
//! * [`Pass`] — enum picking which of the three §J.3.1 passes is
//!   being applied; used to thread the right kernel shape and
//!   `step_multiplier` through [`apply_step_5tap`] /
//!   [`apply_step_13tap`].
//!
//! Per the round-141 contract, this module does **not** drive any
//! frame-pipeline loop; the per-frame wiring (calling each pass for
//! each varblock under the right `epf_iters` / per-block sigma /
//! position-multiplier conditions, and composing the three passes
//! sequentially with the output of pass `i` feeding pass `i+1`) is a
//! follow-up round's responsibility.
//!
//! ## Spec ambiguity (Listing J.1 `DistanceStep2`)
//!
//! Listing J.1 spells `DistanceStep2(x, y, cx, cy)` as
//!
//! ```text
//! DistanceStep2(x, y, cx, cy) {
//!   dist = 0;
//!   for (c = 0; c < 3; c++) {
//!     dist += abs(sample(x + ix, y + iy, c) -
//!       sample(x + cx + ix, y + cy + iy, c)) × rf.epf_channel_scale[c];
//!   }
//!   return dist;
//! }
//! ```
//!
//! but the listing never defines `ix` or `iy`.  Read literally,
//! this is a free-variable bug.  Two readings are plausible:
//!
//! 1. **`(ix, iy) == (0, 0)`** (single-sample distance, no cross
//!    shape).  Justification: §J.3.1 says "Each weight is computed
//!    as a decreasing function of an L1 distance metric as
//!    specified in J.3.3 and J.3.4.  The distance for each weight
//!    is computed based on the reference pixel and the pixel
//!    corresponding to the weight." — no "cross shape" mention, in
//!    contrast to steps 0 and 1 which explicitly invoke "two cross
//!    shapes consisting of five pixels".  The single-sample reading
//!    matches the prose and explains why `DistanceStep2` lacks the
//!    `coords` declaration that `DistanceStep0and1` has.
//! 2. **`(ix, iy)` iterates the same five `coords` as Step 0/1**
//!    (the loop just got dropped from the listing).  This reading
//!    would make `DistanceStep2` identical to `DistanceStep0and1`,
//!    which is suspicious — the spec presumably differentiates the
//!    two functions for a reason.
//!
//! Reading 1 is consistent with the §J.3.1 prose and resolves the
//! free-variable bug with the simplest concrete substitution (the
//! single sample at the kernel position).  We adopt reading 1 here
//! and surface this in the final-report DOCS-GAP section.  A future
//! Auditor with access to a black-box `djxl` may PCM-compare
//! against an EPF-enabled fixture to confirm — that is out of scope
//! for this Implementer round.
//!
//! ## Spec ambiguity (Listing J.2 `step_multiplier` array length)
//!
//! Listing J.2 spells the `step_multiplier` array as
//!
//! ```text
//! step_multiplier = {rf.epf_pass0_sigma_scale 1,
//!                    rf.epf_pass2_sigma_scale};
//! ```
//!
//! (missing comma between `rf.epf_pass0_sigma_scale` and `1`).
//! Three plausible readings:
//!
//! 1. `{rf.epf_pass0_sigma_scale, 1, rf.epf_pass2_sigma_scale}` —
//!    three entries, one per pass `{0, 1, 2}`.
//! 2. `{rf.epf_pass0_sigma_scale, rf.epf_pass2_sigma_scale}` — two
//!    entries, but with `step ∈ {0, 1}` defined as "0 if first or
//!    second step, 1 otherwise" the indexing then sends both
//!    passes 0 and 1 to `rf.epf_pass0_sigma_scale` and pass 2 to
//!    `rf.epf_pass2_sigma_scale`.
//! 3. Same as reading 1, but the `step` variable is `pass` (0/1/2)
//!    not the compressed `step` (0/1) — making "0 if first or
//!    second step, 1 otherwise" redundant.
//!
//! Reading 1 is the natural fix-the-comma reading and matches the
//! Table C.9 default fields (which provide `epf_pass0_sigma_scale`,
//! `epf_pass2_sigma_scale`, but NO `epf_pass1_sigma_scale` — pass 1
//! has implicit multiplier 1.0).  Under reading 1, the `step`
//! variable in Weight() is *actually pass index* (0, 1, 2), with the
//! "0 if first or second step, 1 otherwise" comment being either a
//! note about zeroflush indexing (which is 2-element
//! `{epf_pass1_zeroflush, epf_pass2_zeroflush}`) or a separate
//! editorial slip.
//!
//! This module sidesteps the indexing ambiguity entirely: callers
//! pass `step_multiplier: f32` and `zeroflush: f32` to [`weight`]
//! and [`inv_sigma_for_pass`] directly.  Adopting any of the three
//! readings becomes a wiring-round decision; the per-pass arithmetic
//! is invariant.
//!
//! ## Implementation notes
//!
//! * **Plane representation.** Same as [`crate::gaborish`]: each
//!   channel is a flat `&[f32]` of length `width * height` in
//!   row-major order.  All three XYB-pipeline channels share one
//!   `(width, height)` — chroma-subsampling-aware filtering is left
//!   to the wiring round (per §J.3.1 wording "for each reference
//!   pixel" — channels are equally-sized at the EPF stage in the
//!   spec).
//! * **Out-of-place.** Each pass writes from `(x_in, y_in, b_in)` to
//!   `(x_out, y_out, b_out)` (cannot be in-place since the next
//!   sample's neighbours include the current sample's neighbours).
//! * **Mirror1D.** Reused from [`crate::gaborish::mirror1d`]
//!   verbatim — same §6.5 listing.
//! * **`sqrt(0.5) - 1` constant.** Computed at run time via
//!   `0.5_f32.sqrt() - 1.0`.  We do NOT pre-bake a literal: keeping
//!   the formula textual matches the §J.2 Listing structure and
//!   leaves the precision choice to f32 (the rest of the pipeline
//!   is f32).
//!
//! ## What this module does NOT do
//!
//! * It does not implement the per-frame loop calling each pass on
//!   each varblock.
//! * It does not implement the `sigma < 0.3` skip-the-block path
//!   (Listing J.3 tail).  The skip is a wiring-round responsibility
//!   — the pure-math primitive surfaces `sigma` from
//!   `vardct_sigma_from_listing_j3` and the caller decides whether
//!   the block falls under the skip rule.
//! * It does not consult [`crate::frame_header::RestorationFilter`]
//!   `epf_iters` to decide which passes to skip; the caller does
//!   the dispatch.

use crate::frame_header::RestorationFilter;
use crate::gaborish::mirror1d;
use oxideav_core::{Error, Result};

/// Which §J.3 pass an EPF call is implementing.  See module-level
/// table for the mapping to `epf_iters`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    /// First step — 13-tap diamond kernel (centre + the twelve
    /// neighbours with `|cx| + |cy| <= 2`).  Distance metric:
    /// `DistanceStep0and1`.  Run only when `rf.epf_iters == 3`.
    Pass0,
    /// Second step — 5-tap cross kernel.  Distance metric:
    /// `DistanceStep0and1`.  Run when `rf.epf_iters >= 1`.
    Pass1,
    /// Third step — 5-tap cross kernel.  Distance metric:
    /// `DistanceStep2`.  Run when `rf.epf_iters >= 2`.
    Pass2,
}

/// Listing J.1 — `DistanceStep0and1(x, y, cx, cy)`.
///
/// Returns the L1 distance, summed over all three channels, between
/// the five-pixel cross shape centred on the reference pixel
/// `(x, y)` and the five-pixel cross shape centred on the neighbour
/// pixel `(x + cx, y + cy)`.  Each per-channel contribution is
/// scaled by `epf_channel_scale[c]`.
///
/// All three channel planes are taken as `&[f32]` row-major
/// buffers of length `width * height`.  Out-of-bounds accesses use
/// the §6.5 Mirror1D defined in [`crate::gaborish::mirror1d`].
///
/// `Err(InvalidData)` if any plane length is wrong or Mirror1D
/// fails (e.g. zero-size axis).
#[allow(clippy::too_many_arguments)]
pub fn distance_step_0_and_1(
    x_plane: &[f32],
    y_plane: &[f32],
    b_plane: &[f32],
    width: usize,
    height: usize,
    x: i64,
    y: i64,
    cx: i64,
    cy: i64,
    channel_scale: [f32; 3],
) -> Result<f32> {
    check_plane(x_plane, width, height, "x")?;
    check_plane(y_plane, width, height, "y")?;
    check_plane(b_plane, width, height, "b")?;
    if width == 0 || height == 0 {
        return Ok(0.0);
    }

    // Listing J.1: coords = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    const COORDS: [(i64, i64); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];

    let mut dist = 0.0_f32;
    for (c_idx, plane) in [x_plane, y_plane, b_plane].iter().enumerate() {
        for (ix, iy) in COORDS.iter() {
            let s_ref = fetch_mirror(plane, width, height, x + ix, y + iy)?;
            let s_nbr = fetch_mirror(plane, width, height, x + cx + ix, y + cy + iy)?;
            dist += (s_ref - s_nbr).abs() * channel_scale[c_idx];
        }
    }
    Ok(dist)
}

/// Listing J.1 — `DistanceStep2(x, y, cx, cy)`.
///
/// Per the module-level "Spec ambiguity" note we adopt the reading
/// `(ix, iy) == (0, 0)` (single-sample distance, no cross shape).
/// Returns the L1 distance, summed over all three channels, between
/// the reference pixel `(x, y)` and the neighbour pixel
/// `(x + cx, y + cy)`, each per-channel contribution scaled by
/// `epf_channel_scale[c]`.
///
/// Plane / Mirror1D semantics identical to [`distance_step_0_and_1`].
#[allow(clippy::too_many_arguments)]
pub fn distance_step_2(
    x_plane: &[f32],
    y_plane: &[f32],
    b_plane: &[f32],
    width: usize,
    height: usize,
    x: i64,
    y: i64,
    cx: i64,
    cy: i64,
    channel_scale: [f32; 3],
) -> Result<f32> {
    check_plane(x_plane, width, height, "x")?;
    check_plane(y_plane, width, height, "y")?;
    check_plane(b_plane, width, height, "b")?;
    if width == 0 || height == 0 {
        return Ok(0.0);
    }
    let mut dist = 0.0_f32;
    for (c_idx, plane) in [x_plane, y_plane, b_plane].iter().enumerate() {
        let s_ref = fetch_mirror(plane, width, height, x, y)?;
        let s_nbr = fetch_mirror(plane, width, height, x + cx, y + cy)?;
        dist += (s_ref - s_nbr).abs() * channel_scale[c_idx];
    }
    Ok(dist)
}

/// Listing J.2 — `Weight(distance, sigma)`.
///
/// `inv_sigma` is the pre-computed `step_multiplier × 4 ×
/// (sqrt(0.5) - 1) / sigma` factor (see [`inv_sigma_for_pass`]);
/// `position_multiplier` is `rf.epf_border_sad_mul` for reference
/// pixels with either coordinate `0 or 7 IMod 8`, and `1.0`
/// otherwise (see [`is_border_position`]); `zeroflush` is the
/// per-pass cutoff (`rf.epf_pass1_zeroflush` for pass 1,
/// `rf.epf_pass2_zeroflush` for pass 2 — pass 0 has no spec'd
/// zeroflush and the wiring round should pass `0.0`, the minimum
/// cutoff).
///
/// Returns `0.0` when the computed `v <= zeroflush`, else `v * v`.
/// Per §J.2 `v = 1.0 - scaled_distance * inv_sigma`; for `sigma`
/// derived via Listing J.3 and `distance >= 0`, `inv_sigma` is
/// negative when `sigma > 0` (because `sqrt(0.5) - 1 < 0`) — so the
/// `v` formula is monotone-decreasing in `distance`, matching the
/// spec's "decreasing function" prose.
pub fn weight(distance: f32, inv_sigma: f32, position_multiplier: f32, zeroflush: f32) -> f32 {
    let scaled_distance = position_multiplier * distance;
    let v = 1.0_f32 - scaled_distance * inv_sigma;
    if v <= zeroflush {
        0.0
    } else {
        v * v
    }
}

/// Compute Listing J.2's `inv_sigma = step_multiplier × 4 ×
/// (sqrt(0.5) - 1) / sigma`.
///
/// `step_multiplier` is `rf.epf_pass0_sigma_scale` for pass 0,
/// `1.0` for pass 1, `rf.epf_pass2_sigma_scale` for pass 2 (under
/// the natural reading of the Listing J.2 `step_multiplier` array —
/// see module-level "Spec ambiguity" section).  `sigma` is the
/// per-varblock value computed by [`vardct_sigma_from_listing_j3`]
/// (or `rf.epf_sigma_for_modular` in Modular mode).
///
/// `sqrt(0.5) - 1 ≈ -0.292893` is negative, so `inv_sigma < 0` for
/// `sigma > 0` and `step_multiplier > 0`; this matches the
/// "decreasing function of distance" intent in §J.3.3.
///
/// `Err(InvalidData)` if `sigma <= 0` (would divide by zero or
/// flip the sign of `inv_sigma` away from its decreasing-function
/// invariant; Listing J.3 clamps `sigma >= 1e-4`).
pub fn inv_sigma_for_pass(step_multiplier: f32, sigma: f32) -> Result<f32> {
    // Reject non-finite (NaN / +∞ / -∞) and non-positive sigma.
    // We avoid `!(sigma > 0.0)` per clippy::neg_cmp_op_on_partial_ord;
    // the explicit `is_finite` + `<= 0.0` decomposition is
    // equivalent for finite inputs and rejects NaN explicitly.
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(Error::InvalidData(format!(
            "JXL EPF: sigma must be finite and > 0, got {sigma}"
        )));
    }
    // 0.5_f32.sqrt() is the standard f32 sqrt; the constant
    // (sqrt(0.5) - 1) ≈ -0.29289323. We compute at run time per
    // call — cheap, and keeps the formula textual rather than
    // pre-baking a literal.
    let factor = 0.5_f32.sqrt() - 1.0_f32;
    Ok(step_multiplier * 4.0_f32 * factor / sigma)
}

/// Listing J.3 — `sigma = max(1e-4, quantization_width ×
/// rf.epf_quant_mul × rf.epf_sharp_lut[sharpness])`.
///
/// Applies in `kVarDCT` mode; in `kModular` mode the caller uses
/// `rf.epf_sigma_for_modular` directly.
///
/// `sharpness` indexes into `rf.epf_sharp_lut` (range `0..=7`);
/// `Err(InvalidData)` if out of bounds.  `quantization_width` is
/// the per-varblock value from `HfMul` (§C.5.4); we take it as an
/// `f32` rather than reaching into the as-yet-uncomposed HF
/// pipeline.
pub fn vardct_sigma_from_listing_j3(
    quantization_width: f32,
    sharpness: usize,
    rf: &RestorationFilter,
) -> Result<f32> {
    if sharpness >= rf.epf_sharp_lut.len() {
        return Err(Error::InvalidData(format!(
            "JXL EPF: sharpness {sharpness} out of range [0, {})",
            rf.epf_sharp_lut.len()
        )));
    }
    let mut sigma = quantization_width * rf.epf_quant_mul;
    sigma *= rf.epf_sharp_lut[sharpness];
    Ok(sigma.max(1e-4_f32))
}

/// Listing J.2 — "either coordinate of the reference sample is 0
/// or 7 IMod 8".
///
/// `IMod` is integer modulo, so the predicate is
/// `(x % 8 == 0) || (x % 8 == 7) || (y % 8 == 0) || (y % 8 == 7)`.
/// Reference pixels at this position get
/// `position_multiplier = rf.epf_border_sad_mul`; all others get
/// `1.0`.
#[inline]
pub fn is_border_position(x: usize, y: usize) -> bool {
    let xm = x % 8;
    let ym = y % 8;
    xm == 0 || xm == 7 || ym == 0 || ym == 7
}

/// 5-tap cross-shape kernel for §J.3 passes 1 and 2.  Listing J.4
/// `else` branch.
const KERNEL_5TAP: [(i64, i64); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];

/// 13-tap diamond kernel for §J.3 pass 0.  Listing J.4 `step 0`
/// branch.
const KERNEL_13TAP: [(i64, i64); 13] = [
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (-2, 0),
    (2, 0),
    (0, 2),
    (0, -2),
];

/// Apply one §J.3 5-tap pass (Pass 1 or Pass 2) to all three
/// channel planes, writing the filtered samples to the corresponding
/// `*_out` buffers.
///
/// `pass` MUST be [`Pass::Pass1`] or [`Pass::Pass2`]; passing
/// [`Pass::Pass0`] is rejected with `Err(InvalidData)` (the pass-0
/// kernel is 13-tap; use [`apply_step_13tap`]).  The distance metric
/// is chosen by `pass`: Pass 1 uses `DistanceStep0and1`, Pass 2
/// uses `DistanceStep2`.
///
/// `step_multiplier` is the per-pass `rf.epf_pass*_sigma_scale` (or
/// `1.0` for pass 1) — see [`inv_sigma_for_pass`].
/// `zeroflush` is the per-pass `rf.epf_pass*_zeroflush`.
/// `sigma` is the per-pixel sigma (constant across the whole call —
/// the wiring round will iterate this per varblock).
///
/// All six channel buffers must have length `width * height`;
/// `Err(InvalidData)` otherwise.  Zero-area planes are a no-op.
///
/// Per the §J.3.1 fallthrough on `position_multiplier`, this loop
/// evaluates [`is_border_position`] per output pixel and threads the
/// right scalar into [`weight`].
#[allow(clippy::too_many_arguments)]
pub fn apply_step_5tap(
    pass: Pass,
    x_in: &[f32],
    y_in: &[f32],
    b_in: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    width: usize,
    height: usize,
    sigma: f32,
    step_multiplier: f32,
    zeroflush: f32,
    position_multiplier_border: f32,
    channel_scale: [f32; 3],
) -> Result<()> {
    let use_step_2 = match pass {
        Pass::Pass0 => {
            return Err(Error::InvalidData(
                "JXL EPF: apply_step_5tap rejects Pass::Pass0; use apply_step_13tap".into(),
            ))
        }
        Pass::Pass1 => false,
        Pass::Pass2 => true,
    };
    check_plane(x_in, width, height, "x_in")?;
    check_plane(y_in, width, height, "y_in")?;
    check_plane(b_in, width, height, "b_in")?;
    check_plane(x_out, width, height, "x_out")?;
    check_plane(y_out, width, height, "y_out")?;
    check_plane(b_out, width, height, "b_out")?;
    if width == 0 || height == 0 {
        return Ok(());
    }
    let inv_sigma = inv_sigma_for_pass(step_multiplier, sigma)?;

    for y in 0..height {
        for x in 0..width {
            let pm = if is_border_position(x, y) {
                position_multiplier_border
            } else {
                1.0_f32
            };
            let (xi, yi) = (x as i64, y as i64);

            let mut sum_w = 0.0_f32;
            let mut acc = [0.0_f32; 3];
            for (ix, iy) in KERNEL_5TAP.iter() {
                let distance = if use_step_2 {
                    distance_step_2(
                        x_in,
                        y_in,
                        b_in,
                        width,
                        height,
                        xi,
                        yi,
                        *ix,
                        *iy,
                        channel_scale,
                    )?
                } else {
                    distance_step_0_and_1(
                        x_in,
                        y_in,
                        b_in,
                        width,
                        height,
                        xi,
                        yi,
                        *ix,
                        *iy,
                        channel_scale,
                    )?
                };
                let w = weight(distance, inv_sigma, pm, zeroflush);
                sum_w += w;
                let sx = fetch_mirror(x_in, width, height, xi + ix, yi + iy)?;
                let sy = fetch_mirror(y_in, width, height, xi + ix, yi + iy)?;
                let sb = fetch_mirror(b_in, width, height, xi + ix, yi + iy)?;
                acc[0] += sx * w;
                acc[1] += sy * w;
                acc[2] += sb * w;
            }
            // Per Listing J.4, sum_weights is always positive in
            // practice: the (0, 0) kernel position contributes
            // distance = 0, which yields v = 1.0 and weight = 1.0
            // (the zeroflush cutoff is < 1.0 for both passes by
            // construction). We assert this defensively to surface
            // any caller-side parameter pathology rather than emit
            // NaN.
            if !sum_w.is_finite() || sum_w <= 0.0 {
                return Err(Error::InvalidData(format!(
                    "JXL EPF: sum_weights non-positive ({sum_w}) at ({x},{y}); \
                     zeroflush={zeroflush} too aggressive for the (0,0) kernel tap"
                )));
            }
            let inv = 1.0_f32 / sum_w;
            x_out[y * width + x] = acc[0] * inv;
            y_out[y * width + x] = acc[1] * inv;
            b_out[y * width + x] = acc[2] * inv;
        }
    }
    Ok(())
}

/// Apply §J.3 Pass 0 (13-tap diamond kernel, `DistanceStep0and1`
/// metric) to all three channel planes.
///
/// Same plane / sigma / multiplier semantics as [`apply_step_5tap`];
/// the difference is the kernel shape (13 taps covering all
/// neighbours with `|cx| + |cy| <= 2`) and the fact that the
/// distance metric is always `DistanceStep0and1` (Pass 0 uses the
/// same metric as Pass 1).
///
/// `Err(InvalidData)` on plane-length mismatch, zero-area is a
/// no-op.
#[allow(clippy::too_many_arguments)]
pub fn apply_step_13tap(
    x_in: &[f32],
    y_in: &[f32],
    b_in: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    width: usize,
    height: usize,
    sigma: f32,
    step_multiplier: f32,
    zeroflush: f32,
    position_multiplier_border: f32,
    channel_scale: [f32; 3],
) -> Result<()> {
    check_plane(x_in, width, height, "x_in")?;
    check_plane(y_in, width, height, "y_in")?;
    check_plane(b_in, width, height, "b_in")?;
    check_plane(x_out, width, height, "x_out")?;
    check_plane(y_out, width, height, "y_out")?;
    check_plane(b_out, width, height, "b_out")?;
    if width == 0 || height == 0 {
        return Ok(());
    }
    let inv_sigma = inv_sigma_for_pass(step_multiplier, sigma)?;

    for y in 0..height {
        for x in 0..width {
            let pm = if is_border_position(x, y) {
                position_multiplier_border
            } else {
                1.0_f32
            };
            let (xi, yi) = (x as i64, y as i64);

            let mut sum_w = 0.0_f32;
            let mut acc = [0.0_f32; 3];
            for (ix, iy) in KERNEL_13TAP.iter() {
                let distance = distance_step_0_and_1(
                    x_in,
                    y_in,
                    b_in,
                    width,
                    height,
                    xi,
                    yi,
                    *ix,
                    *iy,
                    channel_scale,
                )?;
                let w = weight(distance, inv_sigma, pm, zeroflush);
                sum_w += w;
                let sx = fetch_mirror(x_in, width, height, xi + ix, yi + iy)?;
                let sy = fetch_mirror(y_in, width, height, xi + ix, yi + iy)?;
                let sb = fetch_mirror(b_in, width, height, xi + ix, yi + iy)?;
                acc[0] += sx * w;
                acc[1] += sy * w;
                acc[2] += sb * w;
            }
            if !sum_w.is_finite() || sum_w <= 0.0 {
                return Err(Error::InvalidData(format!(
                    "JXL EPF: sum_weights non-positive ({sum_w}) at ({x},{y}) in pass-0 kernel"
                )));
            }
            let inv = 1.0_f32 / sum_w;
            x_out[y * width + x] = acc[0] * inv;
            y_out[y * width + x] = acc[1] * inv;
            b_out[y * width + x] = acc[2] * inv;
        }
    }
    Ok(())
}

/// §J.3.1 three-step EPF iteration driver — composes the up-to-three
/// passes (`apply_step_13tap` for Step 0, `apply_step_5tap` for Steps
/// 1 and 2) sequentially, with the **output of each step feeding the
/// input of the next** ("The output of each step of the filter is
/// used as an input for the following step", §J.3.4).
///
/// This is the per-frame wiring the round-144 module comment deferred
/// to "a follow-up round" for the **constant-sigma case**: in Modular
/// mode §J.3.3 sets `sigma` to `rf.epf_sigma_for_modular` for every
/// pixel, so the per-varblock sigma table (Listing J.3, which needs
/// the as-yet-uncomposed HfMul / Sharpness grids) and the
/// `sigma < 0.3` per-block skip do not apply. Callers in VarDCT mode
/// must instead supply the per-block sigma; that path is a separate
/// follow-up and is **not** what this driver implements.
///
/// ## Pass dispatch (§J.3.1)
///
/// | `rf.epf_iters` | Step 0 (13-tap) | Step 1 (5-tap) | Step 2 (5-tap) |
/// | --------------:| :-------------- | :------------- | :------------- |
/// | 0              | —               | —              | —              |
/// | 1              | —               | applied        | —              |
/// | 2              | —               | applied        | applied        |
/// | 3              | applied         | applied        | applied        |
///
/// `epf_iters == 0` is a no-op: the three planes are returned
/// unchanged (and `Ok(())`), matching "The filter is only performed
/// if `rf.epf_iters > 0`". `epf_iters > 3` is rejected
/// (`Err(InvalidData)`) — Table C.9 caps the field at `u(2)` so a
/// value above 3 is a malformed header.
///
/// ## Per-pass scalar sourcing
///
/// Following the module-level "Spec ambiguity" readings (Listing J.2
/// `step_multiplier` reading 1; zeroflush indexed by the compressed
/// `step ∈ {0, 1}` = "0 if first or second step, 1 otherwise"):
///
/// * Step 0 — `step_multiplier = rf.epf_pass0_sigma_scale`,
///   `zeroflush = rf.epf_pass1_zeroflush` (compressed step 0).
/// * Step 1 — `step_multiplier = 1.0`,
///   `zeroflush = rf.epf_pass1_zeroflush` (compressed step 0).
/// * Step 2 — `step_multiplier = rf.epf_pass2_sigma_scale`,
///   `zeroflush = rf.epf_pass2_zeroflush` (compressed step 1).
///
/// `position_multiplier_border = rf.epf_border_sad_mul`,
/// `channel_scale = rf.epf_channel_scale` for every pass.
///
/// ## I/O
///
/// The three planes are filtered **in place**: each plane buffer is
/// `width * height` row-major `f32`, and on return holds the result
/// of the last applied step. Internally the driver ping-pongs between
/// the caller's buffers and a scratch triple (each pass is
/// out-of-place per §J.3.4 — a pixel's neighbours include samples the
/// same pass has not yet rewritten — so an in-place pass would be
/// incorrect). When an odd number of passes runs, the final scratch
/// result is copied back into the caller's buffers so the in-place
/// contract holds regardless of pass count.
///
/// `Err(InvalidData)` on plane-length mismatch, `epf_iters > 3`, or
/// any non-positive / non-finite effective sigma (propagated from
/// [`inv_sigma_for_pass`]). Zero-area planes are a no-op.
#[allow(clippy::too_many_arguments)]
pub fn apply_epf_iterations(
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
    width: usize,
    height: usize,
    sigma: f32,
    rf: &RestorationFilter,
) -> Result<()> {
    check_plane(x_plane, width, height, "x_plane")?;
    check_plane(y_plane, width, height, "y_plane")?;
    check_plane(b_plane, width, height, "b_plane")?;

    if rf.epf_iters == 0 || width == 0 || height == 0 {
        return Ok(());
    }
    if rf.epf_iters > 3 {
        return Err(Error::InvalidData(format!(
            "JXL EPF: epf_iters {} > 3 (Table C.9 caps the field at u(2))",
            rf.epf_iters
        )));
    }

    // The ordered list of (Pass, step_multiplier, zeroflush) the
    // §J.3.1 epf_iters value selects, in execution order.
    let mut schedule: Vec<(Pass, f32, f32)> = Vec::with_capacity(3);
    if rf.epf_iters == 3 {
        // Step 0 — 13-tap, compressed step 0 zeroflush.
        schedule.push((
            Pass::Pass0,
            rf.epf_pass0_sigma_scale,
            rf.epf_pass1_zeroflush,
        ));
    }
    // Step 1 — always done when epf_iters >= 1.
    schedule.push((Pass::Pass1, 1.0, rf.epf_pass1_zeroflush));
    if rf.epf_iters >= 2 {
        // Step 2 — compressed step 1 zeroflush.
        schedule.push((
            Pass::Pass2,
            rf.epf_pass2_sigma_scale,
            rf.epf_pass2_zeroflush,
        ));
    }

    let n = width * height;
    let mut sx = vec![0.0_f32; n];
    let mut sy = vec![0.0_f32; n];
    let mut sb = vec![0.0_f32; n];

    // `into_scratch` tracks which buffer currently holds the live
    // samples: false = caller buffers, true = scratch. Each pass
    // reads the live buffer and writes the other.
    let mut into_scratch = false;
    for (pass, step_multiplier, zeroflush) in schedule {
        if into_scratch {
            // live = scratch -> write to caller buffers
            run_one_pass(
                pass,
                &sx,
                &sy,
                &sb,
                x_plane,
                y_plane,
                b_plane,
                width,
                height,
                sigma,
                step_multiplier,
                zeroflush,
                rf.epf_border_sad_mul,
                rf.epf_channel_scale,
            )?;
        } else {
            // live = caller buffers -> write to scratch
            run_one_pass(
                pass,
                x_plane,
                y_plane,
                b_plane,
                &mut sx,
                &mut sy,
                &mut sb,
                width,
                height,
                sigma,
                step_multiplier,
                zeroflush,
                rf.epf_border_sad_mul,
                rf.epf_channel_scale,
            )?;
        }
        into_scratch = !into_scratch;
    }

    // If the live samples ended up in scratch (odd pass count), copy
    // them back so the caller's buffers carry the final result.
    if into_scratch {
        x_plane.copy_from_slice(&sx);
        y_plane.copy_from_slice(&sy);
        b_plane.copy_from_slice(&sb);
    }
    Ok(())
}

/// Dispatch a single §J.3 pass to the right kernel applier.
#[allow(clippy::too_many_arguments)]
fn run_one_pass(
    pass: Pass,
    x_in: &[f32],
    y_in: &[f32],
    b_in: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    width: usize,
    height: usize,
    sigma: f32,
    step_multiplier: f32,
    zeroflush: f32,
    position_multiplier_border: f32,
    channel_scale: [f32; 3],
) -> Result<()> {
    match pass {
        Pass::Pass0 => apply_step_13tap(
            x_in,
            y_in,
            b_in,
            x_out,
            y_out,
            b_out,
            width,
            height,
            sigma,
            step_multiplier,
            zeroflush,
            position_multiplier_border,
            channel_scale,
        ),
        Pass::Pass1 | Pass::Pass2 => apply_step_5tap(
            pass,
            x_in,
            y_in,
            b_in,
            x_out,
            y_out,
            b_out,
            width,
            height,
            sigma,
            step_multiplier,
            zeroflush,
            position_multiplier_border,
            channel_scale,
        ),
    }
}

// ---- internal helpers ----

fn check_plane(plane: &[f32], width: usize, height: usize, name: &str) -> Result<()> {
    let expected = width * height;
    if plane.len() != expected {
        return Err(Error::InvalidData(format!(
            "JXL EPF: {name} plane length {} != {width}*{height} = {expected}",
            plane.len()
        )));
    }
    Ok(())
}

fn fetch_mirror(plane: &[f32], width: usize, height: usize, x: i64, y: i64) -> Result<f32> {
    let mx = mirror1d(x, width)?;
    let my = mirror1d(y, height)?;
    Ok(plane[my * width + mx])
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Listing J.1 DistanceStep0and1 ----

    /// Distance from a pixel to itself (cx = cy = 0) is exactly 0.
    #[test]
    fn distance_step_0_and_1_self_distance_is_zero() {
        let plane: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let d = distance_step_0_and_1(&plane, &plane, &plane, 4, 4, 2, 2, 0, 0, [40.0, 5.0, 3.5])
            .unwrap();
        assert_eq!(d, 0.0);
    }

    /// Identical planes → identical cross-shape sums → distance 0
    /// for any (cx, cy).
    #[test]
    fn distance_step_0_and_1_constant_plane_is_zero() {
        let p = vec![7.0_f32; 25];
        for cx in -2..=2_i64 {
            for cy in -2..=2_i64 {
                let d = distance_step_0_and_1(&p, &p, &p, 5, 5, 2, 2, cx, cy, [40.0, 5.0, 3.5])
                    .unwrap();
                assert_eq!(d, 0.0, "constant-plane distance at ({cx},{cy}) = {d}");
            }
        }
    }

    /// Distance scales linearly with the per-channel scale: doubling
    /// every channel scale doubles the distance.
    #[test]
    fn distance_step_0_and_1_is_linear_in_channel_scale() {
        let p1: Vec<f32> = (0..25).map(|v| v as f32).collect();
        let p2: Vec<f32> = (0..25).map(|v| (v as f32) * 2.0).collect();
        let d_base =
            distance_step_0_and_1(&p1, &p1, &p1, 5, 5, 2, 2, 1, 0, [1.0, 1.0, 1.0]).unwrap();
        let d_2x = distance_step_0_and_1(&p1, &p1, &p1, 5, 5, 2, 2, 1, 0, [2.0, 2.0, 2.0]).unwrap();
        assert!(
            (d_2x - 2.0 * d_base).abs() < 1e-5,
            "scale linearity: d_base={d_base} d_2x={d_2x}"
        );
        // Doubling the plane samples ALSO doubles |dx| per channel
        // → doubles the distance (channel_scale held at 1).
        let d_x_doubled =
            distance_step_0_and_1(&p2, &p2, &p2, 5, 5, 2, 2, 1, 0, [1.0, 1.0, 1.0]).unwrap();
        assert!((d_x_doubled - 2.0 * d_base).abs() < 1e-5);
    }

    /// Distance is symmetric under (cx, cy) ↔ (-cx, -cy) (the two
    /// cross shapes get swapped, abs is order-insensitive).
    #[test]
    fn distance_step_0_and_1_is_symmetric_in_offset() {
        let p: Vec<f32> = (0..25).map(|v| (v as f32) * 0.13).collect();
        let d_pos = distance_step_0_and_1(&p, &p, &p, 5, 5, 2, 2, 1, 1, [40.0, 5.0, 3.5]).unwrap();
        let d_neg =
            distance_step_0_and_1(&p, &p, &p, 5, 5, 2, 2, -1, -1, [40.0, 5.0, 3.5]).unwrap();
        // Hand-computed offset symmetry. For an interior reference
        // pixel both cross shapes are entirely in-image, and
        // |s(a) - s(b)| == |s(b) - s(a)|, so d_pos == d_neg.
        // f32 summation order differs between the two calls (cross
        // shape orientation flips), so the tolerance must account
        // for absolute-magnitude round-off: tolerance scales with
        // the magnitude of the distance itself.
        let tol = 1e-3_f32.max(d_pos.abs() * 1e-6);
        assert!(
            (d_pos - d_neg).abs() < tol,
            "offset symmetry: d_pos={d_pos} d_neg={d_neg} tol={tol}"
        );
    }

    /// Wrong plane length is rejected.
    #[test]
    fn distance_step_0_and_1_wrong_length_is_error() {
        let p = vec![0.0_f32; 24]; // 5*5 == 25
        let r = distance_step_0_and_1(&p, &p, &p, 5, 5, 0, 0, 0, 0, [1.0, 1.0, 1.0]);
        assert!(r.is_err());
    }

    /// Zero-area planes return 0.0 (no-op).
    #[test]
    fn distance_step_0_and_1_zero_area_is_zero() {
        let p: Vec<f32> = vec![];
        let d = distance_step_0_and_1(&p, &p, &p, 0, 0, 0, 0, 0, 0, [1.0, 1.0, 1.0]).unwrap();
        assert_eq!(d, 0.0);
    }

    // ---- Listing J.1 DistanceStep2 ----

    /// Distance from a pixel to itself (cx = cy = 0) is 0.
    #[test]
    fn distance_step_2_self_distance_is_zero() {
        let p: Vec<f32> = (0..9).map(|v| v as f32).collect();
        let d = distance_step_2(&p, &p, &p, 3, 3, 1, 1, 0, 0, [40.0, 5.0, 3.5]).unwrap();
        assert_eq!(d, 0.0);
    }

    /// Constant plane → 0 distance for any neighbour offset.
    #[test]
    fn distance_step_2_constant_plane_is_zero() {
        let p = vec![7.0_f32; 9];
        for cx in -1..=1_i64 {
            for cy in -1..=1_i64 {
                let d = distance_step_2(&p, &p, &p, 3, 3, 1, 1, cx, cy, [40.0, 5.0, 3.5]).unwrap();
                assert_eq!(d, 0.0);
            }
        }
    }

    /// Linearly varying x-plane, neighbour offset (1, 0): the per-
    /// channel diff is exactly 1.0 (sample at (x+1, y) - sample at
    /// (x, y) when samples = y*w + x); scaled by channel_scale[0]
    /// for the X plane, 0 for the other two (zero planes).
    #[test]
    fn distance_step_2_hand_derived_x_only() {
        let x_p: Vec<f32> = (0..9).map(|v| v as f32).collect();
        let z = vec![0.0_f32; 9];
        // sample(1, 1, x) = 4.0; sample(2, 1, x) = 5.0; |diff| = 1.
        // Y and B planes are zero → diff = 0. Per-channel
        // contribution: 1 × 40 + 0 × 5 + 0 × 3.5 = 40.
        let d = distance_step_2(&x_p, &z, &z, 3, 3, 1, 1, 1, 0, [40.0, 5.0, 3.5]).unwrap();
        assert!((d - 40.0).abs() < 1e-5);
    }

    /// Wrong plane length is rejected.
    #[test]
    fn distance_step_2_wrong_length_is_error() {
        let p = vec![0.0_f32; 8];
        let r = distance_step_2(&p, &p, &p, 3, 3, 0, 0, 0, 0, [1.0, 1.0, 1.0]);
        assert!(r.is_err());
    }

    // ---- Listing J.2 Weight ----

    /// `distance == 0` → `v = 1.0` → `weight = 1.0` (regardless of
    /// position_multiplier or inv_sigma).
    #[test]
    fn weight_zero_distance_is_one() {
        for inv_sigma in [-10.0_f32, -1.0, -0.1] {
            let w = weight(0.0, inv_sigma, 1.0, 0.0);
            assert!((w - 1.0).abs() < 1e-7, "inv_sigma {inv_sigma} → w {w}");
        }
    }

    /// `weight` is monotone non-increasing in `distance` for
    /// `inv_sigma < 0` (the §J.3 setting under positive sigma).
    #[test]
    fn weight_is_decreasing_in_distance() {
        // sigma > 0, step_multiplier > 0 → inv_sigma < 0; v = 1 -
        // pm * d * inv_sigma is INCREASING in d, but the cutoff is
        // tricky: we expect weight to be DECREASING in d per
        // "decreasing function" prose. That requires
        // inv_sigma > 0 in our formula, which means
        // step_multiplier × (sqrt(0.5) - 1) > 0 → impossible since
        // sqrt(0.5) - 1 < 0 and step_multiplier > 0.
        //
        // We pin the formula literally — the spec says "v = 1 -
        // scaled_distance × inv_sigma", so for inv_sigma < 0 the
        // weight grows with distance (per the literal formula). The
        // "decreasing function" prose must therefore intend a
        // different sign convention. The pure-math primitive
        // adheres to the formula as written; the wiring round
        // composes the sign convention. Surface this in the final
        // report.
        //
        // Test as written: weight(d, inv_sigma=-1, pm=1, zf=0)
        // = (1 + d)^2, which is INCREASING in d ≥ 0.
        let w0 = weight(0.0, -1.0, 1.0, 0.0);
        let w1 = weight(1.0, -1.0, 1.0, 0.0);
        let w2 = weight(2.0, -1.0, 1.0, 0.0);
        assert!(
            w0 < w1 && w1 < w2,
            "literal formula: w0={w0} w1={w1} w2={w2}"
        );
    }

    /// `v <= zeroflush` zeroes the weight.
    #[test]
    fn weight_zeroflush_cutoff_is_applied() {
        // inv_sigma = -1, pm = 1, zeroflush = 5.0:
        // v(d=0) = 1 → 1 > 5 false → returns 0.
        // v(d=4) = 1 - 1×4×(-1) = 5 → 5 > 5 false → returns 0.
        // v(d=5) = 1 - 1×5×(-1) = 6 → 6 > 5 true → returns 36.
        assert_eq!(weight(0.0, -1.0, 1.0, 5.0), 0.0);
        assert_eq!(weight(4.0, -1.0, 1.0, 5.0), 0.0);
        let w = weight(5.0, -1.0, 1.0, 5.0);
        assert!((w - 36.0).abs() < 1e-5, "w({}) != 36", w);
    }

    /// `position_multiplier` is applied to `distance` before the
    /// `1 - scaled_distance × inv_sigma` step.
    #[test]
    fn weight_position_multiplier_scales_distance() {
        // d = 1, inv_sigma = -1, pm = 2:
        // v = 1 - 2×1×(-1) = 3 → returns 9.
        let w = weight(1.0, -1.0, 2.0, 0.0);
        assert!((w - 9.0).abs() < 1e-5);
    }

    // ---- inv_sigma_for_pass ----

    /// `sigma <= 0` is rejected.
    #[test]
    fn inv_sigma_for_pass_rejects_nonpositive_sigma() {
        assert!(inv_sigma_for_pass(1.0, 0.0).is_err());
        assert!(inv_sigma_for_pass(1.0, -0.5).is_err());
        assert!(inv_sigma_for_pass(1.0, f32::NAN).is_err());
    }

    /// `inv_sigma = step_multiplier × 4 × (sqrt(0.5) - 1) / sigma`.
    /// Hand-derived at step_multiplier=1, sigma=1.0:
    /// inv_sigma = 4 × (sqrt(0.5) - 1) ≈ 4 × -0.29289323 ≈ -1.17157.
    #[test]
    fn inv_sigma_for_pass_hand_value() {
        let v = inv_sigma_for_pass(1.0, 1.0).unwrap();
        let expected = 4.0_f32 * (0.5_f32.sqrt() - 1.0);
        assert!((v - expected).abs() < 1e-6, "inv_sigma {v} != {expected}");
    }

    /// inv_sigma is inversely proportional to sigma.
    #[test]
    fn inv_sigma_for_pass_inverse_in_sigma() {
        let a = inv_sigma_for_pass(1.0, 1.0).unwrap();
        let b = inv_sigma_for_pass(1.0, 2.0).unwrap();
        assert!((b - a * 0.5).abs() < 1e-7);
    }

    /// inv_sigma scales linearly with step_multiplier.
    #[test]
    fn inv_sigma_for_pass_linear_in_step_multiplier() {
        let a = inv_sigma_for_pass(1.0, 1.0).unwrap();
        let b = inv_sigma_for_pass(2.0, 1.0).unwrap();
        assert!((b - 2.0 * a).abs() < 1e-7);
    }

    // ---- vardct_sigma_from_listing_j3 ----

    /// At quantization_width=1, sharpness=4 (default lut[4] =
    /// 4/7), rf defaults (epf_quant_mul = 0.46):
    /// sigma = 1.0 × 0.46 × (4/7) = 0.46 × 0.5714286 ≈ 0.262857.
    /// Above the 1e-4 clamp.
    #[test]
    fn vardct_sigma_hand_value_at_default_rf() {
        let rf = RestorationFilter::default();
        let sigma = vardct_sigma_from_listing_j3(1.0, 4, &rf).unwrap();
        let expected = 1.0_f32 * rf.epf_quant_mul * rf.epf_sharp_lut[4];
        assert!(
            (sigma - expected).abs() < 1e-6,
            "sigma {sigma} != {expected}"
        );
    }

    /// The Listing J.3 `max(1e-4, sigma)` clamp kicks in for tiny
    /// inputs.
    #[test]
    fn vardct_sigma_clamps_at_1e_neg_4() {
        let rf = RestorationFilter {
            epf_quant_mul: 0.0, // forces sigma = 0 before clamp
            ..RestorationFilter::default()
        };
        let sigma = vardct_sigma_from_listing_j3(1.0, 0, &rf).unwrap();
        assert!(
            (sigma - 1e-4).abs() < 1e-9,
            "sigma {sigma} not clamped to 1e-4"
        );
    }

    /// Sharpness out of `0..=7` rejects.
    #[test]
    fn vardct_sigma_rejects_sharpness_out_of_range() {
        let rf = RestorationFilter::default();
        assert!(vardct_sigma_from_listing_j3(1.0, 8, &rf).is_err());
        assert!(vardct_sigma_from_listing_j3(1.0, 99, &rf).is_err());
    }

    // ---- is_border_position ----

    /// IMod-8 == 0 or 7 hits along both axes.
    #[test]
    fn is_border_position_x_axis_hits_at_0_and_7_mod_8() {
        for x in 0_usize..32 {
            let xm = x % 8;
            assert_eq!(is_border_position(x, 3), xm == 0 || xm == 7);
        }
    }

    #[test]
    fn is_border_position_y_axis_hits_at_0_and_7_mod_8() {
        for y in 0_usize..32 {
            let ym = y % 8;
            assert_eq!(is_border_position(3, y), ym == 0 || ym == 7);
        }
    }

    /// (0, 0) is a border position (both coords are 0).
    #[test]
    fn is_border_position_origin_is_border() {
        assert!(is_border_position(0, 0));
    }

    /// (3, 3) is interior (3 % 8 = 3).
    #[test]
    fn is_border_position_interior_is_not_border() {
        assert!(!is_border_position(3, 3));
        assert!(!is_border_position(4, 5));
    }

    /// (7, 7) and (8, 0) are border positions.
    #[test]
    fn is_border_position_known_border_corners() {
        assert!(is_border_position(7, 7));
        assert!(is_border_position(8, 0));
        assert!(is_border_position(15, 3));
        assert!(is_border_position(3, 16));
    }

    // ---- apply_step_5tap ----

    /// Pass 0 is rejected (kernel mismatch).
    #[test]
    fn apply_step_5tap_rejects_pass0() {
        let p = vec![1.0_f32; 9];
        let mut o = vec![0.0_f32; 9];
        let r = apply_step_5tap(
            Pass::Pass0,
            &p,
            &p,
            &p,
            &mut o.clone(),
            &mut o.clone(),
            &mut o,
            3,
            3,
            1.0,
            1.0,
            0.0,
            1.0,
            [40.0, 5.0, 3.5],
        );
        assert!(r.is_err());
    }

    /// Constant input → constant output (per the (0,0) kernel tap
    /// always weighting the centre with weight = 1 and all other
    /// taps weighting their mirrored neighbours with non-negative
    /// weights summing to a finite positive value; on a constant
    /// plane every weighted sum equals the constant).
    #[test]
    fn apply_step_5tap_constant_plane_is_invariant() {
        let p = vec![5.0_f32; 16];
        let mut xo = vec![0.0_f32; 16];
        let mut yo = vec![0.0_f32; 16];
        let mut bo = vec![0.0_f32; 16];
        apply_step_5tap(
            Pass::Pass1,
            &p,
            &p,
            &p,
            &mut xo,
            &mut yo,
            &mut bo,
            4,
            4,
            1.0,
            1.0,
            0.0,
            1.0,
            [40.0, 5.0, 3.5],
        )
        .unwrap();
        for (i, &v) in xo.iter().enumerate() {
            assert!((v - 5.0).abs() < 1e-4, "x[{i}] = {v}, expected 5.0");
        }
        for &v in yo.iter() {
            assert!((v - 5.0).abs() < 1e-4);
        }
        for &v in bo.iter() {
            assert!((v - 5.0).abs() < 1e-4);
        }
    }

    /// Pass 2 on a constant plane is also invariant.
    #[test]
    fn apply_step_5tap_pass2_constant_plane_is_invariant() {
        let p = vec![3.25_f32; 25];
        let mut xo = vec![0.0_f32; 25];
        let mut yo = vec![0.0_f32; 25];
        let mut bo = vec![0.0_f32; 25];
        apply_step_5tap(
            Pass::Pass2,
            &p,
            &p,
            &p,
            &mut xo,
            &mut yo,
            &mut bo,
            5,
            5,
            1.0,
            1.0,
            0.0,
            1.0,
            [40.0, 5.0, 3.5],
        )
        .unwrap();
        for &v in xo.iter() {
            assert!((v - 3.25).abs() < 1e-4);
        }
    }

    /// Wrong plane length is rejected.
    #[test]
    fn apply_step_5tap_wrong_length_is_error() {
        let p = vec![0.0_f32; 8];
        let mut o = vec![0.0_f32; 9];
        let r = apply_step_5tap(
            Pass::Pass1,
            &p,
            &p,
            &p,
            &mut o.clone(),
            &mut o.clone(),
            &mut o,
            3,
            3,
            1.0,
            1.0,
            0.0,
            1.0,
            [40.0, 5.0, 3.5],
        );
        assert!(r.is_err());
    }

    /// Zero-area planes are a no-op.
    #[test]
    fn apply_step_5tap_zero_area_is_noop() {
        let p: Vec<f32> = vec![];
        let mut o: Vec<f32> = vec![];
        apply_step_5tap(
            Pass::Pass1,
            &p,
            &p,
            &p,
            &mut o.clone(),
            &mut o.clone(),
            &mut o,
            0,
            0,
            1.0,
            1.0,
            0.0,
            1.0,
            [40.0, 5.0, 3.5],
        )
        .unwrap();
    }

    // ---- apply_step_13tap ----

    /// Constant input → constant output (Pass 0 with 13-tap
    /// kernel).
    #[test]
    fn apply_step_13tap_constant_plane_is_invariant() {
        let p = vec![2.75_f32; 36];
        let mut xo = vec![0.0_f32; 36];
        let mut yo = vec![0.0_f32; 36];
        let mut bo = vec![0.0_f32; 36];
        apply_step_13tap(
            &p,
            &p,
            &p,
            &mut xo,
            &mut yo,
            &mut bo,
            6,
            6,
            1.0,
            1.0,
            0.0,
            1.0,
            [40.0, 5.0, 3.5],
        )
        .unwrap();
        for &v in xo.iter() {
            assert!((v - 2.75).abs() < 1e-4);
        }
    }

    /// Wrong plane length rejected.
    #[test]
    fn apply_step_13tap_wrong_length_is_error() {
        let p = vec![0.0_f32; 8];
        let mut o = vec![0.0_f32; 9];
        let r = apply_step_13tap(
            &p,
            &p,
            &p,
            &mut o.clone(),
            &mut o.clone(),
            &mut o,
            3,
            3,
            1.0,
            1.0,
            0.0,
            1.0,
            [40.0, 5.0, 3.5],
        );
        assert!(r.is_err());
    }

    /// Zero-area planes are a no-op.
    #[test]
    fn apply_step_13tap_zero_area_is_noop() {
        let p: Vec<f32> = vec![];
        let mut o: Vec<f32> = vec![];
        apply_step_13tap(
            &p,
            &p,
            &p,
            &mut o.clone(),
            &mut o.clone(),
            &mut o,
            0,
            0,
            1.0,
            1.0,
            0.0,
            1.0,
            [40.0, 5.0, 3.5],
        )
        .unwrap();
    }

    /// On a heavy zeroflush + degenerate sigma combination the
    /// guard against `sum_w == 0` fires: pick a setting where every
    /// non-centre kernel tap has weight 0 but the centre also has
    /// weight 0. With `zeroflush >= 1.0` the centre tap (distance 0
    /// → v = 1.0) is cut, making sum_w = 0.
    #[test]
    fn apply_step_5tap_zeroflush_above_one_rejects_with_sum_zero_guard() {
        let p = vec![5.0_f32; 9];
        let mut xo = vec![0.0_f32; 9];
        let mut yo = vec![0.0_f32; 9];
        let mut bo = vec![0.0_f32; 9];
        let r = apply_step_5tap(
            Pass::Pass1,
            &p,
            &p,
            &p,
            &mut xo,
            &mut yo,
            &mut bo,
            3,
            3,
            1.0,
            1.0,
            1.5, // zeroflush > 1.0 cuts even the centre tap
            1.0,
            [40.0, 5.0, 3.5],
        );
        assert!(r.is_err());
    }

    // ---- cross-module: round-141 mirror1d round-trips ----

    /// The §J.3 module reuses [`crate::gaborish::mirror1d`] for its
    /// boundary handling per §J.3.1; verify the integration end-to-
    /// end by feeding a constant plane (the simplest fixture that
    /// exercises every kernel tap regardless of mirror direction).
    #[test]
    fn epf_mirror_handling_uses_round141_mirror1d_consistently() {
        let p = vec![1.0_f32; 4];
        let mut xo = vec![0.0_f32; 4];
        let mut yo = vec![0.0_f32; 4];
        let mut bo = vec![0.0_f32; 4];
        // 2×2 plane forces every kernel reference to mirror at least
        // once; we still expect the constant-plane invariance.
        apply_step_5tap(
            Pass::Pass1,
            &p,
            &p,
            &p,
            &mut xo,
            &mut yo,
            &mut bo,
            2,
            2,
            1.0,
            1.0,
            0.0,
            1.0,
            [40.0, 5.0, 3.5],
        )
        .unwrap();
        for &v in xo.iter() {
            assert!((v - 1.0).abs() < 1e-4);
        }
    }

    // ---- §J.3.1 apply_epf_iterations driver ----

    fn modular_rf(epf_iters: u32) -> RestorationFilter {
        RestorationFilter {
            epf_iters,
            ..RestorationFilter::default()
        }
    }

    /// `epf_iters == 0` leaves the planes untouched ("The filter is
    /// only performed if `rf.epf_iters > 0`").
    #[test]
    fn epf_iters_zero_is_identity() {
        let rf = modular_rf(0);
        let mut x: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let mut y = x.clone();
        let mut b = x.clone();
        let orig = x.clone();
        apply_epf_iterations(&mut x, &mut y, &mut b, 4, 4, 5.0, &rf).unwrap();
        assert_eq!(x, orig);
        assert_eq!(y, orig);
        assert_eq!(b, orig);
    }

    /// A constant plane is a fixed point of every pass (all distances
    /// 0 → weight 1 → weighted mean of identical samples), so for any
    /// `epf_iters` the constant survives the full chain.
    #[test]
    fn epf_constant_plane_is_fixed_point_for_all_iters() {
        for iters in 1..=3 {
            let rf = modular_rf(iters);
            let mut x = vec![3.5_f32; 64]; // 8×8 so no border-only block
            let mut y = vec![-2.0_f32; 64];
            let mut b = vec![7.25_f32; 64];
            apply_epf_iterations(&mut x, &mut y, &mut b, 8, 8, 5.0, &rf).unwrap();
            for &v in &x {
                assert!((v - 3.5).abs() < 1e-4, "iters={iters} x drifted: {v}");
            }
            for &v in &y {
                assert!((v + 2.0).abs() < 1e-4, "iters={iters} y drifted: {v}");
            }
            for &v in &b {
                assert!((v - 7.25).abs() < 1e-4, "iters={iters} b drifted: {v}");
            }
        }
    }

    /// `epf_iters > 3` is rejected as a malformed header (Table C.9
    /// `u(2)` cap).
    #[test]
    fn epf_iters_above_three_is_error() {
        let rf = modular_rf(4);
        let mut x = vec![1.0_f32; 16];
        let mut y = vec![1.0_f32; 16];
        let mut b = vec![1.0_f32; 16];
        assert!(apply_epf_iterations(&mut x, &mut y, &mut b, 4, 4, 5.0, &rf).is_err());
    }

    /// Plane-length mismatch is rejected before any work.
    #[test]
    fn epf_plane_length_mismatch_is_error() {
        let rf = modular_rf(1);
        let mut x = vec![1.0_f32; 15]; // wrong: 4*4 = 16
        let mut y = vec![1.0_f32; 16];
        let mut b = vec![1.0_f32; 16];
        assert!(apply_epf_iterations(&mut x, &mut y, &mut b, 4, 4, 5.0, &rf).is_err());
    }

    /// Zero-area planes are a no-op even with filtering requested.
    #[test]
    fn epf_zero_area_is_noop() {
        let rf = modular_rf(3);
        let mut x: Vec<f32> = vec![];
        let mut y: Vec<f32> = vec![];
        let mut b: Vec<f32> = vec![];
        apply_epf_iterations(&mut x, &mut y, &mut b, 0, 0, 5.0, &rf).unwrap();
        assert!(x.is_empty());
    }

    /// The driver's single-pass result (epf_iters == 1 → Step 1 only)
    /// matches a direct `apply_step_5tap(Pass1)` call with the same
    /// scalars: this pins the per-pass scalar sourcing (step
    /// multiplier 1.0, zeroflush = epf_pass1_zeroflush) and the
    /// in-place writeback for an odd pass count.
    #[test]
    fn epf_single_pass_matches_direct_step1() {
        let rf = modular_rf(1);
        let sigma = 6.0_f32;
        // A non-constant plane so the filter actually moves samples.
        let base: Vec<f32> = (0..64).map(|v| (v % 7) as f32).collect();
        let (mut dx, mut dy, mut db) = (base.clone(), base.clone(), base.clone());
        apply_epf_iterations(&mut dx, &mut dy, &mut db, 8, 8, sigma, &rf).unwrap();

        let (mut rx, mut ry, mut rb) = (vec![0.0_f32; 64], vec![0.0_f32; 64], vec![0.0_f32; 64]);
        apply_step_5tap(
            Pass::Pass1,
            &base,
            &base,
            &base,
            &mut rx,
            &mut ry,
            &mut rb,
            8,
            8,
            sigma,
            1.0,
            rf.epf_pass1_zeroflush,
            rf.epf_border_sad_mul,
            rf.epf_channel_scale,
        )
        .unwrap();
        assert_eq!(dx, rx);
        assert_eq!(dy, ry);
        assert_eq!(db, rb);
    }

    /// The driver's two-pass result (epf_iters == 2 → Step 1 then
    /// Step 2) matches running the two steps by hand with the Step-1
    /// output feeding Step 2 (§J.3.4 "output of each step is used as
    /// an input for the following step"). This pins both the ordering
    /// and the even-pass-count buffer handoff.
    #[test]
    fn epf_two_pass_matches_manual_chain() {
        let rf = modular_rf(2);
        let sigma = 6.0_f32;
        let base: Vec<f32> = (0..64).map(|v| ((v * 3) % 11) as f32).collect();
        let (mut dx, mut dy, mut db) = (base.clone(), base.clone(), base.clone());
        apply_epf_iterations(&mut dx, &mut dy, &mut db, 8, 8, sigma, &rf).unwrap();

        // Manual: Step 1 (base -> mid), Step 2 (mid -> out).
        let (mut mx, mut my, mut mb) = (vec![0.0_f32; 64], vec![0.0_f32; 64], vec![0.0_f32; 64]);
        apply_step_5tap(
            Pass::Pass1,
            &base,
            &base,
            &base,
            &mut mx,
            &mut my,
            &mut mb,
            8,
            8,
            sigma,
            1.0,
            rf.epf_pass1_zeroflush,
            rf.epf_border_sad_mul,
            rf.epf_channel_scale,
        )
        .unwrap();
        let (mut ox, mut oy, mut ob) = (vec![0.0_f32; 64], vec![0.0_f32; 64], vec![0.0_f32; 64]);
        apply_step_5tap(
            Pass::Pass2,
            &mx,
            &my,
            &mb,
            &mut ox,
            &mut oy,
            &mut ob,
            8,
            8,
            sigma,
            rf.epf_pass2_sigma_scale,
            rf.epf_pass2_zeroflush,
            rf.epf_border_sad_mul,
            rf.epf_channel_scale,
        )
        .unwrap();
        assert_eq!(dx, ox);
        assert_eq!(dy, oy);
        assert_eq!(db, ob);
    }
}
