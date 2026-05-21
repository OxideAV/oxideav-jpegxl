//! `GetDCTQuantWeights` + per-`DctSelect` dequantization-matrix
//! materialisation — ISO/IEC 18181-1:2024 §I.2.4 / §I.2.5 + Table I.4
//! + Table I.6.
//!
//! ## Spec mapping
//!
//! This module transcribes Annex I.2.4's `GetDCTQuantWeights()` /
//! `Interpolate()` / `Mult()` listings (page 58 of the 2024 final
//! core PDF), Table I.4 (DctSelect → weights-matrix dimensions, page
//! 57), and Table I.6 (Default matrix parameters for each DctSelect,
//! page 60). The per-mode weights-derivation rules (DCT, DCT4, DCT2,
//! Hornuss, DCT4x8, AFV, RAW) are transcribed from the same listing
//! block on page 58 and the per-mode prose on page 59.
//!
//! ## Pure-math entry points (this round)
//!
//! * [`Mult`] / [`interpolate`] / [`compute_dct_weights`] — the
//!   primitives. Each one-to-one with the Listing on page 58.
//! * [`materialise_weights_for_dct_select`] — given the bundle of
//!   parsed-or-defaulted parameters for one `DctSelect` slot,
//!   produce the X×Y weights matrix per the mode's per-spec rule.
//! * [`materialise_default_weights_for_dct_select`] — convenience
//!   wrapper that materialises Table I.6's default-encoding weights
//!   for one slot (used by the `u(1)=1` HfGlobal fast path).
//! * [`DequantMatrixSet`] / [`materialise_default_dequant_set`] —
//!   the 17-slot full default set (Table I.6) materialised as
//!   dequantization matrices (element-wise reciprocal of the
//!   weights matrix per the §I.2.4 last-paragraph "the
//!   dequantization matrices for channel c are the element-wise
//!   reciprocals of the weights matrix" rule).
//!
//! ## Spec listing typo notes
//!
//! The ISO/IEC FDIS 18181-1:2021 PDF Listing C.10 places the
//! `for (y, x) { ... weights(x, y) = weight; }` weights-matrix loop
//! *inside* the `for (i = 1; i < len; i++) { bands(i) = ... }` bands
//! loop, which would compute the weights matrix `len - 1` times.
//! This is a spec listing typo. The 2024 published edition
//! (`docs/image/jpegxl/ISO_IEC_18181-1-JPEG-XL-Core-2024.pdf`,
//! page 58) corrects this: the bands loop closes before the weights
//! double-loop starts. This module follows the 2024 corrected form.
//!
//! The 2024 listing also drops the `len` argument from `Interpolate`
//! (uses `bands.size()` directly) and writes `pow(B / A, frac_index)`
//! in place of FDIS 2021's `A * (B / A)^frac_index`. The two are
//! mathematically identical (and equivalent to `A^(1 - f) * B^f`);
//! this module factors out `A *` from the exponent for numerical
//! stability when `A` is near zero.
//!
//! ## Numeric-literal precision note
//!
//! Table I.6 quotes many `f64` literals with 15+ digits of mantissa
//! (e.g. `8996.8725711814115328`). Some of these exceed the
//! representable precision of `f64`; clippy's
//! `excessive_precision` lint flags them. The literals are
//! transcribed verbatim from the published spec for provenance
//! reasons (so a reader can grep them back to the PDF), so the
//! lint is allowed at module level. The runtime f64 conversion
//! rounds to nearest representable, which is the behaviour every
//! conforming decoder gets.
//!
//! ## What this module does NOT do
//!
//! * It does not consume any bitstream — that's [`crate::hf_global`].
//! * It does not apply the per-channel HF-coefficient multiply —
//!   that's the round 14+ HF dequantisation step (§F.3).
//! * The Table I.4 row for `RAW` mode is fully described by §I.2.4
//!   ("the dequantization matrices are equal to the params matrices
//!   multiplied by params.denominator"); RAW dispatch is exposed
//!   here but the RAW-params decoder lives in [`crate::hf_global`]
//!   and currently rejects with `Error::Unsupported` (round 15+).

#![allow(clippy::excessive_precision)]

use crate::dct_select::TransformType;
use crate::hf_global::{DequantMatrixParams, EncodingMode};
use oxideav_core::{Error, Result};

/// `Mult(v)` per ISO/IEC 18181-1:2024 §I.2.4 listing (page 58):
///
/// ```text
/// Mult(v) { if (v > 0) return 1 + v; else return 1 / (1 - v); }
/// ```
#[inline]
pub fn mult(v: f64) -> f64 {
    if v > 0.0 {
        1.0 + v
    } else {
        1.0 / (1.0 - v)
    }
}

/// `Interpolate(pos, max, bands)` per ISO/IEC 18181-1:2024 §I.2.4
/// listing (page 58):
///
/// ```text
/// Interpolate(pos, max, bands) {
///   if (bands.size() == 1) return bands[0];
///   scaled_pos = pos * (bands.size() - 1) / max;
///   scaled_index = floor(scaled_pos);
///   frac_index = scaled_pos - scaled_index;
///   A = bands[scaled_index];
///   B = bands[scaled_index + 1];
///   interpolated_value = A * pow(B / A, frac_index);
///   return interpolated_value;
/// }
/// ```
///
/// Returns an error if `bands` is empty or `max <= 0`. Per Listing
/// C.10 the only caller `compute_dct_weights` passes `bands` directly
/// from the per-slot parameters and `max = sqrt(2) + 1e-6`, so the
/// `max > 0` invariant is structurally guaranteed; we still check
/// defensively.
pub fn interpolate(pos: f64, max: f64, bands: &[f64]) -> Result<f64> {
    if bands.is_empty() {
        return Err(Error::InvalidData("JXL Interpolate: bands is empty".into()));
    }
    if bands.len() == 1 {
        return Ok(bands[0]);
    }
    if max <= 0.0 {
        return Err(Error::InvalidData(format!(
            "JXL Interpolate: max must be positive (got {max})"
        )));
    }
    let scaled_pos = pos * (bands.len() as f64 - 1.0) / max;
    let scaled_index_f = scaled_pos.floor();
    // `scaled_index` should be in [0, bands.len() - 2]; clamp
    // defensively because `pos` may equal `max` exactly (which
    // would otherwise index `bands.len() - 1` and crash on the
    // `[scaled_index + 1]` read).
    let max_index = bands.len() - 2;
    let scaled_index = (scaled_index_f as i64).clamp(0, max_index as i64) as usize;
    let frac_index = scaled_pos - scaled_index as f64;
    let a = bands[scaled_index];
    let b = bands[scaled_index + 1];
    // 2024 listing form: A * pow(B / A, frac_index). For numerical
    // safety, when A == 0, fall back to direct A^(1-f) * B^f which
    // is well-defined for B == 0 (yields 0).
    if a == 0.0 {
        if b == 0.0 {
            return Ok(0.0);
        }
        // A = 0 means bands[scaled_index] = 0; by the bands
        // recurrence (each band = previous * Mult(param), Mult
        // never returns 0 for finite v) this should never occur on
        // a valid params input. Return 0 to match the limiting
        // behaviour of `A^(1-f) * B^f` as A → 0+.
        return Ok(0.0);
    }
    let ratio = b / a;
    // Reject negative ratio (would produce NaN through pow with
    // non-integer exponent). The post-bands `[[ bands[i] > 0 ]]`
    // invariant on page 58 guarantees this; report defensively.
    if ratio < 0.0 {
        return Err(Error::InvalidData(format!(
            "JXL Interpolate: negative ratio {ratio} (A={a}, B={b})"
        )));
    }
    Ok(a * ratio.powf(frac_index))
}

/// `GetDCTQuantWeights(params)` per ISO/IEC 18181-1:2024 §I.2.4
/// listing (page 58, post-typo-fix form):
///
/// ```text
/// GetDCTQuantWeights(params) {
///   bands.clear();
///   bands.push_back(params[0]);
///   for (i = 1; i < params.size(); i++) {
///     bands.push_back(bands[i - 1] * Mult(params[i]));
///     /* bands[i] > 0 */
///   }
///   for (y = 0; y < Y; y++) {
///     for (x = 0; x < X; x++) {
///       dx = x / (X - 1);
///       dy = y / (Y - 1);
///       distance = sqrt(dx*dx + dy*dy);
///       weight = Interpolate(distance, sqrt(2) + 1e-6, bands);
///       weights(x, y) = weight;
///     }
///   }
///   return weights;
/// }
/// ```
///
/// `params` is the row of parameters for a single channel.
/// `(x_dim, y_dim)` are the X×Y dimensions of the target weights
/// matrix.
///
/// Output: row-major `(x_dim * y_dim)` `Vec<f64>`. Cell `(x, y)`
/// lives at index `y * x_dim + x`.
///
/// Errors when `params` is empty, when either dimension is zero,
/// or when the bands recurrence violates the spec's
/// `bands[i] > 0` invariant.
pub fn compute_dct_weights(params: &[f64], x_dim: u32, y_dim: u32) -> Result<Vec<f64>> {
    if params.is_empty() {
        return Err(Error::InvalidData(
            "JXL GetDCTQuantWeights: params is empty".into(),
        ));
    }
    if x_dim == 0 || y_dim == 0 {
        return Err(Error::InvalidData(format!(
            "JXL GetDCTQuantWeights: dimensions must be positive (got {x_dim}x{y_dim})"
        )));
    }
    let mut bands: Vec<f64> = Vec::with_capacity(params.len());
    bands.push(params[0]);
    for i in 1..params.len() {
        let next = bands[i - 1] * mult(params[i]);
        if next <= 0.0 || !next.is_finite() {
            return Err(Error::InvalidData(format!(
                "JXL GetDCTQuantWeights: bands[{i}] = {next} violates `bands[i] > 0` invariant"
            )));
        }
        bands.push(next);
    }
    let total = (x_dim as usize) * (y_dim as usize);
    let mut weights = vec![0.0f64; total];
    // sqrt(2) + 1e-6 per spec.
    let max_distance = std::f64::consts::SQRT_2 + 1e-6;
    for y in 0..y_dim {
        for x in 0..x_dim {
            // The spec writes `dx = x / (X - 1)` which would be
            // 0/0 on X == 1; the only X == 1 caller (DCT4x8 in
            // certain configurations) has `params.size() == 1`
            // so Interpolate's `len == 1` short-circuit fires
            // before `pos` matters. Be defensive when `X-1 == 0`
            // by pinning dx (resp. dy) to 0.
            let dx = if x_dim > 1 {
                x as f64 / (x_dim as f64 - 1.0)
            } else {
                0.0
            };
            let dy = if y_dim > 1 {
                y as f64 / (y_dim as f64 - 1.0)
            } else {
                0.0
            };
            let distance = (dx * dx + dy * dy).sqrt();
            let w = interpolate(distance, max_distance, &bands)?;
            weights[(y as usize) * (x_dim as usize) + (x as usize)] = w;
        }
    }
    Ok(weights)
}

/// Table I.4 column 3 — weights-matrix dimensions `(X, Y)` for a
/// `DctSelect` value (page 57 of the 2024 final core PDF):
///
/// | Parameters index | DctSelect             | Matrix size (rows × cols) |
/// |------------------|-----------------------|---------------------------|
/// | 0                | DCT8×8                | 8×8                       |
/// | 1                | Hornuss               | 8×8                       |
/// | 2                | DCT2×2                | 8×8                       |
/// | 3                | DCT4×4                | 8×8                       |
/// | 4                | DCT16×16              | 16×16                     |
/// | 5                | DCT32×32              | 32×32                     |
/// | 6                | DCT16×8, DCT8×16      | 8×16                      |
/// | 7                | DCT32×8, DCT8×32      | 8×32                      |
/// | 8                | DCT16×32, DCT32×16    | 16×32                     |
/// | 9                | DCT4×8, DCT8×4        | 8×8                       |
/// | 10               | AFV0..AFV3            | 8×8                       |
/// | 11               | DCT64×64              | 64×64                     |
/// | 12               | DCT32×64, DCT64×32    | 32×64                     |
/// | 13               | DCT128×128            | 128×128                   |
/// | 14               | DCT64×128, DCT128×64  | 64×128                    |
/// | 15               | DCT256×256            | 256×256                   |
/// | 16               | DCT128×256, DCT256×128| 128×256                   |
///
/// Returned as `(x_dim, y_dim)` where x_dim is "cols" and y_dim is
/// "rows".  The DCT family carries the longer side first in its
/// transform name (`DCT16×8` = 16 rows × 8 cols), but Table I.4's
/// "Matrix size" column already canonicalises the smaller-first
/// shape per the shared per-slot entry; this helper returns
/// `(x_dim = cols, y_dim = rows)` matching `compute_dct_weights`'s
/// X × Y argument convention.
pub fn weights_matrix_dims_for_slot(slot_index: u32) -> Result<(u32, u32)> {
    Ok(match slot_index {
        // 0..3 — 8×8 family.
        0..=3 => (8, 8),
        // 4 — DCT16×16.
        4 => (16, 16),
        // 5 — DCT32×32.
        5 => (32, 32),
        // 6 — DCT16×8/DCT8×16: Table I.4 says 8×16 (rows × cols).
        // x_dim = 16 cols, y_dim = 8 rows.
        6 => (16, 8),
        // 7 — DCT32×8/DCT8×32: Table I.4 says 8×32 (rows × cols).
        7 => (32, 8),
        // 8 — DCT16×32/DCT32×16: Table I.4 says 16×32 (rows × cols).
        8 => (32, 16),
        // 9 — DCT4×8/DCT8×4: 8×8.
        9 => (8, 8),
        // 10 — AFV0..AFV3: 8×8.
        10 => (8, 8),
        // 11 — DCT64×64.
        11 => (64, 64),
        // 12 — DCT32×64/DCT64×32: Table I.4 says 32×64 (rows × cols).
        12 => (64, 32),
        // 13 — DCT128×128.
        13 => (128, 128),
        // 14 — DCT64×128/DCT128×64: Table I.4 says 64×128.
        14 => (128, 64),
        // 15 — DCT256×256.
        15 => (256, 256),
        // 16 — DCT128×256/DCT256×128: Table I.4 says 128×256.
        16 => (256, 128),
        other => {
            return Err(Error::InvalidData(format!(
                "JXL dequant-matrix slot index {other} out of range 0..=16 (Table I.4)"
            )));
        }
    })
}

/// Map a [`TransformType`] (Table C.16 0..=26 variant) to the
/// Table I.4 slot index (0..=16) that holds its weights-matrix
/// parameters. Multiple transforms share a single slot (e.g.
/// DCT16×8 and DCT8×16 both use slot 6).
pub fn slot_for_transform(t: TransformType) -> u32 {
    match t {
        TransformType::Dct8x8 => 0,
        TransformType::Hornuss => 1,
        TransformType::Dct2x2 => 2,
        TransformType::Dct4x4 => 3,
        TransformType::Dct16x16 => 4,
        TransformType::Dct32x32 => 5,
        TransformType::Dct16x8 | TransformType::Dct8x16 => 6,
        TransformType::Dct32x8 | TransformType::Dct8x32 => 7,
        TransformType::Dct16x32 | TransformType::Dct32x16 => 8,
        TransformType::Dct4x8 | TransformType::Dct8x4 => 9,
        TransformType::Afv0 | TransformType::Afv1 | TransformType::Afv2 | TransformType::Afv3 => 10,
        TransformType::Dct64x64 => 11,
        TransformType::Dct32x64 | TransformType::Dct64x32 => 12,
        TransformType::Dct128x128 => 13,
        TransformType::Dct64x128 | TransformType::Dct128x64 => 14,
        TransformType::Dct256x256 => 15,
        TransformType::Dct128x256 | TransformType::Dct256x128 => 16,
    }
}

/// Per-channel weights matrix for a single dequantization-matrix
/// slot. `mode == EncodingMode::Library` means "use Table I.6
/// defaults" — call [`materialise_default_weights_for_dct_select`]
/// in that case.
///
/// `params` carries the per-channel parsed parameters (the
/// `DequantMatrixParams` bundle for that slot). `channel` is in
/// {0, 1, 2} for X/Y/B. `(x_dim, y_dim)` are the dimensions from
/// [`weights_matrix_dims_for_slot`].
///
/// Returns an `(x_dim * y_dim)` row-major `Vec<f64>`.
///
/// Per §I.2.4 last paragraph the dequantization matrix for channel
/// `c` is the element-wise reciprocal of the weights matrix
/// computed for channel `c`; that reciprocal is applied by
/// [`materialise_dequant_for_channel`] downstream, not here.
pub fn materialise_weights_for_dct_select(
    bundle: &DequantMatrixParams,
    channel: usize,
    x_dim: u32,
    y_dim: u32,
) -> Result<Vec<f64>> {
    if channel >= 3 {
        return Err(Error::InvalidData(format!(
            "JXL materialise_weights: channel {channel} out of range [0, 3)"
        )));
    }
    match bundle.mode {
        EncodingMode::Library => Err(Error::InvalidData(
            "JXL materialise_weights: Library mode has no in-stream params; \
             call materialise_default_weights_for_dct_select() instead"
                .into(),
        )),
        EncodingMode::Hornuss => materialise_hornuss(bundle, channel, x_dim, y_dim),
        EncodingMode::Dct2 => materialise_dct2(bundle, channel, x_dim, y_dim),
        EncodingMode::Dct4 => materialise_dct4(bundle, channel, x_dim, y_dim),
        EncodingMode::Dct4x8 => materialise_dct4x8(bundle, channel, x_dim, y_dim),
        EncodingMode::Afv => materialise_afv(bundle, channel, x_dim, y_dim),
        EncodingMode::Dct => materialise_dct(bundle, channel, x_dim, y_dim),
        EncodingMode::Raw => Err(Error::Unsupported(
            "JXL materialise_weights: RAW mode (modular sub-bitstream) deferred to round 15+"
                .into(),
        )),
    }
}

/// Encoding mode DCT per §I.2.4 page 58: the weights matrix for
/// channel c is the matrix of the correct size computed using
/// `GetDctQuantWeights`, using row c of `dct_params` as input.
fn materialise_dct(
    bundle: &DequantMatrixParams,
    channel: usize,
    x_dim: u32,
    y_dim: u32,
) -> Result<Vec<f64>> {
    let row = row_of_dct_params(bundle, channel)?;
    compute_dct_weights(&row, x_dim, y_dim)
}

/// Encoding mode DCT4 per §I.2.4 page 58: copy into position (x, y)
/// the value in position (x Idiv 2, y Idiv 2) of the 4×4 matrix
/// computed by `GetDctQuantWeights` using row c of `dct_params`;
/// coefficients (0,1) and (1,0) are divided by `params(c, 0)`, and
/// the (1,1) coefficient is divided by `params(c, 1)`.
///
/// Per Table I.4 the output is 8×8 for slot 3 (DCT4×4) — confirmed
/// by the spec's "the 4 x 4 matrix" wording (the small input expands
/// via floor division into the 8×8 output).
fn materialise_dct4(
    bundle: &DequantMatrixParams,
    channel: usize,
    x_dim: u32,
    y_dim: u32,
) -> Result<Vec<f64>> {
    // `dct_params` row is the input to GetDCTQuantWeights at 4×4.
    let row = row_of_dct_params(bundle, channel)?;
    let small = compute_dct_weights(&row, 4, 4)?;
    // Pad up by floor-divide: out(x, y) = small(x/2, y/2).
    let mut out = vec![0.0f64; (x_dim as usize) * (y_dim as usize)];
    for y in 0..y_dim {
        for x in 0..x_dim {
            let sx = (x / 2) as usize;
            let sy = (y / 2) as usize;
            out[(y as usize) * (x_dim as usize) + (x as usize)] = small[sy * 4 + sx];
        }
    }
    // Divide (0,1) and (1,0) by params(c, 0); divide (1,1) by params(c, 1).
    // `params` is row-major 3 × 2 = 6 floats; row c starts at c * 2.
    if bundle.params_cols != 2 || bundle.params.len() != 6 {
        return Err(Error::InvalidData(format!(
            "JXL DCT4: expected params to be 3x2 (6 elements, cols=2), got {} elements (cols={})",
            bundle.params.len(),
            bundle.params_cols
        )));
    }
    let p0 = bundle.params[channel * 2] as f64;
    let p1 = bundle.params[channel * 2 + 1] as f64;
    if p0 == 0.0 || p1 == 0.0 {
        return Err(Error::InvalidData(format!(
            "JXL DCT4: per-channel divisor is zero (p0={p0}, p1={p1})"
        )));
    }
    // Defensive: skip the corrections if the output is smaller than
    // 2x2 — shouldn't happen for the canonical 8x8 slot but the
    // helper is reusable.
    if x_dim >= 2 && y_dim >= 2 {
        let stride = x_dim as usize;
        out[1] /= p0;
        out[stride] /= p0;
        out[stride + 1] /= p1;
    }
    Ok(out)
}

/// Encoding mode DCT2 per §I.2.4 page 58: `params(c, i)` are copied
/// into positions defined by 6 spec rules indexed by `i` 0..5.
/// Coefficient (0,0) is implicitly left as 0 by the spec listing;
/// the inverse to dequant matrix is the element-wise reciprocal so
/// (0,0) needs an explicit value — per the spec text and Table I.6's
/// `{}` dct_params + 6-column params per channel, the (0,0) cell is
/// covered by `i == 0`'s "positions (0,1) and (1, 0)" rule by virtue
/// of being THE remaining unmentioned 8×8 cell, OR it's a separate
/// implicit fill. Reading the spec strictly: only the listed
/// rectangles get set; (0,0) is left at the default and the
/// element-wise reciprocal that downstream applies will produce
/// `Inf` from `1 / 0`. To avoid that, we explicitly fill (0,0) with
/// the `params(c, 0)` value matching the "i == 0" cells — which
/// matches the only sensible interpretation (the DC weight is the
/// largest = lowest frequency = same as (0,1)/(1,0)).
///
/// SPECGAP: page 58 listing for DCT2 doesn't specify the (0,0)
/// position. We populate it with `params(c, 0)` (same as i==0
/// positions) to make the matrix invertible. Recommend a spec
/// clarification.
fn materialise_dct2(
    bundle: &DequantMatrixParams,
    channel: usize,
    x_dim: u32,
    y_dim: u32,
) -> Result<Vec<f64>> {
    if bundle.params_cols != 6 || bundle.params.len() != 18 {
        return Err(Error::InvalidData(format!(
            "JXL DCT2: expected params to be 3x6 (18 elements, cols=6), got {} (cols={})",
            bundle.params.len(),
            bundle.params_cols
        )));
    }
    if x_dim < 8 || y_dim < 8 {
        return Err(Error::InvalidData(format!(
            "JXL DCT2: expected at least 8x8 output (got {x_dim}x{y_dim})"
        )));
    }
    let p = |i: usize| bundle.params[channel * 6 + i] as f64;
    let stride = x_dim as usize;
    let mut out = vec![0.0f64; (x_dim as usize) * (y_dim as usize)];
    let set = |out: &mut [f64], x: usize, y: usize, v: f64| {
        out[y * stride + x] = v;
    };
    // (0,0) — see SPECGAP note above. Fill with p(0).
    set(&mut out, 0, 0, p(0));
    // i == 0: positions (0,1) and (1, 0).
    set(&mut out, 0, 1, p(0));
    set(&mut out, 1, 0, p(0));
    // i == 1: position (1,1).
    set(&mut out, 1, 1, p(1));
    // i == 2: all positions in rectangle ((2,0), (4,2)), and symmetric.
    //   The "and symmetric" wording per page 59 means "swap x and y".
    //   ((2,0), (4,2)) covers x in [2, 4) × y in [0, 2) — i.e.
    //   (2,0), (3,0), (2,1), (3,1).
    //   Symmetric: (0,2), (0,3), (1,2), (1,3).
    for y in 0..2 {
        for x in 2..4 {
            set(&mut out, x, y, p(2));
            set(&mut out, y, x, p(2));
        }
    }
    // i == 3: rectangle ((2,2), (4,4)).
    for y in 2..4 {
        for x in 2..4 {
            set(&mut out, x, y, p(3));
        }
    }
    // i == 4: rectangle ((4,0), (8,4)), and symmetric.
    for y in 0..4 {
        for x in 4..8 {
            set(&mut out, x, y, p(4));
            set(&mut out, y, x, p(4));
        }
    }
    // i == 5: rectangle ((4,4), (8,8)).
    for y in 4..8 {
        for x in 4..8 {
            set(&mut out, x, y, p(5));
        }
    }
    Ok(out)
}

/// Encoding mode Hornuss per §I.2.4 page 59: coefficient (1,1) is
/// equal to params(c, 2). Coefficients (0,1) and (1,0) are equal to
/// params(c, 1), and all other coefficients to params(c, 0).
/// Coefficient (0,0) is 1.
fn materialise_hornuss(
    bundle: &DequantMatrixParams,
    channel: usize,
    x_dim: u32,
    y_dim: u32,
) -> Result<Vec<f64>> {
    if bundle.params_cols != 3 || bundle.params.len() != 9 {
        return Err(Error::InvalidData(format!(
            "JXL Hornuss: expected params to be 3x3 (9 elements, cols=3), got {} (cols={})",
            bundle.params.len(),
            bundle.params_cols
        )));
    }
    if x_dim < 2 || y_dim < 2 {
        return Err(Error::InvalidData(format!(
            "JXL Hornuss: expected at least 2x2 output (got {x_dim}x{y_dim})"
        )));
    }
    let p = |i: usize| bundle.params[channel * 3 + i] as f64;
    let stride = x_dim as usize;
    let mut out = vec![p(0); (x_dim as usize) * (y_dim as usize)];
    // (0,0) = 1.
    out[0] = 1.0;
    // (0,1) = (1,0) = params(c, 1).
    out[stride] = p(1);
    out[1] = p(1);
    // (1,1) = params(c, 2).
    out[stride + 1] = p(2);
    Ok(out)
}

/// Encoding mode DCT4x8 per §I.2.4 page 59: the weights matrix is
/// obtained by copying into position (x, y) the value in position
/// (x, y Idiv 2) in the 4 × 8 matrix computed by `GetDctQuantWeights`
/// using row c of `dct_params`; coefficient (0,1) is then divided
/// by params(c, 0).
fn materialise_dct4x8(
    bundle: &DequantMatrixParams,
    channel: usize,
    x_dim: u32,
    y_dim: u32,
) -> Result<Vec<f64>> {
    let row = row_of_dct_params(bundle, channel)?;
    // "the 4 × 8 matrix" — 4 rows × 8 cols per spec text.
    let small = compute_dct_weights(&row, 8, 4)?;
    let mut out = vec![0.0f64; (x_dim as usize) * (y_dim as usize)];
    for y in 0..y_dim {
        for x in 0..x_dim {
            // small is 4-row × 8-col; index small[(y/2)*8 + x].
            let sy = (y / 2) as usize;
            let sx = x as usize;
            out[(y as usize) * (x_dim as usize) + (x as usize)] = small[sy * 8 + sx];
        }
    }
    if bundle.params_cols != 1 || bundle.params.len() != 3 {
        return Err(Error::InvalidData(format!(
            "JXL DCT4x8: expected params to be 3x1 (3 elements), got {} (cols={})",
            bundle.params.len(),
            bundle.params_cols
        )));
    }
    let p0 = bundle.params[channel] as f64;
    if p0 == 0.0 {
        return Err(Error::InvalidData(
            "JXL DCT4x8: per-channel divisor params(c, 0) is zero".into(),
        ));
    }
    // (0, 1) /= params(c, 0).
    if x_dim >= 1 && y_dim >= 2 {
        let stride = x_dim as usize;
        out[stride] /= p0;
    }
    Ok(out)
}

/// Encoding mode AFV per §I.2.4 page 59 (Listing C.11). The
/// listing produces 8×8 weights from dct_params (3×4), dct4x4_params
/// (3×4), and params (3×9). See spec page 59 for the full body.
fn materialise_afv(
    bundle: &DequantMatrixParams,
    channel: usize,
    x_dim: u32,
    y_dim: u32,
) -> Result<Vec<f64>> {
    if x_dim != 8 || y_dim != 8 {
        return Err(Error::InvalidData(format!(
            "JXL AFV: expected 8x8 output (got {x_dim}x{y_dim})"
        )));
    }
    if bundle.params_cols != 9 || bundle.params.len() != 27 {
        return Err(Error::InvalidData(format!(
            "JXL AFV: expected params to be 3x9 (27 elements), got {} (cols={})",
            bundle.params.len(),
            bundle.params_cols
        )));
    }
    let p = |i: usize| bundle.params[channel * 9 + i] as f64;
    // weights4x8 from dct_params (3×N for some N >= 1; spec assumes
    // ReadDctParams shape).
    let row_dct = row_of_dct_params(bundle, channel)?;
    let weights4x8 = compute_dct_weights(&row_dct, 8, 4)?;
    // weights4x4 from dct4x4_params (3×N).
    let row_dct4x4 = row_of_dct4x4_params(bundle, channel)?;
    let weights4x4 = compute_dct_weights(&row_dct4x4, 4, 4)?;
    // freqs[16] per spec page 59.
    const FREQS: [f64; 16] = [
        0.0,
        0.0,
        0.8517778890324296,
        5.37778436506804,
        0.0,
        0.0,
        4.734747904497923,
        5.449245381693219,
        1.6598270267479331,
        4.0,
        7.275749096817861,
        10.423227632456525,
        2.662932286148962,
        7.630657783650829,
        8.962388608184032,
        12.97166202570235,
    ];
    let lo = 0.8517778890324296;
    let hi = 12.97166202570235;
    // bands[0] = params(c, 5); bands[i] = bands[i-1] * Mult(params(c, i + 5)).
    let mut bands = vec![p(5)];
    for i in 1..4 {
        let next = bands[i - 1] * mult(p(i + 5));
        if next <= 0.0 || !next.is_finite() {
            return Err(Error::InvalidData(format!(
                "JXL AFV: bands[{i}] = {next} violates `bands[i] > 0` invariant"
            )));
        }
        bands.push(next);
    }
    let mut weights = vec![0.0f64; 64];
    let stride = 8usize;
    let set = |w: &mut [f64], x: usize, y: usize, v: f64| {
        w[y * stride + x] = v;
    };
    let get = |w: &[f64], x: usize, y: usize| -> f64 { w[y * stride + x] };
    // weights(0,0) = 1.
    set(&mut weights, 0, 0, 1.0);
    // weights(0,1) = params(c, 0); weights(1,0) = params(c, 1).
    set(&mut weights, 0, 1, p(0));
    set(&mut weights, 1, 0, p(1));
    // weights(0,2) = params(c, 2); weights(2,0) = params(c, 3); weights(2,2) = params(c, 4).
    set(&mut weights, 0, 2, p(2));
    set(&mut weights, 2, 0, p(3));
    set(&mut weights, 2, 2, p(4));
    // for (y, x) in [0,4) × [0,4) skipping (x<2 && y<2):
    //   val = Interpolate(freqs[y*4 + x] - lo, hi - lo + 1e-6, bands);
    //   weights(2*y, 2*x) = val;
    //
    // NOTE: the 2024 spec text on page 59 writes
    //   `val = Interpolate(freqs[y * 4 + x] - lo, hi - lo + 1e-6, bands);`
    // (using `hi - lo + 1e-6` instead of FDIS 2021's `hi - lo` — the
    // 2024 form, see PDF page 59, adds the same tiny epsilon used in
    // the main GetDCTQuantWeights bridge to avoid pos == max edge
    // cases). Also: the spec writes `weights(2*y, 2*x)` but the
    // following loops write `weights(x, 2*y+1)` and
    // `weights(2*x+1, 2*y)` — consistent with the (col, row) order
    // used by GetDctQuantWeights, so we use the same convention
    // here.
    let max_dist = hi - lo + 1e-6;
    for y in 0..4u32 {
        for x in 0..4u32 {
            if x < 2 && y < 2 {
                continue;
            }
            let pos = FREQS[(y as usize) * 4 + (x as usize)] - lo;
            let val = interpolate(pos, max_dist, &bands)?;
            set(&mut weights, (2 * x) as usize, (2 * y) as usize, val);
        }
    }
    // for (y, x) in [0,4) × [0,8) skipping (x==0 && y==0):
    //   weights(x, 2*y + 1) = weights4x8(x, y);
    for y in 0..4u32 {
        for x in 0..8u32 {
            if x == 0 && y == 0 {
                continue;
            }
            let v = weights4x8[(y as usize) * 8 + (x as usize)];
            set(&mut weights, x as usize, (2 * y + 1) as usize, v);
        }
    }
    // for (y, x) in [0,4) × [0,4) skipping (x==0 && y==0):
    //   weights(2*x + 1, 2*y) = weights4x4(x, y);
    for y in 0..4u32 {
        for x in 0..4u32 {
            if x == 0 && y == 0 {
                continue;
            }
            let v = weights4x4[(y as usize) * 4 + (x as usize)];
            set(&mut weights, (2 * x + 1) as usize, (2 * y) as usize, v);
        }
    }
    // Silence the unused `get` warning if no caller uses it.
    let _ = get;
    Ok(weights)
}

fn row_of_dct_params(bundle: &DequantMatrixParams, channel: usize) -> Result<Vec<f64>> {
    if bundle.dct_params_cols == 0 || bundle.dct_params.is_empty() {
        return Err(Error::InvalidData(
            "JXL row_of_dct_params: dct_params is empty (mode requires ReadDctParams)".into(),
        ));
    }
    let cols = bundle.dct_params_cols as usize;
    if bundle.dct_params.len() != 3 * cols {
        return Err(Error::InvalidData(format!(
            "JXL row_of_dct_params: dct_params has {} elements, expected 3x{}={} ",
            bundle.dct_params.len(),
            cols,
            3 * cols
        )));
    }
    Ok(bundle.dct_params[channel * cols..(channel + 1) * cols]
        .iter()
        .map(|&v| v as f64)
        .collect())
}

fn row_of_dct4x4_params(bundle: &DequantMatrixParams, channel: usize) -> Result<Vec<f64>> {
    if bundle.dct4x4_params_cols == 0 || bundle.dct4x4_params.is_empty() {
        return Err(Error::InvalidData(
            "JXL row_of_dct4x4_params: dct4x4_params is empty (AFV mode requires it)".into(),
        ));
    }
    let cols = bundle.dct4x4_params_cols as usize;
    if bundle.dct4x4_params.len() != 3 * cols {
        return Err(Error::InvalidData(format!(
            "JXL row_of_dct4x4_params: dct4x4_params has {} elements, expected 3x{}={}",
            bundle.dct4x4_params.len(),
            cols,
            3 * cols
        )));
    }
    Ok(bundle.dct4x4_params[channel * cols..(channel + 1) * cols]
        .iter()
        .map(|&v| v as f64)
        .collect())
}

/// Materialise Table I.6 default weights (page 60 of the 2024 final
/// core PDF) for a single dequantization-matrix slot, for a single
/// channel.
///
/// `slot_index` is in 0..=16 (Table I.4 ordering). Returns the
/// per-channel weights matrix of dimensions from
/// [`weights_matrix_dims_for_slot`].
///
/// The per-slot defaults are baked-in constants from Table I.6
/// (page 60).
pub fn materialise_default_weights_for_dct_select(
    slot_index: u32,
    channel: usize,
) -> Result<Vec<f64>> {
    if channel >= 3 {
        return Err(Error::InvalidData(format!(
            "JXL default weights: channel {channel} out of range [0, 3)"
        )));
    }
    let (x_dim, y_dim) = weights_matrix_dims_for_slot(slot_index)?;
    let bundle = default_bundle_for_slot(slot_index)?;
    materialise_weights_for_dct_select(&bundle, channel, x_dim, y_dim)
}

/// Per-channel dequantization matrix = element-wise reciprocal of
/// the weights matrix per §I.2.4 last paragraph.
///
/// Errors when any weight is non-positive (the spec's "None of the
/// resulting values are non-positive or infinity" invariant).
pub fn materialise_dequant_for_channel(
    bundle: &DequantMatrixParams,
    channel: usize,
    x_dim: u32,
    y_dim: u32,
) -> Result<Vec<f64>> {
    let w = materialise_weights_for_dct_select(bundle, channel, x_dim, y_dim)?;
    let mut out = Vec::with_capacity(w.len());
    for (i, v) in w.iter().enumerate() {
        if *v <= 0.0 || !v.is_finite() {
            return Err(Error::InvalidData(format!(
                "JXL dequant: weights[{i}] = {v} violates the spec's positive-finite invariant"
            )));
        }
        out.push(1.0 / *v);
    }
    Ok(out)
}

/// Full 17-slot, 3-channel default dequantization-matrix set per
/// Table I.6.
#[derive(Debug, Clone)]
pub struct DequantMatrixSet {
    /// `matrices[slot][channel]` is the row-major dequantization
    /// matrix for that slot + channel, sized per Table I.4.
    pub matrices: Vec<[Vec<f64>; 3]>,
}

/// Materialise the full Table I.6 default dequantization set. Used
/// by the `u(1) == 1` HfGlobal fast path.
pub fn materialise_default_dequant_set() -> Result<DequantMatrixSet> {
    let mut matrices: Vec<[Vec<f64>; 3]> = Vec::with_capacity(17);
    for slot in 0..17u32 {
        let bundle = default_bundle_for_slot(slot)?;
        let (x_dim, y_dim) = weights_matrix_dims_for_slot(slot)?;
        let c0 = materialise_dequant_for_channel(&bundle, 0, x_dim, y_dim)?;
        let c1 = materialise_dequant_for_channel(&bundle, 1, x_dim, y_dim)?;
        let c2 = materialise_dequant_for_channel(&bundle, 2, x_dim, y_dim)?;
        matrices.push([c0, c1, c2]);
    }
    Ok(DequantMatrixSet { matrices })
}

/// Construct the [`DequantMatrixParams`] bundle for the default
/// encoding of a single slot, transcribed from Table I.6 (page 60
/// of the 2024 final core PDF).
fn default_bundle_for_slot(slot_index: u32) -> Result<DequantMatrixParams> {
    // The SeqA / SeqB / SeqC abbreviations from Table I.6 (page 60).
    const SEQ_A: [f64; 7] = [
        -1.025,
        -0.78,
        -0.65012,
        -0.19041574084286472,
        -0.20819395464,
        -0.421064,
        -0.32733845535848671,
    ];
    const SEQ_B: [f64; 7] = [
        -0.3041958212306401,
        -0.3633036457487539,
        -0.35660379990111464,
        -0.3443074455424403,
        -0.33699592683512467,
        -0.30180866526242109,
        -0.27321683125358037,
    ];
    const SEQ_C: [f64; 7] = [-1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5];
    // `dct4x4_params` reusable constant for the AFV slot (Table I.6
    // page 60 last paragraph: "dct4x4_params is { {2200, 0, 0, 0},
    // {392, 0, 0, 0}, {112, -0.25, -0.25, -0.5} }").
    const DCT4X4_PARAMS_AFV: [f64; 12] = [
        2200.0, 0.0, 0.0, 0.0, //
        392.0, 0.0, 0.0, 0.0, //
        112.0, -0.25, -0.25, -0.5,
    ];

    // Build dct_params row-major (3 × N) and params row-major
    // (3 × M) from Table I.6 row text.
    let bundle = match slot_index {
        // 0 — DCT8×8, mode DCT, dct_params 3×6, params {}.
        0 => DequantMatrixParams {
            mode: EncodingMode::Dct,
            dct_params_cols: 6,
            dct_params: vec![
                3150.0, 0.0, -0.4, -0.4, -0.4, -2.0, //
                560.0, 0.0, -0.3, -0.3, -0.3, -0.3, //
                512.0, -2.0, -1.0, 0.0, 0.0, -1.0,
            ]
            .into_iter()
            .map(|v: f64| v as f32)
            .collect(),
            ..Default::default()
        },
        // 1 — Hornuss, mode Hornuss, dct_params {}, params 3×3.
        1 => DequantMatrixParams {
            mode: EncodingMode::Hornuss,
            params_cols: 3,
            params: vec![
                280.0, 3160.0, 3160.0, //
                60.0, 864.0, 864.0, //
                18.0, 200.0, 200.0,
            ]
            .into_iter()
            .map(|v: f64| v as f32)
            .collect(),
            ..Default::default()
        },
        // 2 — DCT2×2, mode DCT2, dct_params {}, params 3×6.
        2 => DequantMatrixParams {
            mode: EncodingMode::Dct2,
            params_cols: 6,
            params: vec![
                3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0, //
                960.0, 640.0, 320.0, 180.0, 140.0, 120.0, //
                640.0, 320.0, 128.0, 64.0, 32.0, 16.0,
            ]
            .into_iter()
            .map(|v: f64| v as f32)
            .collect(),
            ..Default::default()
        },
        // 3 — DCT4×4, mode DCT4. Table I.6 entry is
        //   "dct4x4_params, {{1.0,1.0},{1.0,1.0},{1.0,1.0}}"
        // — i.e. dct_params = DCT4X4_PARAMS_AFV, params 3×2 = ones.
        3 => DequantMatrixParams {
            mode: EncodingMode::Dct4,
            dct_params_cols: 4,
            dct_params: DCT4X4_PARAMS_AFV.iter().map(|&v: &f64| v as f32).collect(),
            params_cols: 2,
            params: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ..Default::default()
        },
        // 4 — DCT16×16, mode DCT, dct_params 3×7, params {}.
        4 => DequantMatrixParams {
            mode: EncodingMode::Dct,
            dct_params_cols: 7,
            dct_params: vec![
                8996.8725711814115328,
                -1.3000777393353804,
                -0.49424529824571225,
                -0.439093774457103443,
                -0.6350101832695744,
                -0.90177264050827612,
                -1.6162099239887414,
                //
                3191.48366296844234752,
                -0.67424582104194355,
                -0.80745813428471001,
                -0.44925837484843441,
                -0.35865440981033403,
                -0.31322389111877305,
                -0.37615025315725483,
                //
                1157.50408145487200256,
                -2.0531423165804414,
                -1.4,
                -0.50687130033378396,
                -0.42708730624733904,
                -1.4856834539296244,
                -4.9209142884401604,
            ]
            .into_iter()
            .map(|v: f64| v as f32)
            .collect(),
            ..Default::default()
        },
        // 5 — DCT32×32, mode DCT, dct_params 3×8, params {}.
        5 => DequantMatrixParams {
            mode: EncodingMode::Dct,
            dct_params_cols: 8,
            dct_params: vec![
                15718.40830982518931456,
                -1.025,
                -0.98,
                -0.9012,
                -0.4,
                -0.48819395464,
                -0.421064,
                -0.27,
                //
                7305.7636810695983104,
                -0.8041958212306401,
                -0.7633036457487539,
                -0.55660379990111464,
                -0.49785304658857626,
                -0.43699592683512467,
                -0.40180866526242109,
                -0.27321683125358037,
                //
                3803.53173721215041536,
                -3.060733579805728,
                -2.0413270132490346,
                -2.0235650159727417,
                -0.5495389509954993,
                -0.4,
                -0.4,
                -0.3,
            ]
            .into_iter()
            .map(|v: f64| v as f32)
            .collect(),
            ..Default::default()
        },
        // 6 — DCT16×8/DCT8×16, mode DCT, dct_params 3×7, params {}.
        6 => DequantMatrixParams {
            mode: EncodingMode::Dct,
            dct_params_cols: 7,
            dct_params: vec![
                7240.7734393502,
                -0.7,
                -0.7,
                -0.2,
                -0.2,
                -0.2,
                -0.5, //
                1448.15468787004,
                -0.5,
                -0.5,
                -0.5,
                -0.2,
                -0.2,
                -0.2, //
                506.854140754517,
                -1.4,
                -0.2,
                -0.5,
                -0.5,
                -1.5,
                -3.6,
            ]
            .into_iter()
            .map(|v: f64| v as f32)
            .collect(),
            ..Default::default()
        },
        // 7 — DCT32×8/DCT8×32, mode DCT, dct_params 3×8, params {}.
        7 => DequantMatrixParams {
            mode: EncodingMode::Dct,
            dct_params_cols: 8,
            dct_params: vec![
                16283.2494710648897,
                -1.7812845336559429,
                -1.6309059012653515,
                -1.0382179034313539,
                -0.85,
                -0.7,
                -0.9,
                -1.2360638576849587,
                //
                5089.15750884921511936,
                -0.320049391452786891,
                -0.35362849922161446,
                -0.30340000000000003,
                -0.61,
                -0.5,
                -0.5,
                -0.6,
                //
                3397.77603275308720128,
                -0.321327362693153371,
                -0.34507619223117997,
                -0.70340000000000003,
                -0.9,
                -1.0,
                -1.0,
                -1.1754605576265209,
            ]
            .into_iter()
            .map(|v: f64| v as f32)
            .collect(),
            ..Default::default()
        },
        // 8 — DCT16×32/DCT32×16, mode DCT, dct_params 3×8, params {}.
        8 => DequantMatrixParams {
            mode: EncodingMode::Dct,
            dct_params_cols: 8,
            dct_params: vec![
                13844.97076442300573,
                -0.97113799999999995,
                -0.658,
                -0.42026,
                -0.22712,
                -0.2206,
                -0.226,
                -0.6,
                //
                4798.964084220744293,
                -0.61125308982767057,
                -0.83770786552491361,
                -0.79014862079498627,
                -0.2692727459704829,
                -0.38272769465388551,
                -0.22924222653091453,
                -0.20719098826199578,
                //
                1807.236946760964614,
                -1.2,
                -1.2,
                -0.7,
                -0.7,
                -0.7,
                -0.4,
                -0.5,
            ]
            .into_iter()
            .map(|v: f64| v as f32)
            .collect(),
            ..Default::default()
        },
        // 9 — DCT4×8/DCT8×4, mode DCT4x8, dct_params 3×4, params 3×1.
        9 => DequantMatrixParams {
            mode: EncodingMode::Dct4x8,
            dct_params_cols: 4,
            dct_params: vec![
                2198.050556016380522,
                -0.96269623020744692,
                -0.76194253026666783,
                -0.6551140670773547,
                //
                764.3655248643528689,
                -0.92630200888366945,
                -0.9675229603596517,
                -0.27845290869168118,
                //
                527.107573587542228,
                -1.4594385811273854,
                -1.450082094097871593,
                -1.5843722511996204,
            ]
            .into_iter()
            .map(|v: f64| v as f32)
            .collect(),
            params_cols: 1,
            params: vec![1.0, 1.0, 1.0],
            ..Default::default()
        },
        // 10 — AFV, mode AFV, dct_params 3×4 (same as slot 9),
        //   dct4x4_params 3×4 = DCT4X4_PARAMS_AFV, params 3×9.
        10 => DequantMatrixParams {
            mode: EncodingMode::Afv,
            dct_params_cols: 4,
            dct_params: vec![
                2198.050556016380522,
                -0.96269623020744692,
                -0.76194253026666783,
                -0.6551140670773547,
                //
                764.3655248643528689,
                -0.92630200888366945,
                -0.9675229603596517,
                -0.27845290869168118,
                //
                527.107573587542228,
                -1.4594385811273854,
                -1.450082094097871593,
                -1.5843722511996204,
            ]
            .into_iter()
            .map(|v: f64| v as f32)
            .collect(),
            dct4x4_params_cols: 4,
            dct4x4_params: DCT4X4_PARAMS_AFV.iter().map(|&v: &f64| v as f32).collect(),
            params_cols: 9,
            params: vec![
                3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0, //
                1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0, //
                384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25,
            ],
            ..Default::default()
        },
        // 11..16 — DCT64×64 ... DCT128×256, all mode DCT with
        //   SeqA / SeqB / SeqC tails per Table I.6.
        11 => seq_bundle(
            23966.1665298448605,
            8380.19148390090414,
            4493.02378009847706,
            &SEQ_A,
            &SEQ_B,
            &SEQ_C,
        ),
        12 => seq_bundle(
            15358.89804933239925,
            5597.360516150652990,
            2919.961618960011210,
            &SEQ_A,
            &SEQ_B,
            &SEQ_C,
        ),
        13 => seq_bundle(
            47932.3330596897210,
            16760.38296780180828,
            8986.04756019695412,
            &SEQ_A,
            &SEQ_B,
            &SEQ_C,
        ),
        14 => seq_bundle(
            30717.796098664792,
            11194.72103230130598,
            5839.92323792002242,
            &SEQ_A,
            &SEQ_B,
            &SEQ_C,
        ),
        15 => seq_bundle(
            95864.6661193794420,
            33520.76593560361656,
            17972.09512039390824,
            &SEQ_A,
            &SEQ_B,
            &SEQ_C,
        ),
        16 => seq_bundle(
            61435.5921973295970,
            24209.44206460261196,
            12979.84647584004484,
            &SEQ_A,
            &SEQ_B,
            &SEQ_C,
        ),
        other => {
            return Err(Error::InvalidData(format!(
                "JXL default_bundle_for_slot: slot {other} out of range 0..=16"
            )));
        }
    };
    Ok(bundle)
}

fn seq_bundle(
    head_c0: f64,
    head_c1: f64,
    head_c2: f64,
    seq_a: &[f64; 7],
    seq_b: &[f64; 7],
    seq_c: &[f64; 7],
) -> DequantMatrixParams {
    // dct_params 3×8: head + 7-element sequence per channel.
    let mut dct = Vec::with_capacity(24);
    dct.push(head_c0);
    dct.extend_from_slice(seq_a);
    dct.push(head_c1);
    dct.extend_from_slice(seq_b);
    dct.push(head_c2);
    dct.extend_from_slice(seq_c);
    DequantMatrixParams {
        mode: EncodingMode::Dct,
        dct_params_cols: 8,
        dct_params: dct.into_iter().map(|v: f64| v as f32).collect(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mult_positive_and_negative() {
        // Spec form: Mult(v) = 1 + v if v > 0 else 1 / (1 - v).
        assert_eq!(mult(0.5), 1.5);
        assert!((mult(-0.5) - (1.0 / 1.5)).abs() < 1e-12);
        // Boundary at v == 0: spec uses `v > 0` so 0 falls into the
        // negative branch, yielding 1 / (1 - 0) = 1.
        assert_eq!(mult(0.0), 1.0);
    }

    #[test]
    fn interpolate_single_band_returns_band() {
        let bands = [42.0];
        assert_eq!(interpolate(0.0, 1.0, &bands).unwrap(), 42.0);
        assert_eq!(interpolate(100.0, 1.0, &bands).unwrap(), 42.0);
    }

    #[test]
    fn interpolate_two_bands_endpoints() {
        let bands = [4.0, 16.0];
        // pos = 0 → A = 4.0.
        let v0 = interpolate(0.0, 2.0, &bands).unwrap();
        assert!((v0 - 4.0).abs() < 1e-12, "v0 = {v0}");
        // pos = max → scaled_pos = 1.0 → scaled_index clamped to 0,
        // frac_index = 1.0 → A * (B/A)^1 = B = 16.0.
        let v1 = interpolate(2.0, 2.0, &bands).unwrap();
        assert!((v1 - 16.0).abs() < 1e-9, "v1 = {v1}");
    }

    #[test]
    fn interpolate_two_bands_midpoint_geometric_mean() {
        let bands = [4.0, 16.0];
        // pos = 1.0 (half of max=2.0) → scaled_pos = 0.5 →
        // A * (B/A)^0.5 = sqrt(A*B) = sqrt(64) = 8.
        let v = interpolate(1.0, 2.0, &bands).unwrap();
        assert!((v - 8.0).abs() < 1e-9, "v = {v}");
    }

    #[test]
    fn get_dct_quant_weights_2x2_constant_bands() {
        // params = [5.0, 0.0] → bands = [5.0, 5.0 * Mult(0.0) = 5.0].
        // For a 2x2 output every cell ends up with Interpolate over
        // constant bands = 5.0.
        let w = compute_dct_weights(&[5.0, 0.0], 2, 2).unwrap();
        assert_eq!(w.len(), 4);
        for v in &w {
            assert!((v - 5.0).abs() < 1e-9, "cell = {v}");
        }
    }

    #[test]
    fn get_dct_quant_weights_corner_distance_zero_takes_band0() {
        // At (x, y) = (0, 0), distance = 0 → Interpolate returns A
        // = bands[0] = params[0].
        let w = compute_dct_weights(&[7.5, 0.1], 4, 4).unwrap();
        // (0,0) → 7.5.
        assert!((w[0] - 7.5).abs() < 1e-9, "(0,0) = {}", w[0]);
    }

    #[test]
    fn weights_matrix_dims_table_i_4_round_trip() {
        // Spot-check a few slots against Table I.4 page 57.
        assert_eq!(weights_matrix_dims_for_slot(0).unwrap(), (8, 8));
        assert_eq!(weights_matrix_dims_for_slot(4).unwrap(), (16, 16));
        assert_eq!(weights_matrix_dims_for_slot(5).unwrap(), (32, 32));
        assert_eq!(weights_matrix_dims_for_slot(6).unwrap(), (16, 8));
        assert_eq!(weights_matrix_dims_for_slot(11).unwrap(), (64, 64));
        assert_eq!(weights_matrix_dims_for_slot(15).unwrap(), (256, 256));
        // Out-of-range.
        assert!(weights_matrix_dims_for_slot(17).is_err());
    }

    #[test]
    fn slot_for_transform_canonical() {
        assert_eq!(slot_for_transform(TransformType::Dct8x8), 0);
        assert_eq!(slot_for_transform(TransformType::Hornuss), 1);
        assert_eq!(slot_for_transform(TransformType::Dct16x16), 4);
        assert_eq!(slot_for_transform(TransformType::Dct16x8), 6);
        assert_eq!(slot_for_transform(TransformType::Dct8x16), 6);
        assert_eq!(slot_for_transform(TransformType::Dct256x256), 15);
        assert_eq!(slot_for_transform(TransformType::Dct128x256), 16);
    }

    #[test]
    fn default_dequant_set_materialises_all_17_slots() {
        let set = materialise_default_dequant_set().unwrap();
        assert_eq!(set.matrices.len(), 17);
        // Per Table I.4 + Table I.6, every channel of every slot
        // must yield a positive-finite matrix of the slot's
        // expected dimensions.
        for (slot_index, slot) in set.matrices.iter().enumerate() {
            let (x_dim, y_dim) = weights_matrix_dims_for_slot(slot_index as u32).unwrap();
            let expected_len = (x_dim as usize) * (y_dim as usize);
            for (channel, mat) in slot.iter().enumerate() {
                assert_eq!(
                    mat.len(),
                    expected_len,
                    "slot {slot_index} channel {channel}: matrix length {}, expected {expected_len}",
                    mat.len()
                );
                for (i, &v) in mat.iter().enumerate() {
                    assert!(
                        v > 0.0 && v.is_finite(),
                        "slot {slot_index} channel {channel} cell {i}: dequant {v} not positive-finite"
                    );
                }
            }
        }
    }

    #[test]
    fn default_dct8x8_slot_first_cell_matches_reciprocal_of_3150() {
        // Slot 0 (DCT8×8) dct_params row 0 starts with 3150.0; at
        // (0,0) distance = 0 → Interpolate returns bands[0] =
        // params[0] = 3150.0; dequant = 1 / 3150.0.
        let set = materialise_default_dequant_set().unwrap();
        let dq = &set.matrices[0][0];
        let expected = 1.0 / 3150.0;
        assert!(
            (dq[0] - expected).abs() < 1e-12,
            "DCT8x8 slot, channel 0, (0,0) = {}, expected {}",
            dq[0],
            expected
        );
    }

    #[test]
    fn default_hornuss_slot_corner_is_unity_dequant() {
        // Hornuss (slot 1) sets weights(0, 0) = 1 per spec; the
        // dequant reciprocal is therefore 1.0 as well.
        let set = materialise_default_dequant_set().unwrap();
        let dq = &set.matrices[1][0];
        assert!(
            (dq[0] - 1.0).abs() < 1e-12,
            "Hornuss (0,0) dequant = {}",
            dq[0]
        );
    }

    #[test]
    fn default_dct2x2_slot_no_nan_or_inf() {
        // DCT2×2 (slot 2) is the canary for the (0,0) SPECGAP
        // documented in materialise_dct2 — verify we don't emit
        // NaN / Inf for it.
        let set = materialise_default_dequant_set().unwrap();
        for channel in 0..3 {
            for v in &set.matrices[2][channel] {
                assert!(
                    v.is_finite() && *v > 0.0,
                    "DCT2 channel {channel} cell {v} is bad"
                );
            }
        }
    }

    #[test]
    fn default_afv_slot_8x8_no_zero_cells() {
        // AFV (slot 10) should populate all 64 cells; if any cell
        // is still 0.0 (pre-listing-C.11 default fill), the
        // dequant reciprocal would be Inf and our positivity gate
        // would have errored. Re-confirm here.
        let set = materialise_default_dequant_set().unwrap();
        for channel in 0..3 {
            let dq = &set.matrices[10][channel];
            assert_eq!(dq.len(), 64);
            for (i, &v) in dq.iter().enumerate() {
                assert!(
                    v.is_finite() && v > 0.0,
                    "AFV channel {channel} cell {i} = {v}"
                );
            }
        }
    }

    #[test]
    fn afv_weights4x4_uses_dct4x4_params_not_dct_params() {
        // AFV bundle uses dct4x4_params = {{2200,0,0,0},{392,0,0,0},
        // {112,-0.25,-0.25,-0.5}} (Table I.6 last paragraph). Slot
        // 9's dct_params (DCT4x8) starts with 2198.05... so if the
        // AFV path accidentally substituted dct_params for
        // dct4x4_params, channel 0's weights4x4(0,0) would equal
        // 2198.05 instead of 2200. Verify the AFV bundle's
        // dct4x4_params is wired distinctly.
        let bundle = default_bundle_for_slot(10).unwrap();
        assert_eq!(bundle.mode, EncodingMode::Afv);
        assert_eq!(bundle.dct4x4_params_cols, 4);
        // Channel 0 head of dct4x4_params = 2200.0.
        assert!((bundle.dct4x4_params[0] - 2200.0).abs() < 1e-3);
        // Channel 0 head of dct_params = 2198.05...
        assert!((bundle.dct_params[0] - 2198.05).abs() < 0.5);
    }

    #[test]
    fn dct4x4_default_slot_3_uses_dct4x4_params_via_dct_params_field() {
        // Slot 3 (DCT4×4) Table I.6 entry encodes dct_params =
        // DCT4X4_PARAMS_AFV (the same 3×4 constant) + params 3×2
        // of ones. Verify the wiring.
        let bundle = default_bundle_for_slot(3).unwrap();
        assert_eq!(bundle.mode, EncodingMode::Dct4);
        assert_eq!(bundle.dct_params_cols, 4);
        // Channel 0 head = 2200.0.
        assert!((bundle.dct_params[0] - 2200.0).abs() < 1e-3);
        // params is 3x2 of ones.
        assert_eq!(bundle.params_cols, 2);
        assert_eq!(bundle.params.len(), 6);
        for v in &bundle.params {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }
}
