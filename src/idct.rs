//! Inverse DCT dispatch — ISO/IEC 18181-1:2024 Annex I.7 + I.9.
//!
//! Round 12 lands the **plain-DCT IDCT dispatcher** for the family of
//! variable block sizes JPEG XL VarDCT supports (Table I.4 / Table C.16
//! transforms 0..=26). Round 13 extends the dispatcher to the non-DCT
//! transforms — `DCT2×2` (Listing I.9.3), `DCT4×4` (I.9.4), `Hornuss`
//! (I.9.5), `DCT8×4` (I.9.6), `DCT4×8` (I.9.7). The four `AFVn` variants
//! (Listing I.9.8) require a 256-entry `AFVBasis` table whose
//! transcription from the FDIS PDF is deferred to a later round to
//! avoid a high-risk OCR pass; they continue to return
//! `Err(Unsupported)`.
//!
//! ## Spec mapping
//!
//! From FDIS Annex I.7.2 (One-dimensional DCT and IDCT), with input
//! vector `out` of size `s` (a power of two), the inverse DCT is:
//!
//! ```text
//! in[k] = out[0] + sum_{n=1..s-1} sqrt(2) * out[n] * cos(pi*n*(k+0.5)/s)
//! ```
//!
//! Equivalently, written so the `n=0` term joins the sum with scale
//! factor `1`: `in[k] = sum_{n=0..s-1} f(n) * out[n] * cos(pi*n*(k+0.5)/s)`
//! where `f(0) = 1` and `f(n>=1) = sqrt(2)`.
//!
//! Per Annex I.7.3, the 2-D IDCT for an `R × C` matrix is:
//!
//! ```text
//! if (C > R) dct2 = Transpose(coefficients);
//! else        dct2 = coefficients;
//! dct1_t = ColumnIDCT(dct2);
//! dct1   = Transpose(dct1_t);
//! varblock = ColumnIDCT(dct1);
//! ```
//!
//! Where `ColumnIDCT` applies the 1-D IDCT to each column of its input.
//!
//! ## Round 12 + 13 scope
//!
//! Round 12 wires:
//!
//! * [`idct_1d`] — the spec-conformant 1-D IDCT for power-of-two sizes
//!   `s ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}`. Implemented as an
//!   `O(s²)` direct cosine sum so the scaffolding is self-contained.
//! * [`idct_2d`] — the spec-conformant 2-D IDCT per Annex I.7.3,
//!   handling rectangular `R × C` blocks where `C >= R` (the listing's
//!   own pre-transpose normalises the asymmetric case).
//! * [`idct_for_transform`] — dispatch on a [`TransformType`] for the
//!   variable-DCT block sizes used by VarDCT. This is the consumer-facing
//!   entry point that PassGroup HF decode (round 13+) will call once
//!   the dequantised coefficient block is available.
//!
//! Round 13 adds the non-DCT IDCT path:
//!
//! * [`aux_idct_2x2`] — `AuxIDCT2x2(block, S)` per Annex I.9.3. A
//!   2×2 Hadamard-style butterfly that operates on the top-left `S×S`
//!   region of an 8×8 coefficient buffer. The remainder is unmodified.
//! * [`idct_dct2x2`] — three nested invocations of [`aux_idct_2x2`]
//!   `(block, 2)`, `(block, 4)`, `(block, 8)` per the I.9.3 closing
//!   recipe.
//! * [`idct_dct4x4`] — Annex I.9.4: a per-2×2-quadrant IDCT_2D over a
//!   4×4 sub-block built from interleaved coefficients with a DC patch
//!   from `AuxIDCT2x2(coefficients, 2)`.
//! * [`idct_hornuss`] — Annex I.9.5: per-2×2-quadrant residual-sum
//!   replacement of the centre sample plus block-LF + residual addition
//!   to fill the remaining 15 cells.
//! * [`idct_dct8x4`] — Annex I.9.6: Hadamard-style separation of the
//!   left two columns into two 4×8 vertical sub-blocks each subjected
//!   to `IDCT_2D`.
//! * [`idct_dct4x8`] — Annex I.9.7: dual of `dct8x4`, splitting the
//!   first two coefficient rows into two 4×8 horizontal sub-blocks.
//!
//! AFV0..AFV3 (I.9.8) remain `Err(Unsupported)` pending an
//! independently-verified `AFVBasis[16][16]` table (PDF transcription of
//! 256 floats is too high-risk for this round; deferred so the table
//! can be cross-checked against an orthonormality property test as a
//! followup).

use oxideav_core::{Error, Result};

use crate::dct_select::TransformType;

/// Spec-conformant 1-D inverse DCT per Annex I.2.1.
///
/// Computes `in[k] = sum_{n=0..s-1} f(n) * out[n] * cos(pi*n*(k+0.5)/s)`
/// where `f(0) = 1` and `f(n>=1) = sqrt(2)`. Operates in-place on a
/// borrowed slice; the input slice's length is taken as `s`.
///
/// `s` must be a power of two and `>= 1`. Returns `Err(InvalidData)`
/// otherwise.
pub fn idct_1d(input: &[f32]) -> Result<Vec<f32>> {
    let s = input.len();
    if s == 0 {
        return Err(Error::InvalidData("JXL idct_1d: empty input vector".into()));
    }
    if !s.is_power_of_two() {
        return Err(Error::InvalidData(format!(
            "JXL idct_1d: input length {s} is not a power of two"
        )));
    }
    if s > 256 {
        return Err(Error::InvalidData(format!(
            "JXL idct_1d: input length {s} exceeds 256 (largest VarDCT axis)"
        )));
    }
    let mut out = vec![0.0f32; s];
    let s_f = s as f32;
    let sqrt2 = 2f32.sqrt();
    for (k, slot) in out.iter_mut().enumerate() {
        let mut acc = input[0]; // n=0 term, scale 1, cos(0) = 1
        for (n, &val) in input.iter().enumerate().skip(1) {
            let angle = std::f32::consts::PI * (n as f32) * (k as f32 + 0.5) / s_f;
            acc += sqrt2 * val * angle.cos();
        }
        *slot = acc;
    }
    Ok(out)
}

/// Spec-conformant 2-D inverse DCT per Annex I.2.2 Listing I.4.
///
/// ## Coefficient and output layout
///
/// JPEG XL VarDCT stores coefficients in **natural ordering** (Annex
/// I.2.4) over a `bwidth × bheight` grid where
/// `bwidth = max(8, max(N, M))` and `bheight = max(8, min(N, M))`.
/// Since `bwidth >= bheight`, the coefficient buffer is laid out
/// row-major with `bheight` rows and `bwidth` cols — i.e. the *long*
/// axis is always the column axis.
///
/// `output_rows × output_cols` are the **pixel** dimensions of the
/// resulting sample block. They map to a Table-C.16 `DCTNxM` transform
/// where N = output_rows and M = output_cols.
///
/// `coefficients` length must equal `output_rows * output_cols`. The
/// buffer is interpreted in spec coefficient layout
/// `(min(R,C), max(R,C))` row-major.
///
/// Returns the `output_rows × output_cols` sample matrix in row-major
/// order.
///
/// Per Listing I.4:
///
/// ```text
/// if (C > R) dct2 = Transpose(coefficients);
/// else        dct2 = coefficients;
/// dct1_t  = ColumnIDCT(dct2);
/// dct1    = Transpose(dct1_t);
/// varblock = ColumnIDCT(dct1);
/// ```
///
/// where `ColumnIDCT` applies the 1-D IDCT to each column. Note that
/// when `C > R`, the coefficient buffer is in `(R × C)` row-major and
/// the `Transpose` flips it to `(C × R)`. When `C <= R`, the
/// coefficient buffer is already in `(C × R)` row-major (per the
/// natural-ordering convention above), so no transpose is needed.
///
/// `R` and `C` here name the **output** dimensions (rows × cols).
pub fn idct_2d(coefficients: &[f32], output_rows: usize, output_cols: usize) -> Result<Vec<f32>> {
    if output_rows == 0 || output_cols == 0 {
        return Err(Error::InvalidData(format!(
            "JXL idct_2d: output_rows = {output_rows}, output_cols = {output_cols} \
             (both must be > 0)"
        )));
    }
    if !output_rows.is_power_of_two() || !output_cols.is_power_of_two() {
        return Err(Error::InvalidData(format!(
            "JXL idct_2d: output_rows = {output_rows}, output_cols = {output_cols} \
             (both must be powers of two)"
        )));
    }
    if coefficients.len() != output_rows * output_cols {
        return Err(Error::InvalidData(format!(
            "JXL idct_2d: coefficients length {} != output_rows {} * output_cols {} = {}",
            coefficients.len(),
            output_rows,
            output_cols,
            output_rows * output_cols
        )));
    }

    // Listing I.4 step 1: dct2 is shape (long × short) = (max(R,C) × min(R,C))
    // for both branches. Since coefficient input is in spec layout
    // (short × long) row-major, transpose to obtain (long × short).
    let short = output_rows.min(output_cols);
    let long = output_rows.max(output_cols);

    // dct2[r,c] (long × short) = coefficients[c,r] (short × long).
    let mut dct2 = vec![0.0f32; long * short];
    for r in 0..long {
        for c in 0..short {
            dct2[r * short + c] = coefficients[c * long + r];
        }
    }

    // Step 2: dct1_t = ColumnIDCT(dct2). Each of `short` columns
    // (length `long`) is 1-D IDCT'd independently.
    let mut dct1_t = vec![0.0f32; long * short];
    let mut col_buf = vec![0.0f32; long];
    for c in 0..short {
        for r in 0..long {
            col_buf[r] = dct2[r * short + c];
        }
        let col_idct = idct_1d(&col_buf)?;
        for r in 0..long {
            dct1_t[r * short + c] = col_idct[r];
        }
    }

    // Step 3: dct1 = Transpose(dct1_t). dct1_t is (long × short),
    // dct1 is (short × long).
    let mut dct1 = vec![0.0f32; short * long];
    for r in 0..long {
        for c in 0..short {
            dct1[c * long + r] = dct1_t[r * short + c];
        }
    }

    // Step 4: varblock = ColumnIDCT(dct1). dct1 has `long` columns each
    // of length `short`. Result is (short × long).
    let mut varblock = vec![0.0f32; short * long];
    let mut col_buf2 = vec![0.0f32; short];
    for c in 0..long {
        for r in 0..short {
            col_buf2[r] = dct1[r * long + c];
        }
        let col_idct = idct_1d(&col_buf2)?;
        for r in 0..short {
            varblock[r * long + c] = col_idct[r];
        }
    }

    // varblock is (short × long) row-major. The output is (R × C)
    // = (output_rows × output_cols) row-major. Two cases:
    //   * R <= C (R = short, C = long): varblock is (short × long) =
    //     (R × C). Direct copy.
    //   * R > C  (R = long, C = short): varblock is (short × long) =
    //     (C × R). Transpose to (R × C).
    if output_rows <= output_cols {
        // varblock layout already matches output (R × C).
        Ok(varblock)
    } else {
        // Transpose (C × R) → (R × C).
        let mut out = vec![0.0f32; output_rows * output_cols];
        for r in 0..output_cols {
            for c in 0..output_rows {
                out[c * output_cols + r] = varblock[r * output_rows + c];
            }
        }
        Ok(out)
    }
}

/// Block dimensions in **pixels** for a Table C.16 transform that uses
/// the plain DCT path (Annex I.7.3 / Listing I.9.2). Returns
/// `(pixel_rows, pixel_cols)`.
///
/// Round 12 covers the DCT transforms only. Non-DCT transforms (Hornuss,
/// DCT2×2, DCT4×4, DCT4×8/8×4, AFV) are not handled by this function
/// because their IDCT path is *not* a 1-D-then-1-D plain IDCT — it is
/// the dispatch in Listings I.9.3..I.9.8. Their pixel block is always
/// 8×8; see [`non_dct_pixel_dims`] / [`idct_for_transform`].
pub fn dct_pixel_dims(t: TransformType) -> Option<(usize, usize)> {
    match t {
        TransformType::Dct8x8 => Some((8, 8)),
        TransformType::Dct16x16 => Some((16, 16)),
        TransformType::Dct32x32 => Some((32, 32)),
        TransformType::Dct16x8 => Some((16, 8)),
        TransformType::Dct8x16 => Some((8, 16)),
        TransformType::Dct32x8 => Some((32, 8)),
        TransformType::Dct8x32 => Some((8, 32)),
        TransformType::Dct32x16 => Some((32, 16)),
        TransformType::Dct16x32 => Some((16, 32)),
        TransformType::Dct64x64 => Some((64, 64)),
        TransformType::Dct64x32 => Some((64, 32)),
        TransformType::Dct32x64 => Some((32, 64)),
        TransformType::Dct128x128 => Some((128, 128)),
        TransformType::Dct128x64 => Some((128, 64)),
        TransformType::Dct64x128 => Some((64, 128)),
        TransformType::Dct256x256 => Some((256, 256)),
        TransformType::Dct256x128 => Some((256, 128)),
        TransformType::Dct128x256 => Some((128, 256)),
        // Non-DCT transforms: Hornuss, DCT2x2, DCT4x4, DCT4x8, DCT8x4,
        // AFV0..AFV3. Their IDCT path lives in I.9.3..I.9.8 and the
        // dispatcher [`idct_for_transform`] routes them through
        // dedicated helpers ([`idct_dct2x2`], [`idct_dct4x4`],
        // [`idct_hornuss`], [`idct_dct8x4`], [`idct_dct4x8`]) instead.
        TransformType::Hornuss
        | TransformType::Dct2x2
        | TransformType::Dct4x4
        | TransformType::Dct4x8
        | TransformType::Dct8x4
        | TransformType::Afv0
        | TransformType::Afv1
        | TransformType::Afv2
        | TransformType::Afv3 => None,
    }
}

/// Pixel dimensions for the **non-DCT** transforms — they are all
/// 8×8 (Table I.4 entries 1, 2, 3, 9, 10) per the closing entries of
/// Listings I.9.3..I.9.8.
///
/// Returns `None` for any plain-DCT type (use [`dct_pixel_dims`]).
pub fn non_dct_pixel_dims(t: TransformType) -> Option<(usize, usize)> {
    match t {
        TransformType::Hornuss
        | TransformType::Dct2x2
        | TransformType::Dct4x4
        | TransformType::Dct4x8
        | TransformType::Dct8x4
        | TransformType::Afv0
        | TransformType::Afv1
        | TransformType::Afv2
        | TransformType::Afv3 => Some((8, 8)),
        _ => None,
    }
}

/// Apply the appropriate IDCT path to a coefficient block whose shape
/// matches the pixel shape of [`TransformType`] `t`.
///
/// * Plain-DCT types route through [`idct_2d`] and return an
///   `(rows × cols)` row-major sample buffer (sizes per
///   [`dct_pixel_dims`]).
/// * `DCT2×2` / `DCT4×4` / `Hornuss` / `DCT8×4` / `DCT4×8` route through
///   their dedicated helpers and return an 8×8 sample buffer
///   (`coefficients` and the result are both length 64, row-major).
/// * `AFV0..AFV3` continue to return `Err(Unsupported)` pending the
///   `AFVBasis` table (round 14+).
pub fn idct_for_transform(t: TransformType, coefficients: &[f32]) -> Result<Vec<f32>> {
    if let Some((rows, cols)) = dct_pixel_dims(t) {
        return idct_2d(coefficients, rows, cols);
    }
    match t {
        TransformType::Dct2x2 => idct_dct2x2(coefficients),
        TransformType::Dct4x4 => idct_dct4x4(coefficients),
        TransformType::Hornuss => idct_hornuss(coefficients),
        TransformType::Dct8x4 => idct_dct8x4(coefficients),
        TransformType::Dct4x8 => idct_dct4x8(coefficients),
        TransformType::Afv0 | TransformType::Afv1 | TransformType::Afv2 | TransformType::Afv3 => {
            Err(Error::Unsupported(format!(
                "JXL idct_for_transform: AFV transform {t:?} (Annex I.9.8) requires a 256-entry AFVBasis table whose verified transcription is deferred to a later round"
            )))
        }
        _ => unreachable!("dct_pixel_dims covers all plain-DCT TransformType variants"),
    }
}

/// `AuxIDCT2x2(block, S)` per Annex I.9.3.
///
/// `block` is an 8×8 row-major buffer (length 64). `S` (`size`) is a
/// power of two and `<= 8`. The function operates on the top-left
/// `S × S` cells; the remaining cells are passed through unchanged.
///
/// The transform groups the top-left `S × S` cells into `(S/2) × (S/2)`
/// 2×2 quadrants whose corners are at strided positions
/// `(x, y), (S/2 + x, y), (x, S/2 + y), (S/2 + x, S/2 + y)` and
/// applies a Hadamard-style butterfly:
///
/// ```text
/// r00 = c00 + c01 + c10 + c11
/// r01 = c00 + c01 - c10 - c11
/// r10 = c00 - c01 + c10 - c11
/// r11 = c00 - c01 - c10 + c11
/// ```
///
/// emitted into an interleaved layout at `(2x, 2y), (2x+1, 2y),
/// (2x, 2y+1), (2x+1, 2y+1)`.
pub fn aux_idct_2x2(block: &[f32], size: usize) -> Result<Vec<f32>> {
    if block.len() != 64 {
        return Err(Error::InvalidData(format!(
            "JXL aux_idct_2x2: block length {} != 64",
            block.len()
        )));
    }
    if size == 0 || !size.is_power_of_two() || size > 8 {
        return Err(Error::InvalidData(format!(
            "JXL aux_idct_2x2: size {size} must be a power of two in {{1,2,4,8}}"
        )));
    }
    let mut result = block.to_vec();
    if size < 2 {
        return Ok(result);
    }
    let num_2x2 = size / 2;
    // Pre-snapshot the top-left S×S region so the butterfly reads from
    // the *input* values even though it writes back into `result`.
    let mut src = vec![0.0f32; size * size];
    for y in 0..size {
        for x in 0..size {
            src[y * size + x] = block[y * 8 + x];
        }
    }
    let read = |x: usize, y: usize| -> f32 { src[y * size + x] };
    for y in 0..num_2x2 {
        for x in 0..num_2x2 {
            let c00 = read(x, y);
            let c01 = read(num_2x2 + x, y);
            let c10 = read(x, num_2x2 + y);
            let c11 = read(num_2x2 + x, num_2x2 + y);
            let r00 = c00 + c01 + c10 + c11;
            let r01 = c00 + c01 - c10 - c11;
            let r10 = c00 - c01 + c10 - c11;
            let r11 = c00 - c01 - c10 + c11;
            // Write to (x*2, y*2), (x*2+1, y*2), (x*2, y*2+1), (x*2+1, y*2+1)
            result[(y * 2) * 8 + (x * 2)] = r00;
            result[(y * 2) * 8 + (x * 2 + 1)] = r01;
            result[(y * 2 + 1) * 8 + (x * 2)] = r10;
            result[(y * 2 + 1) * 8 + (x * 2 + 1)] = r11;
        }
    }
    Ok(result)
}

/// `IDCT_DCT2×2(coefficients)` per the closing recipe of Annex I.9.3.
///
/// ```text
/// block   = AuxIDCT2x2(coefficients, 2);
/// block   = AuxIDCT2x2(block, 4);
/// samples = AuxIDCT2x2(block, 8);
/// ```
pub fn idct_dct2x2(coefficients: &[f32]) -> Result<Vec<f32>> {
    let b = aux_idct_2x2(coefficients, 2)?;
    let b = aux_idct_2x2(&b, 4)?;
    aux_idct_2x2(&b, 8)
}

/// `IDCT_DCT4×4(coefficients)` per Annex I.9.4.
///
/// The 8×8 coefficient matrix is split into four 4×4 sub-blocks (one
/// per 2×2 quadrant of `AuxIDCT2x2(coefficients, 2)`). Each sub-block
/// `(qx, qy)` is constructed from the strided cells `coefficients(qx +
/// 2*ix, qy + 2*iy)` for `(ix, iy) ∈ [0..4)²`, with the (0,0) entry
/// patched to the corresponding quadrant of `AuxIDCT2x2(coefficients,
/// 2)`. Each sub-block is then `IDCT_2D`'d and tiled back into the
/// 8×8 output via `result(4*qx + ix, 4*qy + iy) = sample(ix, iy)`.
pub fn idct_dct4x4(coefficients: &[f32]) -> Result<Vec<f32>> {
    if coefficients.len() != 64 {
        return Err(Error::InvalidData(format!(
            "JXL idct_dct4x4: coefficients length {} != 64",
            coefficients.len()
        )));
    }
    let dcs = aux_idct_2x2(coefficients, 2)?;
    let mut result = vec![0.0f32; 64];
    for qy in 0..2 {
        for qx in 0..2 {
            let mut block_4x4 = [0.0f32; 16];
            for iy in 0..4 {
                for ix in 0..4 {
                    let from_x = qx + ix * 2;
                    let from_y = qy + iy * 2;
                    block_4x4[iy * 4 + ix] = coefficients[from_y * 8 + from_x];
                }
            }
            // Patch (0,0) to dcs[qy*8+qx] (top-left 2×2 of dcs holds
            // the per-quadrant LF DC).
            block_4x4[0] = dcs[qy * 8 + qx];
            let samples = idct_2d(&block_4x4, 4, 4)?;
            for iy in 0..4 {
                for ix in 0..4 {
                    result[(qy * 4 + iy) * 8 + (qx * 4 + ix)] = samples[iy * 4 + ix];
                }
            }
        }
    }
    Ok(result)
}

/// `IDCT_Hornuss(coefficients)` per Annex I.9.5.
///
/// The 8×8 coefficient matrix is split into four 2×2 quadrants. For
/// each quadrant `(qx, qy)`:
///
/// 1. `block_lf = AuxIDCT2x2(coefficients, 2)[qy*8 + qx]` provides the
///    quadrant's DC.
/// 2. `residual_sum = sum over (iy, ix) ∈ [0..4)² with not (ix==0 &&
///    iy==0) of coefficients(qx + 2*ix, qy + 2*iy)`.
/// 3. The centre sample at `(4*qx + 1, 4*qy + 1)` is set to
///    `block_lf - residual_sum / 16.0`.
/// 4. Each remaining cell `(4*qx + ix, 4*qy + iy)` (for the 15 cells
///    excluding `(1, 1)`) is set to
///    `coefficients(qx + 2*ix, qy + 2*iy) + sample(4*qx + 1, 4*qy + 1)`.
///
/// The `(0, 0)` overwrite path of step 4 is then re-overridden as
/// `coefficients(qx + 2, qy + 2) + sample(4*qx + 1, 4*qy + 1)` per
/// the listing's final two-line corrective for the corner cell.
pub fn idct_hornuss(coefficients: &[f32]) -> Result<Vec<f32>> {
    if coefficients.len() != 64 {
        return Err(Error::InvalidData(format!(
            "JXL idct_hornuss: coefficients length {} != 64",
            coefficients.len()
        )));
    }
    let dcs = aux_idct_2x2(coefficients, 2)?;
    let mut sample = vec![0.0f32; 64];
    let coeff = |x: usize, y: usize| -> f32 { coefficients[y * 8 + x] };
    for qy in 0..2usize {
        for qx in 0..2usize {
            let block_lf = dcs[qy * 8 + qx];
            let mut residual_sum = 0.0f32;
            for iy in 0..4usize {
                let start = if iy == 0 { 1 } else { 0 };
                for ix in start..4 {
                    residual_sum += coeff(qx + ix * 2, qy + iy * 2);
                }
            }
            let centre = block_lf - residual_sum / 16.0;
            sample[(qy * 4 + 1) * 8 + (qx * 4 + 1)] = centre;
            for iy in 0..4usize {
                for ix in 0..4usize {
                    if ix == 1 && iy == 1 {
                        continue;
                    }
                    sample[(qy * 4 + iy) * 8 + (qx * 4 + ix)] =
                        coeff(qx + ix * 2, qy + iy * 2) + centre;
                }
            }
            // Final corrective: overwrite the corner (qx*4, qy*4) per
            // the listing's last two lines, which read coefficients at
            // (qx + 2, qy + 2) — the same coordinate as the (ix=1,
            // iy=1) sample on the strided grid, supplying the top-left
            // sample with the central residual instead of the (0, 0)
            // coefficient.
            sample[(qy * 4) * 8 + (qx * 4)] = coeff(qx + 2, qy + 2) + centre;
        }
    }
    Ok(sample)
}

/// `IDCT_DCT8×4(coefficients)` per Annex I.9.6.
///
/// The first two coefficient columns are split into a Hadamard-style
/// pair `dcs = (c00 + c01, c00 - c01)` providing the DC for each of two
/// 4×8 (rows × cols) **vertical** halves. For half `x ∈ {0, 1}`, an
/// internal 4×8 block is filled with `coefficients(ix, x + iy*2)` for
/// `(iy, ix) ∈ [0..4) × [0..8)` (with `(0, 0)` patched to `dcs[x]`),
/// then `IDCT_2D`'d to a 4×8 sample matrix.
///
/// The two 4×8 halves are tiled into the 8×4 (rows × cols) output as
/// `result(x*4 + 0..4, 0..8)` — i.e. the result is in the spec's
/// `(short × long) = (4 × 8)` natural-ordering layout, suitable for
/// re-shaping into 8 rows × 4 cols by the caller. Per the closing
/// table entry of Listing I.9.6 the output **pixel block** is 8×8
/// (the spec stores 8×4 as the half-width-half-height of an 8×8 cell);
/// here the helper returns the 8×8 reshape with each 4×8 half
/// occupying the top/bottom 4 rows of the result.
///
/// Concretely the output shape is `8 × 8` row-major: rows
/// `[0..4)` ← `samples_8x4[0]` (the `x=0` half, 4 rows × 8 cols),
/// rows `[4..8)` ← `samples_8x4[1]` (the `x=1` half).
pub fn idct_dct8x4(coefficients: &[f32]) -> Result<Vec<f32>> {
    if coefficients.len() != 64 {
        return Err(Error::InvalidData(format!(
            "JXL idct_dct8x4: coefficients length {} != 64",
            coefficients.len()
        )));
    }
    let coef = |x: usize, y: usize| -> f32 { coefficients[y * 8 + x] };
    let coef0 = coef(0, 0);
    let coef1 = coef(0, 1);
    let dcs = [coef0 + coef1, coef0 - coef1];
    let mut result = vec![0.0f32; 64];
    for x in 0..2usize {
        let mut coeffs_4x8 = [0.0f32; 32]; // 4 rows × 8 cols, row-major
        coeffs_4x8[0] = dcs[x];
        for iy in 0..4usize {
            let start = if iy == 0 { 1 } else { 0 };
            for ix in start..8 {
                // coeffs_4x8(ix, iy) = coefficients(ix, x + iy*2)
                coeffs_4x8[iy * 8 + ix] = coef(ix, x + iy * 2);
            }
        }
        let samples = idct_2d(&coeffs_4x8, 4, 8)?;
        // Tile the 4×8 sample half into rows [x*4 .. x*4 + 4) of the
        // 8×8 output.
        for iy in 0..4usize {
            for ix in 0..8usize {
                result[(x * 4 + iy) * 8 + ix] = samples[iy * 8 + ix];
            }
        }
    }
    Ok(result)
}

/// `IDCT_DCT4×8(coefficients)` per Annex I.9.7.
///
/// Dual of [`idct_dct8x4`]: the first two coefficient *rows* (rows 0
/// and 1, both at column 0) hold the Hadamard pair providing the DC for
/// two 4×8 horizontal halves. For half `y ∈ {0, 1}`, the internal 4×8
/// coefficient buffer is filled with `coefficients(ix, y + iy*2)` for
/// `(iy, ix) ∈ [0..4) × [0..8)` (with `(0, 0)` patched to `dcs[y]`),
/// then `IDCT_2D`'d to a 4×8 sample matrix.
///
/// The two 4×8 halves are tiled into the 8×8 output as rows
/// `[y*4 .. y*4 + 4)`. Reading symmetric to [`idct_dct8x4`]:
/// `y + iy*2` produces row sets `{0, 2, 4, 6}` for `y=0` and
/// `{1, 3, 5, 7}` for `y=1`.
pub fn idct_dct4x8(coefficients: &[f32]) -> Result<Vec<f32>> {
    if coefficients.len() != 64 {
        return Err(Error::InvalidData(format!(
            "JXL idct_dct4x8: coefficients length {} != 64",
            coefficients.len()
        )));
    }
    let coef = |x: usize, y: usize| -> f32 { coefficients[y * 8 + x] };
    let coef0 = coef(0, 0);
    let coef1 = coef(0, 1);
    let dcs = [coef0 + coef1, coef0 - coef1];
    let mut result = vec![0.0f32; 64];
    for y in 0..2usize {
        let mut coeffs_4x8 = [0.0f32; 32]; // 4 rows × 8 cols, row-major
        coeffs_4x8[0] = dcs[y];
        for iy in 0..4usize {
            let start = if iy == 0 { 1 } else { 0 };
            for ix in start..8 {
                // coeffs_4x8(ix, iy) = coefficients(ix, y + iy*2).
                coeffs_4x8[iy * 8 + ix] = coef(ix, y + iy * 2);
            }
        }
        let samples = idct_2d(&coeffs_4x8, 4, 8)?;
        // Tile the 4×8 sample half into rows [y*4 .. y*4 + 4) of the
        // 8×8 output.
        for iy in 0..4usize {
            for ix in 0..8usize {
                result[(y * 4 + iy) * 8 + ix] = samples[iy * 8 + ix];
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// Spec-conformant 1-D forward DCT for power-of-two `s`. Used only
    /// in tests as the inverse-of-the-inverse oracle.
    ///
    /// `out[k] = (k == 0 ? 1 : sqrt(2)) * (1/s) * sum_{n=0..s-1} in[n] * cos(pi*k/s * (n + 0.5))`.
    fn forward_dct_1d(input: &[f32]) -> Vec<f32> {
        let s = input.len();
        let s_f = s as f32;
        let sqrt2 = 2f32.sqrt();
        let mut out = vec![0.0f32; s];
        for (k, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (n, &val) in input.iter().enumerate() {
                let angle = std::f32::consts::PI * (k as f32) * (n as f32 + 0.5) / s_f;
                acc += val * angle.cos();
            }
            let scale = if k == 0 { 1.0 } else { sqrt2 };
            *slot = scale * acc / s_f;
        }
        out
    }

    #[test]
    fn idct_1d_rejects_non_power_of_two_length() {
        let input = vec![0.0f32; 7];
        assert!(idct_1d(&input).is_err());
    }

    #[test]
    fn idct_1d_rejects_empty_input() {
        let input: Vec<f32> = Vec::new();
        assert!(idct_1d(&input).is_err());
    }

    #[test]
    fn idct_1d_rejects_too_large() {
        let input = vec![0.0f32; 512];
        assert!(idct_1d(&input).is_err());
    }

    #[test]
    fn idct_1d_dc_only_returns_constant_8() {
        // For DC-only out = [c, 0, 0, ..., 0], spec IDCT yields in[k] = c
        // for every k.
        let mut c = vec![0.0f32; 8];
        c[0] = 5.0;
        let out = idct_1d(&c).unwrap();
        for &v in &out {
            assert!(approx_eq(v, 5.0, 1e-5), "got {v} expected 5.0");
        }
    }

    #[test]
    fn idct_1d_dc_only_constant_for_all_sizes() {
        for s in [1usize, 2, 4, 8, 16, 32, 64, 128, 256] {
            let mut c = vec![0.0f32; s];
            c[0] = 3.0;
            let out = idct_1d(&c).unwrap();
            assert_eq!(out.len(), s, "size {s}");
            for &v in &out {
                assert!(approx_eq(v, 3.0, 1e-3), "size {s}: got {v} expected 3.0");
            }
        }
    }

    #[test]
    fn idct_1d_round_trip_via_forward_dct_8() {
        // Forward DCT followed by IDCT should reproduce the original
        // signal (up to floating-point noise).
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let coeffs = forward_dct_1d(&input);
        let recovered = idct_1d(&coeffs).unwrap();
        for (i, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(approx_eq(a, b, 1e-4), "i={i}: input={a} recovered={b}");
        }
    }

    #[test]
    fn idct_1d_round_trip_via_forward_dct_16_32_64() {
        for s in [16usize, 32, 64] {
            let input: Vec<f32> = (0..s).map(|i| (i as f32) * 0.5 + 1.0).collect();
            let coeffs = forward_dct_1d(&input);
            let recovered = idct_1d(&coeffs).unwrap();
            for (i, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
                assert!(
                    approx_eq(a, b, 1e-3),
                    "size {s}, i={i}: input={a} recovered={b}"
                );
            }
        }
    }

    #[test]
    fn idct_1d_first_ac_known_value_8() {
        // out = [0, 1, 0, ..., 0] → in[k] = sqrt(2) * cos(pi*1*(k+0.5)/8).
        let mut c = vec![0.0f32; 8];
        c[1] = 1.0;
        let out = idct_1d(&c).unwrap();
        let sqrt2 = 2f32.sqrt();
        for (k, &got) in out.iter().enumerate() {
            let expected = sqrt2 * f32::cos(std::f32::consts::PI * (k as f32 + 0.5) / 8.0);
            assert!(
                approx_eq(got, expected, 1e-5),
                "k={k}: got {got} expected {expected}"
            );
        }
    }

    #[test]
    fn idct_2d_rejects_non_power_of_two() {
        let coeffs = vec![0.0f32; 6 * 8];
        assert!(idct_2d(&coeffs, 6, 8).is_err());
    }

    #[test]
    fn idct_2d_rejects_zero_dim() {
        let coeffs: Vec<f32> = Vec::new();
        assert!(idct_2d(&coeffs, 0, 8).is_err());
        assert!(idct_2d(&coeffs, 8, 0).is_err());
    }

    #[test]
    fn idct_2d_rejects_length_mismatch() {
        let coeffs = vec![0.0f32; 100];
        assert!(idct_2d(&coeffs, 8, 8).is_err());
    }

    #[test]
    fn idct_2d_dc_only_8x8_constant() {
        // Single DC coefficient, IDCT2D should yield a constant block.
        // For DC-only 2-D: in[r][c] = (1)(1) = 1 with c[0][0] = 1; the
        // 2-D IDCT applies 1-D IDCT twice along each axis. With only
        // DC nonzero, both 1-D passes are constant operations: along
        // each column (DC only), the IDCT yields a column of `c[0][0]`;
        // then along each row (DC only), the IDCT yields `c[0][0]`
        // every position.
        let mut coeffs = vec![0.0f32; 8 * 8];
        coeffs[0] = 7.0;
        let out = idct_2d(&coeffs, 8, 8).unwrap();
        for &v in &out {
            assert!(approx_eq(v, 7.0, 1e-4), "got {v}");
        }
    }

    #[test]
    fn idct_2d_dc_only_constant_all_dct_sizes() {
        for &(r, c) in &[
            (8, 8),
            (16, 16),
            (32, 32),
            (8, 16),
            (16, 8),
            (8, 32),
            (32, 8),
            (16, 32),
            (32, 16),
            (64, 64),
            (32, 64),
            (64, 32),
        ] {
            let mut coeffs = vec![0.0f32; r * c];
            coeffs[0] = 4.0;
            let out = idct_2d(&coeffs, r, c).unwrap();
            assert_eq!(out.len(), r * c, "{r}x{c}");
            for &v in &out {
                assert!(approx_eq(v, 4.0, 1e-3), "{r}x{c}: got {v} expected 4.0");
            }
        }
    }

    #[test]
    fn idct_2d_round_trip_8x8_via_forward_2d() {
        // 2-D forward DCT (column-then-row, both unscaled forward) then
        // 2-D inverse should recover the original.
        let mut input = vec![0.0f32; 8 * 8];
        for r in 0..8 {
            for c in 0..8 {
                input[r * 8 + c] = (r as f32) + (c as f32) * 0.1;
            }
        }
        let coeffs = forward_dct_2d(&input, 8, 8);
        let recovered = idct_2d(&coeffs, 8, 8).unwrap();
        for (i, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(approx_eq(a, b, 1e-3), "i={i}: input={a} recovered={b}");
        }
    }

    #[test]
    fn idct_2d_round_trip_16x8_via_forward_2d() {
        // R=16, C=8 (R > C → no pre-transpose in Listing I.4).
        let mut input = vec![0.0f32; 16 * 8];
        for r in 0..16 {
            for c in 0..8 {
                input[r * 8 + c] = (r as f32) * 0.7 - (c as f32) * 0.3 + 1.0;
            }
        }
        let coeffs = forward_dct_2d(&input, 16, 8);
        let recovered = idct_2d(&coeffs, 16, 8).unwrap();
        for (i, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(approx_eq(a, b, 1e-2), "i={i}: input={a} recovered={b}");
        }
    }

    #[test]
    fn idct_2d_round_trip_8x16_via_forward_2d() {
        // R=8, C=16 (C > R → pre-transpose in Listing I.4).
        let mut input = vec![0.0f32; 8 * 16];
        for r in 0..8 {
            for c in 0..16 {
                input[r * 16 + c] = (r as f32) - (c as f32) * 0.2 + 2.0;
            }
        }
        let coeffs = forward_dct_2d(&input, 8, 16);
        let recovered = idct_2d(&coeffs, 8, 16).unwrap();
        for (i, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(approx_eq(a, b, 1e-2), "i={i}: input={a} recovered={b}");
        }
    }

    #[test]
    fn idct_2d_round_trip_16x16_via_forward_2d() {
        let mut input = vec![0.0f32; 16 * 16];
        for r in 0..16 {
            for c in 0..16 {
                input[r * 16 + c] = ((r * c) as f32).sin();
            }
        }
        let coeffs = forward_dct_2d(&input, 16, 16);
        let recovered = idct_2d(&coeffs, 16, 16).unwrap();
        for (i, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(approx_eq(a, b, 1e-2), "i={i}: input={a} recovered={b}");
        }
    }

    #[test]
    fn idct_2d_round_trip_32x32_via_forward_2d() {
        let mut input = vec![0.0f32; 32 * 32];
        for r in 0..32 {
            for c in 0..32 {
                input[r * 32 + c] = ((r + c) as f32) * 0.05;
            }
        }
        let coeffs = forward_dct_2d(&input, 32, 32);
        let recovered = idct_2d(&coeffs, 32, 32).unwrap();
        for (i, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(approx_eq(a, b, 1e-2), "i={i}: input={a} recovered={b}");
        }
    }

    /// Spec forward 2-D DCT (Listing I.3). Returns coefficients in
    /// **spec coefficient layout** `(short × long)` row-major where
    /// `short = min(R, C)` and `long = max(R, C)`.
    ///
    /// `samples` is `(R × C)` row-major; the forward DCT produces a
    /// coefficient buffer that, when fed to [`super::idct_2d`] with the
    /// same `(R, C)`, recovers the input.
    fn forward_dct_2d(samples: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let short = rows.min(cols);
        let long = rows.max(cols);

        // Reshape input into working layout (long × short) so we can do
        // ColumnDCT followed by transpose followed by ColumnDCT —
        // matching the spec algorithm with axes oriented identically to
        // [`super::idct_2d`]'s working space (long × short).
        //
        // For R <= C (rows = short, cols = long): samples is (short × long).
        //   working[r,c] (long × short) = samples[c,r] (short × long).
        // For R > C (rows = long, cols = short): samples is (long × short).
        //   working[r,c] (long × short) = samples[r,c] (long × short).
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

        // ColumnDCT on `working` (long × short): each of `short` columns
        // has length `long`.
        let mut dct1 = vec![0.0f32; long * short];
        let mut col_buf = vec![0.0f32; long];
        for c in 0..short {
            for r in 0..long {
                col_buf[r] = working[r * short + c];
            }
            let cdct = forward_dct_1d(&col_buf);
            for r in 0..long {
                dct1[r * short + c] = cdct[r];
            }
        }
        // Transpose → dct1_t (short × long).
        let mut dct1_t = vec![0.0f32; short * long];
        for r in 0..long {
            for c in 0..short {
                dct1_t[c * long + r] = dct1[r * short + c];
            }
        }
        // ColumnDCT on dct1_t: each of `long` columns has length `short`.
        let mut dct2 = vec![0.0f32; short * long];
        let mut col_buf2 = vec![0.0f32; short];
        for c in 0..long {
            for r in 0..short {
                col_buf2[r] = dct1_t[r * long + c];
            }
            let cdct = forward_dct_1d(&col_buf2);
            for r in 0..short {
                dct2[r * long + c] = cdct[r];
            }
        }
        // dct2 is (short × long) row-major — exactly the spec coefficient
        // layout. Return as-is.
        dct2
    }

    #[test]
    fn idct_for_transform_dct8x8_dispatches() {
        let mut coeffs = vec![0.0f32; 64];
        coeffs[0] = 1.0;
        let out = idct_for_transform(TransformType::Dct8x8, &coeffs).unwrap();
        assert_eq!(out.len(), 64);
        for &v in &out {
            assert!(approx_eq(v, 1.0, 1e-4));
        }
    }

    #[test]
    fn idct_for_transform_dct16x16_dispatches() {
        let mut coeffs = vec![0.0f32; 256];
        coeffs[0] = 2.0;
        let out = idct_for_transform(TransformType::Dct16x16, &coeffs).unwrap();
        assert_eq!(out.len(), 256);
        for &v in &out {
            assert!(approx_eq(v, 2.0, 1e-4));
        }
    }

    #[test]
    fn idct_for_transform_dct32x32_dispatches() {
        let mut coeffs = vec![0.0f32; 32 * 32];
        coeffs[0] = 3.0;
        let out = idct_for_transform(TransformType::Dct32x32, &coeffs).unwrap();
        assert_eq!(out.len(), 32 * 32);
        for &v in &out {
            assert!(approx_eq(v, 3.0, 1e-3));
        }
    }

    #[test]
    fn idct_for_transform_dct8x16_dispatches() {
        let mut coeffs = vec![0.0f32; 8 * 16];
        coeffs[0] = 5.0;
        let out = idct_for_transform(TransformType::Dct8x16, &coeffs).unwrap();
        assert_eq!(out.len(), 8 * 16);
        for &v in &out {
            assert!(approx_eq(v, 5.0, 1e-3));
        }
    }

    #[test]
    fn idct_for_transform_dct16x8_dispatches() {
        let mut coeffs = vec![0.0f32; 16 * 8];
        coeffs[0] = 6.0;
        let out = idct_for_transform(TransformType::Dct16x8, &coeffs).unwrap();
        assert_eq!(out.len(), 16 * 8);
        for &v in &out {
            assert!(approx_eq(v, 6.0, 1e-3));
        }
    }

    #[test]
    fn idct_for_transform_hornuss_dispatches_to_idct_hornuss() {
        // Round 13: Hornuss now resolves through idct_hornuss (Annex
        // I.9.5). The smoke test below asserts dispatch + length; the
        // semantic correctness is in idct_hornuss_*.
        let coeffs = vec![0.0f32; 64];
        let out = idct_for_transform(TransformType::Hornuss, &coeffs).unwrap();
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn idct_for_transform_dct2x2_dispatches_to_idct_dct2x2() {
        let coeffs = vec![0.0f32; 64];
        let out = idct_for_transform(TransformType::Dct2x2, &coeffs).unwrap();
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn idct_for_transform_dct4x4_dispatches_to_idct_dct4x4() {
        let coeffs = vec![0.0f32; 64];
        let out = idct_for_transform(TransformType::Dct4x4, &coeffs).unwrap();
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn idct_for_transform_dct4x8_dispatches_to_idct_dct4x8() {
        let coeffs = vec![0.0f32; 64];
        let out = idct_for_transform(TransformType::Dct4x8, &coeffs).unwrap();
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn idct_for_transform_dct8x4_dispatches_to_idct_dct8x4() {
        let coeffs = vec![0.0f32; 64];
        let out = idct_for_transform(TransformType::Dct8x4, &coeffs).unwrap();
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn idct_for_transform_afv_all_unsupported() {
        for t in [
            TransformType::Afv0,
            TransformType::Afv1,
            TransformType::Afv2,
            TransformType::Afv3,
        ] {
            let coeffs = vec![0.0f32; 64];
            let r = idct_for_transform(t, &coeffs);
            assert!(matches!(r, Err(Error::Unsupported(_))), "{t:?}");
        }
    }

    #[test]
    fn dct_pixel_dims_table_c16_dct_entries() {
        assert_eq!(dct_pixel_dims(TransformType::Dct8x8), Some((8, 8)));
        assert_eq!(dct_pixel_dims(TransformType::Dct16x16), Some((16, 16)));
        assert_eq!(dct_pixel_dims(TransformType::Dct32x32), Some((32, 32)));
        assert_eq!(dct_pixel_dims(TransformType::Dct16x8), Some((16, 8)));
        assert_eq!(dct_pixel_dims(TransformType::Dct8x16), Some((8, 16)));
        assert_eq!(dct_pixel_dims(TransformType::Dct64x64), Some((64, 64)));
        assert_eq!(dct_pixel_dims(TransformType::Dct256x256), Some((256, 256)));
        assert_eq!(dct_pixel_dims(TransformType::Dct128x256), Some((128, 256)));
    }

    #[test]
    fn dct_pixel_dims_returns_none_for_non_dct() {
        assert_eq!(dct_pixel_dims(TransformType::Hornuss), None);
        assert_eq!(dct_pixel_dims(TransformType::Dct2x2), None);
        assert_eq!(dct_pixel_dims(TransformType::Dct4x4), None);
        assert_eq!(dct_pixel_dims(TransformType::Dct4x8), None);
        assert_eq!(dct_pixel_dims(TransformType::Dct8x4), None);
        assert_eq!(dct_pixel_dims(TransformType::Afv0), None);
        assert_eq!(dct_pixel_dims(TransformType::Afv1), None);
        assert_eq!(dct_pixel_dims(TransformType::Afv2), None);
        assert_eq!(dct_pixel_dims(TransformType::Afv3), None);
    }

    // ---------- Round 13 — non-DCT IDCT helpers (Annex I.9.3..I.9.7) ----------

    #[test]
    fn non_dct_pixel_dims_returns_8x8_for_each_non_dct() {
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
    fn non_dct_pixel_dims_returns_none_for_plain_dct() {
        for t in [
            TransformType::Dct8x8,
            TransformType::Dct16x16,
            TransformType::Dct32x32,
            TransformType::Dct16x8,
            TransformType::Dct8x16,
        ] {
            assert_eq!(non_dct_pixel_dims(t), None, "{t:?}");
        }
    }

    #[test]
    fn aux_idct_2x2_rejects_bad_block_length() {
        let bad = vec![0.0f32; 32];
        assert!(aux_idct_2x2(&bad, 2).is_err());
    }

    #[test]
    fn aux_idct_2x2_rejects_bad_size() {
        let block = vec![0.0f32; 64];
        assert!(aux_idct_2x2(&block, 0).is_err());
        assert!(aux_idct_2x2(&block, 3).is_err());
        assert!(aux_idct_2x2(&block, 16).is_err());
    }

    #[test]
    fn aux_idct_2x2_size_1_passes_through() {
        let mut block = vec![0.0f32; 64];
        block[0] = 5.0;
        block[1] = 7.0;
        let out = aux_idct_2x2(&block, 1).unwrap();
        // Size 1 → num_2x2 = 0 → no work; result == input.
        assert_eq!(out, block);
    }

    #[test]
    fn aux_idct_2x2_size_2_butterfly_known() {
        // Build an 8×8 block whose top-left 2×2 is:
        //   1 2
        //   3 4
        // num_2x2 = 1. There's one quadrant (x=0, y=0) with c00=1,
        // c01=2, c10=3, c11=4.
        // r00 = 1+2+3+4 = 10
        // r01 = 1+2-3-4 = -4
        // r10 = 1-2+3-4 = -2
        // r11 = 1-2-3+4 =  0
        // Output positions: (0,0)=r00, (1,0)=r01, (0,1)=r10, (1,1)=r11.
        let mut block = vec![0.0f32; 64];
        block[0] = 1.0; // (x=0,y=0)
        block[1] = 2.0; // (x=1,y=0)
        block[8] = 3.0; // (x=0,y=1)
        block[9] = 4.0; // (x=1,y=1)
        let out = aux_idct_2x2(&block, 2).unwrap();
        assert!(approx_eq(out[0], 10.0, 1e-6));
        assert!(approx_eq(out[1], -4.0, 1e-6));
        assert!(approx_eq(out[8], -2.0, 1e-6));
        assert!(approx_eq(out[9], 0.0, 1e-6));
    }

    #[test]
    fn aux_idct_2x2_preserves_outside_top_left() {
        let mut block = vec![0.0f32; 64];
        // Set a value outside the top-left 2×2 (size=2).
        block[7] = 99.0; // (x=7, y=0)
        block[63] = 42.0; // (x=7, y=7)
        let out = aux_idct_2x2(&block, 2).unwrap();
        assert!(approx_eq(out[7], 99.0, 1e-6));
        assert!(approx_eq(out[63], 42.0, 1e-6));
    }

    #[test]
    fn aux_idct_2x2_dc_only_size_2() {
        // c00 = 8, c01 = c10 = c11 = 0.
        // r00 = r01 = r10 = r11 = 8 (Hadamard of (8, 0, 0, 0)).
        let mut block = vec![0.0f32; 64];
        block[0] = 8.0;
        let out = aux_idct_2x2(&block, 2).unwrap();
        for &(x, y) in &[(0usize, 0), (1, 0), (0, 1), (1, 1)] {
            assert!(approx_eq(out[y * 8 + x], 8.0, 1e-6), "({x},{y})");
        }
    }

    #[test]
    fn idct_dct2x2_dc_only_constant_8x8() {
        // (0,0) = 64; AuxIDCT2x2(_, 2) → 4 corners are 64 + 0 + 0 + 0
        // = 64, then ..(_, 4), (_, 8) propagate that to all 64 cells.
        let mut coeffs = vec![0.0f32; 64];
        coeffs[0] = 64.0;
        let out = idct_dct2x2(&coeffs).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!(approx_eq(v, 64.0, 1e-4), "i={i}: got {v}");
        }
    }

    #[test]
    fn idct_dct2x2_length_validation() {
        let bad = vec![0.0f32; 32];
        assert!(idct_dct2x2(&bad).is_err());
    }

    #[test]
    fn idct_dct4x4_length_validation() {
        assert!(idct_dct4x4(&[0.0f32; 32]).is_err());
        assert!(idct_dct4x4(&[0.0f32; 100]).is_err());
    }

    #[test]
    fn idct_dct4x4_dc_only_constant_8x8() {
        // (0, 0) = 4; AuxIDCT2x2(_, 2) puts 4 in all four DCS cells.
        // Each per-quadrant 4×4 IDCT block has (0,0) = 4 (the dcs
        // patch) and all other cells = 0. Spec 4×4 IDCT_2D of
        // [4, 0, 0, ..., 0] → 4 in every cell. So all 64 result cells
        // = 4.
        let mut coeffs = vec![0.0f32; 64];
        coeffs[0] = 4.0;
        let out = idct_dct4x4(&coeffs).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!(approx_eq(v, 4.0, 1e-4), "i={i}: got {v}");
        }
    }

    #[test]
    fn idct_dct4x4_per_quadrant_dc_independent() {
        // Place a DC value in each of the four 2×2 corners (the cells
        // that AuxIDCT2x2(_, 2) reads). The Hadamard butterfly mixes
        // them, but each quadrant's 4×4 IDCT then sees a unique DC.
        // Build coefficients with c00=4, c10=4, c01=4, c11=4 in the
        // 2×2: dcs(0,0) = 16, dcs(1,0) = 0, dcs(0,1) = 0, dcs(1,1) = 0.
        // Quadrant (0,0): dc = 16 → fills with 16.
        // Quadrants (1,0), (0,1), (1,1): dc = 0 → fills with 0.
        let mut coeffs = vec![0.0f32; 64];
        coeffs[0] = 4.0; // (x=0,y=0) "c00"
        coeffs[1] = 4.0; // (x=1,y=0) "c01"
        coeffs[8] = 4.0; // (x=0,y=1) "c10"
        coeffs[9] = 4.0; // (x=1,y=1) "c11"
        let out = idct_dct4x4(&coeffs).unwrap();
        // Top-left 4×4 should be ~16; other three 4×4 quadrants ~0.
        for iy in 0..4 {
            for ix in 0..4 {
                assert!(
                    approx_eq(out[iy * 8 + ix], 16.0, 1e-3),
                    "TL ({ix},{iy}): {}",
                    out[iy * 8 + ix]
                );
            }
        }
        for iy in 0..4 {
            for ix in 4..8 {
                assert!(
                    approx_eq(out[iy * 8 + ix], 0.0, 1e-3),
                    "TR ({ix},{iy}): {}",
                    out[iy * 8 + ix]
                );
            }
        }
        for iy in 4..8 {
            for ix in 0..4 {
                assert!(
                    approx_eq(out[iy * 8 + ix], 0.0, 1e-3),
                    "BL ({ix},{iy}): {}",
                    out[iy * 8 + ix]
                );
            }
        }
        for iy in 4..8 {
            for ix in 4..8 {
                assert!(
                    approx_eq(out[iy * 8 + ix], 0.0, 1e-3),
                    "BR ({ix},{iy}): {}",
                    out[iy * 8 + ix]
                );
            }
        }
    }

    #[test]
    fn idct_hornuss_length_validation() {
        assert!(idct_hornuss(&[0.0f32; 0]).is_err());
        assert!(idct_hornuss(&[0.0f32; 65]).is_err());
    }

    #[test]
    fn idct_hornuss_returns_64_samples() {
        let coeffs = vec![1.0f32; 64];
        let out = idct_hornuss(&coeffs).unwrap();
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn idct_hornuss_dc_only_per_quadrant_constant() {
        // (0,0) = 4 → AuxIDCT2x2(_, 2): all four 2×2 cells are 4.
        // For quadrant (0,0): block_lf = 4, residual_sum = 0, centre =
        // 4. The 15 non-centre cells = coefficient + centre = 0 + 4 =
        // 4 (since coefficients(0+ix*2, 0+iy*2) = 0 except at ix=iy=0
        // which is overwritten by the corrective). So all cells in
        // quadrant (0,0) ≈ 4.
        let mut coeffs = vec![0.0f32; 64];
        coeffs[0] = 4.0;
        let out = idct_hornuss(&coeffs).unwrap();
        for iy in 0..4 {
            for ix in 0..4 {
                assert!(
                    approx_eq(out[iy * 8 + ix], 4.0, 1e-4),
                    "TL ({ix},{iy}): {}",
                    out[iy * 8 + ix]
                );
            }
        }
    }

    #[test]
    fn idct_dct8x4_length_validation() {
        assert!(idct_dct8x4(&[0.0f32; 8]).is_err());
    }

    #[test]
    fn idct_dct8x4_returns_64_samples() {
        let out = idct_dct8x4(&vec![0.0f32; 64]).unwrap();
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn idct_dct8x4_dc_only_constant() {
        // c(0,0) = 1, c(0,1) = 1 → dcs = (2, 0). Half x=0 has DC=2 and
        // all other cells 0 → 4×8 IDCT yields 2 everywhere; half x=1
        // has DC=0 → 0 everywhere. Since DC=2 and other cells are 0
        // for half 0, IDCT_2D(coeffs_4x8 with first-cell=2, rest=0) =
        // 2 in every cell. So result rows 0..4 ≈ 2 and rows 4..8 ≈ 0.
        let mut coeffs = vec![0.0f32; 64];
        coeffs[0] = 1.0; // (0, 0)
        coeffs[8] = 1.0; // (0, 1)
        let out = idct_dct8x4(&coeffs).unwrap();
        for iy in 0..4 {
            for ix in 0..8 {
                assert!(
                    approx_eq(out[iy * 8 + ix], 2.0, 1e-4),
                    "row<4 ({ix},{iy}): {}",
                    out[iy * 8 + ix]
                );
            }
        }
        for iy in 4..8 {
            for ix in 0..8 {
                assert!(
                    approx_eq(out[iy * 8 + ix], 0.0, 1e-4),
                    "row>=4 ({ix},{iy}): {}",
                    out[iy * 8 + ix]
                );
            }
        }
    }

    #[test]
    fn idct_dct4x8_length_validation() {
        assert!(idct_dct4x8(&[0.0f32; 8]).is_err());
    }

    #[test]
    fn idct_dct4x8_returns_64_samples() {
        let out = idct_dct4x8(&vec![0.0f32; 64]).unwrap();
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn idct_dct4x8_dc_only_constant() {
        // c(0,0) = 1, c(0,1) = 1 → dcs = (2, 0). Half y=0 has DC=2;
        // half y=1 has DC=0. Each half's 4×8 IDCT yields a 4×8 block
        // where DC-only fills with 2 (or 0). Tiled into rows 0..4 and
        // rows 4..8 respectively.
        let mut coeffs = vec![0.0f32; 64];
        coeffs[0] = 1.0; // (0, 0)
        coeffs[8] = 1.0; // (0, 1)
        let out = idct_dct4x8(&coeffs).unwrap();
        for iy in 0..4 {
            for ix in 0..8 {
                assert!(
                    approx_eq(out[iy * 8 + ix], 2.0, 1e-4),
                    "row<4 ({ix},{iy}): {}",
                    out[iy * 8 + ix]
                );
            }
        }
        for iy in 4..8 {
            for ix in 0..8 {
                assert!(
                    approx_eq(out[iy * 8 + ix], 0.0, 1e-4),
                    "row>=4 ({ix},{iy}): {}",
                    out[iy * 8 + ix]
                );
            }
        }
    }

    #[test]
    fn idct_for_transform_afv_message_mentions_round_13_blockage() {
        let coeffs = vec![0.0f32; 64];
        let r = idct_for_transform(TransformType::Afv0, &coeffs);
        match r {
            Err(Error::Unsupported(msg)) => {
                assert!(
                    msg.contains("AFVBasis"),
                    "expected AFVBasis mention, got: {msg}"
                );
            }
            other => panic!("expected Err(Unsupported), got {other:?}"),
        }
    }
}
