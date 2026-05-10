//! Inverse DCT dispatch — ISO/IEC 18181-1:2024 Annex I.2.1 + I.2.2.
//!
//! Round 12 lands the **IDCT dispatcher** for the family of variable
//! block sizes JPEG XL VarDCT supports (Table C.16 transforms 0..=26).
//!
//! ## Spec mapping
//!
//! From FDIS Annex I.2.1 (One-dimensional DCT and IDCT), with input
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
//! Per Annex I.2.2 Listing I.4, the 2-D IDCT for an `R × C` matrix is:
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
//! ## Round 12 scope
//!
//! Round 12 wires:
//!
//! * [`idct_1d`] — the spec-conformant 1-D IDCT for power-of-two sizes
//!   `s ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}`. Implemented as an
//!   `O(s²)` direct cosine sum so the scaffolding is self-contained.
//! * [`idct_2d`] — the spec-conformant 2-D IDCT per Listing I.4,
//!   handling rectangular `R × C` blocks where `C >= R` (the listing's
//!   own pre-transpose normalises the asymmetric case).
//! * [`idct_for_transform`] — dispatch on a [`TransformType`] for the
//!   variable-DCT block sizes used by VarDCT. This is the consumer-facing
//!   entry point that PassGroup HF decode (round 13+) will call once
//!   the dequantised coefficient block is available.
//!
//! Round 12 covers the **plain DCT** transforms (transform types 0,
//! 4..=11, 18..=26 from Table C.16). The non-DCT transforms (Hornuss,
//! DCT2x2, DCT4x4, DCT4x8/8x4, AFV0..3) — Listings I.7..I.13 in the
//! spec — are deferred to round 13 since they consume an 8×8 coefficient
//! block whose layout depends on subsequent HF coefficient decode.

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
/// the plain DCT path (Listing I.4, I.2.3.2). Returns
/// `(pixel_rows, pixel_cols)`.
///
/// Round 12 covers the DCT transforms only. Non-DCT transforms (Hornuss,
/// DCT2×2, DCT4×4, DCT4×8/8×4, AFV) are not handled by this function
/// because their IDCT path is *not* a 1-D-then-1-D Listing-I.4 IDCT.
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
        // AFV0..AFV3. Their IDCT path lives in I.2.3.3..I.2.3.8 and is
        // not a plain Listing-I.4 IDCT.
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

/// Apply the Listing-I.4 IDCT to a coefficient block whose shape matches
/// the pixel shape of [`TransformType`] `t`. `coefficients` has length
/// `rows * cols` where `(rows, cols) = dct_pixel_dims(t)`.
///
/// Returns `Err(Unsupported)` for the non-DCT transform types — those
/// land in round 13 alongside the AuxIDCT2x2 / AFV listings (I.2.3.3
/// through I.2.3.8).
pub fn idct_for_transform(t: TransformType, coefficients: &[f32]) -> Result<Vec<f32>> {
    let (rows, cols) = dct_pixel_dims(t).ok_or_else(|| {
        Error::Unsupported(format!(
            "JXL idct_for_transform: TransformType {t:?} uses a non-DCT IDCT path (Listings I.7..I.13) — round 13+ work"
        ))
    })?;
    idct_2d(coefficients, rows, cols)
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
    fn idct_for_transform_hornuss_unsupported() {
        let coeffs = vec![0.0f32; 64];
        let r = idct_for_transform(TransformType::Hornuss, &coeffs);
        assert!(matches!(r, Err(Error::Unsupported(_))));
    }

    #[test]
    fn idct_for_transform_dct2x2_unsupported() {
        let coeffs = vec![0.0f32; 64];
        let r = idct_for_transform(TransformType::Dct2x2, &coeffs);
        assert!(matches!(r, Err(Error::Unsupported(_))));
    }

    #[test]
    fn idct_for_transform_dct4x4_unsupported() {
        let coeffs = vec![0.0f32; 64];
        let r = idct_for_transform(TransformType::Dct4x4, &coeffs);
        assert!(matches!(r, Err(Error::Unsupported(_))));
    }

    #[test]
    fn idct_for_transform_dct4x8_unsupported() {
        let coeffs = vec![0.0f32; 64];
        let r = idct_for_transform(TransformType::Dct4x8, &coeffs);
        assert!(matches!(r, Err(Error::Unsupported(_))));
    }

    #[test]
    fn idct_for_transform_dct8x4_unsupported() {
        let coeffs = vec![0.0f32; 64];
        let r = idct_for_transform(TransformType::Dct8x4, &coeffs);
        assert!(matches!(r, Err(Error::Unsupported(_))));
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
}
