//! LLF coefficients from downsampled image — ISO/IEC FDIS
//! 18181-1:2021 Annex I.2.5 (Listings I.15 + I.16) and ISO/IEC
//! 18181-1:2024 Annex I.2.5.
//!
//! ## Round 121 scope
//!
//! Round 121 lands the **LF→HF fusion pure-math step** that the
//! trailing prose of FDIS Annex F.2 hands off to §I.2.5:
//!
//! > After applying this smoothing, the decoder uses the process
//! > described in I.2.7 to compute the top-left X/8 × Y/8
//! > coefficients of each varblock of size X × Y, using the
//! > corresponding X/8 × Y/8 samples from the dequantized LF image.
//!
//! (FDIS p. 72, last paragraph of §F.2 — the spec text references
//! §I.2.7 which in the 2021 FDIS is renumbered §I.2.5 "LLF
//! coefficients from downsampled image".)
//!
//! In other words: for each varblock of pixel size `bwidth × bheight`,
//! the decoder takes the corresponding `cx × cy = bwidth/8 × bheight/8`
//! block of dequantised LF samples and computes the top-left
//! `cx × cy` block of LLF DCT coefficients of the HF varblock.
//!
//! This is the bridge from the round-12 [`crate::lf_dequant`] output
//! into the round-95 [`crate::hf_dequant`] coefficient surface: the
//! cells of the natural-ordering "LLF" prefix of each HF varblock
//! (FDIS Annex I.2.4) are NOT decoded from the per-block coefficient
//! stream — they are derived from the dequantised LF samples by this
//! procedure.
//!
//! ## Spec listings (FDIS p. 81 — Annex I.2.5, normative)
//!
//! ### Listing I.15 — Scaling coefficients
//!
//! ```text
//! I8(N, u) {
//!   eps = (u == 0) ? sqrt(0.5) : 1;
//!   return sqrt(2.0 / N) * eps * cos(u * π / (2.0 * N));
//! }
//! D8(N, u) { return 1 / (N * I8(N, u)); }
//! I(N, u)  { return (N == 8) ? I8(N, u) : D8(N, u); }
//! D(N, u)  { return (N == 8) ? D8(N, u) : I8(N, u); }
//! C(N, n, x) {
//!   if (n > N) return 1 / C(n, N, x);
//!   if (n == N) return 1;
//!   else return cos(x * π / (2 * N)) * C(N / 2, n, x);
//! }
//! ScaleF(N, n, x) { return sqrt(n * N) * D(N, x) * I(n, x) * C(N, n, x); }
//! ```
//!
//! ### Listing I.16 — Converting LF to LLF (DCT-family transforms)
//!
//! ```text
//! cx = bwidth / 8;
//! cy = bheight / 8;
//! dc = DCT_2D(input);
//! for (y = 0; y < cy; y++)
//!   for (x = 0; x < cx; x++)
//!     output(x, y) = dc(x, y) * ScaleF(cy, bheight, y) * ScaleF(cx, bwidth, x);
//! ```
//!
//! For the non-DCT transforms (IDENTITY, DCT2×2, DCT4×4, DCT8×4,
//! DCT4×8, AFV0..AFV3), the output is equal to the input — they are
//! single-block transforms whose `cx = cy = 1` so the only output
//! cell is the DC of the 1×1 LF input itself.
//!
//! ## Spec mapping by transform
//!
//! Per FDIS §I.2.5 prose, Listing I.16 applies to:
//!
//! > DCT8×8, DCT8×16, DCT8×32, DCT16×8, DCT16×16, DCT32×8, DCT16×32,
//! > DCT32×16, DCT32×32, DCT32×64, DCT64×32, DCT64×64, DCT64×128,
//! > DCT128×64, DCT128×128, DCT128×256, DCT256×128, and DCT256×256.
//!
//! The 18 DCT-family transforms. For these, `cx = bwidth / 8` and
//! `cy = bheight / 8`. For DCT8×8 (`bwidth = bheight = 8`), both are
//! 1 — the LF→LLF step is a degenerate single-cell DCT scaled by
//! `ScaleF(1, 8, 0) ^ 2`.
//!
//! For the remaining nine transforms (IDENTITY in §I.2.5 prose is a
//! placeholder; the spec's concrete list is `DCT2×2, DCT4×4, DCT8×4,
//! DCT4×8, AFV0..3` — `bwidth = bheight = 8` and the input/output
//! mapping is the identity).
//!
//! ## Forward DCT
//!
//! Listing I.16's `DCT_2D` is the *forward* 2-D DCT defined in
//! §I.2.1 + §I.2.2 Listing I.3. The crate's existing [`crate::idct`]
//! module covers the inverse; this module adds the forward path
//! restricted to the dimensions Listing I.16 actually needs
//! (`cx, cy ∈ {1, 2, 4, 8, 16, 32}`).
//!
//! ## What this module does NOT do
//!
//! * It does not decode the per-block HF coefficient stream — the
//!   round-90 [`crate::pass_group_hf`] handles the entropy-coded HF
//!   suffix.
//! * It does not run the inverse DCT to recover pixels — that is
//!   [`crate::idct`].
//! * It does not apply Chroma-from-Luma — that runs after this step
//!   on the per-channel dequantised samples (Annex G).

use oxideav_core::{Error, Result};

use crate::dct_select::TransformType;

/// FDIS Listing I.15 helper: `I8(N, u)`.
///
/// `eps = (u == 0) ? sqrt(0.5) : 1; return sqrt(2.0 / N) * eps *
/// cos(u * π / (2.0 * N));`
#[inline]
pub fn scale_i8(n: u32, u: u32) -> f32 {
    let eps = if u == 0 { 0.5f32.sqrt() } else { 1.0 };
    let n_f = n as f32;
    let u_f = u as f32;
    (2.0 / n_f).sqrt() * eps * (u_f * std::f32::consts::PI / (2.0 * n_f)).cos()
}

/// FDIS Listing I.15 helper: `D8(N, u) = 1 / (N * I8(N, u))`.
#[inline]
pub fn scale_d8(n: u32, u: u32) -> f32 {
    1.0 / ((n as f32) * scale_i8(n, u))
}

/// FDIS Listing I.15 helper: `I(N, u) = (N == 8) ? I8(N, u) : D8(N, u)`.
#[inline]
pub fn scale_i(n: u32, u: u32) -> f32 {
    if n == 8 {
        scale_i8(n, u)
    } else {
        scale_d8(n, u)
    }
}

/// FDIS Listing I.15 helper: `D(N, u) = (N == 8) ? D8(N, u) : I8(N, u)`.
#[inline]
pub fn scale_d(n: u32, u: u32) -> f32 {
    if n == 8 {
        scale_d8(n, u)
    } else {
        scale_i8(n, u)
    }
}

/// FDIS Listing I.15 helper: `C(N, n, x)`.
///
/// ```text
/// if (n > N) return 1 / C(n, N, x);
/// if (n == N) return 1;
/// else return cos(x * π / (2 * N)) * C(N / 2, n, x);
/// ```
///
/// The recursion halves `N` each step; for the dimensions that
/// occur in Listing I.16 (`N ∈ {1, 2, 4, 8, 16, 32}`, `n ∈ {8, 16,
/// 32, 64, 128, 256}`) the depth is at most 5.
pub fn scale_c(n_big: u32, n_small: u32, x: u32) -> f32 {
    // FDIS recursion. The spec's `N`-param (capital) is the first
    // argument; the lowercase `n` is the second. We rename for
    // clarity.
    if n_small > n_big {
        // 1 / C(n_small, n_big, x): swap and invert.
        1.0 / scale_c(n_small, n_big, x)
    } else if n_small == n_big {
        1.0
    } else {
        let n_f = n_big as f32;
        let x_f = x as f32;
        (x_f * std::f32::consts::PI / (2.0 * n_f)).cos() * scale_c(n_big / 2, n_small, x)
    }
}

/// FDIS Listing I.15 helper: `ScaleF(N, n, x)`.
///
/// `sqrt(n * N) * D(N, x) * I(n, x) * C(N, n, x)`.
///
/// In Listing I.16's call sites the arguments are
/// `ScaleF(cy, bheight, y)` and `ScaleF(cx, bwidth, x)`, i.e. the
/// first argument is the LF-block axis (`cx` or `cy ∈ {1..32}`) and
/// the second is the varblock pixel-axis (`bwidth` or `bheight ∈
/// {8..256}`).
#[inline]
pub fn scale_f(n_big: u32, n_small: u32, x: u32) -> f32 {
    let prefactor = ((n_small as f32) * (n_big as f32)).sqrt();
    prefactor * scale_d(n_big, x) * scale_i(n_small, x) * scale_c(n_big, n_small, x)
}

/// Spec-conformant 1-D forward DCT per FDIS Annex I.2.1.
///
/// `out_k = (1/s) * (k == 0 ? 1 : sqrt(2)) * sum_{n=0..s-1} in_n *
/// cos(π * k * (n + 0.5) / s)`.
///
/// `s = input.len()` must be a power of two in `{1, 2, 4, 8, 16,
/// 32}` (the dimensions Listing I.16 supplies). Larger powers of
/// two would be safe by symmetry but are not exercised by the LLF
/// path; reject to make the error message explicit.
pub fn dct_1d(input: &[f32]) -> Result<Vec<f32>> {
    let s = input.len();
    if s == 0 {
        return Err(Error::InvalidData("JXL dct_1d: empty input vector".into()));
    }
    if !s.is_power_of_two() {
        return Err(Error::InvalidData(format!(
            "JXL dct_1d: input length {s} is not a power of two"
        )));
    }
    if s > 32 {
        return Err(Error::InvalidData(format!(
            "JXL dct_1d: input length {s} exceeds 32 (largest LF-axis dim for LLF-from-LF)"
        )));
    }
    let mut out = vec![0.0f32; s];
    let s_f = s as f32;
    let sqrt2 = 2f32.sqrt();
    let inv_s = 1.0 / s_f;
    for (k, slot) in out.iter_mut().enumerate() {
        let scale_k = if k == 0 { 1.0 } else { sqrt2 };
        let mut acc = 0.0f32;
        for (n, &val) in input.iter().enumerate() {
            let angle = std::f32::consts::PI * (k as f32) * ((n as f32) + 0.5) / s_f;
            acc += val * angle.cos();
        }
        *slot = inv_s * scale_k * acc;
    }
    Ok(out)
}

/// Spec-conformant 2-D forward DCT — the algorithmic inverse of
/// [`crate::idct::idct_2d`] per FDIS Annex I.2.2 Listing I.3.
///
/// ```text
/// dct1     = ColumnDCT(samples);
/// dct1_t   = Transpose(dct1);
/// dct2     = ColumnDCT(dct1_t);
/// if (C > R) result = Transpose(dct2);
/// else       result = dct2;
/// ```
///
/// ## Input and output layout
///
/// `samples` is the input pixel block, `input_rows × input_cols`
/// row-major. The output is the coefficient block in **spec natural
/// ordering** — `(min(R,C), max(R,C))` row-major (the "short × long"
/// shape consumed by [`crate::idct::idct_2d`]).
///
/// This is the precise algorithmic inverse of `idct_2d`: feeding
/// `dct_2d(samples, R, C)`'s output back into
/// `idct_2d(coefficients, R, C)` reconstructs `samples` up to f32
/// round-off.
///
/// For Listing I.16 the only dimensions that occur are
/// `rows, cols ∈ {1, 2, 4, 8, 16, 32}` — the LF-axis sizes for the
/// 18 DCT-family transforms.
pub fn dct_2d(samples: &[f32], input_rows: usize, input_cols: usize) -> Result<Vec<f32>> {
    if input_rows == 0 || input_cols == 0 {
        return Err(Error::InvalidData(format!(
            "JXL dct_2d: input_rows = {input_rows}, input_cols = {input_cols} \
             (both must be > 0)"
        )));
    }
    if !input_rows.is_power_of_two() || !input_cols.is_power_of_two() {
        return Err(Error::InvalidData(format!(
            "JXL dct_2d: input_rows = {input_rows}, input_cols = {input_cols} \
             (both must be powers of two)"
        )));
    }
    if samples.len() != input_rows * input_cols {
        return Err(Error::InvalidData(format!(
            "JXL dct_2d: samples length {} != input_rows {} * input_cols {} = {}",
            samples.len(),
            input_rows,
            input_cols,
            input_rows * input_cols
        )));
    }

    // We invert idct_2d step-by-step in reverse order. idct_2d
    // expects `coefficients` in (short × long) row-major and
    // produces `varblock` such that output dims match the caller's
    // requested (output_rows × output_cols).
    //
    // The IDCT's final stage transposes (short × long) varblock to
    // (long × short) when output_rows > output_cols. To invert:
    // start by un-doing that transpose so we work in (short × long)
    // varblock layout regardless of input aspect.
    let short = input_rows.min(input_cols);
    let long = input_rows.max(input_cols);

    // `varblock` (short × long) row-major. If input has rows <= cols
    // (input_rows = short), it's already in (short × long). Otherwise
    // (input_rows > input_cols), transpose to put it in (short × long).
    let varblock: Vec<f32> = if input_rows <= input_cols {
        samples.to_vec()
    } else {
        // Transpose (input_rows × input_cols) → (input_cols × input_rows)
        // = (short × long).
        let mut t = vec![0.0f32; short * long];
        for r in 0..input_rows {
            for c in 0..input_cols {
                t[c * long + r] = samples[r * input_cols + c];
            }
        }
        t
    };

    // Inverse of idct_2d step 4: idct_2d ran ColumnIDCT on `dct1`
    // (short × long) → varblock (short × long), processing each of
    // `long` cols (length `short`). We invert by ColumnDCT.
    let mut dct1 = vec![0.0f32; short * long];
    let mut col_buf = vec![0.0f32; short];
    for c in 0..long {
        for r in 0..short {
            col_buf[r] = varblock[r * long + c];
        }
        let col = dct_1d(&col_buf)?;
        for r in 0..short {
            dct1[r * long + c] = col[r];
        }
    }

    // Inverse of idct_2d step 3: dct1[c * long + r] = dct1_t[r * short + c],
    // i.e. dct1 (short × long) = transpose of dct1_t (long × short).
    // Invert: dct1_t (long × short) = transpose of dct1.
    let mut dct1_t = vec![0.0f32; long * short];
    for r in 0..short {
        for c in 0..long {
            dct1_t[c * short + r] = dct1[r * long + c];
        }
    }

    // Inverse of idct_2d step 2: idct_2d ran ColumnIDCT on `dct2`
    // (long × short) → dct1_t (long × short), processing each of
    // `short` cols (length `long`). Invert by ColumnDCT.
    let mut dct2 = vec![0.0f32; long * short];
    let mut col_buf2 = vec![0.0f32; long];
    for c in 0..short {
        for r in 0..long {
            col_buf2[r] = dct1_t[r * short + c];
        }
        let col = dct_1d(&col_buf2)?;
        for r in 0..long {
            dct2[r * short + c] = col[r];
        }
    }

    // Inverse of idct_2d step 1: dct2[r * short + c] = coefficients[c * long + r].
    // Invert: coefficients[c * long + r] = dct2[r * short + c].
    // coefficients is (short × long) row-major.
    let mut coefficients = vec![0.0f32; short * long];
    for r in 0..long {
        for c in 0..short {
            coefficients[c * long + r] = dct2[r * short + c];
        }
    }

    Ok(coefficients)
}

/// LF→LLF coefficient block dimensions `(cx, cy)` for a transform.
///
/// Per FDIS §I.2.5: for the 18 DCT-family transforms, `cx = bwidth / 8`
/// and `cy = bheight / 8`. For the non-DCT (single-8×8-block) transforms
/// the LF→LLF step is the identity (`cx = cy = 1`).
///
/// Returns `(cx, cy)` matching the order of [`TransformType::block_dims`]
/// (cols, rows). The pair is always at least `(1, 1)`.
pub fn llf_dims(t: TransformType) -> (u32, u32) {
    match t {
        TransformType::Dct8x8 => (1, 1),
        TransformType::Dct16x16 => (2, 2),
        TransformType::Dct32x32 => (4, 4),
        TransformType::Dct64x64 => (8, 8),
        TransformType::Dct128x128 => (16, 16),
        TransformType::Dct256x256 => (32, 32),
        // Rectangular DCTs: dims = (bwidth/8, bheight/8). The
        // `block_dims` enumeration follows `(cols, rows)` and the
        // varblock pixel size is each component × 8.
        TransformType::Dct16x8
        | TransformType::Dct8x16
        | TransformType::Dct32x8
        | TransformType::Dct8x32
        | TransformType::Dct32x16
        | TransformType::Dct16x32
        | TransformType::Dct64x32
        | TransformType::Dct32x64
        | TransformType::Dct128x64
        | TransformType::Dct64x128
        | TransformType::Dct256x128
        | TransformType::Dct128x256 => t.block_dims(),
        // Non-DCT (single 8×8 block, cx = cy = 1, identity map).
        TransformType::Hornuss
        | TransformType::Dct2x2
        | TransformType::Dct4x4
        | TransformType::Dct4x8
        | TransformType::Dct8x4
        | TransformType::Afv0
        | TransformType::Afv1
        | TransformType::Afv2
        | TransformType::Afv3 => (1, 1),
    }
}

/// FDIS Listing I.16 — convert a `cy × cx` block of dequantised LF
/// samples into the top-left `cy × cx` LLF coefficient block of an
/// HF varblock.
///
/// `input` is `cy * cx` samples in row-major order. `t` selects the
/// transform; the LF-block shape is taken from [`llf_dims`].
///
/// For the 18 DCT-family transforms, the procedure is:
///
/// 1. `dc = DCT_2D(input)`.
/// 2. `output(x, y) = dc(x, y) * ScaleF(cy, bheight, y) * ScaleF(cx,
///    bwidth, x)`.
///
/// For the non-DCT transforms (single-8×8-block), the LF block is
/// `1 × 1` and the output is the input unchanged.
///
/// The returned vector is `cy * cx` `f32` values in row-major order
/// matching Listing I.16's `output(x, y)` indexing.
///
/// Note on layout: `dct_2d` returns coefficients in spec natural
/// ordering `(min(R,C), max(R,C))` row-major. For LLF computation
/// the input is already `(cy, cx)` row-major where the LF samples
/// follow the per-channel image grid — so when `cx >= cy` the DCT
/// output is `(cy × cx)` row-major directly; otherwise it is the
/// transpose. We canonicalise back to `(cy × cx)` row-major here so
/// the per-element ScaleF multiplication is unambiguous regardless
/// of varblock aspect.
pub fn llf_from_lf(input: &[f32], t: TransformType) -> Result<Vec<f32>> {
    let (cx, cy) = llf_dims(t);
    let cx_u = cx as usize;
    let cy_u = cy as usize;
    if input.len() != cx_u * cy_u {
        return Err(Error::InvalidData(format!(
            "JXL llf_from_lf: input length {} != cx ({cx}) * cy ({cy}) = {}",
            input.len(),
            cx_u * cy_u
        )));
    }

    // Non-DCT transforms: output == input per FDIS §I.2.5 closing
    // sentence ("For DctSelect types IDENTITY, DCT2×2, DCT4×4,
    // DCT8×4, DCT4×8, AFV0, AFV1, AFV2, AFV3, the output is equal
    // to the input.").
    let is_dct_family = matches!(
        t,
        TransformType::Dct8x8
            | TransformType::Dct16x16
            | TransformType::Dct32x32
            | TransformType::Dct64x64
            | TransformType::Dct128x128
            | TransformType::Dct256x256
            | TransformType::Dct16x8
            | TransformType::Dct8x16
            | TransformType::Dct32x8
            | TransformType::Dct8x32
            | TransformType::Dct32x16
            | TransformType::Dct16x32
            | TransformType::Dct64x32
            | TransformType::Dct32x64
            | TransformType::Dct128x64
            | TransformType::Dct64x128
            | TransformType::Dct256x128
            | TransformType::Dct128x256
    );
    if !is_dct_family {
        return Ok(input.to_vec());
    }

    // Forward 2-D DCT. For `cx == cy == 1` (DCT8×8) this is a
    // single-cell pass-through scaled by 1/1 = 1; the helper
    // returns dc[0] = input[0] verbatim.
    //
    // `dct_2d` takes (input_rows, input_cols). For LLF the LF input
    // block is (cy × cx) row-major (cy rows × cx cols) — matching
    // the per-channel LF image grid.
    //
    // `dct_2d` returns coefficients in spec natural ordering:
    // (min(cy, cx), max(cy, cx)) row-major. We re-interpret as
    // (cy × cx) row-major for Listing I.16's straight `dc(x, y)`
    // indexing.
    let dc_raw = dct_2d(input, cy_u, cx_u)?;
    let short = cy_u.min(cx_u);
    let long = cy_u.max(cx_u);
    let dc_yx: Vec<f32> = if cy_u <= cx_u {
        // (short × long) = (cy × cx) — already in the desired layout.
        dc_raw
    } else {
        // (short × long) = (cx × cy). Transpose to (cy × cx).
        let mut t = vec![0.0f32; cy_u * cx_u];
        for r in 0..short {
            for c in 0..long {
                t[c * cx_u + r] = dc_raw[r * long + c];
            }
        }
        t
    };

    // Listing I.16: output(x, y) = dc(x, y) * ScaleF(cy, bheight, y)
    //                              * ScaleF(cx, bwidth, x)
    // where bwidth = 8 * cx and bheight = 8 * cy.
    let bwidth = 8 * cx;
    let bheight = 8 * cy;
    // Precompute the per-axis ScaleF vectors so we don't recompute
    // them for every (x, y) cell.
    let scale_x: Vec<f32> = (0..cx).map(|x| scale_f(cx, bwidth, x)).collect();
    let scale_y: Vec<f32> = (0..cy).map(|y| scale_f(cy, bheight, y)).collect();

    let mut out = vec![0.0f32; cy_u * cx_u];
    for y in 0..cy_u {
        for x in 0..cx_u {
            out[y * cx_u + x] = dc_yx[y * cx_u + x] * scale_y[y] * scale_x[x];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Listing I.15 helper tests --------------------------------

    #[test]
    fn scale_i8_n_eq_8_u_eq_0_is_half_root() {
        // I8(8, 0) = sqrt(2/8) * sqrt(0.5) * cos(0) = sqrt(0.25) *
        //            sqrt(0.5) * 1 = 0.5 * sqrt(0.5) = sqrt(0.5)/2
        //          = 0.35355339...
        let v = scale_i8(8, 0);
        let expected = (0.5f32).sqrt() / 2.0;
        assert!((v - expected).abs() < 1e-7, "got {v}, expected {expected}");
    }

    #[test]
    fn scale_i8_n_eq_8_u_eq_1_general_form() {
        // I8(8, 1) = sqrt(2/8) * 1 * cos(π/16)
        //          = 0.5 * cos(π/16)
        let v = scale_i8(8, 1);
        let expected = 0.5 * (std::f32::consts::PI / 16.0).cos();
        assert!((v - expected).abs() < 1e-7);
    }

    #[test]
    fn scale_d8_is_reciprocal() {
        // D8(8, u) = 1 / (8 * I8(8, u)) by definition.
        for u in 0..8 {
            let i8 = scale_i8(8, u);
            let d8 = scale_d8(8, u);
            assert!(
                (d8 - 1.0 / (8.0 * i8)).abs() < 1e-7,
                "u={u}: D8 {d8} != 1/(8*I8) {}",
                1.0 / (8.0 * i8)
            );
        }
    }

    #[test]
    fn scale_i_and_d_switch_at_n_eq_8() {
        // I(8, u) = I8; D(8, u) = D8; for N != 8, the roles swap.
        for u in 0..2 {
            assert_eq!(scale_i(8, u), scale_i8(8, u));
            assert_eq!(scale_d(8, u), scale_d8(8, u));
            // For N=16: I(16, u) = D8(16, u); D(16, u) = I8(16, u).
            assert_eq!(scale_i(16, u), scale_d8(16, u));
            assert_eq!(scale_d(16, u), scale_i8(16, u));
        }
    }

    #[test]
    fn scale_c_equal_n_returns_one() {
        // C(N, N, x) = 1 by spec.
        for n in [1u32, 2, 4, 8, 16, 32] {
            for x in 0..n {
                assert_eq!(scale_c(n, n, x), 1.0, "C({n},{n},{x}) != 1");
            }
        }
    }

    #[test]
    fn scale_c_swap_when_n_small_gt_n_big() {
        // C(N, n, x) with n > N returns 1 / C(n, N, x).
        // Spot check: C(4, 16, 1) = 1 / C(16, 4, 1).
        let forward = scale_c(16, 4, 1);
        let reciprocal = scale_c(4, 16, 1);
        assert!((reciprocal - 1.0 / forward).abs() < 1e-6);
    }

    #[test]
    fn scale_c_n_big_eq_2_n_small_eq_1_x_eq_0() {
        // C(2, 1, 0):
        //   n_small (1) <= n_big (2) so recurse.
        //   cos(0 * π / 4) * C(1, 1, 0) = 1 * 1 = 1.
        assert_eq!(scale_c(2, 1, 0), 1.0);
    }

    #[test]
    fn scale_f_dct8x8_corner() {
        // For DCT8×8, cx = cy = 1, bwidth = bheight = 8. The only
        // cell is (0, 0). ScaleF(1, 8, 0):
        //   prefactor = sqrt(8 * 1) = sqrt(8)
        //   D(1, 0) = I8(1, 0) = sqrt(2) * sqrt(0.5) * cos(0) = 1
        //   I(8, 0) = I8(8, 0) = sqrt(0.5)/2
        //   C(1, 8, 0) = 1 / C(8, 1, 0) — let's compute C(8, 1, 0):
        //     C(8, 1, 0): n_small=1 <= n_big=8, recurse.
        //       cos(0 * π / 16) * C(4, 1, 0) = 1 * C(4, 1, 0)
        //       C(4, 1, 0) = cos(0) * C(2, 1, 0) = 1 * 1 = 1
        //     So C(8, 1, 0) = 1 → C(1, 8, 0) = 1.
        //   ScaleF = sqrt(8) * 1 * (sqrt(0.5)/2) * 1
        //          = sqrt(8) * sqrt(0.5) / 2
        //          = sqrt(4) / 2 = 1.
        let v = scale_f(1, 8, 0);
        assert!(
            (v - 1.0).abs() < 1e-6,
            "ScaleF(1, 8, 0) = {v}, expected 1.0"
        );
    }

    // ---- 1-D DCT tests -------------------------------------------

    #[test]
    fn dct_1d_size_1_is_identity() {
        let v = dct_1d(&[3.5]).unwrap();
        assert_eq!(v, vec![3.5]);
    }

    #[test]
    fn dct_1d_constant_signal_produces_only_dc() {
        // For a constant signal of value `c`, DCT output is
        // [c, 0, 0, ...].
        //   out[0] = (1/8) * 1 * sum(c) = (1/8) * 8c = c.
        //   out[k>0] = (1/8) * sqrt(2) * c * sum(cos(...)) = 0
        //              because sum_{n} cos(π k (n+0.5)/8) = 0 for
        //              k in 1..s.
        let signal = vec![2.5f32; 8];
        let out = dct_1d(&signal).unwrap();
        assert!((out[0] - 2.5).abs() < 1e-6);
        for (k, v) in out.iter().enumerate().skip(1) {
            assert!(v.abs() < 1e-5, "out[{k}] = {v}, expected ≈ 0");
        }
    }

    #[test]
    fn dct_then_idct_round_trips_byte_exact_signal() {
        // DCT/IDCT should be exact inverses up to float rounding.
        // We use a small signal whose values are representable
        // exactly in f32 to keep the residual tight.
        let signal = vec![1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];
        let dct = dct_1d(&signal).unwrap();
        let back = crate::idct::idct_1d(&dct).unwrap();
        for (i, (a, b)) in signal.iter().zip(back.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "i={i}: orig {a} != roundtrip {b}");
        }
    }

    #[test]
    fn dct_1d_rejects_non_power_of_two() {
        assert!(dct_1d(&[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn dct_1d_rejects_empty() {
        assert!(dct_1d(&[]).is_err());
    }

    #[test]
    fn dct_1d_rejects_too_large() {
        let big = vec![0.0f32; 64];
        assert!(dct_1d(&big).is_err());
    }

    // ---- 2-D DCT tests -------------------------------------------

    #[test]
    fn dct_2d_size_1x1_is_identity() {
        let out = dct_2d(&[7.0], 1, 1).unwrap();
        assert_eq!(out, vec![7.0]);
    }

    #[test]
    fn dct_2d_constant_block_produces_only_dc() {
        // For a 4×4 constant block, the DC coefficient is the value
        // and all other coefficients are zero.
        let block = vec![3.0f32; 4 * 4];
        let out = dct_2d(&block, 4, 4).unwrap();
        // For a square block, layout is (cols × rows) row-major =
        // (4 × 4); out[0] is the DC.
        assert!((out[0] - 3.0).abs() < 1e-6);
        for v in &out[1..] {
            assert!(v.abs() < 1e-5, "non-DC = {v}, expected 0");
        }
    }

    #[test]
    fn dct_2d_rectangular_4x2_produces_expected_dc() {
        // 2 rows × 4 cols constant block. DC after a 2-D DCT is
        // the average value.
        let block = vec![5.0f32; 2 * 4];
        let out = dct_2d(&block, 2, 4).unwrap();
        // Layout returned in (min × max) = (2 × 4) row-major, DC at
        // [0].
        assert!((out[0] - 5.0).abs() < 1e-6, "DC = {}", out[0]);
        for (i, v) in out.iter().enumerate().skip(1) {
            assert!(v.abs() < 1e-5, "out[{i}] = {v} (expected 0)");
        }
    }

    // ---- llf_dims tests ------------------------------------------

    #[test]
    fn llf_dims_square_dct() {
        assert_eq!(llf_dims(TransformType::Dct8x8), (1, 1));
        assert_eq!(llf_dims(TransformType::Dct16x16), (2, 2));
        assert_eq!(llf_dims(TransformType::Dct32x32), (4, 4));
        assert_eq!(llf_dims(TransformType::Dct64x64), (8, 8));
        assert_eq!(llf_dims(TransformType::Dct128x128), (16, 16));
        assert_eq!(llf_dims(TransformType::Dct256x256), (32, 32));
    }

    #[test]
    fn llf_dims_rectangular_dct() {
        // DCT16×8 (16 rows × 8 cols) → cy=2, cx=1 → (cx, cy) = (1, 2).
        assert_eq!(llf_dims(TransformType::Dct16x8), (1, 2));
        // DCT8×16 → (2, 1).
        assert_eq!(llf_dims(TransformType::Dct8x16), (2, 1));
        // DCT64×128 (64 rows × 128 cols) → (16, 8).
        assert_eq!(llf_dims(TransformType::Dct64x128), (16, 8));
        // DCT128×256 → (32, 16).
        assert_eq!(llf_dims(TransformType::Dct128x256), (32, 16));
    }

    #[test]
    fn llf_dims_non_dct_identity_one_by_one() {
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
            assert_eq!(llf_dims(t), (1, 1), "non-DCT transform {t:?}");
        }
    }

    // ---- llf_from_lf tests ---------------------------------------

    #[test]
    fn llf_from_lf_dct8x8_single_cell_identity_after_scale() {
        // For DCT8×8, cx = cy = 1; ScaleF(1, 8, 0) = 1.0 (shown
        // above). So the output is the input.
        let out = llf_from_lf(&[42.0], TransformType::Dct8x8).unwrap();
        assert_eq!(out.len(), 1);
        assert!(
            (out[0] - 42.0).abs() < 1e-5,
            "DCT8×8 LLF should be input * 1.0, got {}",
            out[0]
        );
    }

    #[test]
    fn llf_from_lf_non_dct_passes_through() {
        // For Hornuss / DCT2×2 / DCT4×4 / etc., output = input.
        let cases = [
            TransformType::Hornuss,
            TransformType::Dct2x2,
            TransformType::Dct4x4,
            TransformType::Dct4x8,
            TransformType::Dct8x4,
            TransformType::Afv0,
            TransformType::Afv1,
            TransformType::Afv2,
            TransformType::Afv3,
        ];
        for t in cases {
            let v = llf_from_lf(&[-3.5], t).unwrap();
            assert_eq!(v, vec![-3.5], "non-DCT {t:?} should pass through");
        }
    }

    #[test]
    fn llf_from_lf_dct16x16_constant_block() {
        // For DCT16×16: cx = cy = 2 → 2×2 LF block. Set all four
        // samples to c=4.0; forward DCT of a 2×2 constant block
        // gives [c, 0, 0, 0] in row-major.
        //
        // The Listing I.16 output is then dc(x,y) * ScaleF(2,16,y)
        // * ScaleF(2,16,x). Only (0,0) is non-zero, so the LLF
        // block is [c * SF(2,16,0)^2, 0, 0, 0].
        let block = vec![4.0f32; 4];
        let out = llf_from_lf(&block, TransformType::Dct16x16).unwrap();
        let sf00 = scale_f(2, 16, 0);
        let expected_dc = 4.0 * sf00 * sf00;
        assert!(
            (out[0] - expected_dc).abs() < 1e-5,
            "DC = {}, expected {}",
            out[0],
            expected_dc
        );
        for (i, v) in out.iter().enumerate().skip(1) {
            assert!(v.abs() < 1e-4, "out[{i}] = {v} (expected 0)");
        }
    }

    #[test]
    fn llf_from_lf_dct16x16_known_byte_exact_signal() {
        // Hand-verifiable: 2×2 LF block { (0,0)=1, (1,0)=0, (0,1)=0,
        // (1,1)=0 }. The single non-zero sample sits at (x=0, y=0).
        //
        // 2-D forward DCT of this block: every output coefficient
        // equals (1/2) * scale_k * (1/2) * scale_l * cos(0) * cos(0)
        // by the separable kernel — except the k or l = 0 cases get
        // a factor 1 instead of sqrt(2).
        //
        // out[0,0] = (1/2)(1/2)(1)(1) * 1 * 1 = 1/4
        // out[1,0] = (1/2)(1/2)(sqrt(2))(1) * cos(π * 1 * 0.5 / 2)
        //          * 1 = (sqrt(2)/4) * cos(π/4)
        //          = (sqrt(2)/4) * (sqrt(2)/2) = 1/4.
        // out[0,1] = symmetric = 1/4.
        // out[1,1] = (1/2)(1/2)(sqrt(2))(sqrt(2)) * cos(π/4) * cos(π/4)
        //          = (1/2) * (sqrt(2)/2)^2 = (1/2) * (1/2) = 1/4.
        //
        // So all four DCT coefficients are 1/4 = 0.25 exactly.
        let block = [1.0f32, 0.0, 0.0, 0.0];
        let out = llf_from_lf(&block, TransformType::Dct16x16).unwrap();
        let sf0 = scale_f(2, 16, 0);
        let sf1 = scale_f(2, 16, 1);
        // out[y*cx + x] = 0.25 * ScaleF_y * ScaleF_x.
        let expected = [
            0.25 * sf0 * sf0,
            0.25 * sf0 * sf1,
            0.25 * sf1 * sf0,
            0.25 * sf1 * sf1,
        ];
        for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "i={i}: got {got}, expected {want}",
            );
        }
    }

    #[test]
    fn llf_from_lf_rejects_wrong_input_length() {
        // DCT16×16 needs 2×2 = 4 samples; give it 3.
        let block = vec![1.0f32, 2.0, 3.0];
        assert!(llf_from_lf(&block, TransformType::Dct16x16).is_err());
    }

    #[test]
    fn llf_from_lf_dct16x8_rectangular() {
        // DCT16×8: 16 rows × 8 cols → cy=2, cx=1. Input is 2×1
        // (2 rows, 1 col).
        let block = vec![6.0f32, 6.0];
        let out = llf_from_lf(&block, TransformType::Dct16x8).unwrap();
        assert_eq!(out.len(), 2);
        // For a constant block, DC = 6.0, AC = 0; scaled:
        //   out[0,0] = 6.0 * SF(2, 16, 0) * SF(1, 8, 0)
        //   out[1,0] = 0 * SF(2, 16, 1) * SF(1, 8, 0)  = 0.
        let dc_sf = scale_f(2, 16, 0) * scale_f(1, 8, 0);
        let expected_dc = 6.0 * dc_sf;
        assert!(
            (out[0] - expected_dc).abs() < 1e-5,
            "DC = {}, expected {}",
            out[0],
            expected_dc
        );
        assert!(out[1].abs() < 1e-5, "AC = {} (expected 0)", out[1]);
    }

    #[test]
    fn llf_from_lf_dct8x16_rectangular_swap_axes() {
        // DCT8×16: 8 rows × 16 cols → cy=1, cx=2. Input is 1×2.
        let block = vec![6.0f32, 6.0];
        let out = llf_from_lf(&block, TransformType::Dct8x16).unwrap();
        assert_eq!(out.len(), 2);
        let dc_sf = scale_f(1, 8, 0) * scale_f(2, 16, 0);
        let expected_dc = 6.0 * dc_sf;
        assert!((out[0] - expected_dc).abs() < 1e-5);
        assert!(out[1].abs() < 1e-5);
    }

    #[test]
    fn llf_from_lf_dct32x32_dimensions() {
        // DCT32×32: cx=cy=4 → 16-element LF block, 16-element LLF
        // block. Sanity-only: a flat input produces a single non-zero
        // output at (0,0).
        let block = vec![2.0f32; 16];
        let out = llf_from_lf(&block, TransformType::Dct32x32).unwrap();
        assert_eq!(out.len(), 16);
        // out[0] = 2.0 * SF(4, 32, 0)^2.
        let s = scale_f(4, 32, 0);
        assert!((out[0] - 2.0 * s * s).abs() < 1e-5);
        for (i, v) in out.iter().enumerate().skip(1) {
            assert!(v.abs() < 1e-3, "i={i}: got {v}, expected 0");
        }
    }
}
