//! Per-block VarDCT decode walk — chains the §C.8.3 decoded
//! quantised-coefficient block through Annex F.3 HF dequantisation and
//! the Annex I.2 inverse DCT to produce a block of spatial residual
//! samples.
//!
//! ## Scope (round 286)
//!
//! The earlier rounds landed every constituent primitive of the
//! VarDCT path in isolation:
//!
//! * the §C.8.3 per-block coefficient decode loop produces a
//!   [`crate::pass_group_hf::DecodedHfBlock`] — quantised integer
//!   coefficients placed in **raster index space**
//!   (`coeffs[i]` is the coefficient at raster cell `i = y * bwidth +
//!   x`, per the `DecodedHfBlock` documentation);
//! * [`crate::hf_dequant::dequant_hf_coefficient`] applies the
//!   Annex F.3 / Listing F.2 per-sample dequantisation formula to a
//!   *single* coefficient given its dequant-matrix entry;
//! * [`crate::dct_quant_weights::DequantMatrixSet`] materialises the
//!   per-slot, per-channel dequantisation matrices (the element-wise
//!   inverses of the C.6.2 weights matrices);
//! * [`crate::idct::idct_for_transform`] applies the appropriate
//!   inverse-DCT family for a [`TransformType`] to a raster-order
//!   coefficient block.
//!
//! What was missing — and what this module adds — is the **per-block
//! decode-walk stage** that composes those primitives: take a decoded
//! quantised block plus its transform type, channel, and `HfMul`,
//! dequantise *every* coefficient of the block (F.3 across the whole
//! raster), and run the inverse transform to obtain the block's
//! spatial residual samples.
//!
//! ## Coordinate convention (all plain-DCT transforms)
//!
//! This module covers **every plain-separable-DCT** transform — the
//! square `DCT8×8` / `DCT16×16` / `DCT32×32`, the rectangular
//! `DCT16×8` / `DCT8×16` / `DCT32×8` / `DCT8×32` / `DCT32×16` /
//! `DCT16×32`, and the larger `DCT64×64` … `DCT256×256` family —
//! i.e. exactly the set for which
//! [`crate::idct::dct_pixel_dims`] returns `Some`. For every one of
//! these the **coefficient grid** and the **dequantisation matrix**
//! share one unambiguous `bwidth × bheight` "wide" row-major layout
//! (`i = y * bwidth + x`), and that layout is precisely the
//! "spec coefficient layout" `(short × long)` row-major that
//! [`crate::idct::idct_for_transform`] expects as input:
//!
//! * [`crate::coeff_order::varblock_size_for_order`] (keyed by
//!   [`crate::coeff_order::order_id_for_transform`]) gives the
//!   coefficient grid `(bwidth, bheight)` with
//!   `bwidth = max(8, max(N, M))`, `bheight = max(8, min(N, M))` — so
//!   `bwidth >= bheight` always ("wide"). The decoded block is stored
//!   row-major as `coeffs[y * bwidth + x]`.
//! * [`crate::dct_quant_weights::weights_matrix_dims_for_slot`] gives
//!   the dequant-matrix dims as `(x_dim = cols, y_dim = rows)`; for
//!   every DCT slot these equal `(bwidth, bheight)` and the matrix is
//!   stored row-major as `matrix[y * x_dim + x]` — the **same** wide
//!   layout as the coefficient grid.
//! * [`crate::idct::dct_pixel_dims`] gives the pixel dims
//!   `(R, C) = (N, M)`. For a rectangular transform `DCT16×8` and its
//!   transpose `DCT8×16` the coefficient grid and dequant matrix are
//!   *identical* (both slot 6, both `16 × 8` wide); only the pixel
//!   orientation `(R, C)` passed to the IDCT differs. `idct_2d`
//!   consumes the wide `(short × long)` block and emits the correctly
//!   oriented `(R × C)` pixel block, so no per-cell transpose is
//!   needed in this module.
//!
//! Because the coefficient grid and the dequant matrix share the wide
//! layout, the per-coefficient dequant mapping is the identity:
//! dequant-matrix entry for raster cell `i` multiplies decoded
//! coefficient `coeffs[i]`, and the result feeds raster cell `i` of
//! the inverse-DCT input. The **non-DCT** transforms (`Hornuss`,
//! `DCT2×2`, `DCT4×4`, `DCT4×8`, `DCT8×4`, `AFV0..AFV3`) are deferred:
//! their dequant matrix is canonicalised to `8 × 8` while their IDCT
//! path is the §I.2.3 dispatch (`idct_dct2x2`, `idct_afv`, …) rather
//! than a plain separable IDCT, and the AFV / DCT2×2 sub-block
//! coefficient extraction does not reduce to a flat identity over an
//! `8 × 8` grid. Those are pinned in a follow-up round.
//!
//! ## Pipeline order (Annex F.3, then Annex I.2.3.2)
//!
//! 1. For each raster cell `i` of the block, dequantise the integer
//!    coefficient `coeffs[i]` with
//!    [`crate::hf_dequant::dequant_hf_coefficient`], passing the
//!    dequant-matrix entry `matrix[i]` (cast to `f32`).
//! 2. Feed the resulting raster-order `f32` block to
//!    [`crate::idct::idct_for_transform`], which returns the block's
//!    `dim × dim` spatial residual samples (row-major).
//!
//! Chroma-from-luma (Annex G) and the Gaborish / EPF restoration
//! filters run *after* this stage on the assembled per-channel image;
//! they remain caller-side concerns above this primitive.

use crate::coeff_order::{order_id_for_transform, varblock_size_for_order};
use crate::dct_quant_weights::{slot_for_transform, DequantMatrixSet};
use crate::dct_select::TransformType;
use crate::hf_dequant::{dequant_hf_coefficient, QmScaleFactors};
use crate::idct::{dct_pixel_dims, idct_for_transform};
use crate::metadata_fdis::OpsinInverseMatrix;
use crate::pass_group_hf::DecodedHfBlock;
use oxideav_core::{Error, Result};

/// The "wide" coefficient-grid dimensions `(bwidth, bheight)` of a
/// plain-DCT transform `t` covered by this per-block decode walk, or
/// `None` for any non-DCT transform (Hornuss / DCT2×2 / DCT4×4 /
/// DCT4×8 / DCT8×4 / AFV0..AFV3) whose dequant-vs-IDCT layout is
/// deferred to a follow-up round.
///
/// The covered set is exactly the transforms for which
/// [`dct_pixel_dims`] is `Some` — the plain separable-IDCT family.
/// For every covered transform the returned `(bwidth, bheight)` is the
/// shared row-major layout of both the decoded coefficient block and
/// the dequant matrix (`bwidth >= bheight`).
pub fn covered_grid_dims(t: TransformType) -> Option<(usize, usize)> {
    // The covered set is precisely "plain separable DCT", which is the
    // domain of `dct_pixel_dims`. For those, the coefficient grid is
    // the `varblock_size_for_order` of the transform's order id.
    dct_pixel_dims(t)?;
    let (bw, bh) = varblock_size_for_order(order_id_for_transform(t));
    Some((bw as usize, bh as usize))
}

/// Side of a covered **square** plain-DCT transform (8, 16 or 32), or
/// `None` otherwise. Retained for callers that only need the
/// square-DCT subset; prefer [`covered_grid_dims`] for the full
/// plain-DCT set.
pub fn covered_square_dim(t: TransformType) -> Option<usize> {
    match t {
        TransformType::Dct8x8 => Some(8),
        TransformType::Dct16x16 => Some(16),
        TransformType::Dct32x32 => Some(32),
        _ => None,
    }
}

/// Reject any transform outside the covered plain-DCT set with a
/// precise [`Error::Unsupported`], returning the wide coefficient-grid
/// dimensions `(bwidth, bheight)` for a covered transform.
fn require_covered(t: TransformType) -> Result<(usize, usize)> {
    covered_grid_dims(t).ok_or_else(|| {
        Error::Unsupported(format!(
            "JXL block_dequant: per-block decode walk covers the plain \
             separable-DCT transforms only; {t:?} is a non-DCT transform \
             (Hornuss / DCT2×2 / DCT4×4 / DCT4×8 / DCT8×4 / AFV) whose \
             dequant-vs-IDCT layout is deferred to a follow-up round"
        ))
    })
}

/// Dequantise a whole decoded coefficient block per Annex F.3.
///
/// Applies [`dequant_hf_coefficient`] to every raster cell of the
/// block, using the per-cell dequant-matrix entry from the slot that
/// `t` maps to (via [`slot_for_transform`]) for the given `channel`.
/// Returns the `dim × dim` raster-order `f32` coefficient block ready
/// for the inverse DCT.
///
/// * `decoded.coeffs` is the §C.8.3 raster-index-space quantised block
///   (`coeffs[i]` is the integer coefficient at cell `i = y * dim +
///   x`); its length must equal `dim * dim`.
/// * `channel` is `0 = X`, `1 = Y`, `2 = B` (the Listing C.13 / F.2
///   channel index).
/// * `hf_mul` is the per-varblock `HfMul` value (from the LfGroup
///   DctSelect/HfMul grid).
/// * `set` is the materialised [`DequantMatrixSet`]; `oim` and `qm`
///   carry the F.3 bias and per-channel `0.8^(qm_scale - 2)` inputs.
///
/// Errors:
/// * [`Error::Unsupported`] for any transform outside the covered
///   square-DCT set.
/// * [`Error::InvalidData`] if `channel >= 3`, if `decoded.coeffs`
///   length does not match `dim * dim`, or if the slot's
///   dequant-matrix is not the expected `dim * dim` length (a defence
///   against a mis-materialised set).
#[allow(clippy::too_many_arguments)]
pub fn dequant_block_for_transform(
    decoded: &DecodedHfBlock,
    t: TransformType,
    channel: usize,
    hf_mul: i32,
    set: &DequantMatrixSet,
    oim: &OpsinInverseMatrix,
    qm: &QmScaleFactors,
) -> Result<Vec<f32>> {
    let (bwidth, bheight) = require_covered(t)?;
    if channel >= 3 {
        return Err(Error::InvalidData(format!(
            "JXL dequant_block_for_transform: channel {channel} must be < 3"
        )));
    }
    let n = bwidth * bheight;
    if decoded.coeffs.len() != n {
        return Err(Error::InvalidData(format!(
            "JXL dequant_block_for_transform: decoded.coeffs length {} != \
             bwidth * bheight ({bwidth} * {bheight} = {n}) for {t:?}",
            decoded.coeffs.len()
        )));
    }
    let slot = slot_for_transform(t) as usize;
    if slot >= set.matrices.len() {
        return Err(Error::InvalidData(format!(
            "JXL dequant_block_for_transform: slot {slot} out of range for \
             dequant set of {} matrices",
            set.matrices.len()
        )));
    }
    let matrix = &set.matrices[slot][channel];
    if matrix.len() != n {
        return Err(Error::InvalidData(format!(
            "JXL dequant_block_for_transform: dequant matrix length {} != \
             bwidth * bheight ({n}) for slot {slot} channel {channel}",
            matrix.len()
        )));
    }

    let mut out = vec![0.0f32; n];
    for i in 0..n {
        // Per the `DecodedHfBlock` documentation `coeffs[i]` is the
        // wide-grid raster cell `i = y * bwidth + x`; the dequant
        // matrix shares that same `bwidth × bheight` row-major layout
        // (`matrix[y * x_dim + x]` with `x_dim == bwidth`), so the
        // per-cell mapping is the identity for both square and
        // rectangular plain-DCT transforms.
        out[i] = dequant_hf_coefficient(
            decoded.coeffs[i],
            channel,
            hf_mul,
            matrix[i] as f32,
            oim,
            qm,
        );
    }
    Ok(out)
}

/// Per-block VarDCT decode walk: dequantise the decoded coefficient
/// block (Annex F.3) and apply the inverse DCT (Annex I.2.3.2) to
/// obtain the block's `dim × dim` spatial residual samples (row-major).
///
/// This is the composition of [`dequant_block_for_transform`] and
/// [`idct_for_transform`]; the returned samples are the per-channel
/// residual contribution of this varblock *before* chroma-from-luma
/// and the restoration filters.
///
/// Errors propagate verbatim from [`dequant_block_for_transform`] and
/// [`idct_for_transform`].
#[allow(clippy::too_many_arguments)]
pub fn decode_block_to_residual(
    decoded: &DecodedHfBlock,
    t: TransformType,
    channel: usize,
    hf_mul: i32,
    set: &DequantMatrixSet,
    oim: &OpsinInverseMatrix,
    qm: &QmScaleFactors,
) -> Result<Vec<f32>> {
    let dequantised = dequant_block_for_transform(decoded, t, channel, hf_mul, set, oim, qm)?;
    // The dequantised block is `bwidth * bheight` cells in the wide
    // `(short × long)` layout that `idct_for_transform` consumes; we
    // assert defensively that the IDCT's pixel-cell count matches that
    // coefficient count so a future table edit cannot silently feed a
    // mis-sized block to the IDCT. (`bwidth * bheight == R * C` because
    // `{bwidth, bheight} == {R, C}` as a set for every plain-DCT
    // transform — they differ only by which side is "long".)
    let (bwidth, bheight) = require_covered(t)?;
    debug_assert_eq!(
        dct_pixel_dims(t).map(|(r, c)| r * c),
        Some(bwidth * bheight),
        "JXL decode_block_to_residual: dct_pixel_dims({t:?}) pixel count must equal \
         bwidth * bheight ({bwidth} * {bheight})"
    );
    idct_for_transform(t, &dequantised)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct_quant_weights::materialise_default_dequant_set;
    use crate::pass_group_hf::DecodedHfBlock;

    fn oim() -> OpsinInverseMatrix {
        OpsinInverseMatrix::default()
    }

    fn qm() -> QmScaleFactors {
        // Default frame: x_qm_scale = 3 → 0.8^1, b_qm_scale = 2 → 0.8^0.
        QmScaleFactors {
            x_factor: 0.8,
            b_factor: 1.0,
        }
    }

    fn block(coeffs: Vec<i32>) -> DecodedHfBlock {
        DecodedHfBlock {
            coeffs,
            remaining_non_zeros: 0,
            coeffs_read: 0,
        }
    }

    #[test]
    fn covered_square_dim_is_the_three_square_dcts() {
        assert_eq!(covered_square_dim(TransformType::Dct8x8), Some(8));
        assert_eq!(covered_square_dim(TransformType::Dct16x16), Some(16));
        assert_eq!(covered_square_dim(TransformType::Dct32x32), Some(32));
        // Rectangular + non-DCT are not in the *square* subset.
        assert_eq!(covered_square_dim(TransformType::Dct16x8), None);
        assert_eq!(covered_square_dim(TransformType::Hornuss), None);
    }

    #[test]
    fn covered_grid_dims_spans_every_plain_dct() {
        use crate::dct_select::TransformType as T;
        // The covered set is exactly the transforms with a plain-DCT
        // pixel shape; for each the grid is the wide `(bwidth, bheight)`
        // with `bwidth >= bheight`.
        let cases = [
            (T::Dct8x8, (8, 8)),
            (T::Dct16x16, (16, 16)),
            (T::Dct32x32, (32, 32)),
            (T::Dct16x8, (16, 8)),
            (T::Dct8x16, (16, 8)),
            (T::Dct32x8, (32, 8)),
            (T::Dct8x32, (32, 8)),
            (T::Dct32x16, (32, 16)),
            (T::Dct16x32, (32, 16)),
            (T::Dct64x64, (64, 64)),
            (T::Dct64x32, (64, 32)),
            (T::Dct32x64, (64, 32)),
            (T::Dct128x128, (128, 128)),
            (T::Dct128x64, (128, 64)),
            (T::Dct64x128, (128, 64)),
            (T::Dct256x256, (256, 256)),
            (T::Dct256x128, (256, 128)),
            (T::Dct128x256, (256, 128)),
        ];
        for (t, (bw, bh)) in cases {
            assert_eq!(covered_grid_dims(t), Some((bw, bh)), "{t:?}");
            assert!(bw >= bh, "{t:?}: grid must be wide");
            // A rectangular transform and its transpose share one grid.
            assert!(dct_pixel_dims(t).is_some(), "{t:?} must be a plain DCT");
        }
        // Non-DCT transforms are NOT covered.
        for t in [
            T::Hornuss,
            T::Dct2x2,
            T::Dct4x4,
            T::Dct4x8,
            T::Dct8x4,
            T::Afv0,
            T::Afv1,
            T::Afv2,
            T::Afv3,
        ] {
            assert_eq!(covered_grid_dims(t), None, "{t:?}");
        }
    }

    #[test]
    fn rect_transform_and_its_transpose_share_one_grid_and_matrix() {
        // DCT16×8 and DCT8×16 differ only in pixel orientation; the
        // coefficient grid + dequant matrix are identical (slot 6).
        let set = materialise_default_dequant_set().unwrap();
        assert_eq!(
            covered_grid_dims(TransformType::Dct16x8),
            covered_grid_dims(TransformType::Dct8x16)
        );
        assert_eq!(
            slot_for_transform(TransformType::Dct16x8),
            slot_for_transform(TransformType::Dct8x16)
        );
        // Same coefficients dequantise identically for both names.
        let mut coeffs = vec![0i32; 128];
        coeffs[3] = 5;
        coeffs[70] = -2;
        let b = block(coeffs);
        let a = dequant_block_for_transform(&b, TransformType::Dct16x8, 1, 3, &set, &oim(), &qm())
            .unwrap();
        let c = dequant_block_for_transform(&b, TransformType::Dct8x16, 1, 3, &set, &oim(), &qm())
            .unwrap();
        assert_eq!(a, c, "transpose pair must share the dequant result");
        // But the IDCT pixel orientation differs.
        assert_ne!(
            dct_pixel_dims(TransformType::Dct16x8),
            dct_pixel_dims(TransformType::Dct8x16)
        );
    }

    #[test]
    fn dequant_rejects_non_dct_transform() {
        let set = materialise_default_dequant_set().unwrap();
        // A non-DCT transform (8×8 grid) is rejected.
        let b = block(vec![0i32; 64]);
        let err =
            dequant_block_for_transform(&b, TransformType::Hornuss, 1, 1, &set, &oim(), &qm())
                .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }

    #[test]
    fn dequant_rejects_bad_channel() {
        let set = materialise_default_dequant_set().unwrap();
        let b = block(vec![0i32; 64]);
        let err = dequant_block_for_transform(&b, TransformType::Dct8x8, 3, 1, &set, &oim(), &qm())
            .unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn dequant_rejects_wrong_coeff_length() {
        let set = materialise_default_dequant_set().unwrap();
        // DCT8×8 expects 64 coefficients; give it 63.
        let b = block(vec![0i32; 63]);
        let err = dequant_block_for_transform(&b, TransformType::Dct8x8, 1, 1, &set, &oim(), &qm())
            .unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn dequant_all_zero_block_is_all_zero() {
        // A block with no non-zero coefficients dequantises to all
        // zeros: bias_adjust(0) = 0, and 0 × anything = 0.
        let set = materialise_default_dequant_set().unwrap();
        for (t, n) in [
            (TransformType::Dct8x8, 64),
            (TransformType::Dct16x16, 256),
            (TransformType::Dct32x32, 1024),
        ] {
            let b = block(vec![0i32; n]);
            for ch in 0..3 {
                let out = dequant_block_for_transform(&b, t, ch, 5, &set, &oim(), &qm()).unwrap();
                assert_eq!(out.len(), n, "{t:?} ch {ch}");
                assert!(out.iter().all(|&v| v == 0.0), "{t:?} ch {ch} not all zero");
            }
        }
    }

    #[test]
    fn dequant_single_coefficient_matches_per_sample_formula() {
        // Place one non-zero coefficient at raster cell 5 of a DCT8×8
        // Y-channel block and confirm the whole-block dequant matches
        // the per-sample `dequant_hf_coefficient` at that cell exactly.
        let set = materialise_default_dequant_set().unwrap();
        let slot = slot_for_transform(TransformType::Dct8x8) as usize;
        let channel = 1usize;
        let hf_mul = 7;
        let mut coeffs = vec![0i32; 64];
        coeffs[5] = 9;
        let b = block(coeffs);
        let out = dequant_block_for_transform(
            &b,
            TransformType::Dct8x8,
            channel,
            hf_mul,
            &set,
            &oim(),
            &qm(),
        )
        .unwrap();
        let entry = set.matrices[slot][channel][5] as f32;
        let expected = dequant_hf_coefficient(9, channel, hf_mul, entry, &oim(), &qm());
        assert_eq!(out[5], expected);
        // Every other cell stays zero.
        for (i, &v) in out.iter().enumerate() {
            if i != 5 {
                assert_eq!(v, 0.0, "cell {i} should be zero");
            }
        }
    }

    #[test]
    fn dequant_per_cell_uses_distinct_matrix_entries() {
        // Two equal coefficients at two cells with distinct
        // dequant-matrix entries must produce two distinct outputs
        // (proves the per-cell matrix indexing is the identity raster
        // map, not a single shared entry).
        let set = materialise_default_dequant_set().unwrap();
        let channel = 1usize;
        let slot = slot_for_transform(TransformType::Dct16x16) as usize;
        let matrix = &set.matrices[slot][channel];
        // Find two cells with different matrix entries (the DC cell 0
        // and a high-frequency cell differ in every default matrix).
        let hf_cell = (1..256)
            .find(|&i| (matrix[i] - matrix[0]).abs() > 1e-12)
            .expect("default DCT16×16 matrix must have a non-uniform entry");
        let mut coeffs = vec![0i32; 256];
        coeffs[0] = 4;
        coeffs[hf_cell] = 4;
        let b = block(coeffs);
        let out = dequant_block_for_transform(
            &b,
            TransformType::Dct16x16,
            channel,
            3,
            &set,
            &oim(),
            &qm(),
        )
        .unwrap();
        assert_ne!(
            out[0], out[hf_cell],
            "distinct matrix entries must yield distinct dequantised values"
        );
    }

    #[test]
    fn residual_all_zero_block_is_all_zero_samples() {
        // An all-zero coefficient block inverse-transforms to all-zero
        // spatial samples for every covered square transform.
        let set = materialise_default_dequant_set().unwrap();
        for (t, n) in [
            (TransformType::Dct8x8, 64),
            (TransformType::Dct16x16, 256),
            (TransformType::Dct32x32, 1024),
        ] {
            let b = block(vec![0i32; n]);
            let out = decode_block_to_residual(&b, t, 1, 4, &set, &oim(), &qm()).unwrap();
            assert_eq!(out.len(), n, "{t:?}");
            assert!(
                out.iter().all(|&v| v.abs() < 1e-6),
                "{t:?} residual not all zero"
            );
        }
    }

    #[test]
    fn residual_rect_all_zero_block_is_all_zero_samples() {
        // Rectangular + large-DCT all-zero blocks inverse-transform to
        // all-zero spatial samples (pixel-cell count == grid-cell
        // count).
        let set = materialise_default_dequant_set().unwrap();
        for t in [
            TransformType::Dct16x8,
            TransformType::Dct8x16,
            TransformType::Dct32x8,
            TransformType::Dct8x32,
            TransformType::Dct32x16,
            TransformType::Dct16x32,
            TransformType::Dct64x64,
            TransformType::Dct64x32,
            TransformType::Dct32x64,
        ] {
            let (bw, bh) = covered_grid_dims(t).unwrap();
            let n = bw * bh;
            let b = block(vec![0i32; n]);
            let out = decode_block_to_residual(&b, t, 1, 4, &set, &oim(), &qm()).unwrap();
            // Output pixel count equals the grid cell count.
            assert_eq!(out.len(), n, "{t:?}");
            assert!(
                out.iter().all(|&v| v.abs() < 1e-6),
                "{t:?} residual not all zero"
            );
        }
    }

    #[test]
    fn residual_rect_dc_only_block_is_flat() {
        // A pure-DC rectangular block inverse-transforms to a flat
        // (constant) pixel block, just like the square case — this pins
        // the wide-grid dequant → oriented-IDCT chain end-to-end for
        // rectangular transforms.
        let set = materialise_default_dequant_set().unwrap();
        for t in [
            TransformType::Dct16x8,
            TransformType::Dct8x16,
            TransformType::Dct32x16,
        ] {
            let (bw, bh) = covered_grid_dims(t).unwrap();
            let n = bw * bh;
            let mut coeffs = vec![0i32; n];
            coeffs[0] = 10;
            let b = block(coeffs);
            let out = decode_block_to_residual(&b, t, 1, 3, &set, &oim(), &qm()).unwrap();
            assert_eq!(out.len(), n, "{t:?}");
            let first = out[0];
            assert!(first.abs() > 1e-9, "{t:?} DC residual unexpectedly zero");
            for (i, &v) in out.iter().enumerate() {
                assert!(
                    (v - first).abs() < 1e-2,
                    "{t:?} cell {i} = {v} not flat (first = {first})"
                );
            }
        }
    }

    #[test]
    fn residual_dc_only_block_is_flat() {
        // A block with only the DC coefficient (cell 0) non-zero
        // inverse-transforms to a *flat* spatial block: every sample
        // equals the same constant (the IDCT of a pure-DC input is a
        // constant). This pins the dequant → IDCT chain end-to-end.
        let set = materialise_default_dequant_set().unwrap();
        for (t, n) in [
            (TransformType::Dct8x8, 64),
            (TransformType::Dct16x16, 256),
            (TransformType::Dct32x32, 1024),
        ] {
            let mut coeffs = vec![0i32; n];
            coeffs[0] = 10;
            let b = block(coeffs);
            let out = decode_block_to_residual(&b, t, 1, 3, &set, &oim(), &qm()).unwrap();
            assert_eq!(out.len(), n, "{t:?}");
            let first = out[0];
            // The DC dequantised value is non-zero, so the flat block
            // is non-zero too.
            assert!(first.abs() > 1e-9, "{t:?} DC residual unexpectedly zero");
            for (i, &v) in out.iter().enumerate() {
                assert!(
                    (v - first).abs() < 1e-3,
                    "{t:?} cell {i} = {v} not flat (first = {first})"
                );
            }
        }
    }

    #[test]
    fn residual_matches_manual_dequant_then_idct() {
        // End-to-end equivalence: decode_block_to_residual must equal
        // dequant_block_for_transform followed by idct_for_transform.
        let set = materialise_default_dequant_set().unwrap();
        let mut coeffs = vec![0i32; 64];
        coeffs[0] = 6;
        coeffs[1] = -3;
        coeffs[9] = 2;
        let b = block(coeffs);
        let chained =
            decode_block_to_residual(&b, TransformType::Dct8x8, 1, 5, &set, &oim(), &qm()).unwrap();
        let dq = dequant_block_for_transform(&b, TransformType::Dct8x8, 1, 5, &set, &oim(), &qm())
            .unwrap();
        let manual = idct_for_transform(TransformType::Dct8x8, &dq).unwrap();
        assert_eq!(chained.len(), manual.len());
        for (i, (&a, &c)) in chained.iter().zip(manual.iter()).enumerate() {
            assert_eq!(a, c, "cell {i}");
        }
    }

    #[test]
    fn residual_rejects_uncovered_transform() {
        let set = materialise_default_dequant_set().unwrap();
        let b = block(vec![0i32; 64]);
        let err = decode_block_to_residual(&b, TransformType::Afv0, 1, 1, &set, &oim(), &qm())
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }
}
