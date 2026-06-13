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
//! ## Coordinate convention (square DCT transforms)
//!
//! Round 286 covers the **square plain-DCT** transforms — `DCT8×8`,
//! `DCT16×16`, `DCT32×32` — where the coefficient block, the
//! dequantisation matrix, and the inverse-DCT input all share one
//! unambiguous `dim × dim` row-major layout (`i = y * dim + x`):
//!
//! * [`crate::coeff_order::varblock_size_for_order`] gives the
//!   coefficient grid `(bwidth, bheight)`; for a square transform
//!   `bwidth == bheight == dim`.
//! * [`crate::dct_quant_weights::weights_matrix_dims_for_slot`] gives
//!   the dequant-matrix dims; for the square slots (0 → 8×8, 4 →
//!   16×16, 5 → 32×32) this is the same `dim × dim`.
//! * [`crate::idct::dct_pixel_dims`] gives `(rows, cols) = (dim, dim)`.
//!
//! Because all three layouts coincide, the per-coefficient mapping is
//! the identity: dequant-matrix entry for raster cell `i` multiplies
//! decoded coefficient `coeffs[i]`, and the result feeds raster cell
//! `i` of the inverse-DCT input. The rectangular and non-DCT
//! transforms (whose coefficient grid is stored "wide" while their
//! pixel block may be "tall", and whose IDCT path is the I.2.3
//! dispatch rather than a plain separable IDCT) are deferred to a
//! follow-up round so their orientation handling can be pinned
//! independently.
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

use crate::dct_quant_weights::{slot_for_transform, DequantMatrixSet};
use crate::dct_select::TransformType;
use crate::hf_dequant::{dequant_hf_coefficient, QmScaleFactors};
use crate::idct::{dct_pixel_dims, idct_for_transform};
use crate::metadata_fdis::OpsinInverseMatrix;
use crate::pass_group_hf::DecodedHfBlock;
use oxideav_core::{Error, Result};

/// The square plain-DCT transforms covered by the round-286 per-block
/// decode walk. Returns the side `dim` (8, 16 or 32) for a covered
/// transform, or `None` for any rectangular / non-DCT transform whose
/// orientation handling is deferred to a follow-up round.
pub fn covered_square_dim(t: TransformType) -> Option<usize> {
    match t {
        TransformType::Dct8x8 => Some(8),
        TransformType::Dct16x16 => Some(16),
        TransformType::Dct32x32 => Some(32),
        _ => None,
    }
}

/// Reject any transform outside the round-286 covered set with a
/// precise [`Error::Unsupported`].
fn require_covered(t: TransformType) -> Result<usize> {
    covered_square_dim(t).ok_or_else(|| {
        Error::Unsupported(format!(
            "JXL block_dequant: per-block decode walk covers the square \
             plain-DCT transforms (DCT8×8, DCT16×16, DCT32×32) only; {t:?} \
             is a rectangular / non-DCT transform whose orientation is \
             deferred to a follow-up round"
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
    let dim = require_covered(t)?;
    if channel >= 3 {
        return Err(Error::InvalidData(format!(
            "JXL dequant_block_for_transform: channel {channel} must be < 3"
        )));
    }
    let n = dim * dim;
    if decoded.coeffs.len() != n {
        return Err(Error::InvalidData(format!(
            "JXL dequant_block_for_transform: decoded.coeffs length {} != \
             dim * dim ({n}) for {t:?}",
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
             dim * dim ({n}) for slot {slot} channel {channel}",
            matrix.len()
        )));
    }

    let mut out = vec![0.0f32; n];
    for i in 0..n {
        // Per the `DecodedHfBlock` documentation `coeffs[i]` is the
        // raster cell `i`; the square-transform dequant matrix shares
        // that same `i = y * dim + x` layout, so the per-cell mapping
        // is the identity.
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
    // For the covered square transforms `dct_pixel_dims` returns
    // `(dim, dim)`; we assert that invariant defensively so a future
    // table edit cannot silently feed a mis-sized block to the IDCT.
    let dim = require_covered(t)?;
    debug_assert_eq!(
        dct_pixel_dims(t),
        Some((dim, dim)),
        "JXL decode_block_to_residual: dct_pixel_dims({t:?}) must be ({dim}, {dim})"
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
    fn covered_set_is_the_three_square_dcts() {
        assert_eq!(covered_square_dim(TransformType::Dct8x8), Some(8));
        assert_eq!(covered_square_dim(TransformType::Dct16x16), Some(16));
        assert_eq!(covered_square_dim(TransformType::Dct32x32), Some(32));
        // A representative rectangular + a non-DCT transform are not
        // covered.
        assert_eq!(covered_square_dim(TransformType::Dct16x8), None);
        assert_eq!(covered_square_dim(TransformType::Dct8x16), None);
        assert_eq!(covered_square_dim(TransformType::Hornuss), None);
        assert_eq!(covered_square_dim(TransformType::Afv0), None);
    }

    #[test]
    fn dequant_rejects_uncovered_transform() {
        let set = materialise_default_dequant_set().unwrap();
        let b = block(vec![0i32; 128]);
        let err =
            dequant_block_for_transform(&b, TransformType::Dct16x8, 1, 1, &set, &oim(), &qm())
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
