//! End-to-end per-LfGroup VarDCT reconstruction from cross-pass
//! accumulated coefficients to spatial residual planes —
//! ISO/IEC FDIS 18181-1:2021 §C.8.3 + Annex F.3 + Annex I.2.
//!
//! ## Scope
//!
//! This module is the **integration layer** that ties together the
//! per-stage VarDCT primitives into a single per-LfGroup reconstruction
//! call, driven from the cross-pass accumulated quantised coefficients:
//!
//! 1. [`crate::cross_pass::accumulate_three_channel_multi_pass`] folds
//!    the multi-pass `out[p][i]` per-pass [`DecodedHfBlock`] stack into
//!    one accumulated quantised coefficient grid per varblock per channel
//!    (the §C.8.3 per-pass-`shift[]`-then-sum rule).
//! 2. For each varblock this driver extracts that varblock's per-channel
//!    LLF (DC subband) block from the LfGroup's dequantised LF image
//!    (Listing I.16, [`crate::vardct::compose_lf_to_llf_block`]).
//! 3. [`crate::block_dequant::decode_block_to_residual_with_llf`] runs
//!    the F.3 dequant → §I.2.4 LLF-prefix merge → §I.2.3.2 inverse DCT,
//!    producing the varblock's `R × C` spatial residual block — for
//!    **every** [`TransformType`], square / non-square / non-DCT alike.
//! 4. [`crate::residual_plane::assemble_three_channel_planes_with_lf`]
//!    places each block into the per-channel padded plane, and Annex G
//!    chroma-from-luma restores the X / B planes from Y.
//!
//! The output is the three XYB residual planes on the LfGroup's padded
//! block grid; the caller crops to `lf_w × lf_h` (§6.2,
//! [`crate::residual_plane::ChannelResidualPlanes::crop_to`]) and runs
//! Gaborish (Annex J.2) + EPF (Annex J.3) above this primitive.
//!
//! ## Why this is the non-square + cross-pass milestone closer
//!
//! Each per-stage primitive was already non-square-correct in isolation
//! (the DctSelect grid places rectangular footprints, the IDCT carries
//! the Listing I.4 pre/post-transpose for `R != C`, the LLF extraction
//! reads a `cy × cx` sub-block, the dequant matrix is the wide
//! `bwidth × bheight` layout). What was missing was the **wiring** that
//! drives them from a single accumulated-coefficient source so a
//! multi-pass frame containing non-square transforms reconstructs to
//! spatial samples in one call. This module is that wiring; it owns no
//! bit reads, no entropy state, and no spec re-derivation.

use oxideav_core::{Error, Result};

use crate::block_dequant::decode_block_to_residual_with_llf;
use crate::cross_pass::accumulate_three_channel_multi_pass;
use crate::dct_quant_weights::DequantMatrixSet;
use crate::dct_select::DctSelectGrid;
use crate::frame_header::Passes;
use crate::hf_dequant::QmScaleFactors;
use crate::lf_dequant::LfDequantOutput;
use crate::lf_global::LfChannelCorrelation;
use crate::metadata_fdis::OpsinInverseMatrix;
use crate::multi_pass_decode::MultiPassThreeChannelOutput;
use crate::pass_group_hf::DecodedHfBlock;
use crate::residual_plane::{
    apply_chroma_from_luma, assemble_three_channel_planes_with_lf, ChannelResidualPlanes,
};
use crate::varblock_walk::Varblock;

/// F.3 dequantisation inputs shared by every varblock of an LfGroup: the
/// materialised dequant-matrix set, the opsin-inverse bias matrix, and
/// the per-channel `0.8^(qm_scale - 2)` scale factors.
///
/// Grouped into one struct so the per-LfGroup reconstruction call has a
/// manageable arity (the F.3 stage needs all three for every block).
pub struct DequantContext<'a> {
    /// Materialised dequant-matrix set (one matrix per slot per channel).
    pub set: &'a DequantMatrixSet,
    /// Opsin-inverse matrix carrying the F.3 quant bias.
    pub oim: &'a OpsinInverseMatrix,
    /// Per-channel `0.8^(qm_scale - 2)` scale factors.
    pub qm: &'a QmScaleFactors,
}

/// Reconstruct one LfGroup's three XYB residual planes from its
/// **cross-pass accumulated** quantised coefficients and dequantised LF
/// image.
///
/// `multi_pass[p][i]` is the per-pass per-varblock decode output (the
/// [`MultiPassThreeChannelOutput`] shape); this driver first folds it
/// across passes via
/// [`accumulate_three_channel_multi_pass`] (applying the Table C.6
/// `shift[]` per-pass left-shift and the §C.8.3 cross-pass sum), then
/// runs the per-varblock dequant → LLF-merge → IDCT → placement →
/// chroma-from-luma pipeline.
///
/// * `passes` supplies `num_passes` + the per-pass `shift[]` vector.
/// * `grid` is the shared per-LfGroup [`DctSelectGrid`]; the accumulated
///   varblocks are placed in its raster walk order (and this driver
///   asserts the accumulated varblock count matches the grid's
///   TopLeft-cell count, so a mis-sized multi-pass output surfaces before
///   any decode work).
/// * `lf` is the LfGroup's dequantised LF image
///   ([`crate::lf_dequant::dequant_lf`] output); the three LF channels
///   must share dims (the non-subsampled case — enforced downstream by
///   [`assemble_three_channel_planes_with_lf`]).
/// * `dq` carries the F.3 dequant inputs.
/// * `x_from_y` / `b_from_y` are the per-64×64-tile CfL factor channels;
///   `cfl` is the [`LfChannelCorrelation`] base/colour factors.
///
/// Returns the three XYB residual planes on the padded block grid (the
/// caller crops + filters). Errors from any stage propagate verbatim;
/// the accumulated-coefficient count / placement invariants are checked
/// before the per-block walk begins.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_lf_group_cross_pass(
    passes: &Passes,
    grid: &DctSelectGrid,
    lf: &LfDequantOutput,
    dq: &DequantContext<'_>,
    x_from_y: &[i32],
    b_from_y: &[i32],
    cfl: &LfChannelCorrelation,
    multi_pass: &MultiPassThreeChannelOutput,
) -> Result<ChannelResidualPlanes> {
    // Fold the per-pass coefficient stack into one accumulated quantised
    // grid per varblock per channel.
    let accumulated = accumulate_three_channel_multi_pass(passes, multi_pass)?;

    // The accumulated varblock list is in the §C.8.3 raster walk order —
    // identical to the order the placement driver walks the grid. Verify
    // the count matches the grid's TopLeft-cell count so a mismatch can't
    // silently mis-index the per-block lookup mid-walk.
    let expected = crate::varblock_walk::count_varblocks(grid) as usize;
    if accumulated.len() != expected {
        return Err(Error::InvalidData(format!(
            "JXL vardct_reconstruct: accumulated varblock count {} != grid \
             TopLeft-cell count {expected}",
            accumulated.len()
        )));
    }

    // The placement driver walks the grid once per channel, invoking the
    // closure in raster order each time. We index the accumulated list by
    // a per-channel walk counter; because every channel walks the same
    // grid in the same order, the counter advances in lockstep with the
    // accumulated list's order. We also cross-check that the closure's
    // varblock matches the accumulated entry's recorded placement (defence
    // against a future placement-order change).
    let mut counters = [0usize; 3];
    assemble_three_channel_planes_with_lf(grid, lf, |c, vb: &Varblock, llf: &[f32]| {
        let idx = counters[c];
        if idx >= accumulated.len() {
            return Err(Error::InvalidData(format!(
                "JXL vardct_reconstruct: channel {c} walk overran accumulated \
                 varblock list ({} entries)",
                accumulated.len()
            )));
        }
        let (acc_vb, acc_channels) = &accumulated[idx];
        if acc_vb.x != vb.x || acc_vb.y != vb.y || acc_vb.transform != vb.transform {
            return Err(Error::InvalidData(format!(
                "JXL vardct_reconstruct: channel {c} varblock {idx} placement \
                 ({},{},{:?}) differs from accumulated ({},{},{:?})",
                vb.x, vb.y, vb.transform, acc_vb.x, acc_vb.y, acc_vb.transform
            )));
        }
        counters[c] += 1;

        // Wrap the accumulated quantised grid in a DecodedHfBlock so the
        // existing F.3 dequant primitive consumes it unchanged. The
        // remaining_non_zeros / coeffs_read fields are decode-side
        // bookkeeping the dequant ignores.
        let decoded = DecodedHfBlock {
            coeffs: acc_channels[c].clone(),
            remaining_non_zeros: 0,
            coeffs_read: 0,
        };
        decode_block_to_residual_with_llf(
            &decoded,
            vb.transform,
            c,
            vb.hf_mul,
            dq.set,
            dq.oim,
            dq.qm,
            llf,
        )
    })
    .and_then(|mut planes| {
        apply_chroma_from_luma(&mut planes, x_from_y, b_from_y, cfl)?;
        Ok(planes)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_context_resolver::ThreeChannelVarblock;
    use crate::dct_quant_weights::materialise_default_dequant_set;
    use crate::dct_select::{DctSelectCell, TransformType};

    fn passes(num_passes: u32, shift: Vec<u32>) -> Passes {
        Passes {
            num_passes,
            num_ds: 0,
            shift,
            downsample: Vec::new(),
            last_pass: Vec::new(),
        }
    }

    fn grid(cells: Vec<DctSelectCell>, hf_mul: Vec<i32>, w: u32, h: u32) -> DctSelectGrid {
        DctSelectGrid {
            cells,
            hf_mul,
            width_blocks: w,
            height_blocks: h,
        }
    }

    fn vb(x: u32, y: u32, t: TransformType) -> Varblock {
        Varblock {
            x,
            y,
            transform: t,
            hf_mul: 1,
        }
    }

    fn block(coeffs: Vec<i32>) -> DecodedHfBlock {
        DecodedHfBlock {
            coeffs,
            remaining_non_zeros: 0,
            coeffs_read: 0,
        }
    }

    fn tcv(v: Varblock, x: Vec<i32>, y: Vec<i32>, b: Vec<i32>) -> ThreeChannelVarblock {
        (v, [block(x), block(y), block(b)], [0, 0, 0])
    }

    fn oim() -> OpsinInverseMatrix {
        OpsinInverseMatrix::default()
    }

    fn qm() -> QmScaleFactors {
        QmScaleFactors {
            x_factor: 0.8,
            b_factor: 1.0,
        }
    }

    fn lf_flat(value: f32, w: u32, h: u32) -> LfDequantOutput {
        let n = (w * h) as usize;
        LfDequantOutput {
            samples: [vec![value; n], vec![value; n], vec![value; n]],
            widths: [w, w, w],
            heights: [h, h, h],
        }
    }

    /// A single-pass single-DCT8×8 varblock reconstructs to a flat plane
    /// (constant LF, zero HF) — sanity check the wiring runs through.
    #[test]
    fn single_pass_dct8x8_reconstructs() {
        let set = materialise_default_dequant_set().unwrap();
        let p = passes(1, vec![]);
        let g = grid(
            vec![DctSelectCell::TopLeft(TransformType::Dct8x8)],
            vec![1],
            1,
            1,
        );
        let lf = lf_flat(4.0, 1, 1);
        let mp: MultiPassThreeChannelOutput = vec![vec![tcv(
            vb(0, 0, TransformType::Dct8x8),
            vec![0; 64],
            vec![0; 64],
            vec![0; 64],
        )]];
        let dq = DequantContext {
            set: &set,
            oim: &oim(),
            qm: &qm(),
        };
        let planes = reconstruct_lf_group_cross_pass(
            &p,
            &g,
            &lf,
            &dq,
            &[0i32; 1],
            &[0i32; 1],
            &LfChannelCorrelation::default(),
            &mp,
        )
        .unwrap();
        assert_eq!(planes.dims(), (8, 8));
        // Constant LF + zero HF → the block is the DC reconstructed flat;
        // every sample of the Y plane equals the same constant.
        let v0 = planes.y().get(0, 0).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                let v = planes.y().get(x, y).unwrap();
                assert!((v - v0).abs() < 1e-3, "Y ({x},{y}) = {v} != {v0}");
            }
        }
    }

    /// A non-square DCT8×16 varblock (16 px wide × 8 px tall; footprint
    /// 2×1 cells) reconstructs to spatial samples through the full
    /// cross-pass → dequant → IDCT → placement walk. This is the
    /// milestone's non-square reconstruction path.
    #[test]
    fn non_square_dct8x16_reconstructs_to_spatial_samples() {
        let set = materialise_default_dequant_set().unwrap();
        let p = passes(1, vec![]);
        // DCT8×16 footprint is (bcols=2, brows=1) → a 2×1 grid.
        let g = grid(
            vec![
                DctSelectCell::TopLeft(TransformType::Dct8x16),
                DctSelectCell::Continuation,
            ],
            vec![1, 0],
            2,
            1,
        );
        // LF image is 2×1 samples (one per 8×8 block cell). DCT8×16 reads
        // a cx=2 × cy=1 LF sub-block.
        let lf = LfDequantOutput {
            samples: [vec![3.0, 5.0], vec![3.0, 5.0], vec![3.0, 5.0]],
            widths: [2, 2, 2],
            heights: [1, 1, 1],
        };
        // DCT8×16 coefficient grid is 16 × 8 = 128 cells. Zero HF.
        let mp: MultiPassThreeChannelOutput = vec![vec![tcv(
            vb(0, 0, TransformType::Dct8x16),
            vec![0; 128],
            vec![0; 128],
            vec![0; 128],
        )]];
        let dq = DequantContext {
            set: &set,
            oim: &oim(),
            qm: &qm(),
        };
        // 16×8 padded plane → ceil(16/64)=1 × ceil(8/64)=1 CfL tile.
        let planes = reconstruct_lf_group_cross_pass(
            &p,
            &g,
            &lf,
            &dq,
            &[0i32; 1],
            &[0i32; 1],
            &LfChannelCorrelation::default(),
            &mp,
        )
        .unwrap();
        // Padded plane is 16 px wide (2 cells) × 8 px tall (1 cell).
        assert_eq!(planes.dims(), (16, 8));
        // The Y plane is finite everywhere (the IDCT produced real
        // samples across the full non-square footprint).
        for y in 0..8 {
            for x in 0..16 {
                let v = planes.y().get(x, y).unwrap();
                assert!(v.is_finite(), "Y ({x},{y}) not finite: {v}");
            }
        }
    }

    /// A two-pass frame: the second pass's HF delta is added on top of
    /// pass 0 (with pass 0 left-shifted by shift[0]). The reconstruction
    /// differs from a single-pass-only run, proving the cross-pass
    /// accumulation feeds the spatial output.
    #[test]
    fn two_pass_accumulation_changes_reconstruction() {
        let set = materialise_default_dequant_set().unwrap();
        let g = grid(
            vec![DctSelectCell::TopLeft(TransformType::Dct8x8)],
            vec![1],
            1,
            1,
        );
        let lf = lf_flat(0.0, 1, 1);
        let dq = DequantContext {
            set: &set,
            oim: &oim(),
            qm: &qm(),
        };
        // A single AC coefficient placed at raster cell 1 of the Y channel.
        let mut c_pass0 = vec![0i32; 64];
        c_pass0[1] = 4;
        let mut c_pass1 = vec![0i32; 64];
        c_pass1[1] = 1;

        // Single-pass reference (just pass 0, shift 0).
        let p1 = passes(1, vec![]);
        let mp1: MultiPassThreeChannelOutput = vec![vec![tcv(
            vb(0, 0, TransformType::Dct8x8),
            vec![0; 64],
            c_pass0.clone(),
            vec![0; 64],
        )]];
        let single = reconstruct_lf_group_cross_pass(
            &p1,
            &g,
            &lf,
            &dq,
            &[0i32; 1],
            &[0i32; 1],
            &LfChannelCorrelation::default(),
            &mp1,
        )
        .unwrap();

        // Two-pass: pass 0 shift = 1 (×2), pass 1 (last) shift 0.
        // Accumulated Y cell1 = 4·2 + 1 = 9 != single-pass 4.
        let p2 = passes(2, vec![1]);
        let mp2: MultiPassThreeChannelOutput = vec![
            vec![tcv(
                vb(0, 0, TransformType::Dct8x8),
                vec![0; 64],
                c_pass0,
                vec![0; 64],
            )],
            vec![tcv(
                vb(0, 0, TransformType::Dct8x8),
                vec![0; 64],
                c_pass1,
                vec![0; 64],
            )],
        ];
        let multi = reconstruct_lf_group_cross_pass(
            &p2,
            &g,
            &lf,
            &dq,
            &[0i32; 1],
            &[0i32; 1],
            &LfChannelCorrelation::default(),
            &mp2,
        )
        .unwrap();

        // The accumulated coefficient is larger, so at least one Y sample
        // differs between the single- and two-pass reconstructions.
        let mut differ = false;
        for y in 0..8 {
            for x in 0..8 {
                let a = single.y().get(x, y).unwrap();
                let b = multi.y().get(x, y).unwrap();
                if (a - b).abs() > 1e-4 {
                    differ = true;
                }
            }
        }
        assert!(
            differ,
            "two-pass accumulation must change the spatial reconstruction"
        );
    }

    /// A mismatched accumulated varblock count (grid has more TopLeft
    /// cells than the multi-pass output provides) is rejected before the
    /// per-block walk.
    #[test]
    fn rejects_varblock_count_mismatch_with_grid() {
        let set = materialise_default_dequant_set().unwrap();
        let p = passes(1, vec![]);
        // Grid expects two varblocks.
        let g = grid(
            vec![
                DctSelectCell::TopLeft(TransformType::Dct8x8),
                DctSelectCell::TopLeft(TransformType::Dct8x8),
            ],
            vec![1, 1],
            2,
            1,
        );
        let lf = lf_flat(0.0, 2, 1);
        // Multi-pass output provides only one.
        let mp: MultiPassThreeChannelOutput = vec![vec![tcv(
            vb(0, 0, TransformType::Dct8x8),
            vec![0; 64],
            vec![0; 64],
            vec![0; 64],
        )]];
        let dq = DequantContext {
            set: &set,
            oim: &oim(),
            qm: &qm(),
        };
        let err = reconstruct_lf_group_cross_pass(
            &p,
            &g,
            &lf,
            &dq,
            &[0i32; 1],
            &[0i32; 1],
            &LfChannelCorrelation::default(),
            &mp,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }
}
