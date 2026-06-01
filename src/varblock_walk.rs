//! Per-LfGroup varblock-walk driver —
//! ISO/IEC FDIS 18181-1:2021 §C.5.4 (DctSelect placement prose) +
//! §C.8.3 (per-pass per-channel varblock decode loop).
//!
//! ## Scope (round 208)
//!
//! Round 208 lands the typed scaffolding that walks the per-LfGroup
//! [`crate::dct_select::DctSelectGrid`] in raster order, yielding the
//! `(top_left_x, top_left_y, transform_type, hf_mul)` quadruple at
//! each varblock placement. This is the "per-LfGroup varblock-shape
//! grid" the round-177 / 183 / 190 module notes repeatedly defer to:
//! a pure storage-walk primitive that bridges the round-13 DctSelect
//! placement (`derive_dct_select`) with the round-190 per-pass
//! per-channel NonZeros routing
//! ([`crate::per_pass_non_zeros::PerPassNonZerosGrids`]).
//!
//! No bit reads, no spec re-derivation, no histogram materialisation
//! — same pure-control-flow primitive shape as round-89
//! [`crate::dct_quant_weights`], round-95 [`crate::hf_dequant`],
//! round-121 [`crate::llf_from_lf`], round-138
//! [`crate::chroma_from_luma`], round-141 [`crate::gaborish`],
//! round-144 [`crate::epf`], round-147 [`crate::afv::afv_idct`],
//! round-159 / 164 [`crate::pass_group_hf`], round-177
//! [`crate::non_zeros_grid`], round-183
//! [`crate::per_channel_non_zeros`], and round-190
//! [`crate::per_pass_non_zeros`].
//!
//! ## FDIS prose anchor
//!
//! From §C.5.4 (DctSelect derivation):
//!
//! > The DctSelect and HfMul fields are derived from the first and
//! > second rows of BlockInfo. These two fields have ceil(height / 8)
//! > rows and ceil(width / 8) columns. They are reconstructed by
//! > iterating over the columns of BlockInfo to obtain a varblock
//! > transform type type (the sample at the first row) and a
//! > quantization multiplier mul (the sample at the second row). The
//! > type corresponds to a valid varblock type and covers a rectangle
//! > that does not cross group boundaries; this is the DctSelect
//! > sample and it is stored at the coordinates of the top-left 8 × 8
//! > rectangle of the varblock, which is positioned as much towards
//! > the top and towards the left as possible without overlapping
//! > already-positioned varblocks. The HfMul sample is stored at the
//! > same position and gets the value 1 + mul.
//!
//! From §C.8.3 (per-varblock decode loop):
//!
//! > For each pass `p ∈ [0, num_passes)` the PassGroup decoder scans
//! > every varblock once. Each varblock contributes one block-context
//! > read + one per-channel coefficient decode at its top-left
//! > placement; the placement is read off the DctSelect grid (a
//! > non-top-left cell is skipped because it's a continuation of an
//! > earlier varblock).
//!
//! The round-208 walker captures the structural intersection of those
//! two prose sections: a raster-order iteration that emits one
//! [`Varblock`] per top-left cell.
//!
//! ## Round-177 → 183 → 190 → 208 layering
//!
//! * Round 177 [`crate::non_zeros_grid::NonZerosGrid`] — single-channel,
//!   single-pass position grid.
//! * Round 183 [`crate::per_channel_non_zeros::PerChannelNonZerosGrids`]
//!   — per-channel container of per-position grids.
//! * Round 190 [`crate::per_pass_non_zeros::PerPassNonZerosGrids`] —
//!   per-pass container of per-channel grids.
//! * Round 208 [`VarblockWalk`] (this module) — the per-LfGroup
//!   shape-grid iterator that drives the round-190 typed driver at
//!   each varblock placement.
//!
//! ## Scope boundary
//!
//! The §C.7.2 entropy histogram array, the per-pass `EntropyStream`
//! / `HybridUintState` wiring, and the per-channel `BlockContext()`
//! history threading remain follow-up work — the `decode_symbol` and
//! `read_non_zeros` closures abstract over them at the per-block
//! level, exactly as in rounds 177 / 183 / 190. Round 208 walks the
//! shape grid; it does not materialise histograms or compute
//! `block_context` itself.

use oxideav_core::{Error, Result};

use crate::dct_select::{DctSelectCell, DctSelectGrid, TransformType};
use crate::pass_group_hf::DecodedHfBlock;
use crate::per_pass_non_zeros::PerPassNonZerosGrids;

/// One varblock placement read off a [`DctSelectGrid`]. The varblock
/// is uniquely identified by its top-left cell coordinates `(x, y)`
/// in 8×8-block grid units, plus its [`TransformType`] and per-block
/// `hf_mul` multiplier (= `1 + mul` per §C.5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Varblock {
    /// 8×8-block grid column of the top-left cell.
    pub x: u32,
    /// 8×8-block grid row of the top-left cell.
    pub y: u32,
    /// Transform type assigned to this varblock.
    pub transform: TransformType,
    /// `HfMul = 1 + mul` (§C.5.4 stored at the top-left cell).
    pub hf_mul: i32,
}

/// Raster-order iterator over the top-left varblock placements stored
/// in a [`DctSelectGrid`].
///
/// The walker is borrow-based — `'a` ties the iterator to the grid.
/// Continuation cells (interior of a multi-block varblock) are
/// skipped; Empty cells are an error (round 13's `derive_dct_select`
/// guarantees the grid is fully covered after a successful build, so
/// a residual Empty here would indicate caller-side grid mutation
/// after derivation).
#[derive(Debug, Clone)]
pub struct VarblockWalk<'a> {
    grid: &'a DctSelectGrid,
    cursor: usize,
}

impl<'a> VarblockWalk<'a> {
    /// Construct a walker over the given grid. The cursor starts at
    /// the top-left of the grid; the first `next()` call yields the
    /// varblock whose top-left lives at `(0, 0)`.
    pub fn new(grid: &'a DctSelectGrid) -> Self {
        Self { grid, cursor: 0 }
    }

    /// Yield the next varblock placement, or `None` at end-of-grid.
    /// `Err(InvalidData)` if the cursor lands on an [`DctSelectCell::Empty`]
    /// cell — that would indicate a malformed grid (a grid returned by
    /// [`crate::dct_select::derive_dct_select`] is fully covered).
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<Varblock>> {
        let width = self.grid.width_blocks as usize;
        let total = self.grid.cells.len();
        while self.cursor < total {
            let idx = self.cursor;
            self.cursor += 1;
            match self.grid.cells[idx] {
                DctSelectCell::TopLeft(transform) => {
                    let x = (idx % width) as u32;
                    let y = (idx / width) as u32;
                    let hf_mul = self.grid.hf_mul[idx];
                    return Ok(Some(Varblock {
                        x,
                        y,
                        transform,
                        hf_mul,
                    }));
                }
                DctSelectCell::Continuation => continue,
                DctSelectCell::Empty => {
                    let x = idx % width;
                    let y = idx / width;
                    return Err(Error::InvalidData(format!(
                        "JXL varblock walk: residual Empty cell at ({x}, {y}) — \
                         grid not fully covered (round-13 derive_dct_select bug or \
                         caller-side mutation)"
                    )));
                }
            }
        }
        Ok(None)
    }

    /// Drain the walker into a `Vec<Varblock>` in raster order.
    pub fn collect(mut self) -> Result<Vec<Varblock>> {
        let mut out = Vec::new();
        while let Some(v) = self.next()? {
            out.push(v);
        }
        Ok(out)
    }
}

/// Number of varblocks in a [`DctSelectGrid`] (= number of top-left
/// cells). Cheap O(n) cell scan. Useful for pre-sizing per-block
/// caller buffers.
pub fn count_varblocks(grid: &DctSelectGrid) -> u32 {
    let mut n = 0u32;
    for c in &grid.cells {
        if matches!(c, DctSelectCell::TopLeft(_)) {
            n += 1;
        }
    }
    n
}

/// Typed per-LfGroup per-pass per-channel varblock decode driver.
///
/// Walks the varblocks of `grid` in raster order; for each varblock
/// the driver invokes the caller's `block_ctx_for_varblock` closure
/// (which packs together the Listing C.13 `BlockContext()` lookup,
/// thereby threading the round-90 `HfBlockContext` bundle the walker
/// does not own) and calls
/// [`PerPassNonZerosGrids::decode_block_at_for_pass_channel`] at the
/// `(p, c, x, y)` quadruple with the returned `block_ctx`.
///
/// The walker keeps the same shape as round 190: every callback is
/// invoked with the matching `c` (channel) and the varblock's
/// `(x, y, transform_type, hf_mul)` wiring, so the closures remain
/// pure routing — no histogram material crosses the boundary.
///
/// Returns the in-order vector of `(Varblock, DecodedHfBlock,
/// raw_non_zeros)` triples — one entry per top-left placement walked.
/// The vector preserves raster order so callers that need to write
/// per-block coefficients into a per-channel buffer get a
/// deterministic layout. A failure mid-walk surfaces as a single
/// error (and the partial vector is discarded).
///
/// Caller responsibilities (still!):
/// * The `block_ctx_for_varblock` closure encapsulates the full
///   Listing C.13 `BlockContext()` lookup — the per-LfGroup
///   `block_ctx_map[order_id][hf_mul - 1]` table read + the
///   `qf_thresholds` / `lf_thresholds` / `qdc[3]` ladder. Round 208
///   does not own the `HfBlockContext` bundle materialisation (that
///   default table is built by round 90's [`crate::hf_pass`], but
///   per-frame overrides + per-varblock `qf` / `qdc` derivation are
///   follow-up work).
/// * `nb_block_ctx` — same source.
/// * The two ANS closures — round 208's walker abstracts the
///   varblock-shape iteration but not the §C.7.2 histograms (#799
///   DOCS-GAP).
#[allow(clippy::too_many_arguments)]
pub fn decode_varblocks_for_pass_channel<H, F, G>(
    grid: &DctSelectGrid,
    nz: &mut PerPassNonZerosGrids,
    p: u32,
    c: u32,
    nb_block_ctx: u32,
    mut block_ctx_for_varblock: H,
    mut read_non_zeros: F,
    mut decode_symbol: G,
) -> Result<Vec<(Varblock, DecodedHfBlock, u32)>>
where
    H: FnMut(&Varblock) -> Result<u32>,
    F: FnMut(u32) -> Result<u32>,
    G: FnMut(u32) -> Result<u32>,
{
    let mut out = Vec::with_capacity(count_varblocks(grid) as usize);
    let mut walk = VarblockWalk::new(grid);
    while let Some(vb) = walk.next()? {
        let ctx = block_ctx_for_varblock(&vb)?;
        let (decoded, raw_non_zeros) = nz.decode_block_at_for_pass_channel(
            p,
            c,
            vb.x,
            vb.y,
            vb.transform,
            ctx,
            nb_block_ctx,
            &mut read_non_zeros,
            &mut decode_symbol,
        )?;
        out.push((vb, decoded, raw_non_zeros));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct_select::{derive_dct_select, TransformType};
    use crate::lf_group::HfMetadata;

    fn make_hf(block_info: Vec<i32>, nb_blocks: u32, info_w: u32) -> HfMetadata {
        HfMetadata {
            nb_blocks,
            x_from_y: vec![0],
            b_from_y: vec![0],
            block_info,
            sharpness: vec![0],
            channel_widths: [1, 1, info_w, 1],
            channel_heights: [1, 1, 2, 1],
        }
    }

    #[test]
    fn walk_single_dct8x8_yields_one_varblock() {
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut walk = VarblockWalk::new(&grid);
        let v0 = walk.next().unwrap().unwrap();
        assert_eq!(v0.x, 0);
        assert_eq!(v0.y, 0);
        assert_eq!(v0.transform, TransformType::Dct8x8);
        assert_eq!(v0.hf_mul, 1);
        assert!(walk.next().unwrap().is_none());
    }

    #[test]
    fn walk_2x2_grid_four_dct8x8_blocks_raster_order() {
        // 16×16 LfGroup, four DCT8×8 in raster order.
        let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let blocks = VarblockWalk::new(&grid).collect().unwrap();
        assert_eq!(blocks.len(), 4);
        // Raster order: (0,0) → (1,0) → (0,1) → (1,1).
        assert_eq!((blocks[0].x, blocks[0].y), (0, 0));
        assert_eq!((blocks[1].x, blocks[1].y), (1, 0));
        assert_eq!((blocks[2].x, blocks[2].y), (0, 1));
        assert_eq!((blocks[3].x, blocks[3].y), (1, 1));
        for b in &blocks {
            assert_eq!(b.transform, TransformType::Dct8x8);
            assert_eq!(b.hf_mul, 1);
        }
    }

    #[test]
    fn walk_skips_continuation_cells_dct16x16() {
        // 16×16 LfGroup with a single DCT16×16 covers all four 8×8
        // cells; only the top-left counts as a varblock.
        let hf = make_hf(vec![4, 0], 1, 1);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let blocks = VarblockWalk::new(&grid).collect().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].x, 0);
        assert_eq!(blocks[0].y, 0);
        assert_eq!(blocks[0].transform, TransformType::Dct16x16);
        assert_eq!(blocks[0].hf_mul, 1);
    }

    #[test]
    fn walk_dct8x16_then_two_dct8x8() {
        // 16×16 grid: first DCT8×16 fills row 0 (covers (0,0)+(1,0));
        // then two DCT8×8 at (0,1) and (1,1).
        let hf = make_hf(vec![7, 0, 0, 0, 0, 0], 3, 3);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let blocks = VarblockWalk::new(&grid).collect().unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].transform, TransformType::Dct8x16);
        assert_eq!((blocks[0].x, blocks[0].y), (0, 0));
        assert_eq!(blocks[1].transform, TransformType::Dct8x8);
        assert_eq!((blocks[1].x, blocks[1].y), (0, 1));
        assert_eq!(blocks[2].transform, TransformType::Dct8x8);
        assert_eq!((blocks[2].x, blocks[2].y), (1, 1));
    }

    #[test]
    fn count_varblocks_matches_walk_collect_len() {
        let hf = make_hf(vec![7, 0, 0, 0, 0, 0], 3, 3);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let n = count_varblocks(&grid);
        let collected = VarblockWalk::new(&grid).collect().unwrap();
        assert_eq!(n as usize, collected.len());
        assert_eq!(n, 3);
    }

    #[test]
    fn count_varblocks_single_dct16x16_is_one() {
        let hf = make_hf(vec![4, 0], 1, 1);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        assert_eq!(count_varblocks(&grid), 1);
    }

    #[test]
    fn walk_residual_empty_cell_errors() {
        // Synthesise a malformed grid: 2×2 cells with one TopLeft and
        // three Empty cells. The walker should error on the first
        // Empty cell. (derive_dct_select would reject this; we build
        // it by hand to test the walker's defence in depth.)
        let grid = DctSelectGrid {
            cells: vec![
                DctSelectCell::TopLeft(TransformType::Dct8x8),
                DctSelectCell::Empty,
                DctSelectCell::Empty,
                DctSelectCell::Empty,
            ],
            hf_mul: vec![1, 0, 0, 0],
            width_blocks: 2,
            height_blocks: 2,
        };
        let mut walk = VarblockWalk::new(&grid);
        // First call: yields the TopLeft at (0,0).
        let v = walk.next().unwrap().unwrap();
        assert_eq!((v.x, v.y), (0, 0));
        // Second call: hits an Empty cell → error.
        assert!(walk.next().is_err());
    }

    #[test]
    fn walk_empty_grid_no_varblocks() {
        // A grid of all-Continuation cells (synthesised) yields no
        // varblocks. derive_dct_select can't produce this but the
        // walker's contract is "no error, no yield".
        let grid = DctSelectGrid {
            cells: vec![DctSelectCell::Continuation; 4],
            hf_mul: vec![0; 4],
            width_blocks: 2,
            height_blocks: 2,
        };
        let mut walk = VarblockWalk::new(&grid);
        assert!(walk.next().unwrap().is_none());
    }

    #[test]
    fn walk_yields_hf_mul_from_top_left() {
        // Verify the walker reads hf_mul from the same cell as the
        // top-left, not from a different cell.
        let hf = make_hf(vec![0, 5], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut walk = VarblockWalk::new(&grid);
        let v = walk.next().unwrap().unwrap();
        assert_eq!(v.hf_mul, 6); // 1 + 5
    }

    #[test]
    fn walk_preserves_transform_diversity() {
        // 32×8 grid with DCT8×8, DCT8×16-then-DCT8×8 mid-row would be
        // ill-formed; use 16×16 with [DCT16×8 (1×2), DCT8×8, DCT8×8].
        // DCT16×8 has dims (cols=1, rows=2). nb_blocks = 3.
        let hf = make_hf(vec![6, 0, 0, 0, 0, 0], 3, 3);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let blocks = VarblockWalk::new(&grid).collect().unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].transform, TransformType::Dct16x8);
        // DCT16×8 (cols=1, rows=2) covers (0,0) + (0,1); next slot
        // is (1,0).
        assert_eq!((blocks[0].x, blocks[0].y), (0, 0));
        assert_eq!(blocks[1].transform, TransformType::Dct8x8);
        assert_eq!((blocks[1].x, blocks[1].y), (1, 0));
        assert_eq!(blocks[2].transform, TransformType::Dct8x8);
        assert_eq!((blocks[2].x, blocks[2].y), (1, 1));
    }

    #[test]
    fn decode_varblocks_for_pass_channel_walks_all_blocks() {
        // Four DCT8×8 in raster order on pass 0 / channel 0; the
        // closures return a constant non-zero count so each call
        // exercises the round-190 typed driver under the round-208
        // walker.
        let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
        // Each varblock's `read_non_zeros` returns 0 (no symbol
        // reads). Each varblock's `decode_symbol` is never called
        // because non_zeros = 0 short-circuits the loop. The
        // `block_ctx_offset_for_transform` closure returns 0 for
        // every transform (a constant block-ctx-map stub).
        let triples = decode_varblocks_for_pass_channel(
            &grid,
            &mut nz,
            0,
            0,
            13,                 // nb_block_ctx
            |_vb| Ok(0),        // block_ctx for varblock
            |_predicted| Ok(0), // read_non_zeros → 0
            |_ctx| Ok(0),       // decode_symbol (unused)
        )
        .unwrap();
        assert_eq!(triples.len(), 4);
        // After the walk: per-channel NonZeros(x, y) at (0,0)..(1,1)
        // should all be 0 (raw_non_zeros = 0 / num_blocks=1 = 0).
        for x in 0..2 {
            for y in 0..2 {
                assert_eq!(nz.get(0, 0, x, y).unwrap(), 0);
            }
        }
    }

    #[test]
    fn decode_varblocks_for_pass_channel_writes_back_per_varblock() {
        // Each varblock returns raw_non_zeros = 8; under
        // num_blocks = 1 (DCT8×8) the per-position update writes
        // 8 to each cell, in raster order.
        use std::cell::Cell;
        let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
        // Closure: read_non_zeros returns 8 (per-block non-zero
        // count); decode_symbol returns ucoeff = 2 (non-zero
        // contributing to the running non_zeros decrement) — 8
        // reads, each decrementing non_zeros until reaching 0.
        let reads_remaining: Cell<i32> = Cell::new(-1);
        let total_decode_calls: Cell<u32> = Cell::new(0);
        let triples = decode_varblocks_for_pass_channel(
            &grid,
            &mut nz,
            0,
            0,
            13,
            |_vb| Ok(0),
            |_predicted| {
                reads_remaining.set(8);
                Ok(8)
            },
            |_ctx| {
                total_decode_calls.set(total_decode_calls.get() + 1);
                let r = reads_remaining.get();
                if r > 0 {
                    reads_remaining.set(r - 1);
                    Ok(2) // non-zero ucoeff → decrements non_zeros
                } else {
                    Ok(0)
                }
            },
        )
        .unwrap();
        assert_eq!(triples.len(), 4);
        // Each varblock decoded 8 ucoeffs.
        assert_eq!(total_decode_calls.get(), 4 * 8);
        // Each cell now stores 8 (ceil(8 / 1) = 8).
        for x in 0..2 {
            for y in 0..2 {
                assert_eq!(nz.get(0, 0, x, y).unwrap(), 8);
            }
        }
    }

    #[test]
    fn decode_varblocks_for_pass_channel_propagates_block_ctx_offset_error() {
        // The block-ctx-offset closure can error; that error should
        // bubble out of the walker.
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
        let r = decode_varblocks_for_pass_channel(
            &grid,
            &mut nz,
            0,
            0,
            13,
            |_vb| Err(Error::InvalidData("test ctx closure error".into())),
            |_| Ok(0),
            |_| Ok(0),
        );
        assert!(r.is_err());
    }

    #[test]
    fn decode_varblocks_for_pass_channel_routes_distinct_transforms() {
        // Mixed-transform grid: DCT8×16 + two DCT8×8 — verify the
        // typed driver sees the right TransformType at each
        // varblock by capturing transforms via a side-channel.
        let hf = make_hf(vec![7, 0, 0, 0, 0, 0], 3, 3);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
        let mut seen_transforms = Vec::new();
        let triples = decode_varblocks_for_pass_channel(
            &grid,
            &mut nz,
            0,
            0,
            13,
            |vb| {
                seen_transforms.push(vb.transform);
                Ok(0)
            },
            |_| Ok(0),
            |_| Ok(0),
        )
        .unwrap();
        assert_eq!(triples.len(), 3);
        assert_eq!(
            seen_transforms,
            vec![
                TransformType::Dct8x16,
                TransformType::Dct8x8,
                TransformType::Dct8x8
            ]
        );
    }
}
