//! Per-pass / per-channel `NonZeros(x, y)` grid bookkeeping —
//! ISO/IEC FDIS 18181-1:2021 §C.8.3 + Listing C.13 + Listing C.14.
//!
//! ## Scope (round 177)
//!
//! Round 177 lands the typed scaffolding that owns the
//! `NonZeros(x, y)` per-channel storage referenced by FDIS Listing
//! C.13's `PredictedNonZeros(x, y)` recurrence and updated by the
//! spec prose right after Listing C.14:
//!
//! > NonZeros(x, y) is then (non_zeros + num_blocks − 1) Idiv num_blocks.
//!
//! The round-159 [`crate::pass_group_hf::predicted_non_zeros`] takes a
//! `non_zeros_at(x, y)` closure; round 164 added a
//! [`TransformType`]-driven entry point for the per-block coefficient
//! loop. This module fills the gap between them: a typed
//! [`NonZerosGrid`] that
//!
//! 1. stores `NonZeros(x, y)` for every per-channel position in a
//!    rectangular varblock region (one cell per varblock origin),
//! 2. issues the [`crate::pass_group_hf::predicted_non_zeros`] lookup
//!    against its own storage (so the caller does not have to
//!    re-implement the four-branch recurrence each round), and
//! 3. updates the cell with the spec post-Listing-C.14 formula given
//!    a `(non_zeros, num_blocks)` pair.
//!
//! Plus a typed driver that threads
//! [`crate::pass_group_hf::read_non_zeros_and_decode_block_for_transform`]
//! through the grid — single call per varblock origin, grid
//! bookkeeping handled internally, caller only owns the two ANS
//! closures.
//!
//! ## Scope boundary
//!
//! The §C.7.2 entropy histogram array, the per-pass
//! `EntropyStream` / `HybridUintState` wiring, the per-LfGroup
//! varblock-shape grid, and the per-channel `BlockContext()` /
//! `non_zeros` history threading remain follow-up work (the
//! `decode_symbol` and `read_non_zeros` closures abstract over them
//! at the per-block level, but the grid presented here is the storage
//! they consume above the per-block primitive).
//!
//! ## §C.8.3 prose — `NonZeros(x, y)` grid
//!
//! From the FDIS §C.8.3 prose right after the per-varblock
//! `non_zeros` read:
//!
//! > The decoder then computes the NonZeros(x, y) field for each
//! > block in the current varblock as follows. [...] NonZeros(x, y)
//! > is then (non_zeros + num_blocks - 1) Idiv num_blocks.
//!
//! Note "for **each block** in the current varblock" — every 8×8
//! block covered by the varblock's footprint receives the same
//! ceiling-divided value (see
//! [`NonZerosGrid::update_after_block_for_transform`]; round-281
//! prose-conformance fix).
//!
//! `PredictedNonZeros(x, y)` from the prose right before Listing
//! C.14:
//!
//! * `(x, y) == (0, 0)` → 32
//! * `x == 0 && y != 0` → `NonZeros(x, y − 1)`
//! * `x != 0 && y == 0` → `NonZeros(x − 1, y)`
//! * otherwise         → `(NonZeros(x, y − 1) + NonZeros(x − 1, y) +
//!                         1) >> 1`
//!
//! Both are implemented here at the grid level and verified against
//! [`crate::pass_group_hf::predicted_non_zeros`] for byte-for-byte
//! agreement on every tested shape.
//!
//! ## Pure-control-flow primitive
//!
//! Same shape as round-89 [`crate::dct_quant_weights`], round-95
//! [`crate::hf_dequant`], round-121 [`crate::llf_from_lf`],
//! round-138 [`crate::chroma_from_luma`], round-141
//! [`crate::gaborish`], round-144 [`crate::epf`], round-147
//! [`crate::afv::afv_idct`], round-159 / 164
//! [`crate::pass_group_hf`]. No bit reads, no spec-derivation; the
//! per-LfGroup driver that calls into this grid from the
//! `decode_codestream` plumbing remains the follow-up.

use oxideav_core::{Error, Result};

use crate::dct_select::TransformType;
use crate::pass_group_hf::{
    predicted_non_zeros, read_non_zeros_and_decode_block_for_transform, transform_block_params,
    DecodedHfBlock,
};

/// `NonZeros(x, y)` storage for a single pass-group / channel pair.
///
/// Indexed by the **varblock origin** `(x, y)` in 8-sample units, not
/// in pixel coordinates. Each cell stores the post-Listing-C.14 value
/// `(non_zeros + num_blocks − 1) Idiv num_blocks` (per FDIS §C.8.3
/// prose) — i.e. the `NonZeros(x, y)` field that
/// `PredictedNonZeros(x, y)` reads against.
///
/// `width` and `height` are the per-channel **varblock-grid**
/// dimensions (so for a 64×64 pixel single-channel group with
/// DCT8×8 throughout, `width = height = 8`).
#[derive(Debug, Clone)]
pub struct NonZerosGrid {
    width: u32,
    height: u32,
    cells: Vec<u32>,
}

impl NonZerosGrid {
    /// Build a `width × height` grid initialised to zero.
    ///
    /// Returns [`Error::InvalidData`] if either dimension is zero —
    /// FDIS §C.8.3 has no useful interpretation for a degenerate
    /// grid, so callers must filter before constructing.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidData(format!(
                "JXL NonZerosGrid: dims must be non-zero, got ({width}, {height})"
            )));
        }
        // Defensively cap; a per-group grid is bounded by §C.8.3 +
        // group_dim (typically 256×256 px = 32×32 varblocks for the
        // smallest DCT). 65535 × 65535 is far past any conformant
        // codestream and protects against integer overflow on the
        // `width × height` allocation.
        if width > u16::MAX as u32 || height > u16::MAX as u32 {
            return Err(Error::InvalidData(format!(
                "JXL NonZerosGrid: dims exceed 65535 ({width}, {height})"
            )));
        }
        let total = (width as usize) * (height as usize);
        Ok(Self {
            width,
            height,
            cells: vec![0; total],
        })
    }

    /// Per-channel varblock-grid width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Per-channel varblock-grid height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Linear index for `(x, y)`. Returns `None` if either coordinate
    /// is out of range.
    fn index(&self, x: u32, y: u32) -> Option<usize> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some((y as usize) * (self.width as usize) + (x as usize))
    }

    /// Read the `NonZeros(x, y)` cell. Returns
    /// [`Error::InvalidData`] for out-of-range `(x, y)`.
    pub fn get(&self, x: u32, y: u32) -> Result<u32> {
        match self.index(x, y) {
            Some(i) => Ok(self.cells[i]),
            None => Err(Error::InvalidData(format!(
                "JXL NonZerosGrid::get: ({x}, {y}) out of range (w={}, h={})",
                self.width, self.height
            ))),
        }
    }

    /// Direct cell write. The natural caller is the post-Listing-C.14
    /// update — see [`NonZerosGrid::update_after_block`] for the
    /// spec-formula entry point. Provided here as the verbatim
    /// primitive so unit tests can seed arbitrary grids.
    pub fn set(&mut self, x: u32, y: u32, value: u32) -> Result<()> {
        match self.index(x, y) {
            Some(i) => {
                self.cells[i] = value;
                Ok(())
            }
            None => Err(Error::InvalidData(format!(
                "JXL NonZerosGrid::set: ({x}, {y}) out of range (w={}, h={})",
                self.width, self.height
            ))),
        }
    }

    /// `PredictedNonZeros(x, y)` per the FDIS prose right before
    /// Listing C.14. Delegates to
    /// [`predicted_non_zeros`] with `|xx, yy| self.get(xx,
    /// yy).unwrap_or(0)` so the four-branch recurrence is the single
    /// source of truth.
    ///
    /// `(x, y)` out of range returns [`Error::InvalidData`] — the
    /// per-LfGroup driver iterates in raster order, so it is a bug to
    /// query a position past the grid.
    pub fn predicted(&self, x: u32, y: u32) -> Result<u32> {
        if x >= self.width || y >= self.height {
            return Err(Error::InvalidData(format!(
                "JXL NonZerosGrid::predicted: ({x}, {y}) out of range (w={}, h={})",
                self.width, self.height
            )));
        }
        // The recurrence never asks for an `(x, y-1)` or `(x-1, y)`
        // cell outside the grid (the four `x == 0` / `y == 0` cases
        // cover the borders), but the `non_zeros_at` closure must
        // still be total — we delegate to `get(...).unwrap_or(0)`
        // which is well-defined: an out-of-range query returns 0
        // (the "uninitialised cell" sentinel), but the recurrence
        // never reads such a cell on valid `(x, y)` input anyway.
        Ok(predicted_non_zeros(x, y, |xx, yy| {
            self.get(xx, yy).unwrap_or(0)
        }))
    }

    /// Update a **single** grid cell after a per-block decode, per the
    /// FDIS §C.8.3 prose formula:
    ///
    /// > NonZeros(x, y) = (non_zeros + num_blocks − 1) Idiv num_blocks.
    ///
    /// This is the per-cell assignment primitive. The spec prose
    /// applies the formula to *each block* of the varblock's
    /// footprint — callers decoding a varblock should use
    /// [`Self::update_after_block_for_transform`], which derives the
    /// footprint from the [`TransformType`] and writes every covered
    /// cell. `update_after_block` remains useful when the caller
    /// addresses cells individually (or for `num_blocks` values with
    /// no transform attached, as in tests).
    ///
    /// `non_zeros` here is the **raw** value read from the
    /// `NonZerosContext(predicted)` ANS stream (i.e. the value the
    /// caller passed as `initial_non_zeros` to
    /// [`crate::pass_group_hf::decode_block_coefficients`]) — NOT
    /// the `remaining_non_zeros` field of [`DecodedHfBlock`].
    /// `num_blocks` is the per-[`TransformType`] block count from
    /// [`transform_block_params`].
    ///
    /// Returns [`Error::InvalidData`] for `num_blocks == 0` (the spec
    /// `Idiv 0` is undefined) or for out-of-range `(x, y)`.
    pub fn update_after_block(
        &mut self,
        x: u32,
        y: u32,
        non_zeros: u32,
        num_blocks: u32,
    ) -> Result<u32> {
        if num_blocks == 0 {
            return Err(Error::InvalidData(
                "JXL NonZerosGrid::update_after_block: num_blocks must be ≥ 1".into(),
            ));
        }
        // `Idiv` is unsigned floor-division per the FDIS notation
        // (§3.2 "Operators"). `(non_zeros + num_blocks - 1) Idiv
        // num_blocks` is the ceiling-divide identity.
        let updated = non_zeros
            .saturating_add(num_blocks - 1)
            .checked_div(num_blocks)
            .unwrap_or(0);
        self.set(x, y, updated)?;
        Ok(updated)
    }

    /// Full-varblock-footprint writeback with `num_blocks` derived
    /// from a [`TransformType`] via [`transform_block_params`].
    ///
    /// Per the FDIS §C.8.3 prose right after the `non_zeros` read:
    ///
    /// > The decoder then computes the NonZeros(x, y) field for
    /// > **each block** in the current varblock [...] NonZeros(x, y)
    /// > is then (non_zeros + num_blocks - 1) Idiv num_blocks.
    ///
    /// i.e. *every* 8×8 block covered by the varblock receives the
    /// same ceiling-divided value — not just the top-left cell.
    /// This matters whenever a neighbouring varblock's
    /// `PredictedNonZeros(x, y)` reads an `(x − 1, y)` / `(x, y − 1)`
    /// cell that is a non-top-left (continuation) cell of a larger
    /// transform: the cell must hold the varblock's value, not the
    /// zero-init sentinel. (Round 281 prose-conformance fix: rounds
    /// 177..264 wrote only the top-left cell.)
    ///
    /// `(x, y)` is the varblock's **top-left** cell; the footprint
    /// `(bcols, brows) = t.block_dims()` cells to the right/below
    /// are all written. Returns [`Error::InvalidData`] when any
    /// covered cell falls outside the grid (the placement-validated
    /// [`crate::dct_select::DctSelectGrid`] guarantees in-grid
    /// footprints, so an out-of-range write indicates caller-side
    /// grid mismatch).
    pub fn update_after_block_for_transform(
        &mut self,
        x: u32,
        y: u32,
        non_zeros: u32,
        t: TransformType,
    ) -> Result<u32> {
        let (num_blocks, _size) = transform_block_params(t);
        if num_blocks == 0 {
            return Err(Error::InvalidData(
                "JXL NonZerosGrid::update_after_block_for_transform: num_blocks must be ≥ 1".into(),
            ));
        }
        let updated = non_zeros
            .saturating_add(num_blocks - 1)
            .checked_div(num_blocks)
            .unwrap_or(0);
        let (bcols, brows) = t.block_dims();
        for j in 0..brows {
            for i in 0..bcols {
                self.set(x + i, y + j, updated)?;
            }
        }
        Ok(updated)
    }

    /// Read-only view onto the raw cell buffer (row-major, `y *
    /// width + x`). Provided for test introspection / debug output;
    /// callers should prefer [`NonZerosGrid::get`] for indexed
    /// access.
    pub fn cells(&self) -> &[u32] {
        &self.cells
    }
}

/// Per-varblock typed driver that threads
/// [`read_non_zeros_and_decode_block_for_transform`] through a
/// [`NonZerosGrid`].
///
/// Single call per varblock origin `(x, y)`:
///
/// 1. compute `predicted = grid.predicted(x, y)`,
/// 2. invoke [`read_non_zeros_and_decode_block_for_transform`] with
///    the caller-supplied `read_non_zeros` and `decode_symbol`
///    closures,
/// 3. call `grid.update_after_block_for_transform(x, y, raw_non_zeros,
///    t)`,
/// 4. return the `(DecodedHfBlock, raw_non_zeros)` pair so the caller
///    can also write the coefficients into its per-channel
///    coefficient buffer.
///
/// Caller responsibilities (still!):
/// * the `BlockContext()` value at `(x, y)` — Listing C.13 reads
///   from the LfGlobal HfBlockContext bundle which round 177 does
///   not wire,
/// * `nb_block_ctx` — same source,
/// * the two ANS closures — round 177's grid abstracts the
///   `NonZeros` storage but not the §C.7.2 histograms (#799
///   DOCS-GAP).
#[allow(clippy::too_many_arguments)]
pub fn decode_block_at<F, G>(
    grid: &mut NonZerosGrid,
    x: u32,
    y: u32,
    t: TransformType,
    block_ctx: u32,
    nb_block_ctx: u32,
    read_non_zeros: F,
    decode_symbol: G,
) -> Result<(DecodedHfBlock, u32)>
where
    F: FnMut(u32) -> Result<u32>,
    G: FnMut(u32) -> Result<u32>,
{
    let predicted = grid.predicted(x, y)?;
    let (decoded, raw_non_zeros) = read_non_zeros_and_decode_block_for_transform(
        t,
        predicted,
        block_ctx,
        nb_block_ctx,
        read_non_zeros,
        decode_symbol,
    )?;
    grid.update_after_block_for_transform(x, y, raw_non_zeros, t)?;
    Ok((decoded, raw_non_zeros))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_zero_dims() {
        assert!(NonZerosGrid::new(0, 5).is_err());
        assert!(NonZerosGrid::new(5, 0).is_err());
        assert!(NonZerosGrid::new(0, 0).is_err());
    }

    #[test]
    fn new_rejects_oversize_dims() {
        // 65535 is allowed.
        assert!(NonZerosGrid::new(65535, 1).is_ok());
        assert!(NonZerosGrid::new(1, 65535).is_ok());
        // 65536 is not.
        assert!(NonZerosGrid::new(65536, 1).is_err());
        assert!(NonZerosGrid::new(1, 65536).is_err());
    }

    #[test]
    fn new_zeroes_all_cells() {
        let g = NonZerosGrid::new(4, 3).unwrap();
        assert_eq!(g.width(), 4);
        assert_eq!(g.height(), 3);
        assert_eq!(g.cells().len(), 12);
        assert!(g.cells().iter().all(|&v| v == 0));
    }

    #[test]
    fn get_and_set_roundtrip() {
        let mut g = NonZerosGrid::new(4, 3).unwrap();
        g.set(2, 1, 42).unwrap();
        assert_eq!(g.get(2, 1).unwrap(), 42);
        // Every other cell is still 0.
        for y in 0..3 {
            for x in 0..4 {
                if (x, y) != (2, 1) {
                    assert_eq!(g.get(x, y).unwrap(), 0);
                }
            }
        }
    }

    #[test]
    fn get_oob_errors() {
        let g = NonZerosGrid::new(4, 3).unwrap();
        assert!(g.get(4, 0).is_err());
        assert!(g.get(0, 3).is_err());
        assert!(g.get(4, 3).is_err());
    }

    #[test]
    fn set_oob_errors() {
        let mut g = NonZerosGrid::new(4, 3).unwrap();
        assert!(g.set(4, 0, 1).is_err());
        assert!(g.set(0, 3, 1).is_err());
    }

    #[test]
    fn predicted_origin_is_32() {
        // PredictedNonZeros(0, 0) = 32 (Listing C.13 prelude).
        let g = NonZerosGrid::new(4, 3).unwrap();
        assert_eq!(g.predicted(0, 0).unwrap(), 32);
    }

    #[test]
    fn predicted_top_row_reads_left_neighbour() {
        // y == 0 && x != 0 → NonZeros(x - 1, 0).
        let mut g = NonZerosGrid::new(4, 3).unwrap();
        g.set(0, 0, 7).unwrap();
        g.set(1, 0, 11).unwrap();
        g.set(2, 0, 13).unwrap();
        assert_eq!(g.predicted(1, 0).unwrap(), 7);
        assert_eq!(g.predicted(2, 0).unwrap(), 11);
        assert_eq!(g.predicted(3, 0).unwrap(), 13);
    }

    #[test]
    fn predicted_left_col_reads_above_neighbour() {
        // x == 0 && y != 0 → NonZeros(0, y - 1).
        let mut g = NonZerosGrid::new(4, 3).unwrap();
        g.set(0, 0, 5).unwrap();
        g.set(0, 1, 8).unwrap();
        assert_eq!(g.predicted(0, 1).unwrap(), 5);
        assert_eq!(g.predicted(0, 2).unwrap(), 8);
    }

    #[test]
    fn predicted_interior_averages_above_and_left_rounded_up() {
        // (x != 0 && y != 0) → (NonZeros(x, y - 1) +
        //                       NonZeros(x - 1, y) + 1) >> 1.
        let mut g = NonZerosGrid::new(4, 3).unwrap();
        g.set(1, 0, 10).unwrap(); // above
        g.set(0, 1, 3).unwrap(); // left
                                 // (10 + 3 + 1) >> 1 = 7
        assert_eq!(g.predicted(1, 1).unwrap(), 7);

        // Odd-sum exercise: above = 5, left = 4 → (5 + 4 + 1) >> 1 = 5.
        let mut g2 = NonZerosGrid::new(4, 3).unwrap();
        g2.set(2, 0, 5).unwrap();
        g2.set(1, 1, 4).unwrap();
        assert_eq!(g2.predicted(2, 1).unwrap(), 5);
    }

    #[test]
    fn predicted_oob_errors() {
        let g = NonZerosGrid::new(4, 3).unwrap();
        assert!(g.predicted(4, 0).is_err());
        assert!(g.predicted(0, 3).is_err());
    }

    #[test]
    fn predicted_agrees_with_pass_group_hf_helper() {
        // PredictedNonZeros via the grid must match the round-159
        // [`crate::pass_group_hf::predicted_non_zeros`] applied to
        // the same backing storage. Seed a 3×3 grid with arbitrary
        // values and check every position.
        let mut g = NonZerosGrid::new(3, 3).unwrap();
        let seed: [[u32; 3]; 3] = [[10, 4, 9], [7, 2, 6], [3, 1, 5]];
        for y in 0..3 {
            for x in 0..3 {
                g.set(x, y, seed[y as usize][x as usize]).unwrap();
            }
        }
        for y in 0..3 {
            for x in 0..3 {
                let via_grid = g.predicted(x, y).unwrap();
                let via_helper = predicted_non_zeros(x, y, |xx, yy| seed[yy as usize][xx as usize]);
                assert_eq!(via_grid, via_helper, "mismatch at ({x}, {y})");
            }
        }
    }

    #[test]
    fn update_after_block_rejects_zero_num_blocks() {
        let mut g = NonZerosGrid::new(2, 2).unwrap();
        assert!(g.update_after_block(0, 0, 5, 0).is_err());
    }

    #[test]
    fn update_after_block_oob_errors() {
        let mut g = NonZerosGrid::new(2, 2).unwrap();
        assert!(g.update_after_block(2, 0, 5, 1).is_err());
        assert!(g.update_after_block(0, 2, 5, 1).is_err());
    }

    #[test]
    fn update_after_block_dct8x8_is_identity() {
        // num_blocks == 1 → (non_zeros + 0) Idiv 1 = non_zeros.
        let mut g = NonZerosGrid::new(2, 2).unwrap();
        let updated = g.update_after_block(0, 0, 17, 1).unwrap();
        assert_eq!(updated, 17);
        assert_eq!(g.get(0, 0).unwrap(), 17);
    }

    #[test]
    fn update_after_block_dct16x16_ceil_div_4() {
        // num_blocks == 4 (DCT16×16). The formula is ceil(nz / 4).
        // nz = 0  → 0
        // nz = 1  → 1
        // nz = 3  → 1
        // nz = 4  → 1
        // nz = 5  → 2
        // nz = 16 → 4
        let mut g = NonZerosGrid::new(2, 2).unwrap();
        for (nz, want) in [(0, 0), (1, 1), (3, 1), (4, 1), (5, 2), (16, 4)] {
            let v = g.update_after_block(0, 0, nz, 4).unwrap();
            assert_eq!(v, want, "ceil({nz} / 4) != {want}");
            assert_eq!(g.get(0, 0).unwrap(), want);
        }
    }

    #[test]
    fn update_after_block_dct32x32_ceil_div_16() {
        // num_blocks == 16 (DCT32×32). ceil(nz / 16).
        let mut g = NonZerosGrid::new(1, 1).unwrap();
        for (nz, want) in [(0, 0), (1, 1), (15, 1), (16, 1), (17, 2), (1024, 64)] {
            let v = g.update_after_block(0, 0, nz, 16).unwrap();
            assert_eq!(v, want, "ceil({nz} / 16) != {want}");
        }
    }

    #[test]
    fn update_after_block_for_transform_dispatches_via_table() {
        // DCT8×8 → num_blocks = 1, DCT16×16 → num_blocks = 4,
        // DCT32×32 → num_blocks = 16. Verify via the
        // TransformType-driven entry point that all three reduce to
        // the same ceil-division formula, and that the value is
        // written to *every* covered cell per the §C.8.3 "for each
        // block in the current varblock" prose (grid sized for the
        // largest 4×4-cell footprint).
        let mut g = NonZerosGrid::new(4, 4).unwrap();
        let nz = 17;
        let v = g
            .update_after_block_for_transform(0, 0, nz, TransformType::Dct8x8)
            .unwrap();
        assert_eq!(v, 17, "DCT8x8 num_blocks=1: ceil(17/1)=17");
        assert_eq!(g.get(0, 0).unwrap(), 17);
        assert_eq!(g.get(1, 0).unwrap(), 0, "DCT8x8 footprint is 1×1");
        let v = g
            .update_after_block_for_transform(0, 0, nz, TransformType::Dct16x16)
            .unwrap();
        assert_eq!(v, 5, "DCT16x16 num_blocks=4: ceil(17/4)=5");
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            assert_eq!(g.get(x, y).unwrap(), 5, "DCT16x16 cell ({x},{y})");
        }
        assert_eq!(g.get(2, 0).unwrap(), 0, "outside the 2×2 footprint");
        let v = g
            .update_after_block_for_transform(0, 0, nz, TransformType::Dct32x32)
            .unwrap();
        assert_eq!(v, 2, "DCT32x32 num_blocks=16: ceil(17/16)=2");
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(g.get(x, y).unwrap(), 2, "DCT32x32 cell ({x},{y})");
            }
        }
    }

    #[test]
    fn update_after_block_for_transform_footprint_spilling_grid_errors() {
        // A DCT16×16 footprint (2×2 cells) anchored at the last
        // column of a 2×2 grid spills outside → clean error, no
        // panic.
        let mut g = NonZerosGrid::new(2, 2).unwrap();
        let r = g.update_after_block_for_transform(1, 0, 4, TransformType::Dct16x16);
        assert!(r.is_err());
    }

    #[test]
    fn update_after_block_for_transform_rectangular_footprint() {
        // Per Table C.16 / `TransformType::block_dims()` (cols, rows):
        // DCT16×8 (16 rows × 8 cols) covers 1 col × 2 rows; DCT8×16
        // (8 rows × 16 cols) covers 2 cols × 1 row. Verify the
        // rectangular footprints land on the right cells.
        let mut g = NonZerosGrid::new(2, 2).unwrap();
        g.update_after_block_for_transform(0, 0, 3, TransformType::Dct16x8)
            .unwrap();
        assert_eq!(g.get(0, 0).unwrap(), 2, "ceil(3/2) = 2");
        assert_eq!(g.get(0, 1).unwrap(), 2, "second row of 1×2-cell footprint");
        assert_eq!(g.get(1, 0).unwrap(), 0, "column 1 untouched by DCT16x8");
        let mut g = NonZerosGrid::new(2, 2).unwrap();
        g.update_after_block_for_transform(0, 1, 3, TransformType::Dct8x16)
            .unwrap();
        assert_eq!(g.get(0, 1).unwrap(), 2);
        assert_eq!(
            g.get(1, 1).unwrap(),
            2,
            "second column of 2×1-cell footprint"
        );
        assert_eq!(g.get(0, 0).unwrap(), 0, "row 0 untouched by DCT8x16");
    }

    #[test]
    fn predicted_after_update_chains_correctly() {
        // Seed (0, 0) via the spec formula at DCT8×8 (num_blocks = 1)
        // and verify the next-position predict reads it back.
        let mut g = NonZerosGrid::new(3, 1).unwrap();
        // Origin: PredictedNonZeros(0, 0) = 32, regardless of cell.
        assert_eq!(g.predicted(0, 0).unwrap(), 32);
        // After decoding 13 non-zeros at (0, 0) with DCT8×8:
        // NonZeros(0, 0) = ceil(13 / 1) = 13.
        g.update_after_block_for_transform(0, 0, 13, TransformType::Dct8x8)
            .unwrap();
        assert_eq!(g.get(0, 0).unwrap(), 13);
        // PredictedNonZeros(1, 0) reads NonZeros(0, 0) = 13.
        assert_eq!(g.predicted(1, 0).unwrap(), 13);
    }

    #[test]
    fn decode_block_at_dct8x8_chains_grid_state() {
        // End-to-end smoke through `decode_block_at` at DCT8×8 with a
        // hand-rolled `read_non_zeros` (returns a constant per call)
        // and `decode_symbol` (returns zeros, so the inner block loop
        // walks the natural-order tail with empty coefficients).
        //
        // We decode 2 varblocks at (0, 0) and (1, 0). After (0, 0) the
        // grid cell stores ceil(non_zeros / 1) = the same non_zeros
        // value; (1, 0)'s prediction reads it back.
        let mut g = NonZerosGrid::new(2, 1).unwrap();

        // First call: non_zeros = 3 → after decode, NonZeros(0, 0) = 3.
        // The closure-side state is verified by recording the
        // contexts seen by `read_non_zeros`.
        let mut non_zeros_ctxs_seen: Vec<u32> = Vec::new();
        let read_non_zeros = |ctx: u32| -> Result<u32> {
            non_zeros_ctxs_seen.push(ctx);
            Ok(3u32)
        };
        let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let (_decoded, raw_non_zeros) = decode_block_at(
            &mut g,
            0,
            0,
            TransformType::Dct8x8,
            /* block_ctx = */ 0,
            /* nb_block_ctx = */ 1,
            read_non_zeros,
            decode_symbol,
        )
        .unwrap();
        assert_eq!(raw_non_zeros, 3);
        assert_eq!(g.get(0, 0).unwrap(), 3, "ceil(3/1) = 3");

        // Second call at (1, 0): predicted = NonZeros(0, 0) = 3.
        // Verify the `read_non_zeros` context closure sees the
        // [`crate::pass_group_hf::non_zeros_context`] value for
        // predicted = 3 (the `predicted < 8` branch returns
        // block_ctx + nb_block_ctx × predicted = 0 + 1 × 3 = 3).
        let mut second_ctxs: Vec<u32> = Vec::new();
        let read_non_zeros_2 = |ctx: u32| -> Result<u32> {
            second_ctxs.push(ctx);
            Ok(5u32)
        };
        let decode_symbol_2 = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let (_decoded, raw_non_zeros_2) = decode_block_at(
            &mut g,
            1,
            0,
            TransformType::Dct8x8,
            0,
            1,
            read_non_zeros_2,
            decode_symbol_2,
        )
        .unwrap();
        assert_eq!(raw_non_zeros_2, 5);
        assert_eq!(g.get(1, 0).unwrap(), 5, "ceil(5/1) = 5");
        // And the closure saw the predicted-derived context value.
        assert_eq!(
            second_ctxs,
            vec![3],
            "non_zeros context for predicted = 3 must be {}",
            3
        );
    }

    #[test]
    fn decode_block_at_dct16x16_ceil_divides_num_blocks() {
        // DCT16×16 has num_blocks = 4. After decoding with raw
        // non_zeros = 17, the post-Listing-C.14 grid cells must store
        // ceil(17 / 4) = 5, not 17 — on every cell of the 2×2
        // footprint per the §C.8.3 "for each block in the current
        // varblock" prose. Verifies the typed driver routes the
        // TransformType-derived num_blocks through the grid update —
        // not the identity-by-default DCT8×8 path.
        let mut g = NonZerosGrid::new(2, 2).unwrap();
        let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(17u32) };
        let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let (_decoded, raw_non_zeros) = decode_block_at(
            &mut g,
            0,
            0,
            TransformType::Dct16x16,
            0,
            1,
            read_non_zeros,
            decode_symbol,
        )
        .unwrap();
        assert_eq!(raw_non_zeros, 17);
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            assert_eq!(g.get(x, y).unwrap(), 5, "ceil(17/4) = 5 at ({x},{y})");
        }
    }

    #[test]
    fn decode_block_at_oob_errors() {
        let mut g = NonZerosGrid::new(2, 2).unwrap();
        let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let r = decode_block_at(
            &mut g,
            2, // out of range
            0,
            TransformType::Dct8x8,
            0,
            1,
            read_non_zeros,
            decode_symbol,
        );
        assert!(r.is_err());
    }

    #[test]
    fn update_does_not_overflow_at_u32_max() {
        // Spec formula uses `Idiv`; we use `saturating_add` so
        // pathological `non_zeros = u32::MAX, num_blocks = 1` does
        // not panic. The result is u32::MAX (saturated) Idiv 1 =
        // u32::MAX — well past any conformant codestream but the
        // primitive must not panic.
        let mut g = NonZerosGrid::new(1, 1).unwrap();
        let v = g.update_after_block(0, 0, u32::MAX, 1).unwrap();
        assert_eq!(v, u32::MAX);
    }
}
