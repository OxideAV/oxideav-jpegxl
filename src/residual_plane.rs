//! Per-LfGroup VarDCT residual-plane assembly — places each varblock's
//! `R × C` spatial residual block (the output of the Annex F.3 dequant +
//! Annex I.2.3 inverse-DCT stage) into a single-channel spatial plane at
//! the varblock's pixel origin.
//!
//! ## Scope (round 306)
//!
//! [`crate::block_dequant::decode_block_to_residual`] (rounds 286 / 293 /
//! 300) is the per-block decode walk: given a decoded quantised block, a
//! [`TransformType`], a channel, and the per-varblock `HfMul`, it
//! dequantises every coefficient (F.3) and runs the inverse transform
//! (I.2.3.2) to produce the block's spatial residual samples — an
//! `R × C` row-major buffer where `(R, C)` is the transform's pixel
//! shape. That primitive is now complete for **every** [`TransformType`].
//!
//! What was still missing — and what this module adds — is the **spatial
//! placement** stage directly above it: a single channel's varblocks are
//! laid out across the LfGroup by the [`DctSelectGrid`] (one top-left
//! cell per varblock, in 8×8-block grid units), and each varblock's
//! decoded residual block must be written into the channel's spatial
//! plane at the pixel origin `(bx * 8, by * 8)` of its top-left cell.
//!
//! This module is the pure-geometry composition layer between
//! [`crate::block_dequant`] (which produces one residual block) and the
//! restoration filters — chroma-from-luma (Annex G), Gaborish (Annex
//! J.2), and EPF (Annex J.3) — which all consume an **assembled
//! per-channel plane**.
//!
//! ## Plane geometry (padded block grid)
//!
//! The [`DctSelectGrid`] has `width_blocks × height_blocks` cells where
//! `width_blocks = ceil(lf_w / 8)` and `height_blocks = ceil(lf_h / 8)`
//! (per §C.5.4). A varblock whose footprint reaches the right or bottom
//! edge of the grid therefore covers pixels past the LfGroup's actual
//! `lf_w × lf_h` extent. The decoder reconstructs into a buffer sized to
//! the **padded block grid** and crops to `lf_w × lf_h` as a final step.
//!
//! To keep this primitive free of any crop ambiguity, the assembled
//! plane is exactly the padded block grid:
//!
//! * `plane_width  = width_blocks  * 8`
//! * `plane_height = height_blocks * 8`
//!
//! Every covered varblock's residual block fits inside this plane by
//! construction (`derive_dct_select` already rejects a varblock whose
//! footprint spills past the grid), so placement is total and
//! unconditional — no per-edge clamping. The caller crops the assembled
//! plane to `lf_w × lf_h` afterwards.
//!
//! ## Block / pixel geometry invariant
//!
//! For a varblock with transform `t`:
//!
//! * [`crate::dct_select::TransformType::block_dims`] gives the footprint
//!   `(bcols, brows)` in 8×8-cell units;
//! * [`crate::idct::dct_pixel_dims`] (DCT family) or
//!   [`crate::idct::non_dct_pixel_dims`] (non-DCT family) gives the pixel
//!   shape `(R, C) = (rows, cols)`.
//!
//! These are consistent: `C == bcols * 8` and `R == brows * 8` for every
//! [`TransformType`]. The residual block produced by
//! [`crate::block_dequant::decode_block_to_residual`] is `R × C`
//! row-major (`block[r * C + c]`), and it is written to plane pixel
//! `(px = bx * 8 + c, py = by * 8 + r)` — i.e.
//! `plane[py * plane_width + px] = block[r * C + c]`.
//!
//! ## Not in scope
//!
//! The residual produced by the IDCT already carries the LLF (DC
//! subband) contribution at its prefix cells — the LLF is loaded into
//! the top-left coefficients of the block *before* the inverse transform
//! (§I.2.5 + the per-block decode walk), so no separate DC add happens
//! at placement time. Chroma-from-luma, Gaborish, and EPF run on the
//! assembled plane and remain caller-side concerns above this primitive.
//! This module performs **no** bit reads, **no** spec re-derivation, and
//! **no** histogram materialisation.

use crate::block_dequant::covered_grid_dims;
use crate::dct_select::{DctSelectGrid, TransformType};
use crate::idct::{dct_pixel_dims, non_dct_pixel_dims};
use crate::varblock_walk::{Varblock, VarblockWalk};
use oxideav_core::{Error, Result};

/// A single-channel spatial residual plane sized to the padded block
/// grid of a [`DctSelectGrid`].
///
/// `samples` is row-major `width × height` `f32`; `samples[y * width +
/// x]` is the residual at pixel `(x, y)`. Both dimensions are multiples
/// of 8 (`width = width_blocks * 8`, `height = height_blocks * 8`).
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualPlane {
    /// Plane width in pixels (`= grid.width_blocks * 8`).
    pub width: usize,
    /// Plane height in pixels (`= grid.height_blocks * 8`).
    pub height: usize,
    /// Row-major residual samples, length `width * height`.
    pub samples: Vec<f32>,
}

impl ResidualPlane {
    /// Allocate a zero-filled plane sized to the padded block grid of
    /// `grid`.
    ///
    /// Errors with [`Error::InvalidData`] if `width_blocks * height_blocks
    /// * 64` overflows `usize`.
    pub fn for_grid(grid: &DctSelectGrid) -> Result<Self> {
        let width = (grid.width_blocks as usize)
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidData("JXL residual plane: width overflow".into()))?;
        let height = (grid.height_blocks as usize)
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidData("JXL residual plane: height overflow".into()))?;
        let n = width
            .checked_mul(height)
            .ok_or_else(|| Error::InvalidData("JXL residual plane: area overflow".into()))?;
        Ok(Self {
            width,
            height,
            samples: vec![0.0f32; n],
        })
    }

    /// Read the residual at pixel `(x, y)`, or `None` if out of range.
    pub fn get(&self, x: usize, y: usize) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.samples[y * self.width + x])
    }
}

/// Pixel shape `(R, C) = (rows, cols)` of a transform's residual block,
/// from either the plain-DCT or the non-DCT pixel-dims table.
///
/// Returns `None` only if a future [`TransformType`] lacks a pixel-dims
/// mapping in both tables (no current variant does).
pub fn block_pixel_dims(t: TransformType) -> Option<(usize, usize)> {
    dct_pixel_dims(t).or_else(|| non_dct_pixel_dims(t))
}

/// Write one `R × C` row-major residual block into `plane` at the pixel
/// origin of varblock `vb` (`px = vb.x * 8`, `py = vb.y * 8`).
///
/// `block` must have length `R * C` where `(R, C) = block_pixel_dims(t)`
/// for `vb.transform`. The block is copied verbatim:
/// `plane[(py + r) * width + (px + c)] = block[r * C + c]`.
///
/// Errors:
/// * [`Error::Unsupported`] if `vb.transform` has no pixel-dims mapping.
/// * [`Error::InvalidData`] if `block.len() != R * C`, or if the block's
///   footprint at `(px, py)` would extend past the plane bounds (a
///   malformed grid / plane mismatch — a grid from `derive_dct_select`
///   plus a plane from [`ResidualPlane::for_grid`] never trips this).
pub fn place_block(plane: &mut ResidualPlane, vb: &Varblock, block: &[f32]) -> Result<()> {
    let (rows, cols) = block_pixel_dims(vb.transform).ok_or_else(|| {
        Error::Unsupported(format!(
            "JXL residual plane: {:?} has no pixel-dims mapping",
            vb.transform
        ))
    })?;
    if block.len() != rows * cols {
        return Err(Error::InvalidData(format!(
            "JXL residual plane: block length {} != R * C ({rows} * {cols} = {}) for {:?}",
            block.len(),
            rows * cols,
            vb.transform
        )));
    }
    let px = (vb.x as usize)
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidData("JXL residual plane: px overflow".into()))?;
    let py = (vb.y as usize)
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidData("JXL residual plane: py overflow".into()))?;
    if px + cols > plane.width || py + rows > plane.height {
        return Err(Error::InvalidData(format!(
            "JXL residual plane: {:?} block at pixel ({px},{py}) covering {cols}×{rows} \
             spills past plane {}×{}",
            vb.transform, plane.width, plane.height
        )));
    }
    let pw = plane.width;
    for r in 0..rows {
        let dst = (py + r) * pw + px;
        let src = r * cols;
        plane.samples[dst..dst + cols].copy_from_slice(&block[src..src + cols]);
    }
    Ok(())
}

/// Assemble a full single-channel residual plane by walking `grid` in
/// raster order and invoking `residual_at(&vb)` once per varblock to
/// obtain each block's `R × C` spatial residual samples (the output of
/// [`crate::block_dequant::decode_block_to_residual`]), placing each into
/// the returned [`ResidualPlane`].
///
/// The closure owns the per-varblock decode (coefficient lookup, F.3
/// dequant, IDCT); this driver owns only the grid walk and the spatial
/// placement geometry. Closure errors propagate verbatim without writing
/// a partial block. The walk order is the canonical §C.8.3 raster order
/// (top-left cells row-major; continuation cells skipped).
///
/// Defensive: rejects any varblock whose transform lacks a pixel-dims
/// mapping or whose residual length / footprint is inconsistent with the
/// plane (via [`place_block`]).
pub fn assemble_channel_plane<F>(grid: &DctSelectGrid, mut residual_at: F) -> Result<ResidualPlane>
where
    F: FnMut(&Varblock) -> Result<Vec<f32>>,
{
    // Validate every covered varblock has a grid-dims mapping up front so
    // a malformed transform surfaces before any allocation work.
    let mut plane = ResidualPlane::for_grid(grid)?;
    let mut walk = VarblockWalk::new(grid);
    while let Some(vb) = walk.next()? {
        // Defence-in-depth: confirm the transform is one this stack can
        // place (covered_grid_dims == has a pixel-dims mapping).
        if covered_grid_dims(vb.transform).is_none() {
            return Err(Error::Unsupported(format!(
                "JXL residual plane: {:?} is not a covered transform",
                vb.transform
            )));
        }
        let block = residual_at(&vb)?;
        place_block(&mut plane, &vb, &block)?;
    }
    Ok(plane)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct_select::{DctSelectCell, DctSelectGrid};

    /// Build a grid directly from a cell list + hf_mul + dims (bypassing
    /// `derive_dct_select` so tests pin placement geometry independently
    /// of BlockInfo parsing).
    fn grid(cells: Vec<DctSelectCell>, hf_mul: Vec<i32>, w: u32, h: u32) -> DctSelectGrid {
        DctSelectGrid {
            cells,
            hf_mul,
            width_blocks: w,
            height_blocks: h,
        }
    }

    /// A 1×1 grid holding a single DCT8×8 varblock.
    fn single_dct8x8() -> DctSelectGrid {
        grid(
            vec![DctSelectCell::TopLeft(TransformType::Dct8x8)],
            vec![1],
            1,
            1,
        )
    }

    fn vb(x: u32, y: u32, t: TransformType) -> Varblock {
        Varblock {
            x,
            y,
            transform: t,
            hf_mul: 1,
        }
    }

    #[test]
    fn block_pixel_dims_covers_every_transform() {
        use TransformType as T;
        // DCT family from dct_pixel_dims.
        assert_eq!(block_pixel_dims(T::Dct8x8), Some((8, 8)));
        assert_eq!(block_pixel_dims(T::Dct16x8), Some((16, 8)));
        assert_eq!(block_pixel_dims(T::Dct8x16), Some((8, 16)));
        assert_eq!(block_pixel_dims(T::Dct256x256), Some((256, 256)));
        // Non-DCT family — all 8×8.
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
            assert_eq!(block_pixel_dims(t), Some((8, 8)), "{t:?}");
        }
    }

    #[test]
    fn pixel_dims_match_block_dims_times_8_for_every_transform() {
        // The placement geometry invariant: C == bcols*8, R == brows*8.
        use TransformType as T;
        for t in [
            T::Dct8x8,
            T::Dct16x16,
            T::Dct32x32,
            T::Dct16x8,
            T::Dct8x16,
            T::Dct32x8,
            T::Dct8x32,
            T::Dct32x16,
            T::Dct16x32,
            T::Dct64x64,
            T::Dct64x32,
            T::Dct32x64,
            T::Dct128x128,
            T::Dct128x64,
            T::Dct64x128,
            T::Dct256x256,
            T::Dct256x128,
            T::Dct128x256,
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
            let (rows, cols) = block_pixel_dims(t).unwrap();
            let (bcols, brows) = t.block_dims();
            assert_eq!(cols, bcols as usize * 8, "{t:?} cols vs block_dims");
            assert_eq!(rows, brows as usize * 8, "{t:?} rows vs block_dims");
        }
    }

    #[test]
    fn for_grid_sizes_to_padded_block_grid() {
        // 3 cols × 2 rows of cells → 24 × 16 pixel plane.
        let g = grid(
            vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 6],
            vec![1; 6],
            3,
            2,
        );
        let p = ResidualPlane::for_grid(&g).unwrap();
        assert_eq!(p.width, 24);
        assert_eq!(p.height, 16);
        assert_eq!(p.samples.len(), 24 * 16);
        assert!(p.samples.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn place_single_dct8x8_at_origin() {
        let g = single_dct8x8();
        let mut p = ResidualPlane::for_grid(&g).unwrap();
        // A block whose value equals its raster index.
        let block: Vec<f32> = (0..64).map(|i| i as f32).collect();
        place_block(&mut p, &vb(0, 0, TransformType::Dct8x8), &block).unwrap();
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(p.get(c, r), Some((r * 8 + c) as f32), "({c},{r})");
            }
        }
    }

    #[test]
    fn place_block_at_offset_origin() {
        // 3×3 cell grid; place a DCT8×8 block at grid cell (2, 1) → pixel
        // origin (16, 8).
        let g = grid(
            vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 9],
            vec![1; 9],
            3,
            3,
        );
        let mut p = ResidualPlane::for_grid(&g).unwrap();
        let block = vec![7.0f32; 64];
        place_block(&mut p, &vb(2, 1, TransformType::Dct8x8), &block).unwrap();
        // The 8×8 region [16..24) × [8..16) is 7.0; everything else 0.
        for y in 0..p.height {
            for x in 0..p.width {
                let inside = (16..24).contains(&x) && (8..16).contains(&y);
                let want = if inside { 7.0 } else { 0.0 };
                assert_eq!(p.get(x, y), Some(want), "({x},{y})");
            }
        }
    }

    #[test]
    fn place_rectangular_dct16x8_writes_tall_block() {
        // DCT16×8 = 16 rows × 8 cols = 8px wide × 16px tall; block_dims
        // (1, 2) = 1 col × 2 rows of cells. Grid must be at least 1×2.
        let g = grid(
            vec![
                DctSelectCell::TopLeft(TransformType::Dct16x8),
                DctSelectCell::Continuation,
            ],
            vec![1, 0],
            1,
            2,
        );
        let mut p = ResidualPlane::for_grid(&g).unwrap();
        assert_eq!((p.width, p.height), (8, 16));
        // R×C = 16×8 row-major block: value = r (the row).
        let (rows, cols) = block_pixel_dims(TransformType::Dct16x8).unwrap();
        assert_eq!((rows, cols), (16, 8));
        let block: Vec<f32> = (0..rows * cols).map(|i| (i / cols) as f32).collect();
        place_block(&mut p, &vb(0, 0, TransformType::Dct16x8), &block).unwrap();
        for y in 0..16 {
            for x in 0..8 {
                assert_eq!(p.get(x, y), Some(y as f32), "({x},{y})");
            }
        }
    }

    #[test]
    fn place_rejects_wrong_block_length() {
        let g = single_dct8x8();
        let mut p = ResidualPlane::for_grid(&g).unwrap();
        let err =
            place_block(&mut p, &vb(0, 0, TransformType::Dct8x8), &vec![0.0; 63]).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn place_rejects_footprint_spill() {
        // A 1×1 plane (8×8 px) cannot hold a DCT16×16 (16×16 px) block.
        let g = single_dct8x8();
        let mut p = ResidualPlane::for_grid(&g).unwrap();
        let block = vec![0.0f32; 256];
        let err = place_block(&mut p, &vb(0, 0, TransformType::Dct16x16), &block).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn assemble_uniform_dct8x8_grid_in_raster_order() {
        // 2×2 grid of DCT8×8; the closure returns a constant block whose
        // value encodes the varblock raster index, proving placement
        // order + position.
        let cells = vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 4];
        let g = grid(cells, vec![1; 4], 2, 2);
        let mut idx = 0u32;
        let p = assemble_channel_plane(&g, |v| {
            // Raster order must be (0,0), (1,0), (0,1), (1,1).
            let expect = v.y * 2 + v.x;
            assert_eq!(expect, idx, "raster order at vb ({},{})", v.x, v.y);
            idx += 1;
            Ok(vec![expect as f32; 64])
        })
        .unwrap();
        assert_eq!(idx, 4, "exactly 4 varblocks visited");
        assert_eq!((p.width, p.height), (16, 16));
        // Quadrant values: top-left=0, top-right=1, bottom-left=2,
        // bottom-right=3.
        assert_eq!(p.get(0, 0), Some(0.0));
        assert_eq!(p.get(8, 0), Some(1.0));
        assert_eq!(p.get(0, 8), Some(2.0));
        assert_eq!(p.get(8, 8), Some(3.0));
    }

    #[test]
    fn assemble_mixed_transform_grid() {
        // A DCT16×16 (2×2 cells) at (0,0) plus a DCT8×8 at (2,0) and a
        // DCT8×8 at (3,0) on a 4×2 grid; remaining cells DCT8×8.
        // Layout (cells, row-major, w=4 h=2):
        //   row0: TL(16) Cont    TL(8)  TL(8)
        //   row1: Cont   Cont    TL(8)  TL(8)
        let cells = vec![
            DctSelectCell::TopLeft(TransformType::Dct16x16),
            DctSelectCell::Continuation,
            DctSelectCell::TopLeft(TransformType::Dct8x8),
            DctSelectCell::TopLeft(TransformType::Dct8x8),
            DctSelectCell::Continuation,
            DctSelectCell::Continuation,
            DctSelectCell::TopLeft(TransformType::Dct8x8),
            DctSelectCell::TopLeft(TransformType::Dct8x8),
        ];
        let g = grid(cells, vec![1, 0, 1, 1, 0, 0, 1, 1], 4, 2);
        let p = assemble_channel_plane(&g, |v| {
            let (rows, cols) = block_pixel_dims(v.transform).unwrap();
            // Distinct constant per transform: 16×16 → 16, 8×8 → 8.
            Ok(vec![rows as f32; rows * cols])
        })
        .unwrap();
        assert_eq!((p.width, p.height), (32, 16));
        // The DCT16×16 fills pixels [0,16)×[0,16) with 16.0.
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(p.get(x, y), Some(16.0), "DCT16 ({x},{y})");
            }
        }
        // The DCT8×8 at grid (2,0) fills [16,24)×[0,8) with 8.0.
        for y in 0..8 {
            for x in 16..24 {
                assert_eq!(p.get(x, y), Some(8.0), "DCT8 ({x},{y})");
            }
        }
        // The DCT8×8 at grid (2,1) fills [16,24)×[8,16) with 8.0.
        assert_eq!(p.get(16, 8), Some(8.0));
    }

    #[test]
    fn assemble_propagates_closure_error() {
        let g = single_dct8x8();
        let err = assemble_channel_plane(&g, |_| {
            Err(Error::InvalidData("boom".into())) as Result<Vec<f32>>
        })
        .unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn assemble_empty_grid_is_empty_plane() {
        let g = grid(vec![], vec![], 0, 0);
        let p = assemble_channel_plane(&g, |_| unreachable!("no varblocks")).unwrap();
        assert_eq!((p.width, p.height), (0, 0));
        assert!(p.samples.is_empty());
    }

    #[test]
    fn assemble_rejects_residual_empty_cell() {
        // A grid with an Empty cell (malformed) errors during the walk.
        let g = grid(vec![DctSelectCell::Empty], vec![0], 1, 1);
        let err = assemble_channel_plane(&g, |_| Ok(vec![0.0; 64])).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn get_out_of_range_is_none() {
        let g = single_dct8x8();
        let p = ResidualPlane::for_grid(&g).unwrap();
        assert_eq!(p.get(8, 0), None);
        assert_eq!(p.get(0, 8), None);
        assert!(p.get(7, 7).is_some());
    }
}
