//! Per-group sub-rectangle views of the per-LfGroup VarDCT state —
//! ISO/IEC FDIS 18181-1:2021 §C.8.1 group geometry.
//!
//! ## Scope (round 389)
//!
//! A VarDCT frame with `num_groups > 1` carries one PassGroup section
//! per `(pass, group)` pair (§C.3.1). Per §C.8.1, *"width and height
//! refers to the size of the current group (at most kGroupDim ×
//! kGroupDim), and all coordinates are relative to the top-left corner
//! of the group"* — so the §C.8.3 varblock raster walk, the
//! `PredictedNonZeros(x, y)` neighbour lookups (which reset at the
//! group's top-left: `x == y == 0 → 32`), and the NonZeros grids are
//! all **group-local**. The per-LfGroup decode drivers built in rounds
//! 208–385 walk a whole [`DctSelectGrid`]; this module supplies the
//! group-rect views that let those drivers run unchanged, one group at
//! a time:
//!
//! * [`group_rects_in_blocks`] — the raster-order group grid (§C.3.1
//!   "the groups (in raster order)"), in 8×8-block units.
//! * [`slice_dct_select_rect`] — a group's sub-[`DctSelectGrid`],
//!   validating the §C.5.4 invariant that no varblock crosses a group
//!   boundary (*"covers a rectangle that does not cross group
//!   boundaries"*).
//! * [`slice_lf_rect`] — the group's rectangle of the dequantised LF
//!   image (one LF sample per 8×8 block), feeding the Listing I.16
//!   LLF seed of the group's varblocks.
//! * [`slice_tiles_rect`] — the group's rectangle of a per-64×64-tile
//!   channel (`XFromY` / `BFromY` from HfMetadata); group boundaries
//!   (multiples of 256 px = 32 blocks) are always tile-aligned
//!   (tiles are 8 blocks), so the slice is exact.
//!
//! The sliced views keep all varblock coordinates group-relative,
//! which is exactly the coordinate system §C.8.1 prescribes for the
//! entropy decode; the caller pastes the reconstructed group planes
//! back at the group's pixel offset.

use oxideav_core::{Error, Result};

use crate::dct_select::DctSelectGrid;
use crate::lf_dequant::LfDequantOutput;

/// One group's rectangle within an LfGroup, in 8×8-block units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupRect {
    /// Group index in §C.3.1 raster order.
    pub index: u32,
    /// Left edge, in blocks, relative to the LfGroup's block grid.
    pub bx0: u32,
    /// Top edge, in blocks.
    pub by0: u32,
    /// Width in blocks (≤ group_dim / 8; smaller on the right edge).
    pub width_blocks: u32,
    /// Height in blocks (≤ group_dim / 8; smaller on the bottom edge).
    pub height_blocks: u32,
}

/// The §C.3.1 raster-order group grid of a `width_px × height_px`
/// LfGroup (or single-LfGroup frame), as block-unit rectangles.
///
/// `group_dim_px` is `frame_header.group_dim()` (kGroupDim, 256 for
/// VarDCT). Edge groups are clipped to the image extent, matching §6.2
/// ("with the possibility of a smaller-than-kGroupDim group on the
/// right or bottom of the image").
pub fn group_rects_in_blocks(
    width_px: u32,
    height_px: u32,
    group_dim_px: u32,
) -> Result<Vec<GroupRect>> {
    if width_px == 0 || height_px == 0 {
        return Err(Error::InvalidData(
            "JXL group_rects: zero-dimension image".into(),
        ));
    }
    if group_dim_px == 0 || group_dim_px % 8 != 0 {
        return Err(Error::InvalidData(format!(
            "JXL group_rects: group_dim {group_dim_px} must be a positive multiple of 8"
        )));
    }
    let width_blocks = width_px.div_ceil(8);
    let height_blocks = height_px.div_ceil(8);
    let group_blocks = group_dim_px / 8;
    let groups_x = width_px.div_ceil(group_dim_px);
    let groups_y = height_px.div_ceil(group_dim_px);
    let mut out = Vec::with_capacity((groups_x as usize) * (groups_y as usize));
    let mut index = 0u32;
    for gy in 0..groups_y {
        for gx in 0..groups_x {
            let bx0 = gx * group_blocks;
            let by0 = gy * group_blocks;
            out.push(GroupRect {
                index,
                bx0,
                by0,
                width_blocks: group_blocks.min(width_blocks - bx0),
                height_blocks: group_blocks.min(height_blocks - by0),
            });
            index += 1;
        }
    }
    Ok(out)
}

/// Slice a group's sub-[`DctSelectGrid`] out of the per-LfGroup grid.
///
/// Validates the §C.5.4 invariant that every varblock lies entirely
/// inside or entirely outside the rect (*"covers a rectangle that does
/// not cross group boundaries"*): a varblock whose footprint straddles
/// the rect boundary is a malformed grid (or a caller-side rect that is
/// not a §C.8.1 group rectangle) and is rejected rather than silently
/// mis-walked. The returned grid's coordinates are rect-relative, which
/// is the §C.8.1 group-local coordinate system.
pub fn slice_dct_select_rect(grid: &DctSelectGrid, rect: &GroupRect) -> Result<DctSelectGrid> {
    let gw = grid.width_blocks;
    let gh = grid.height_blocks;
    let (bx0, by0, w, h) = (rect.bx0, rect.by0, rect.width_blocks, rect.height_blocks);
    if w == 0 || h == 0 {
        return Err(Error::InvalidData(
            "JXL slice_dct_select_rect: empty rect".into(),
        ));
    }
    let (x1, y1) = (
        bx0.checked_add(w)
            .ok_or_else(|| Error::InvalidData("JXL slice_dct_select_rect: rect overflow".into()))?,
        by0.checked_add(h)
            .ok_or_else(|| Error::InvalidData("JXL slice_dct_select_rect: rect overflow".into()))?,
    );
    if x1 > gw || y1 > gh {
        return Err(Error::InvalidData(format!(
            "JXL slice_dct_select_rect: rect [{bx0},{by0})+({w}×{h}) exceeds grid {gw}×{gh}"
        )));
    }

    // §C.5.4 straddle validation: walk every varblock of the parent
    // grid and require its footprint to be entirely inside or entirely
    // outside the rect.
    let mut walk = crate::varblock_walk::VarblockWalk::new(grid);
    while let Some(vb) = walk.next()? {
        let (bcols, brows) = vb.transform.block_dims();
        let vx1 = vb.x + bcols;
        let vy1 = vb.y + brows;
        let overlaps = vb.x < x1 && vx1 > bx0 && vb.y < y1 && vy1 > by0;
        let inside = vb.x >= bx0 && vx1 <= x1 && vb.y >= by0 && vy1 <= y1;
        if overlaps && !inside {
            return Err(Error::InvalidData(format!(
                "JXL slice_dct_select_rect: varblock at ({},{}) {:?} ({}×{} blocks) straddles \
                 the group rect [{bx0},{x1})×[{by0},{y1}) — §C.5.4 forbids varblocks crossing \
                 group boundaries",
                vb.x, vb.y, vb.transform, bcols, brows
            )));
        }
    }

    let mut cells = Vec::with_capacity((w as usize) * (h as usize));
    let mut hf_mul = Vec::with_capacity((w as usize) * (h as usize));
    for y in by0..y1 {
        let row = (y as usize) * (gw as usize);
        for x in bx0..x1 {
            let idx = row + x as usize;
            cells.push(grid.cells[idx]);
            hf_mul.push(grid.hf_mul[idx]);
        }
    }
    Ok(DctSelectGrid {
        cells,
        hf_mul,
        width_blocks: w,
        height_blocks: h,
    })
}

/// Slice a group's rectangle of the dequantised LF image (one sample
/// per 8×8 block, channels `[X, Y, B]`). All three channels must carry
/// the full LfGroup block dims (the non-subsampled case the integrated
/// VarDCT path handles) and cover the rect.
pub fn slice_lf_rect(lf: &LfDequantOutput, rect: &GroupRect) -> Result<LfDequantOutput> {
    let (bx0, by0, w, h) = (rect.bx0, rect.by0, rect.width_blocks, rect.height_blocks);
    let mut samples: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (c, slot) in samples.iter_mut().enumerate() {
        let cw = lf.widths[c];
        let ch = lf.heights[c];
        if bx0 + w > cw || by0 + h > ch {
            return Err(Error::InvalidData(format!(
                "JXL slice_lf_rect: rect [{bx0},{by0})+({w}×{h}) exceeds LF channel {c} \
                 dims {cw}×{ch}"
            )));
        }
        let mut out = Vec::with_capacity((w as usize) * (h as usize));
        for y in by0..by0 + h {
            let row = (y as usize) * (cw as usize);
            out.extend_from_slice(&lf.samples[c][row + bx0 as usize..row + (bx0 + w) as usize]);
        }
        *slot = out;
    }
    Ok(LfDequantOutput {
        samples,
        widths: [w; 3],
        heights: [h; 3],
    })
}

/// Slice a group's rectangle of a per-64×64-pixel-tile channel
/// (HfMetadata `XFromY` / `BFromY`). `tiles` is the LfGroup-level tile
/// channel, `ceil(grid_width_blocks / 8) × ceil(grid_height_blocks / 8)`
/// row-major. The rect's block origin must be tile-aligned (group
/// origins are multiples of 32 blocks, tiles are 8 blocks — always
/// true for §C.8.1 group rects); the slice has
/// `ceil(w/8) × ceil(h/8)` tiles, matching what the per-group
/// reconstruction expects for the sliced grid.
pub fn slice_tiles_rect(
    tiles: &[i32],
    grid_width_blocks: u32,
    grid_height_blocks: u32,
    rect: &GroupRect,
) -> Result<Vec<i32>> {
    let tiles_w = grid_width_blocks.div_ceil(8).max(1);
    let tiles_h = grid_height_blocks.div_ceil(8).max(1);
    if tiles.len() != (tiles_w as usize) * (tiles_h as usize) {
        return Err(Error::InvalidData(format!(
            "JXL slice_tiles_rect: tile channel has {} samples, expected {tiles_w}×{tiles_h}",
            tiles.len()
        )));
    }
    if rect.bx0 % 8 != 0 || rect.by0 % 8 != 0 {
        return Err(Error::InvalidData(format!(
            "JXL slice_tiles_rect: rect origin ({}, {}) not 64×64-tile aligned",
            rect.bx0, rect.by0
        )));
    }
    let tx0 = rect.bx0 / 8;
    let ty0 = rect.by0 / 8;
    let tw = rect.width_blocks.div_ceil(8).max(1);
    let th = rect.height_blocks.div_ceil(8).max(1);
    if tx0 + tw > tiles_w || ty0 + th > tiles_h {
        return Err(Error::InvalidData(format!(
            "JXL slice_tiles_rect: tile rect [{tx0},{ty0})+({tw}×{th}) exceeds tile grid \
             {tiles_w}×{tiles_h}"
        )));
    }
    let mut out = Vec::with_capacity((tw as usize) * (th as usize));
    for ty in ty0..ty0 + th {
        let row = (ty as usize) * (tiles_w as usize);
        out.extend_from_slice(&tiles[row + tx0 as usize..row + (tx0 + tw) as usize]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct_select::TransformType;

    #[test]
    fn group_grid_matches_fixture_geometry() {
        // 1024×768 at kGroupDim 256 → 4×3 groups of 32×32 blocks each.
        let rects = group_rects_in_blocks(1024, 768, 256).unwrap();
        assert_eq!(rects.len(), 12);
        for (i, r) in rects.iter().enumerate() {
            assert_eq!(r.index as usize, i);
            assert_eq!((r.width_blocks, r.height_blocks), (32, 32));
        }
        // §C.3.1 raster order.
        assert_eq!((rects[0].bx0, rects[0].by0), (0, 0));
        assert_eq!((rects[3].bx0, rects[3].by0), (96, 0));
        assert_eq!((rects[4].bx0, rects[4].by0), (0, 32));
        assert_eq!((rects[11].bx0, rects[11].by0), (96, 64));
    }

    #[test]
    fn edge_groups_are_clipped() {
        // 300×260 → 2×2 groups; right column 300-256=44 px → 6 blocks,
        // bottom row 260-256=4 px → 1 block.
        let rects = group_rects_in_blocks(300, 260, 256).unwrap();
        assert_eq!(rects.len(), 4);
        assert_eq!((rects[0].width_blocks, rects[0].height_blocks), (32, 32));
        assert_eq!((rects[1].width_blocks, rects[1].height_blocks), (6, 32));
        assert_eq!((rects[2].width_blocks, rects[2].height_blocks), (32, 1));
        assert_eq!((rects[3].width_blocks, rects[3].height_blocks), (6, 1));
    }

    #[test]
    fn single_group_covers_whole_small_image() {
        let rects = group_rects_in_blocks(256, 256, 256).unwrap();
        assert_eq!(rects.len(), 1);
        assert_eq!(
            rects[0],
            GroupRect {
                index: 0,
                bx0: 0,
                by0: 0,
                width_blocks: 32,
                height_blocks: 32
            }
        );
    }

    /// Build a 4×2-block grid: an 8×8 varblock at (0,0), a 16×16 at
    /// (1,0)…(2,1), an 8×8 at (3,0), then 8×8s filling the rest.
    fn grid_with_16x16_at_1_0() -> DctSelectGrid {
        use crate::dct_select::DctSelectCell::{Continuation as C, TopLeft as T};
        let d8 = T(TransformType::Dct8x8);
        let d16 = T(TransformType::Dct16x16);
        DctSelectGrid {
            cells: vec![
                d8, d16, C, d8, //
                d8, C, C, d8,
            ],
            hf_mul: vec![1, 2, 0, 3, 4, 0, 0, 5],
            width_blocks: 4,
            height_blocks: 2,
        }
    }

    #[test]
    fn slice_keeps_contained_varblocks_and_rebases_coords() {
        let grid = grid_with_16x16_at_1_0();
        // Rect covering the middle 16×16 varblock plus its right 8×8s.
        let rect = GroupRect {
            index: 0,
            bx0: 1,
            by0: 0,
            width_blocks: 3,
            height_blocks: 2,
        };
        let sub = slice_dct_select_rect(&grid, &rect).unwrap();
        assert_eq!((sub.width_blocks, sub.height_blocks), (3, 2));
        let vbs = crate::varblock_walk::VarblockWalk::new(&sub)
            .collect()
            .unwrap();
        assert_eq!(vbs.len(), 3);
        // Rect-relative coordinates.
        assert_eq!((vbs[0].x, vbs[0].y), (0, 0));
        assert_eq!(vbs[0].transform, TransformType::Dct16x16);
        assert_eq!(vbs[0].hf_mul, 2);
        assert_eq!((vbs[1].x, vbs[1].y), (2, 0));
        assert_eq!((vbs[2].x, vbs[2].y), (2, 1));
    }

    #[test]
    fn slice_rejects_straddling_varblock() {
        let grid = grid_with_16x16_at_1_0();
        // Rect splitting the 16×16 varblock down the middle.
        let rect = GroupRect {
            index: 0,
            bx0: 2,
            by0: 0,
            width_blocks: 2,
            height_blocks: 2,
        };
        let err = slice_dct_select_rect(&grid, &rect).unwrap_err();
        assert!(format!("{err}").contains("straddles"));
    }

    #[test]
    fn slice_rejects_out_of_range_rect() {
        let grid = grid_with_16x16_at_1_0();
        let rect = GroupRect {
            index: 0,
            bx0: 3,
            by0: 0,
            width_blocks: 2,
            height_blocks: 2,
        };
        assert!(slice_dct_select_rect(&grid, &rect).is_err());
    }

    #[test]
    fn lf_slice_extracts_rect_per_channel() {
        let lf = LfDequantOutput {
            samples: [
                (0..12).map(|v| v as f32).collect(),
                (100..112).map(|v| v as f32).collect(),
                (200..212).map(|v| v as f32).collect(),
            ],
            widths: [4; 3],
            heights: [3; 3],
        };
        let rect = GroupRect {
            index: 0,
            bx0: 1,
            by0: 1,
            width_blocks: 2,
            height_blocks: 2,
        };
        let sub = slice_lf_rect(&lf, &rect).unwrap();
        assert_eq!(sub.widths, [2; 3]);
        assert_eq!(sub.heights, [2; 3]);
        assert_eq!(sub.samples[0], vec![5.0, 6.0, 9.0, 10.0]);
        assert_eq!(sub.samples[1], vec![105.0, 106.0, 109.0, 110.0]);
        assert_eq!(sub.samples[2], vec![205.0, 206.0, 209.0, 210.0]);
    }

    #[test]
    fn lf_slice_rejects_out_of_range() {
        let lf = LfDequantOutput {
            samples: [vec![0.0; 12], vec![0.0; 12], vec![0.0; 12]],
            widths: [4; 3],
            heights: [3; 3],
        };
        let rect = GroupRect {
            index: 0,
            bx0: 3,
            by0: 0,
            width_blocks: 2,
            height_blocks: 2,
        };
        assert!(slice_lf_rect(&lf, &rect).is_err());
    }

    #[test]
    fn tile_slice_takes_group_aligned_tiles() {
        // 128×96-block LfGroup grid → 16×12 tiles. A 32×32-block group
        // at (32, 32) blocks → tiles [4..8) × [4..8).
        let tiles: Vec<i32> = (0..16 * 12).collect();
        let rect = GroupRect {
            index: 5,
            bx0: 32,
            by0: 32,
            width_blocks: 32,
            height_blocks: 32,
        };
        let sub = slice_tiles_rect(&tiles, 128, 96, &rect).unwrap();
        assert_eq!(sub.len(), 16);
        assert_eq!(sub[0], 4 * 16 + 4);
        assert_eq!(sub[3], 4 * 16 + 7);
        assert_eq!(sub[15], 7 * 16 + 7);
    }

    #[test]
    fn tile_slice_partial_edge_group_rounds_up() {
        // 40×12-block grid (e.g. 320×96 px) → 5×2 tiles. Right-edge
        // group of 8×12 blocks at bx0=32 → 1×2 tiles.
        let tiles: Vec<i32> = (0..5 * 2).collect();
        let rect = GroupRect {
            index: 1,
            bx0: 32,
            by0: 0,
            width_blocks: 8,
            height_blocks: 12,
        };
        let sub = slice_tiles_rect(&tiles, 40, 12, &rect).unwrap();
        assert_eq!(sub, vec![4, 9]);
    }

    #[test]
    fn tile_slice_rejects_unaligned_origin() {
        let tiles: Vec<i32> = (0..4).collect();
        let rect = GroupRect {
            index: 0,
            bx0: 4,
            by0: 0,
            width_blocks: 8,
            height_blocks: 8,
        };
        assert!(slice_tiles_rect(&tiles, 16, 16, &rect).is_err());
    }
}
