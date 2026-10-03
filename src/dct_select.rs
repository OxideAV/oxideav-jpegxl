//! `DctSelect` / `HfMul` derivation from `BlockInfo` — ISO/IEC 18181-1:2024
//! Annex C.5.4 prose + Table C.16.
//!
//! ## Spec mapping
//!
//! Per FDIS C.5.4:
//!
//! > The DctSelect and HfMul fields are derived from the first and second
//! > rows of BlockInfo. These two fields have ceil(height / 8) rows and
//! > ceil(width / 8) columns. They are reconstructed by iterating over
//! > the columns of BlockInfo to obtain a varblock transform type type
//! > (the sample at the first row) and a quantization multiplier mul (the
//! > sample at the second row). The type corresponds to a valid varblock
//! > type and covers a rectangle that does not cross group boundaries;
//! > this is the DctSelect sample and it is stored at the coordinates of
//! > the top-left 8 × 8 rectangle of the varblock, which is positioned as
//! > much towards the top and towards the left as possible without
//! > overlapping already-positioned varblocks. The HfMul sample is stored
//! > at the same position and gets the value 1 + mul.
//!
//! Table C.16 lists numerical values 0..26 with the corresponding 8×8-
//! block-grid dimensions (`block_cols × block_rows`). Round 13 wires the
//! 27-entry table verbatim and walks BlockInfo column-by-column placing
//! varblocks raster-order at the next-available 8×8 cell.

use crate::error::{Error, Result};
use crate::lf_group::HfMetadata;

/// Transform type per FDIS Table C.16. Numerical values 0..=26 are
/// assigned to integral varblock transforms; later rounds will dispatch
/// IDCT + dequant matrix selection on this enum. Round 13 only uses the
/// associated `block_cols / block_rows` for placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TransformType {
    /// 0 — DCT8×8, dim 1×1.
    Dct8x8 = 0,
    /// 1 — Hornuss, dim 1×1.
    Hornuss = 1,
    /// 2 — DCT2×2, dim 1×1.
    Dct2x2 = 2,
    /// 3 — DCT4×4, dim 1×1.
    Dct4x4 = 3,
    /// 4 — DCT16×16, dim 2×2.
    Dct16x16 = 4,
    /// 5 — DCT32×32, dim 4×4.
    Dct32x32 = 5,
    /// 6 — DCT16×8, dim 2×1.
    Dct16x8 = 6,
    /// 7 — DCT8×16, dim 1×2.
    Dct8x16 = 7,
    /// 8 — DCT32×8, dim 4×1.
    Dct32x8 = 8,
    /// 9 — DCT8×32, dim 1×4.
    Dct8x32 = 9,
    /// 10 — DCT32×16, dim 4×2.
    Dct32x16 = 10,
    /// 11 — DCT16×32, dim 2×4.
    Dct16x32 = 11,
    /// 12 — DCT4×8, dim 1×1.
    Dct4x8 = 12,
    /// 13 — DCT8×4, dim 1×1.
    Dct8x4 = 13,
    /// 14 — AFV0, dim 1×1.
    Afv0 = 14,
    /// 15 — AFV1, dim 1×1.
    Afv1 = 15,
    /// 16 — AFV2, dim 1×1.
    Afv2 = 16,
    /// 17 — AFV3, dim 1×1.
    Afv3 = 17,
    /// 18 — DCT64×64, dim 8×8.
    Dct64x64 = 18,
    /// 19 — DCT64×32, dim 8×4.
    Dct64x32 = 19,
    /// 20 — DCT32×64, dim 4×8.
    Dct32x64 = 20,
    /// 21 — DCT128×128, dim 16×16.
    Dct128x128 = 21,
    /// 22 — DCT128×64, dim 16×8.
    Dct128x64 = 22,
    /// 23 — DCT64×128, dim 8×16.
    Dct64x128 = 23,
    /// 24 — DCT256×256, dim 32×32.
    Dct256x256 = 24,
    /// 25 — DCT256×128, dim 32×16.
    Dct256x128 = 25,
    /// 26 — DCT128×256, dim 16×32.
    Dct128x256 = 26,
}

impl TransformType {
    /// Convert a Table C.16 numerical value to `TransformType`. Returns
    /// `Err(InvalidData)` for values outside 0..=26.
    pub fn from_index(idx: i32) -> Result<Self> {
        Ok(match idx {
            0 => Self::Dct8x8,
            1 => Self::Hornuss,
            2 => Self::Dct2x2,
            3 => Self::Dct4x4,
            4 => Self::Dct16x16,
            5 => Self::Dct32x32,
            6 => Self::Dct16x8,
            7 => Self::Dct8x16,
            8 => Self::Dct32x8,
            9 => Self::Dct8x32,
            10 => Self::Dct32x16,
            11 => Self::Dct16x32,
            12 => Self::Dct4x8,
            13 => Self::Dct8x4,
            14 => Self::Afv0,
            15 => Self::Afv1,
            16 => Self::Afv2,
            17 => Self::Afv3,
            18 => Self::Dct64x64,
            19 => Self::Dct64x32,
            20 => Self::Dct32x64,
            21 => Self::Dct128x128,
            22 => Self::Dct128x64,
            23 => Self::Dct64x128,
            24 => Self::Dct256x256,
            25 => Self::Dct256x128,
            26 => Self::Dct128x256,
            _ => {
                return Err(Error::InvalidData(format!(
                    "JXL DctSelect: transform type index {idx} out of range 0..=26 (Table C.16)"
                )));
            }
        })
    }

    /// Numerical value 0..=26 (Table C.16 column 2).
    pub fn index(self) -> u8 {
        self as u8
    }

    /// Varblock dimensions in 8×8-block grid units `(cols, rows)` per
    /// Table C.16 column 3.
    ///
    /// Note: Table C.16 lists the third column as
    /// `dimension_in_dct_select` formatted "rows × columns" textually
    /// but the placement loop iterates as "cols × rows" — we return
    /// `(cols, rows)` to match the placement algorithm's column-major
    /// scan order. For DCT16×8 (numerical value 6), the spec text reads
    /// "2×1" meaning 2 rows × 1 column = a varblock that's twice as tall
    /// as wide. This matches the transform's name (16 rows × 8 cols).
    pub fn block_dims(self) -> (u32, u32) {
        match self {
            Self::Dct8x8 | Self::Hornuss | Self::Dct2x2 | Self::Dct4x4 => (1, 1),
            Self::Dct4x8 | Self::Dct8x4 => (1, 1),
            Self::Afv0 | Self::Afv1 | Self::Afv2 | Self::Afv3 => (1, 1),
            Self::Dct16x16 => (2, 2),
            Self::Dct32x32 => (4, 4),
            // Per Table C.16 the dim text is "rows × columns".
            // DCT16×8 (transform name = 16 rows × 8 cols) → 2 rows × 1 col.
            Self::Dct16x8 => (1, 2),
            // DCT8×16 (8 rows × 16 cols) → 1 row × 2 cols.
            Self::Dct8x16 => (2, 1),
            // DCT32×8 (32 rows × 8 cols) → 4 rows × 1 col.
            Self::Dct32x8 => (1, 4),
            // DCT8×32 (8 rows × 32 cols) → 1 row × 4 cols.
            Self::Dct8x32 => (4, 1),
            // DCT32×16 (32 rows × 16 cols) → 4 rows × 2 cols.
            Self::Dct32x16 => (2, 4),
            // DCT16×32 (16 rows × 32 cols) → 2 rows × 4 cols.
            Self::Dct16x32 => (4, 2),
            Self::Dct64x64 => (8, 8),
            Self::Dct64x32 => (4, 8),
            Self::Dct32x64 => (8, 4),
            Self::Dct128x128 => (16, 16),
            Self::Dct128x64 => (8, 16),
            Self::Dct64x128 => (16, 8),
            Self::Dct256x256 => (32, 32),
            Self::Dct256x128 => (16, 32),
            Self::Dct128x256 => (32, 16),
        }
    }
}

/// Per-LfGroup `DctSelect` + `HfMul` grids per FDIS C.5.4 prose.
///
/// Both grids have `ceil(width / 8) × ceil(height / 8)` cells laid out
/// row-major. Cells covered by the *interior* of a multi-block varblock
/// store [`DctSelectCell::Continuation`]; the top-left cell of each
/// varblock stores [`DctSelectCell::TopLeft`] with the transform type +
/// HfMul value.
#[derive(Debug, Clone)]
pub struct DctSelectGrid {
    /// Per-cell DctSelect entry. Length = `width_blocks * height_blocks`.
    pub cells: Vec<DctSelectCell>,
    /// HfMul value at the top-left cell of each varblock; 0 elsewhere.
    /// Per spec `HfMul = 1 + mul` and is stored at the top-left of the
    /// varblock.
    pub hf_mul: Vec<i32>,
    /// Number of 8×8 cells horizontally.
    pub width_blocks: u32,
    /// Number of 8×8 cells vertically.
    pub height_blocks: u32,
}

/// Per-cell entry in the [`DctSelectGrid`]. `TopLeft(transform)` for the
/// top-left cell of a varblock; `Continuation` for any other cell covered
/// by the same varblock's footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DctSelectCell {
    /// Cell hasn't yet been claimed by any varblock (placement-pass
    /// transient state). After `derive` returns `Ok`, no Empty cell
    /// remains.
    Empty,
    /// Top-left cell of a varblock.
    TopLeft(TransformType),
    /// Cell is covered by the interior of a multi-block varblock whose
    /// top-left lives elsewhere.
    Continuation,
}

/// Derive [`DctSelectGrid`] from an [`HfMetadata`] block-info channel
/// per FDIS C.5.4 prose. The placement algorithm walks an 8×8-cell grid
/// of size `(ceil(lf_w/8), ceil(lf_h/8))` in raster order; for each
/// varblock (column of BlockInfo) we look up the next empty cell, verify
/// the varblock fits without crossing the LfGroup boundary or
/// overlapping a previously-placed varblock, and stamp every covered
/// cell.
///
/// `lf_w` / `lf_h` are the LfGroup pixel dimensions; the returned grid
/// has `ceil(lf_w/8) × ceil(lf_h/8)` cells.
pub fn derive_dct_select(hf: &HfMetadata, lf_w: u32, lf_h: u32) -> Result<DctSelectGrid> {
    let width_blocks = lf_w.div_ceil(8);
    let height_blocks = lf_h.div_ceil(8);
    let total = (width_blocks as usize)
        .checked_mul(height_blocks as usize)
        .ok_or_else(|| Error::InvalidData("JXL DctSelect: width × height overflow".into()))?;
    let mut cells = vec![DctSelectCell::Empty; total];
    let mut hf_mul = vec![0i32; total];

    // Round-12 stored BlockInfo as a flat row-major Vec<i32> with 2 rows
    // and `nb_blocks` columns. Row 0 (transform type indices) starts at
    // offset 0; row 1 (mul - 1) starts at offset `nb_blocks`.
    let nb_blocks = hf.nb_blocks as usize;
    let row_stride = hf.channel_widths[2] as usize;
    if hf.channel_heights[2] != 2 {
        return Err(Error::InvalidData(format!(
            "JXL DctSelect: BlockInfo has {} rows, expected 2",
            hf.channel_heights[2]
        )));
    }
    if hf.block_info.len() < 2 * row_stride {
        return Err(Error::InvalidData(format!(
            "JXL DctSelect: BlockInfo has {} samples, expected at least 2 × {} = {}",
            hf.block_info.len(),
            row_stride,
            2 * row_stride
        )));
    }
    if nb_blocks > row_stride {
        return Err(Error::InvalidData(format!(
            "JXL DctSelect: nb_blocks {nb_blocks} exceeds BlockInfo row stride {row_stride}"
        )));
    }

    // Placement scan position: walk cells row-major, skipping any cell
    // already filled by a previous varblock's footprint.
    let mut scan = 0usize;
    for col in 0..nb_blocks {
        // Advance scan to the next empty cell.
        while scan < total && cells[scan] != DctSelectCell::Empty {
            scan += 1;
        }
        if scan >= total {
            return Err(Error::InvalidData(format!(
                "JXL DctSelect: BlockInfo column {col} placed past end of {width_blocks}×{height_blocks} grid \
                 ({total} cells); nb_blocks={nb_blocks}"
            )));
        }
        let type_idx = hf.block_info[col];
        let mul_minus_1 = hf.block_info[row_stride + col];
        let transform = TransformType::from_index(type_idx)?;
        let (bcols, brows) = transform.block_dims();

        let bx = (scan % width_blocks as usize) as u32;
        let by = (scan / width_blocks as usize) as u32;

        // Bounds: varblock must stay inside the LfGroup grid.
        if bx + bcols > width_blocks || by + brows > height_blocks {
            return Err(Error::InvalidData(format!(
                "JXL DctSelect: varblock {col} (type {:?} at ({bx},{by}) covering {bcols}×{brows}) \
                 spills outside {width_blocks}×{height_blocks} grid",
                transform
            )));
        }

        // Overlap check + paint.
        for dy in 0..brows {
            for dx in 0..bcols {
                let cell_idx = ((by + dy) * width_blocks + (bx + dx)) as usize;
                if cells[cell_idx] != DctSelectCell::Empty {
                    return Err(Error::InvalidData(format!(
                        "JXL DctSelect: varblock {col} (type {:?}) overlaps already-placed varblock at \
                         ({},{}) inside {width_blocks}×{height_blocks} grid",
                        transform,
                        bx + dx,
                        by + dy
                    )));
                }
                cells[cell_idx] = if dx == 0 && dy == 0 {
                    DctSelectCell::TopLeft(transform)
                } else {
                    DctSelectCell::Continuation
                };
            }
        }
        // HfMul is stored at the top-left position only.
        let tl_idx = (by * width_blocks + bx) as usize;
        // Spec: HfMul = 1 + mul. mul is `BlockInfo` row 1, treated as
        // signed but the spec only assigns valid quant multipliers in
        // range that yield positive HfMul. Coerce to i32 saturating,
        // reject if HfMul would underflow to 0 or negative.
        let hfmul_i64 = (mul_minus_1 as i64) + 1;
        if hfmul_i64 < 1 {
            return Err(Error::InvalidData(format!(
                "JXL DctSelect: varblock {col} HfMul = {hfmul_i64} (= 1 + mul = 1 + {mul_minus_1}) \
                 must be >= 1"
            )));
        }
        if hfmul_i64 > i32::MAX as i64 {
            return Err(Error::InvalidData(format!(
                "JXL DctSelect: varblock {col} HfMul = {hfmul_i64} overflows i32"
            )));
        }
        hf_mul[tl_idx] = hfmul_i64 as i32;
    }

    // After placement: every cell must be filled. If not, the spec says
    // BlockInfo failed to cover the LfGroup — reject.
    if let Some(empty_idx) = cells.iter().position(|c| *c == DctSelectCell::Empty) {
        let bx = empty_idx % width_blocks as usize;
        let by = empty_idx / width_blocks as usize;
        return Err(Error::InvalidData(format!(
            "JXL DctSelect: BlockInfo failed to cover ({bx},{by}) — nb_blocks={nb_blocks} \
             insufficient for {width_blocks}×{height_blocks} grid"
        )));
    }

    Ok(DctSelectGrid {
        cells,
        hf_mul,
        width_blocks,
        height_blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hf(block_info: Vec<i32>, nb_blocks: u32, info_w: u32) -> HfMetadata {
        // Channel order: [XFromY, BFromY, BlockInfo, Sharpness].
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
    fn transform_type_from_index_round_trip() {
        for i in 0..=26 {
            let t = TransformType::from_index(i).unwrap();
            assert_eq!(t.index() as i32, i, "round-trip: i={i}, t={t:?}");
        }
    }

    #[test]
    fn transform_type_invalid_index_rejected() {
        assert!(TransformType::from_index(-1).is_err());
        assert!(TransformType::from_index(27).is_err());
    }

    #[test]
    fn block_dims_match_table_c16_unit_blocks() {
        // All 1×1 transforms.
        for t in [
            TransformType::Dct8x8,
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
            assert_eq!(t.block_dims(), (1, 1), "{t:?}");
        }
    }

    #[test]
    fn block_dims_match_table_c16_multi_block() {
        assert_eq!(TransformType::Dct16x16.block_dims(), (2, 2));
        assert_eq!(TransformType::Dct32x32.block_dims(), (4, 4));
        assert_eq!(TransformType::Dct64x64.block_dims(), (8, 8));
        assert_eq!(TransformType::Dct128x128.block_dims(), (16, 16));
        assert_eq!(TransformType::Dct256x256.block_dims(), (32, 32));
    }

    #[test]
    fn block_dims_dct16x8_two_rows_one_col() {
        // DCT16×8 = 16 rows × 8 cols at output. As DctSelect dims: 2 rows × 1 col.
        // (cols, rows) ordering → (1, 2).
        assert_eq!(TransformType::Dct16x8.block_dims(), (1, 2));
        assert_eq!(TransformType::Dct8x16.block_dims(), (2, 1));
    }

    #[test]
    fn derive_dct_select_one_block_8x8() {
        // Single-block 8×8 LfGroup — one DCT8×8 varblock.
        let hf = make_hf(vec![0, 0], 1, 1);
        let g = derive_dct_select(&hf, 8, 8).unwrap();
        assert_eq!(g.width_blocks, 1);
        assert_eq!(g.height_blocks, 1);
        assert_eq!(g.cells.len(), 1);
        assert_eq!(g.cells[0], DctSelectCell::TopLeft(TransformType::Dct8x8));
        assert_eq!(g.hf_mul[0], 1); // 1 + mul = 1 + 0
    }

    #[test]
    fn derive_dct_select_2x2_grid_four_8x8_blocks() {
        // 16×16 LfGroup, 2×2 cell grid, four DCT8×8 blocks.
        // BlockInfo: row 0 = type indices [0,0,0,0], row 1 = mul-1 [0,0,0,0].
        // Layout in flat block_info: [type0 type1 type2 type3 mul0 mul1 mul2 mul3]
        let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
        let g = derive_dct_select(&hf, 16, 16).unwrap();
        assert_eq!(g.width_blocks, 2);
        assert_eq!(g.height_blocks, 2);
        for i in 0..4 {
            assert_eq!(
                g.cells[i],
                DctSelectCell::TopLeft(TransformType::Dct8x8),
                "cell {i}"
            );
            assert_eq!(g.hf_mul[i], 1);
        }
    }

    #[test]
    fn derive_dct_select_one_dct16x16_covers_2x2_grid() {
        // 16×16 LfGroup, single DCT16×16 covering all four 8×8 cells.
        // BlockInfo: 1 column × 2 rows = [type, mul].
        let hf = make_hf(vec![4, 0], 1, 1);
        let g = derive_dct_select(&hf, 16, 16).unwrap();
        assert_eq!(g.cells[0], DctSelectCell::TopLeft(TransformType::Dct16x16));
        assert_eq!(g.cells[1], DctSelectCell::Continuation);
        assert_eq!(g.cells[2], DctSelectCell::Continuation);
        assert_eq!(g.cells[3], DctSelectCell::Continuation);
        // HfMul stored at top-left only.
        assert_eq!(g.hf_mul[0], 1);
        assert_eq!(g.hf_mul[1], 0);
        assert_eq!(g.hf_mul[2], 0);
        assert_eq!(g.hf_mul[3], 0);
    }

    #[test]
    fn derive_dct_select_dct8x16_then_dct8x8_fills_2x2_grid() {
        // 16×16 grid (2×2 cells). First varblock: DCT8×16 (1 row × 2 cols)
        // → fills cells (0,0) and (1,0). Second varblock: DCT8×8 → fills
        // (0,1). Third varblock: DCT8×8 → fills (1,1). nb_blocks = 3.
        // BlockInfo: types [7,0,0], muls [0,0,0]. Width = 3.
        let hf = make_hf(vec![7, 0, 0, 0, 0, 0], 3, 3);
        let g = derive_dct_select(&hf, 16, 16).unwrap();
        // cells layout (row-major): (0,0)=TL(DCT8x16), (1,0)=Cont,
        //                           (0,1)=TL(DCT8x8), (1,1)=TL(DCT8x8).
        assert_eq!(g.cells[0], DctSelectCell::TopLeft(TransformType::Dct8x16));
        assert_eq!(g.cells[1], DctSelectCell::Continuation);
        assert_eq!(g.cells[2], DctSelectCell::TopLeft(TransformType::Dct8x8));
        assert_eq!(g.cells[3], DctSelectCell::TopLeft(TransformType::Dct8x8));
    }

    #[test]
    fn derive_dct_select_invalid_type_index_rejected() {
        let hf = make_hf(vec![99, 0], 1, 1);
        assert!(derive_dct_select(&hf, 8, 8).is_err());
    }

    #[test]
    fn derive_dct_select_grid_underfilled_rejected() {
        // 2×2 grid but only one DCT8×8 — three cells remain Empty → reject.
        let hf = make_hf(vec![0, 0], 1, 1);
        let r = derive_dct_select(&hf, 16, 16);
        assert!(r.is_err());
    }

    #[test]
    fn derive_dct_select_overflow_rejected() {
        // 8×8 grid, one DCT16×16 — covers (0,0)+(0,1)+(1,0)+(1,1) but
        // grid is only 1×1 cell. Should fail the spill check.
        let hf = make_hf(vec![4, 0], 1, 1);
        let r = derive_dct_select(&hf, 8, 8);
        assert!(r.is_err());
    }

    #[test]
    fn derive_dct_select_hf_mul_with_nonzero_mul() {
        // Single DCT8×8 with mul-1 = 5 → HfMul = 6.
        let hf = make_hf(vec![0, 5], 1, 1);
        let g = derive_dct_select(&hf, 8, 8).unwrap();
        assert_eq!(g.hf_mul[0], 6);
    }
}
