//! Round 208 integration tests — per-LfGroup varblock-walk driver
//! (ISO/IEC FDIS 18181-1:2021 §C.5.4 + §C.8.3).
//!
//! These exercise the [`varblock_walk::VarblockWalk`] +
//! [`varblock_walk::decode_varblocks_for_pass_channel`] surface
//! end-to-end against the round-13 [`dct_select::DctSelectGrid`] and
//! the round-190 [`per_pass_non_zeros::PerPassNonZerosGrids`].
//!
//! Pure-control-flow primitive: no bit reads, no histogram
//! materialisation. The closures abstract over the §C.7.2 entropy
//! decode (#799 DOCS-GAP).

use oxideav_jpegxl::dct_select::{derive_dct_select, DctSelectCell, DctSelectGrid, TransformType};
use oxideav_jpegxl::lf_group::HfMetadata;
use oxideav_jpegxl::per_pass_non_zeros::PerPassNonZerosGrids;
use oxideav_jpegxl::varblock_walk::{
    count_varblocks, decode_varblocks_for_pass_channel, Varblock, VarblockWalk,
};

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
fn r208_single_dct8x8_walk_yields_one_varblock() {
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let blocks: Vec<Varblock> = VarblockWalk::new(&grid).collect().unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].transform, TransformType::Dct8x8);
    assert_eq!(blocks[0].x, 0);
    assert_eq!(blocks[0].y, 0);
    assert_eq!(blocks[0].hf_mul, 1);
}

#[test]
fn r208_raster_order_walk_preserves_row_major_layout() {
    // Build a 4×4 grid (32×32 LfGroup) of all DCT8×8 → 16 varblocks
    // in raster order. Confirm the walker yields each
    // (x, y) pair in row-major order.
    let nb = 16;
    let mut block_info = vec![0i32; (nb * 2) as usize];
    // row 0 (types) = all 0s; row 1 (muls) = all 0s. Already so.
    // info_w (row stride) must be >= nb.
    let _ = &mut block_info; // suppress unused-mut warning
    let hf = make_hf(block_info, nb, nb);
    let grid = derive_dct_select(&hf, 32, 32).unwrap();
    let blocks: Vec<Varblock> = VarblockWalk::new(&grid).collect().unwrap();
    assert_eq!(blocks.len(), 16);
    for (i, b) in blocks.iter().enumerate() {
        let expected_x = (i % 4) as u32;
        let expected_y = (i / 4) as u32;
        assert_eq!((b.x, b.y), (expected_x, expected_y), "block {i}");
        assert_eq!(b.transform, TransformType::Dct8x8);
    }
}

#[test]
fn r208_dct16x16_covers_2x2_cells_yields_single_varblock() {
    let hf = make_hf(vec![4, 0], 1, 1);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let blocks: Vec<Varblock> = VarblockWalk::new(&grid).collect().unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].transform, TransformType::Dct16x16);
    // The DctSelectGrid has 4 cells (2×2); 1 TopLeft + 3 Continuation.
    let n_top = grid
        .cells
        .iter()
        .filter(|c| matches!(c, DctSelectCell::TopLeft(_)))
        .count();
    let n_cont = grid
        .cells
        .iter()
        .filter(|c| matches!(c, DctSelectCell::Continuation))
        .count();
    assert_eq!(n_top, 1);
    assert_eq!(n_cont, 3);
}

#[test]
fn r208_mixed_transforms_walk_in_placement_order() {
    // DCT16×8 (cols=1, rows=2) at (0,0), then two DCT8×8 at (1,0)
    // and (1,1).
    let hf = make_hf(vec![6, 0, 0, 0, 0, 0], 3, 3);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let blocks: Vec<Varblock> = VarblockWalk::new(&grid).collect().unwrap();
    assert_eq!(blocks.len(), 3);
    assert_eq!(blocks[0].transform, TransformType::Dct16x8);
    assert_eq!((blocks[0].x, blocks[0].y), (0, 0));
    assert_eq!(blocks[1].transform, TransformType::Dct8x8);
    assert_eq!((blocks[1].x, blocks[1].y), (1, 0));
    assert_eq!(blocks[2].transform, TransformType::Dct8x8);
    assert_eq!((blocks[2].x, blocks[2].y), (1, 1));
}

#[test]
fn r208_count_matches_walk_for_mixed_grid() {
    let hf = make_hf(vec![6, 0, 0, 0, 0, 0], 3, 3);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let n = count_varblocks(&grid);
    let collected: Vec<Varblock> = VarblockWalk::new(&grid).collect().unwrap();
    assert_eq!(n as usize, collected.len());
    assert_eq!(n, 3);
}

#[test]
fn r208_walk_empty_continuation_only_grid_yields_nothing() {
    // Synthesised malformed-but-tolerated grid: all-Continuation
    // cells produce no varblocks but no error (the walker's contract
    // is "skip Continuation" — an Empty would error).
    let grid = DctSelectGrid {
        cells: vec![DctSelectCell::Continuation; 4],
        hf_mul: vec![0; 4],
        width_blocks: 2,
        height_blocks: 2,
    };
    let blocks: Vec<Varblock> = VarblockWalk::new(&grid).collect().unwrap();
    assert_eq!(blocks.len(), 0);
    assert_eq!(count_varblocks(&grid), 0);
}

#[test]
fn r208_walk_residual_empty_cell_errors() {
    // Synthesised malformed grid with an Empty cell: walker errors
    // on first encounter.
    let grid = DctSelectGrid {
        cells: vec![
            DctSelectCell::TopLeft(TransformType::Dct8x8),
            DctSelectCell::Empty,
        ],
        hf_mul: vec![1, 0],
        width_blocks: 2,
        height_blocks: 1,
    };
    let mut walk = VarblockWalk::new(&grid);
    assert!(walk.next().unwrap().is_some());
    assert!(walk.next().is_err());
}

#[test]
fn r208_decode_varblocks_walks_four_dct8x8_blocks() {
    let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
    let triples = decode_varblocks_for_pass_channel(
        &grid,
        &mut nz,
        0,
        0,
        13,
        |_vb| Ok(0),
        |_| Ok(0),
        |_| Ok(0),
    )
    .unwrap();
    assert_eq!(triples.len(), 4);
    // Verify per-block coordinates match the walk.
    for (i, (vb, _decoded, _raw)) in triples.iter().enumerate() {
        let expected_x = (i % 2) as u32;
        let expected_y = (i / 2) as u32;
        assert_eq!((vb.x, vb.y), (expected_x, expected_y));
    }
}

#[test]
fn r208_decode_varblocks_routes_per_pass_per_channel() {
    // Two passes, three channels. Run the walker for (pass=1,
    // channel=2) and confirm only that cell's grid is mutated.
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
    let triples = decode_varblocks_for_pass_channel(
        &grid,
        &mut nz,
        1,
        2,
        13,
        |_vb| Ok(0),
        |_| Ok(7), // raw_non_zeros = 7
        |_| Ok(0), // ucoeff = 0 → no decrement loop
    )
    .unwrap();
    assert_eq!(triples.len(), 1);
    // Routed cell mutated.
    assert_eq!(nz.get(1, 2, 0, 0).unwrap(), 7);
    // Other (pass, channel) cells untouched.
    assert_eq!(nz.get(0, 0, 0, 0).unwrap(), 0);
    assert_eq!(nz.get(0, 1, 0, 0).unwrap(), 0);
    assert_eq!(nz.get(0, 2, 0, 0).unwrap(), 0);
    assert_eq!(nz.get(1, 0, 0, 0).unwrap(), 0);
    assert_eq!(nz.get(1, 1, 0, 0).unwrap(), 0);
}

#[test]
fn r208_decode_varblocks_propagates_ctx_closure_error() {
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
    let r = decode_varblocks_for_pass_channel(
        &grid,
        &mut nz,
        0,
        0,
        13,
        |_vb| {
            Err(oxideav_core::Error::InvalidData(
                "ctx error from integration test".into(),
            ))
        },
        |_| Ok(0),
        |_| Ok(0),
    );
    assert!(r.is_err());
}

#[test]
fn r208_decode_varblocks_walks_dct16x16_single_block() {
    // DCT16×16 covers a 2×2 cell footprint; only one varblock to
    // walk. The round-208 driver should pass the DCT16×16
    // TransformType through to the round-190 typed driver, which
    // then expects `predicted = NonZerosGrid::predicted(0, 0) = 32`
    // for the first varblock.
    let hf = make_hf(vec![4, 0], 1, 1);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
    let mut transforms_seen = Vec::new();
    let triples = decode_varblocks_for_pass_channel(
        &grid,
        &mut nz,
        0,
        0,
        13,
        |vb| {
            transforms_seen.push(vb.transform);
            Ok(0)
        },
        |_predicted| Ok(0),
        |_| Ok(0),
    )
    .unwrap();
    assert_eq!(triples.len(), 1);
    assert_eq!(transforms_seen, vec![TransformType::Dct16x16]);
    assert_eq!(triples[0].0.transform, TransformType::Dct16x16);
    assert_eq!((triples[0].0.x, triples[0].0.y), (0, 0));
    assert_eq!(triples[0].0.hf_mul, 1);
}

#[test]
fn r208_decode_varblocks_preserves_hf_mul_across_walk() {
    // Set up two varblocks with distinct hf_mul values; confirm
    // both reach the closure unmodified.
    // block_info row 0 = [0, 0], row 1 = [3, 7] (mul-1 = 3, 7).
    // info_w = 2 (row stride).
    let hf = make_hf(vec![0, 0, 3, 7], 2, 2);
    let grid = derive_dct_select(&hf, 16, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 1).unwrap();
    let mut muls_seen = Vec::new();
    let _ = decode_varblocks_for_pass_channel(
        &grid,
        &mut nz,
        0,
        0,
        13,
        |vb| {
            muls_seen.push(vb.hf_mul);
            Ok(0)
        },
        |_| Ok(0),
        |_| Ok(0),
    )
    .unwrap();
    assert_eq!(muls_seen, vec![4, 8]); // 1+3, 1+7
}
