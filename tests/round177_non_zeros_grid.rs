//! Round 177 — per-pass / per-channel `NonZeros(x, y)` grid bookkeeping.
//!
//! Integration tests for the typed [`NonZerosGrid`] scaffolding +
//! [`decode_block_at`] driver — ISO/IEC FDIS 18181-1:2021 §C.8.3,
//! Listing C.13 prelude (`PredictedNonZeros`) + the prose right after
//! Listing C.14 (`NonZeros(x, y) = (non_zeros + num_blocks − 1) Idiv
//! num_blocks`).
//!
//! The unit-test suite at `src/non_zeros_grid.rs::tests` covers each
//! primitive in isolation. These integration tests exercise the grid
//! through the public crate surface and pin the multi-position
//! interaction between [`predicted_non_zeros`] (round 159) →
//! [`decode_block_at`] (round 177) → grid update → next prediction.

use oxideav_core::Result;
use oxideav_jpegxl::dct_select::TransformType;
use oxideav_jpegxl::non_zeros_grid::{decode_block_at, NonZerosGrid};
use oxideav_jpegxl::pass_group_hf::predicted_non_zeros;

#[test]
fn round177_grid_origin_predicted_is_32_for_any_shape() {
    // FDIS Listing C.13 prelude: `PredictedNonZeros(0, 0) = 32`,
    // regardless of channel / shape. Verify across a sweep of grid
    // dimensions.
    for w in [1u32, 2, 4, 8, 16, 32] {
        for h in [1u32, 2, 4, 8, 16, 32] {
            let g = NonZerosGrid::new(w, h).unwrap();
            assert_eq!(
                g.predicted(0, 0).unwrap(),
                32,
                "(w={w}, h={h}) predicted(0, 0)"
            );
        }
    }
}

#[test]
fn round177_grid_raster_walk_dct8x8_chains_through_all_cells() {
    // DCT8×8: num_blocks = 1. With initial_non_zeros = nz at every
    // varblock, the post-Listing-C.14 update writes `nz` directly,
    // so after a raster-walk over (0,0)..(w-1,h-1) every cell stores
    // its sequence number's `nz` (here a constant 7).
    let mut g = NonZerosGrid::new(4, 3).unwrap();
    let mut step = 0u32;
    for y in 0..3 {
        for x in 0..4 {
            let read_nz = |_ctx: u32| -> Result<u32> { Ok(7u32) };
            let dec = |_ctx: u32| -> Result<u32> { Ok(0u32) };
            let (_block, raw) = decode_block_at(
                &mut g,
                x,
                y,
                TransformType::Dct8x8,
                /* block_ctx = */ 0,
                /* nb_block_ctx = */ 1,
                read_nz,
                dec,
            )
            .unwrap();
            assert_eq!(raw, 7);
            step += 1;
        }
    }
    assert_eq!(step, 12);
    // Every cell now stores 7.
    for y in 0..3 {
        for x in 0..4 {
            assert_eq!(g.get(x, y).unwrap(), 7);
        }
    }
}

#[test]
fn round177_grid_predicted_matches_helper_after_raster_walk() {
    // After a non-trivial raster walk with a `step`-varying
    // non_zeros sequence, `predicted_non_zeros` issued against
    // `|x, y| g.get(x, y).unwrap_or(0)` must agree with
    // `g.predicted(x, y)` at every position.
    let mut g = NonZerosGrid::new(3, 3).unwrap();
    let mut step = 0u32;
    for y in 0..3 {
        for x in 0..3 {
            let nz = 4 + step % 9; // arbitrary 4..12 sequence
            let read_nz = |_ctx: u32| -> Result<u32> { Ok(nz) };
            let dec = |_ctx: u32| -> Result<u32> { Ok(0u32) };
            let _ =
                decode_block_at(&mut g, x, y, TransformType::Dct8x8, 0, 1, read_nz, dec).unwrap();
            step += 1;
        }
    }
    // Now compare grid `predicted(...)` vs `predicted_non_zeros(
    //   x, y, |xx, yy| g.get(xx, yy).unwrap_or(0))` at every (x, y).
    for y in 0..3 {
        for x in 0..3 {
            let via_grid = g.predicted(x, y).unwrap();
            let via_helper = predicted_non_zeros(x, y, |xx, yy| g.get(xx, yy).unwrap_or(0));
            assert_eq!(via_grid, via_helper, "predicted disagreement at ({x}, {y})");
        }
    }
}

#[test]
fn round177_grid_dct16x16_ceil_div_4_through_typed_driver() {
    // DCT16×16 has num_blocks = 4. Driving `decode_block_at` with
    // raw non_zeros = 17 must produce a stored grid cell of
    // ceil(17 / 4) = 5, not 17.
    let mut g = NonZerosGrid::new(2, 2).unwrap();
    let read_nz = |_ctx: u32| -> Result<u32> { Ok(17u32) };
    let dec = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let (_block, raw) =
        decode_block_at(&mut g, 0, 0, TransformType::Dct16x16, 0, 1, read_nz, dec).unwrap();
    assert_eq!(raw, 17);
    assert_eq!(g.get(0, 0).unwrap(), 5);
    // The next prediction at (1, 0) reads NonZeros(0, 0) = 5
    // (the y == 0 && x != 0 branch).
    assert_eq!(g.predicted(1, 0).unwrap(), 5);
}

#[test]
fn round177_grid_dct32x32_ceil_div_16_through_typed_driver() {
    // DCT32×32 has num_blocks = 16, size = 1024. The per-block loop
    // (round 159) caps initial_non_zeros at size - num_blocks = 1008,
    // so we test at the boundary: ceil(1008 / 16) = 63. The grid is
    // sized 4×4 for the DCT32×32 footprint; per the §C.8.3 "for each
    // block in the current varblock" prose every covered cell stores
    // the value.
    let mut g = NonZerosGrid::new(4, 4).unwrap();
    let read_nz = |_ctx: u32| -> Result<u32> { Ok(1008u32) };
    let dec = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let (_block, raw) =
        decode_block_at(&mut g, 0, 0, TransformType::Dct32x32, 0, 1, read_nz, dec).unwrap();
    assert_eq!(raw, 1008);
    for y in 0..4 {
        for x in 0..4 {
            assert_eq!(g.get(x, y).unwrap(), 63, "footprint cell ({x},{y})");
        }
    }
}

#[test]
fn round177_grid_interior_average_is_rounded_up() {
    // Listing C.13: (NonZeros(x, y-1) + NonZeros(x-1, y) + 1) >> 1.
    // Seed (1, 0) = 5, (0, 1) = 4, then check predicted(1, 1) = 5
    // ((5 + 4 + 1) >> 1).
    let mut g = NonZerosGrid::new(2, 2).unwrap();
    g.set(1, 0, 5).unwrap();
    g.set(0, 1, 4).unwrap();
    assert_eq!(g.predicted(1, 1).unwrap(), 5);
}

#[test]
fn round177_grid_block_ctx_threads_through_non_zeros_context() {
    // The `decode_block_at` driver computes
    //   NonZerosContext(predicted) = block_ctx
    //     + nb_block_ctx × predicted   (predicted < 8 branch)
    // and passes it as the `ctx` argument to `read_non_zeros`.
    // Verify by capturing the seen context against a known
    // predicted/value pair: at (0, 0) predicted = 32 → the
    // `predicted >= 8` branch fires, ctx = block_ctx + nb_block_ctx
    // × (4 + 32 Idiv 2) = block_ctx + nb_block_ctx × 20.
    let mut g = NonZerosGrid::new(1, 1).unwrap();
    let mut captured: Vec<u32> = Vec::new();
    let read_nz = |ctx: u32| -> Result<u32> {
        captured.push(ctx);
        Ok(0u32)
    };
    let dec = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let _ = decode_block_at(
        &mut g,
        0,
        0,
        TransformType::Dct8x8,
        /* block_ctx = */ 7,
        /* nb_block_ctx = */ 3,
        read_nz,
        dec,
    )
    .unwrap();
    // 7 + 3 × 20 = 67
    assert_eq!(captured, vec![67]);
}

#[test]
fn round177_grid_horizontal_chain_top_row_reads_left_only() {
    // y == 0 && x != 0 → PredictedNonZeros = NonZeros(x - 1, 0).
    // After decoding 5 then 9 then 4 at (0, 0)/(1, 0)/(2, 0) with
    // DCT8×8 (num_blocks = 1, so the stored value == raw non_zeros),
    // predicted at the next position equals the previous cell.
    let mut g = NonZerosGrid::new(4, 1).unwrap();
    for (x, nz) in [(0u32, 5u32), (1, 9), (2, 4)] {
        let read_nz = |_ctx: u32| -> Result<u32> { Ok(nz) };
        let dec = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let _ = decode_block_at(&mut g, x, 0, TransformType::Dct8x8, 0, 1, read_nz, dec).unwrap();
    }
    // predicted(1, 0) = NonZeros(0, 0) = 5.
    // predicted(2, 0) = NonZeros(1, 0) = 9.
    // predicted(3, 0) = NonZeros(2, 0) = 4.
    assert_eq!(g.predicted(1, 0).unwrap(), 5);
    assert_eq!(g.predicted(2, 0).unwrap(), 9);
    assert_eq!(g.predicted(3, 0).unwrap(), 4);
}

#[test]
fn round177_grid_vertical_chain_left_col_reads_above_only() {
    // x == 0 && y != 0 → PredictedNonZeros = NonZeros(0, y - 1).
    let mut g = NonZerosGrid::new(1, 4).unwrap();
    for (y, nz) in [(0u32, 6u32), (1, 11), (2, 2)] {
        let read_nz = |_ctx: u32| -> Result<u32> { Ok(nz) };
        let dec = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let _ = decode_block_at(&mut g, 0, y, TransformType::Dct8x8, 0, 1, read_nz, dec).unwrap();
    }
    assert_eq!(g.predicted(0, 1).unwrap(), 6);
    assert_eq!(g.predicted(0, 2).unwrap(), 11);
    assert_eq!(g.predicted(0, 3).unwrap(), 2);
}

#[test]
fn round177_grid_oob_position_errors() {
    let mut g = NonZerosGrid::new(2, 2).unwrap();
    let read_nz = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let dec = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let r = decode_block_at(
        &mut g,
        2, // x out of range
        0,
        TransformType::Dct8x8,
        0,
        1,
        read_nz,
        dec,
    );
    assert!(r.is_err());
}

#[test]
fn round177_grid_per_channel_grids_are_independent() {
    // The grid carries no channel id — round 177's per-channel
    // independence is achieved by holding one grid per channel.
    // Demonstrate that two grids of the same shape evolve
    // independently when fed different non_zeros streams.
    let mut gy = NonZerosGrid::new(2, 1).unwrap();
    let mut gx = NonZerosGrid::new(2, 1).unwrap();
    let read_y = |_ctx: u32| -> Result<u32> { Ok(11u32) };
    let read_x = |_ctx: u32| -> Result<u32> { Ok(3u32) };
    let dec = |_ctx: u32| -> Result<u32> { Ok(0u32) };

    let _ = decode_block_at(&mut gy, 0, 0, TransformType::Dct8x8, 0, 1, read_y, dec).unwrap();
    let _ = decode_block_at(&mut gx, 0, 0, TransformType::Dct8x8, 0, 1, read_x, dec).unwrap();
    assert_eq!(gy.get(0, 0).unwrap(), 11);
    assert_eq!(gx.get(0, 0).unwrap(), 3);
}

#[test]
fn round177_grid_full_grid_view_returns_row_major_buffer() {
    // Verify the row-major `cells()` accessor layout: writing to
    // (1, 0) and (0, 1) and (1, 1) on a 2×2 grid lands at indices
    // 1, 2, 3 respectively.
    let mut g = NonZerosGrid::new(2, 2).unwrap();
    g.set(1, 0, 10).unwrap();
    g.set(0, 1, 20).unwrap();
    g.set(1, 1, 30).unwrap();
    let cells = g.cells();
    assert_eq!(cells, &[0, 10, 20, 30]);
}
