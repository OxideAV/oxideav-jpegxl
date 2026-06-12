//! Round 183 — integration tests for the per-channel `NonZeros(x, y)`
//! grid container ([`oxideav_jpegxl::per_channel_non_zeros`]).
//!
//! Sibling integration to round 177's `round177_non_zeros_grid.rs`:
//! the single-channel grid is the per-position primitive, the
//! per-channel container is the per-channel routing primitive. Both
//! layers are pure-control-flow against FDIS §C.8.3 + Listing C.13 /
//! Listing C.14.

use oxideav_core::Result;
use oxideav_jpegxl::dct_select::TransformType;
use oxideav_jpegxl::per_channel_non_zeros::{PerChannelNonZerosGrids, DEFAULT_NUM_CHANNELS};

#[test]
fn r183_default_num_channels_constant_is_three() {
    // YCbCr / XYB has three channels; the constant must reflect that
    // for the integration callers that read it.
    assert_eq!(DEFAULT_NUM_CHANNELS, 3);
}

#[test]
fn r183_three_channel_uniform_construction() {
    let p = PerChannelNonZerosGrids::new_uniform(3, 8, 8).unwrap();
    assert_eq!(p.num_channels(), 3);
    for c in 0..3 {
        assert_eq!(p.grid(c).unwrap().width(), 8);
        assert_eq!(p.grid(c).unwrap().height(), 8);
    }
}

#[test]
fn r183_chroma_subsampled_construction_per_channel_dims() {
    // 4:2:0 conceptually: Y at 16×16 varblocks, Cb/Cr at 8×8.
    let p = PerChannelNonZerosGrids::new(&[(16, 16), (8, 8), (8, 8)]).unwrap();
    assert_eq!(p.grid(0).unwrap().width(), 16);
    assert_eq!(p.grid(1).unwrap().width(), 8);
    assert_eq!(p.grid(2).unwrap().width(), 8);
}

#[test]
fn r183_predicted_at_origin_is_thirty_two_per_channel() {
    // PredictedNonZeros(0, 0) = 32 per FDIS Listing C.13 prelude —
    // the per-channel container must propagate this through every
    // channel.
    let p = PerChannelNonZerosGrids::new_uniform(3, 4, 4).unwrap();
    for c in 0..p.num_channels() {
        assert_eq!(p.predicted(c, 0, 0).unwrap(), 32);
    }
}

#[test]
fn r183_per_channel_writes_are_isolated() {
    // Writing to channel 0 does not bleed into channel 1 or 2.
    let mut p = PerChannelNonZerosGrids::new_uniform(3, 2, 2).unwrap();
    p.set(0, 1, 1, 99).unwrap();
    assert_eq!(p.get(0, 1, 1).unwrap(), 99);
    assert_eq!(p.get(1, 1, 1).unwrap(), 0);
    assert_eq!(p.get(2, 1, 1).unwrap(), 0);
}

#[test]
fn r183_per_channel_predicted_horizontal_chain() {
    // Drive a horizontal raster walk on channel 1 only; verify
    // predicted(1, x, 0) reads back the prior cell on channel 1 and
    // unrelated channels remain pinned to the (0, 0) = 32 default.
    let mut p = PerChannelNonZerosGrids::new_uniform(3, 4, 1).unwrap();
    // Seed (0, 0) = 17 on channel 1.
    p.set(1, 0, 0, 17).unwrap();
    assert_eq!(p.predicted(1, 1, 0).unwrap(), 17);
    // Channel 0 / channel 2 still at default.
    assert_eq!(p.predicted(0, 1, 0).unwrap(), 0);
    assert_eq!(p.predicted(2, 1, 0).unwrap(), 0);
}

#[test]
fn r183_update_after_block_for_transform_dispatch() {
    // Per-channel TransformType dispatch. DCT8×8 / DCT16×16 / DCT32×32
    // num_blocks = 1 / 4 / 16; raw_non_zeros = 17 reduces to
    // 17 / 5 / 2 respectively. Grids sized 4×4 for the largest
    // footprint — per the §C.8.3 "for each block in the current
    // varblock" prose every covered cell stores the value.
    let mut p = PerChannelNonZerosGrids::new_uniform(3, 4, 4).unwrap();
    let v0 = p
        .update_after_block_for_transform(0, 0, 0, 17, TransformType::Dct8x8)
        .unwrap();
    let v1 = p
        .update_after_block_for_transform(1, 0, 0, 17, TransformType::Dct16x16)
        .unwrap();
    let v2 = p
        .update_after_block_for_transform(2, 0, 0, 17, TransformType::Dct32x32)
        .unwrap();
    assert_eq!(v0, 17);
    assert_eq!(v1, 5);
    assert_eq!(v2, 2);
    // Footprint writeback: channel 1's DCT16×16 filled its 2×2
    // footprint; channel 2's DCT32×32 the full 4×4; channel 0's
    // DCT8×8 only (0, 0).
    assert_eq!(p.get(0, 1, 1).unwrap(), 0);
    assert_eq!(p.get(1, 1, 1).unwrap(), 5);
    assert_eq!(p.get(2, 3, 3).unwrap(), 2);
}

#[test]
fn r183_typed_driver_routes_per_channel() {
    // Drive decode_block_at_for_channel against channel 2 only;
    // channel 0 and 1 grids remain zero-initialised.
    let mut p = PerChannelNonZerosGrids::new_uniform(3, 2, 2).unwrap();
    let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(11u32) };
    let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let (_decoded, raw_non_zeros) = p
        .decode_block_at_for_channel(
            2,
            0,
            0,
            TransformType::Dct8x8,
            /* block_ctx = */ 0,
            /* nb_block_ctx = */ 1,
            read_non_zeros,
            decode_symbol,
        )
        .unwrap();
    assert_eq!(raw_non_zeros, 11);
    assert_eq!(p.get(2, 0, 0).unwrap(), 11);
    assert_eq!(p.get(0, 0, 0).unwrap(), 0);
    assert_eq!(p.get(1, 0, 0).unwrap(), 0);
}

#[test]
fn r183_typed_driver_propagates_predicted_to_next_position() {
    // After decode_block_at_for_channel at (0, 0) on channel 1,
    // the predicted value at (1, 0) on channel 1 reads back the
    // post-update cell from (0, 0).
    let mut p = PerChannelNonZerosGrids::new_uniform(3, 2, 2).unwrap();
    let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(5u32) };
    let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let _ = p
        .decode_block_at_for_channel(
            1,
            0,
            0,
            TransformType::Dct8x8,
            0,
            1,
            read_non_zeros,
            decode_symbol,
        )
        .unwrap();
    assert_eq!(p.predicted(1, 1, 0).unwrap(), 5);
}

#[test]
fn r183_typed_driver_oob_position_errors() {
    // (x, y) past the grid must error cleanly without panicking.
    let mut p = PerChannelNonZerosGrids::new_uniform(3, 2, 2).unwrap();
    let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let r = p.decode_block_at_for_channel(
        0,
        2,
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
fn r183_two_step_three_channel_raster_walk() {
    // Concurrent raster walks on all three channels at the (0, 0) /
    // (1, 0) positions. Each channel uses its own raw_non_zeros
    // sequence and the post-update + predicted values are
    // channel-keyed.
    let mut p = PerChannelNonZerosGrids::new_uniform(3, 2, 1).unwrap();
    let nz_at_origin = [4u32, 12, 20];
    let nz_at_one = [6u32, 18, 30];
    // (0, 0) on each channel.
    for c in 0..3u32 {
        let nz = nz_at_origin[c as usize];
        let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(nz) };
        let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let _ = p
            .decode_block_at_for_channel(
                c,
                0,
                0,
                TransformType::Dct8x8,
                0,
                1,
                read_non_zeros,
                decode_symbol,
            )
            .unwrap();
    }
    // (1, 0) on each channel.
    for c in 0..3u32 {
        let nz = nz_at_one[c as usize];
        let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(nz) };
        let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let _ = p
            .decode_block_at_for_channel(
                c,
                1,
                0,
                TransformType::Dct8x8,
                0,
                1,
                read_non_zeros,
                decode_symbol,
            )
            .unwrap();
    }
    // After both steps, each channel's grid stores the per-step
    // values at their per-position cells; cross-channel isolation is
    // preserved.
    for c in 0..3u32 {
        assert_eq!(p.get(c, 0, 0).unwrap(), nz_at_origin[c as usize]);
        assert_eq!(p.get(c, 1, 0).unwrap(), nz_at_one[c as usize]);
    }
}

#[test]
fn r183_oob_channel_errors_on_every_entry_point() {
    let mut p = PerChannelNonZerosGrids::new_uniform(3, 2, 2).unwrap();
    assert!(p.grid(99).is_err());
    assert!(p.grid_mut(99).is_err());
    assert!(p.predicted(99, 0, 0).is_err());
    assert!(p.get(99, 0, 0).is_err());
    assert!(p.set(99, 0, 0, 5).is_err());
    assert!(p.update_after_block(99, 0, 0, 1, 1).is_err());
    assert!(p
        .update_after_block_for_transform(99, 0, 0, 1, TransformType::Dct8x8)
        .is_err());
    let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    assert!(p
        .decode_block_at_for_channel(
            99,
            0,
            0,
            TransformType::Dct8x8,
            0,
            1,
            read_non_zeros,
            decode_symbol
        )
        .is_err());
}
