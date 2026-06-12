//! Round 190 — integration tests for the per-pass `NonZeros(x, y)`
//! grid container ([`oxideav_jpegxl::per_pass_non_zeros`]).
//!
//! Sibling integration to round 177's `round177_non_zeros_grid.rs`
//! (single-channel) and round 183's
//! `round183_per_channel_non_zeros.rs` (per-channel). Round 190 layers
//! per-pass routing above the per-channel container: the storage layer
//! for FDIS §C.8.3's per-pass `NonZeros(x, y)` bookkeeping. Pure
//! control-flow primitive — no bit reads, no spec re-derivation, no
//! histogram materialisation.

use oxideav_core::Result;
use oxideav_jpegxl::dct_select::TransformType;
use oxideav_jpegxl::per_pass_non_zeros::PerPassNonZerosGrids;

#[test]
fn r190_new_uniform_two_passes_three_channels() {
    let p = PerPassNonZerosGrids::new_uniform(2, 3, 8, 8).unwrap();
    assert_eq!(p.num_passes(), 2);
    for pp in 0..p.num_passes() {
        assert_eq!(p.pass(pp).unwrap().num_channels(), 3);
    }
}

#[test]
fn r190_per_pass_chroma_subsampled_construction() {
    // 4:2:0-ish per-pass: each pass owns its own Y / Cb / Cr trio.
    let pass0: [(u32, u32); 3] = [(16, 16), (8, 8), (8, 8)];
    let pass1: [(u32, u32); 3] = [(16, 16), (8, 8), (8, 8)];
    let p = PerPassNonZerosGrids::new(&[&pass0[..], &pass1[..]]).unwrap();
    assert_eq!(p.num_passes(), 2);
    assert_eq!(p.pass(0).unwrap().grid(0).unwrap().width(), 16);
    assert_eq!(p.pass(1).unwrap().grid(1).unwrap().width(), 8);
    assert_eq!(p.pass(1).unwrap().grid(2).unwrap().height(), 8);
}

#[test]
fn r190_empty_pass_list_rejected() {
    let r = PerPassNonZerosGrids::new(&[]);
    assert!(r.is_err());
}

#[test]
fn r190_predicted_origin_is_thirty_two_for_every_pass() {
    let p = PerPassNonZerosGrids::new_uniform(3, 3, 4, 4).unwrap();
    for pp in 0..3 {
        for c in 0..3 {
            assert_eq!(p.predicted(pp, c, 0, 0).unwrap(), 32);
        }
    }
}

#[test]
fn r190_per_pass_per_channel_get_set_isolation() {
    // Write to (pass 0, channel 1, (0, 0)) must not leak into pass 1
    // or other channels of pass 0.
    let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
    p.set(0, 1, 0, 0, 42).unwrap();
    assert_eq!(p.get(0, 1, 0, 0).unwrap(), 42);
    assert_eq!(p.get(0, 0, 0, 0).unwrap(), 0);
    assert_eq!(p.get(0, 2, 0, 0).unwrap(), 0);
    assert_eq!(p.get(1, 1, 0, 0).unwrap(), 0);
}

#[test]
fn r190_update_after_block_per_pass_dct16x16_ceil() {
    let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
    let v = p.update_after_block(1, 1, 0, 0, 17, 4).unwrap();
    assert_eq!(v, 5, "ceil(17/4) = 5");
    assert_eq!(p.get(1, 1, 0, 0).unwrap(), 5);
    // Other pass / channels untouched.
    assert_eq!(p.get(0, 1, 0, 0).unwrap(), 0);
    assert_eq!(p.get(1, 0, 0, 0).unwrap(), 0);
}

#[test]
fn r190_update_after_block_for_transform_dispatches_per_pass() {
    // Grids sized 4×4 for the largest (DCT32×32) footprint — per the
    // §C.8.3 "for each block in the current varblock" prose every
    // covered cell stores the ceiling-divided value.
    let mut p = PerPassNonZerosGrids::new_uniform(3, 3, 4, 4).unwrap();
    let v0 = p
        .update_after_block_for_transform(0, 0, 0, 0, 17, TransformType::Dct8x8)
        .unwrap();
    let v1 = p
        .update_after_block_for_transform(1, 1, 0, 0, 17, TransformType::Dct16x16)
        .unwrap();
    let v2 = p
        .update_after_block_for_transform(2, 2, 0, 0, 17, TransformType::Dct32x32)
        .unwrap();
    assert_eq!(v0, 17);
    assert_eq!(v1, 5);
    assert_eq!(v2, 2);
    // Footprint writeback: (pass 1, channel 1) DCT16×16 filled the
    // 2×2 footprint; (pass 2, channel 2) DCT32×32 the full 4×4;
    // (pass 0, channel 0) DCT8×8 only (0, 0).
    assert_eq!(p.get(0, 0, 1, 1).unwrap(), 0);
    assert_eq!(p.get(1, 1, 1, 1).unwrap(), 5);
    assert_eq!(p.get(2, 2, 3, 3).unwrap(), 2);
}

#[test]
fn r190_decode_block_at_for_pass_channel_per_pass_routing() {
    let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
    let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(7u32) };
    let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let (_decoded, raw_non_zeros) = p
        .decode_block_at_for_pass_channel(
            1,
            2,
            0,
            0,
            TransformType::Dct8x8,
            0,
            1,
            read_non_zeros,
            decode_symbol,
        )
        .unwrap();
    assert_eq!(raw_non_zeros, 7);
    assert_eq!(p.get(1, 2, 0, 0).unwrap(), 7);
    // Other passes / channels still zero.
    assert_eq!(p.get(0, 2, 0, 0).unwrap(), 0);
    assert_eq!(p.get(1, 0, 0, 0).unwrap(), 0);
    assert_eq!(p.get(1, 1, 0, 0).unwrap(), 0);
}

#[test]
fn r190_decode_block_at_for_pass_channel_oob_pass_errors() {
    let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
    let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let r = p.decode_block_at_for_pass_channel(
        2,
        0,
        0,
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
fn r190_per_pass_decode_chain_two_step_raster_walk() {
    // Two passes, three channels, walk (0, 0) → (1, 0) with distinct
    // per-pass per-channel raw_non_zeros. Verify the per-pass grids
    // evolve independently and the predicted reads back the matching
    // pass's own history.
    let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 1).unwrap();

    // Step 1 — both passes at (0, 0).
    for (pp, channels) in [(0u32, [4u32, 8, 12]), (1, [3u32, 6, 9])] {
        for (c, nz) in channels.iter().enumerate() {
            let nz = *nz;
            let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(nz) };
            let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
            let _ = p
                .decode_block_at_for_pass_channel(
                    pp,
                    c as u32,
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
    }
    // Pass 0's predicted at (1, 0) reads pass 0's own values.
    assert_eq!(p.predicted(0, 0, 1, 0).unwrap(), 4);
    assert_eq!(p.predicted(0, 1, 1, 0).unwrap(), 8);
    assert_eq!(p.predicted(0, 2, 1, 0).unwrap(), 12);
    // Pass 1's predicted reads pass 1's history.
    assert_eq!(p.predicted(1, 0, 1, 0).unwrap(), 3);
    assert_eq!(p.predicted(1, 1, 1, 0).unwrap(), 6);
    assert_eq!(p.predicted(1, 2, 1, 0).unwrap(), 9);
}

#[test]
fn r190_per_pass_independent_dct16x16_evolution() {
    // Pass 0 uses DCT16×16 (ceil-divide by 4); pass 1 uses DCT8×8
    // (identity). After raw_non_zeros = 17 on (channel 0, (0, 0)) of
    // each pass, the stored cells are 5 on pass 0 (full 2×2
    // footprint per the §C.8.3 prose) and 17 on pass 1 (top-left
    // only — DCT8×8's footprint is a single cell).
    let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
    // Pass 0 DCT16×16.
    let read_a = |_ctx: u32| -> Result<u32> { Ok(17u32) };
    let decode_a = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let _ = p
        .decode_block_at_for_pass_channel(
            0,
            0,
            0,
            0,
            TransformType::Dct16x16,
            0,
            1,
            read_a,
            decode_a,
        )
        .unwrap();
    // Pass 1 DCT8×8.
    let read_b = |_ctx: u32| -> Result<u32> { Ok(17u32) };
    let decode_b = |_ctx: u32| -> Result<u32> { Ok(0u32) };
    let _ = p
        .decode_block_at_for_pass_channel(1, 0, 0, 0, TransformType::Dct8x8, 0, 1, read_b, decode_b)
        .unwrap();
    for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        assert_eq!(
            p.get(0, 0, x, y).unwrap(),
            5,
            "DCT16×16: ceil(17/4) = 5 on footprint cell ({x},{y})"
        );
    }
    assert_eq!(p.get(1, 0, 0, 0).unwrap(), 17, "DCT8×8: identity");
    assert_eq!(p.get(1, 0, 1, 0).unwrap(), 0, "DCT8×8 footprint is 1×1");
}

#[test]
fn r190_ragged_per_pass_channel_counts_supported() {
    // A "DC-only" first pass with one channel followed by a full
    // three-channel main pass — the per-pass container does not
    // enforce a uniform channel count across passes.
    let pass0: [(u32, u32); 1] = [(8, 8)];
    let pass1: [(u32, u32); 3] = [(8, 8), (8, 8), (8, 8)];
    let p = PerPassNonZerosGrids::new(&[&pass0[..], &pass1[..]]).unwrap();
    assert_eq!(p.pass(0).unwrap().num_channels(), 1);
    assert_eq!(p.pass(1).unwrap().num_channels(), 3);
    // Channel index 1 on pass 0 (out-of-range) errors cleanly.
    assert!(p.get(0, 1, 0, 0).is_err());
    // Channel index 1 on pass 1 (in range) works.
    assert_eq!(p.get(1, 1, 0, 0).unwrap(), 0);
}

#[test]
fn r190_invalid_construction_propagates() {
    // Empty per-channel slice on a pass.
    let empty: &[(u32, u32)] = &[];
    let r = PerPassNonZerosGrids::new(&[empty]);
    assert!(r.is_err());
    // Zero-dim on a pass.
    let bad: &[(u32, u32)] = &[(8, 0)];
    let r2 = PerPassNonZerosGrids::new(&[bad]);
    assert!(r2.is_err());
    // Zero passes via new_uniform.
    let r3 = PerPassNonZerosGrids::new_uniform(0, 3, 8, 8);
    assert!(r3.is_err());
}
