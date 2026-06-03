//! Round 221 integration tests — three-channel per-LfGroup
//! varblock decode driver (ISO/IEC FDIS 18181-1:2021 §C.8.3 Listing
//! C.13 + §I.2.2 LfGlobal `HfBlockContext` bundle).
//!
//! These exercise the
//! [`block_context_resolver::decode_varblocks_three_channels_with_resolver`]
//! surface end-to-end against the round-13
//! [`dct_select::DctSelectGrid`], the round-190
//! [`per_pass_non_zeros::PerPassNonZerosGrids`], the round-208
//! [`varblock_walk::Varblock`] descriptor, and the round-214
//! [`block_context_resolver::BlockContextResolver`].
//!
//! Pure-control-flow primitive: no bit reads, no histogram
//! materialisation. The per-channel ANS closures abstract over the
//! §C.7.2 entropy decode (#799 DOCS-GAP).

use oxideav_jpegxl::block_context_resolver::{
    decode_varblocks_three_channels_with_resolver, BlockContextResolver,
};
use oxideav_jpegxl::dct_select::{derive_dct_select, TransformType};
use oxideav_jpegxl::lf_global::HfBlockContext;
use oxideav_jpegxl::lf_group::HfMetadata;
use oxideav_jpegxl::per_pass_non_zeros::PerPassNonZerosGrids;

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

fn default_hbc() -> HfBlockContext {
    HfBlockContext {
        used_default: true,
        block_ctx_map: HfBlockContext::DEFAULT_BLOCK_CTX_MAP.to_vec(),
        nb_block_ctx: (*HfBlockContext::DEFAULT_BLOCK_CTX_MAP.iter().max().unwrap() as u32) + 1,
        lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
        qf_thresholds: Vec::new(),
    }
}

#[test]
fn r221_three_channel_single_varblock_round_trip() {
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
    let out = decode_varblocks_three_channels_with_resolver(
        &grid,
        &mut nz,
        0,
        &resolver,
        |_| Ok([0, 0, 0]),
        |_, _| Ok(0),
        |_, _| Ok(0),
    )
    .unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0.transform, TransformType::Dct8x8);
    assert_eq!(out[0].2, [0, 0, 0]);
}

#[test]
fn r221_three_channel_raster_order_4x4_grid() {
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0; 32], 16, 16);
    let grid = derive_dct_select(&hf, 32, 32).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 4, 4).unwrap();
    let out = decode_varblocks_three_channels_with_resolver(
        &grid,
        &mut nz,
        0,
        &resolver,
        |_| Ok([0, 0, 0]),
        |_, _| Ok(0),
        |_, _| Ok(0),
    )
    .unwrap();
    // 4×4 grid of DCT8×8 → 16 varblocks in raster order.
    assert_eq!(out.len(), 16);
    let mut expected = Vec::with_capacity(16);
    for y in 0..4u32 {
        for x in 0..4u32 {
            expected.push((x, y));
        }
    }
    let observed: Vec<(u32, u32)> = out.iter().map(|e| (e.0.x, e.0.y)).collect();
    assert_eq!(observed, expected);
}

#[test]
fn r221_three_channel_dct16x16_single_block() {
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![4, 0], 1, 1);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
    let out = decode_varblocks_three_channels_with_resolver(
        &grid,
        &mut nz,
        0,
        &resolver,
        |_| Ok([0, 0, 0]),
        |_, _| Ok(0),
        |_, _| Ok(0),
    )
    .unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0.transform, TransformType::Dct16x16);
}

#[test]
fn r221_three_channel_qdc_shared_one_call_per_varblock() {
    // The qdc closure must fire exactly once per varblock, not once
    // per (varblock, channel) pair.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
    let mut qdc_calls = 0u32;
    let _ = decode_varblocks_three_channels_with_resolver(
        &grid,
        &mut nz,
        0,
        &resolver,
        |_| {
            qdc_calls += 1;
            Ok([0, 0, 0])
        },
        |_, _| Ok(0),
        |_, _| Ok(0),
    )
    .unwrap();
    assert_eq!(qdc_calls, 4);
}

#[test]
fn r221_three_channel_per_channel_non_zeros_written_back() {
    // Each channel's `update_after_block` writes non_zeros into the
    // per-channel grid at the varblock's top-left. Verify per-channel
    // values are independently routed.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
    let _ = decode_varblocks_three_channels_with_resolver(
        &grid,
        &mut nz,
        0,
        &resolver,
        |_| Ok([0, 0, 0]),
        |channel, _pred| match channel {
            0 => Ok(10),
            1 => Ok(20),
            2 => Ok(30),
            _ => unreachable!(),
        },
        |_, _| Ok(0),
    )
    .unwrap();
    // DCT8×8 num_blocks = 1; update_after_block writes raw_non_zeros.
    assert_eq!(nz.get(0, 0, 0, 0).unwrap(), 10);
    assert_eq!(nz.get(0, 1, 0, 0).unwrap(), 20);
    assert_eq!(nz.get(0, 2, 0, 0).unwrap(), 30);
}

#[test]
fn r221_three_channel_pass_index_routes_to_correct_pass() {
    // Two passes; write to pass = 1 only. pass = 0 untouched.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
    let _ = decode_varblocks_three_channels_with_resolver(
        &grid,
        &mut nz,
        1,
        &resolver,
        |_| Ok([0, 0, 0]),
        |_, _| Ok(4),
        |_, _| Ok(0),
    )
    .unwrap();
    // pass 0 still zero.
    for c in 0..3 {
        assert_eq!(nz.get(0, c, 0, 0).unwrap(), 0);
    }
    // pass 1 has 4 per channel.
    for c in 0..3 {
        assert_eq!(nz.get(1, c, 0, 0).unwrap(), 4);
    }
}

#[test]
fn r221_three_channel_qdc_error_aborts_before_any_channel_read() {
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
    let mut nz_calls = 0u32;
    let r = decode_varblocks_three_channels_with_resolver(
        &grid,
        &mut nz,
        0,
        &resolver,
        |_| Err(oxideav_core::Error::InvalidData("qdc fail".into())),
        |_, _| {
            nz_calls += 1;
            Ok(0)
        },
        |_, _| Ok(0),
    );
    assert!(r.is_err());
    assert_eq!(nz_calls, 0);
}

#[test]
fn r221_three_channel_x_error_aborts_before_y_and_b() {
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
    let mut per_channel = [0u32; 3];
    let r = decode_varblocks_three_channels_with_resolver(
        &grid,
        &mut nz,
        0,
        &resolver,
        |_| Ok([0, 0, 0]),
        |channel, _pred| {
            per_channel[channel as usize] += 1;
            if channel == 0 {
                Err(oxideav_core::Error::InvalidData("x fail".into()))
            } else {
                Ok(0)
            }
        },
        |_, _| Ok(0),
    );
    assert!(r.is_err());
    assert_eq!(per_channel[0], 1);
    assert_eq!(per_channel[1], 0);
    assert_eq!(per_channel[2], 0);
}

#[test]
fn r221_three_channel_mixed_transforms() {
    // 16×16 grid: DCT16×8 + 2 DCT8×8.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![6, 0, 0, 0, 0, 0], 3, 3);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
    let out = decode_varblocks_three_channels_with_resolver(
        &grid,
        &mut nz,
        0,
        &resolver,
        |_| Ok([0, 0, 0]),
        |_, _| Ok(0),
        |_, _| Ok(0),
    )
    .unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].0.transform, TransformType::Dct16x8);
    assert_eq!((out[0].0.x, out[0].0.y), (0, 0));
    assert_eq!(out[1].0.transform, TransformType::Dct8x8);
    assert_eq!((out[1].0.x, out[1].0.y), (1, 0));
    assert_eq!(out[2].0.transform, TransformType::Dct8x8);
    assert_eq!((out[2].0.x, out[2].0.y), (1, 1));
}

#[test]
fn r221_three_channel_channel_order_x_y_b() {
    // Verify channel order matches §C.8.3 prose (X = 0 first,
    // Y = 1 second, B = 2 last). Capture the channel arg sequence.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
    let mut seen: Vec<u32> = Vec::new();
    let _ = decode_varblocks_three_channels_with_resolver(
        &grid,
        &mut nz,
        0,
        &resolver,
        |_| Ok([0, 0, 0]),
        |channel, _| {
            seen.push(channel);
            Ok(0)
        },
        |_, _| Ok(0),
    )
    .unwrap();
    assert_eq!(seen, vec![0, 1, 2]);
}

#[test]
fn r221_three_channel_qdc_propagates_through_to_resolver() {
    // Custom HfBlockContext with a qf_threshold so that hf_mul
    // differences move idx into different cells. Verify per-channel
    // resolution consistency using a 2-element map.
    let hbc = HfBlockContext {
        used_default: false,
        block_ctx_map: vec![1; 78],
        nb_block_ctx: 2,
        lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
        qf_thresholds: vec![5],
    };
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
    let out = decode_varblocks_three_channels_with_resolver(
        &grid,
        &mut nz,
        0,
        &resolver,
        |_| Ok([0, 0, 0]),
        |_, _| Ok(0),
        |_, _| Ok(0),
    )
    .unwrap();
    assert_eq!(out.len(), 1);
    // All three channels resolved (no panic / no error).
    assert_eq!(out[0].2, [0, 0, 0]);
}

#[test]
fn r221_three_channel_dct16x16_with_non_zeros_round_trip() {
    // DCT16×16: num_blocks = 4, size = 256. Per-channel non_zeros
    // = 4 (one per LLF cell) decrements with non-zero ucoeffs.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![4, 0], 1, 1);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
    let mut per_channel_decode_calls = [0u32; 3];
    let out = decode_varblocks_three_channels_with_resolver(
        &grid,
        &mut nz,
        0,
        &resolver,
        |_| Ok([0, 0, 0]),
        |_channel, _pred| Ok(4),
        |channel, _ctx| {
            per_channel_decode_calls[channel as usize] += 1;
            Ok(7) // non-zero
        },
    )
    .unwrap();
    assert_eq!(out.len(), 1);
    // raw_non_zeros = 4 per channel.
    assert_eq!(out[0].2, [4, 4, 4]);
    // Each channel's decode_symbol called 4 times (loop runs until
    // non_zeros == 0).
    assert_eq!(per_channel_decode_calls, [4, 4, 4]);
    // NonZeros(0, 0) stored after update_after_block_for_transform:
    // for DCT16×16 num_blocks = 4 → (4 + 3) / 4 = 1.
    assert_eq!(nz.get(0, 0, 0, 0).unwrap(), 1);
    assert_eq!(nz.get(0, 1, 0, 0).unwrap(), 1);
    assert_eq!(nz.get(0, 2, 0, 0).unwrap(), 1);
}
