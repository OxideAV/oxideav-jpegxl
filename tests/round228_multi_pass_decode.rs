//! Round 228 — per-LfGroup multi-pass three-channel varblock decode
//! driver integration tests.
//!
//! Covers the outer pass loop that round 228 layers above the
//! round-221 inner per-pass three-channel driver. Each test pins one
//! invariant of the per-pass routing and verifies the per-pass
//! per-channel `NonZeros(x, y)` writeback isolation that round-190's
//! [`per_pass_non_zeros::PerPassNonZerosGrids`] container guarantees
//! on the storage side.
//!
//! All tests are pure control-flow exercises — no fixture decode,
//! no bit reads. The round-228 driver is a histogram-blind primitive
//! that defers per-channel ANS to the caller's closures and per-pass
//! `hfp` selection to the caller's pass-aware closure dispatch.

use oxideav_jpegxl::block_context_resolver::BlockContextResolver;
use oxideav_jpegxl::dct_select::{derive_dct_select, TransformType};
use oxideav_jpegxl::lf_global::HfBlockContext;
use oxideav_jpegxl::lf_group::HfMetadata;
use oxideav_jpegxl::multi_pass_decode::{
    count_decoded_blocks, decode_multi_pass_three_channels_with_resolver,
};
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
        qf_thresholds: vec![],
        lf_thresholds: [vec![], vec![], vec![]],
        block_ctx_map: vec![
            7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 9, 9, 10, 11, 12, 13, 14, 0, 0, 0, 0, 7,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        nb_block_ctx: 15,
    }
}

#[test]
fn r228_two_pass_single_dct8x8_per_pass_writeback_isolated() {
    // num_passes = 2, single DCT8×8 varblock. Pass 0 writes
    // per-channel raw_non_zeros [3, 5, 7]; pass 1 writes [4, 6, 8].
    // Verify writebacks land on the matching pass index.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
    let out = decode_multi_pass_three_channels_with_resolver(
        &grid,
        &mut nz,
        &resolver,
        |_p, _vb| Ok([0, 0, 0]),
        |p, c, _pred| {
            if p == 0 {
                Ok(3 + c * 2)
            } else {
                Ok(4 + c * 2)
            }
        },
        |_p, _c, _coef| Ok(0),
    )
    .unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0][0].2, [3, 5, 7]);
    assert_eq!(out[1][0].2, [4, 6, 8]);
    // Writeback: pass 0 stores 3/5/7 at (0,0); pass 1 stores
    // 4/6/8 — neither leaks into the other.
    assert_eq!(nz.get(0, 0, 0, 0).unwrap(), 3);
    assert_eq!(nz.get(0, 1, 0, 0).unwrap(), 5);
    assert_eq!(nz.get(0, 2, 0, 0).unwrap(), 7);
    assert_eq!(nz.get(1, 0, 0, 0).unwrap(), 4);
    assert_eq!(nz.get(1, 1, 0, 0).unwrap(), 6);
    assert_eq!(nz.get(1, 2, 0, 0).unwrap(), 8);
}

#[test]
fn r228_two_pass_2x2_raster_order_per_pass() {
    // 2×2 DCT8×8 grid, 2 passes. Each pass visits 4 varblocks
    // in raster order.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
    let out = decode_multi_pass_three_channels_with_resolver(
        &grid,
        &mut nz,
        &resolver,
        |_p, _vb| Ok([0, 0, 0]),
        |_p, _c, _pred| Ok(0),
        |_p, _c, _coef| Ok(0),
    )
    .unwrap();
    assert_eq!(out.len(), 2);
    for pass_out in &out {
        let layout: Vec<(u32, u32)> = pass_out.iter().map(|t| (t.0.x, t.0.y)).collect();
        assert_eq!(layout, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
    }
}

#[test]
fn r228_qdc_closure_call_count_per_pass() {
    // 2×2 grid, 3 passes → 4 varblocks × 3 passes = 12 qdc calls.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(3, 3, 2, 2).unwrap();
    let mut qdc_calls_per_pass: [u32; 3] = [0; 3];
    let _ = decode_multi_pass_three_channels_with_resolver(
        &grid,
        &mut nz,
        &resolver,
        |p, _vb| {
            qdc_calls_per_pass[p as usize] += 1;
            Ok([0, 0, 0])
        },
        |_p, _c, _pred| Ok(0),
        |_p, _c, _coef| Ok(0),
    )
    .unwrap();
    assert_eq!(qdc_calls_per_pass, [4, 4, 4]);
}

#[test]
fn r228_three_pass_strict_pass_order() {
    // The pass loop must visit pass 0 → 1 → 2 in strict order; an
    // error in pass 1 must abort before pass 2.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(3, 3, 1, 1).unwrap();
    let mut visited: Vec<u32> = Vec::new();
    let r = decode_multi_pass_three_channels_with_resolver(
        &grid,
        &mut nz,
        &resolver,
        |p, _vb| {
            visited.push(p);
            if p == 1 {
                Err(oxideav_core::Error::InvalidData(
                    "pass-1 simulated failure".into(),
                ))
            } else {
                Ok([0, 0, 0])
            }
        },
        |_p, _c, _pred| Ok(0),
        |_p, _c, _coef| Ok(0),
    );
    assert!(r.is_err());
    // visited[0] = 0, visited[1] = 1 → pass 2 never started.
    assert_eq!(visited, vec![0, 1]);
}

#[test]
fn r228_pass_index_threaded_to_read_non_zeros_closure() {
    // Verify the read_non_zeros closure receives the pass index
    // as the first argument by emitting a pass-distinct value.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(4, 3, 1, 1).unwrap();
    let mut observed: Vec<u32> = Vec::new();
    let _ = decode_multi_pass_three_channels_with_resolver(
        &grid,
        &mut nz,
        &resolver,
        |_p, _vb| Ok([0, 0, 0]),
        |p, c, _pred| {
            observed.push(p * 10 + c);
            Ok(0)
        },
        |_p, _c, _coef| Ok(0),
    )
    .unwrap();
    // 4 passes × 3 channels = 12 calls. Within each pass the
    // per-varblock channel decode order is Y, X, then B per the
    // §C.8.3 prose (channel indices 1, 0, 2): (pass=0, c=1,0,2)
    // then (pass=1, c=1,0,2) etc.
    assert_eq!(observed.len(), 12);
    assert_eq!(observed, vec![1, 0, 2, 11, 10, 12, 21, 20, 22, 31, 30, 32]);
}

#[test]
fn r228_pass_index_threaded_to_decode_symbol_closure() {
    // Symmetric check for decode_symbol: pass index reaches the
    // closure. With read_non_zeros = 1 and decode_symbol always
    // returning a non-zero (1), the inner loop emits exactly one
    // decode_symbol call per channel before non_zeros hits 0 and
    // the loop exits early.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
    let mut symbol_calls_per_pass: [u32; 2] = [0; 2];
    let _ = decode_multi_pass_three_channels_with_resolver(
        &grid,
        &mut nz,
        &resolver,
        |_p, _vb| Ok([0, 0, 0]),
        |_p, _c, _pred| Ok(1),
        |p, _c, _coef| {
            symbol_calls_per_pass[p as usize] += 1;
            // Non-zero ucoeff → decrements non_zeros from 1 to 0,
            // exits the inner loop after one call per channel.
            Ok(1)
        },
    )
    .unwrap();
    // 1 call per channel × 3 channels × 1 varblock = 3 per pass.
    assert_eq!(symbol_calls_per_pass, [3, 3]);
}

#[test]
fn r228_single_pass_dct16x16_preserved() {
    // 1 pass, 1 DCT16×16 varblock — pass-loop pass-through.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    // transform-index 4 = Dct16×16 (2×2 cells), single varblock.
    let hf = make_hf(vec![4, 0], 1, 1);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
    let out = decode_multi_pass_three_channels_with_resolver(
        &grid,
        &mut nz,
        &resolver,
        |_p, _vb| Ok([0, 0, 0]),
        |_p, _c, _pred| Ok(0),
        |_p, _c, _coef| Ok(0),
    )
    .unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), 1);
    assert_eq!(out[0][0].0.transform, TransformType::Dct16x16);
}

#[test]
fn r228_count_decoded_blocks_helper() {
    let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    assert_eq!(count_decoded_blocks(&grid, 0).unwrap(), 0);
    assert_eq!(count_decoded_blocks(&grid, 1).unwrap(), 4);
    assert_eq!(count_decoded_blocks(&grid, 2).unwrap(), 8);
    assert_eq!(count_decoded_blocks(&grid, 5).unwrap(), 20);
}

#[test]
fn r228_per_pass_predicted_non_zeros_default_branch() {
    // Initial PredictedNonZeros(0, 0) = 32 across every pass and
    // channel — this is round-190's per-pass invariant; r228's
    // outer loop must preserve it.
    let nz = PerPassNonZerosGrids::new_uniform(3, 3, 1, 1).unwrap();
    for p in 0..3u32 {
        for c in 0..3u32 {
            assert_eq!(nz.predicted(p, c, 0, 0).unwrap(), 32);
        }
    }
}

#[test]
fn r228_qdc_value_propagation_through_pass_loop() {
    // The qdc[3] triple returned by the per-pass closure must be
    // visible to the resolver invocation within that pass.
    // Default-branch resolver invariance collapses qdc, so we
    // verify by capturing the qdc triple seen at closure-time.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(3, 3, 1, 1).unwrap();
    let mut qdc_seen: Vec<[i32; 3]> = Vec::new();
    let _ = decode_multi_pass_three_channels_with_resolver(
        &grid,
        &mut nz,
        &resolver,
        |p, _vb| {
            let v = [p as i32 + 100, p as i32 + 200, p as i32 + 300];
            qdc_seen.push(v);
            Ok(v)
        },
        |_p, _c, _pred| Ok(0),
        |_p, _c, _coef| Ok(0),
    )
    .unwrap();
    assert_eq!(
        qdc_seen,
        vec![[100, 200, 300], [101, 201, 301], [102, 202, 302]]
    );
}

#[test]
fn r228_mixed_transform_2_pass_layout_consistent() {
    // 2 passes × (DCT16×8 + 2 DCT8×8) layout. Both passes must
    // see the same shape.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![6, 0, 0, 0, 0, 0], 3, 3);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
    let out = decode_multi_pass_three_channels_with_resolver(
        &grid,
        &mut nz,
        &resolver,
        |_p, _vb| Ok([0, 0, 0]),
        |_p, _c, _pred| Ok(0),
        |_p, _c, _coef| Ok(0),
    )
    .unwrap();
    for pass_out in &out {
        assert_eq!(pass_out.len(), 3);
        assert_eq!(pass_out[0].0.transform, TransformType::Dct16x8);
        assert_eq!(pass_out[1].0.transform, TransformType::Dct8x8);
        assert_eq!(pass_out[2].0.transform, TransformType::Dct8x8);
    }
}

#[test]
fn r228_inner_driver_error_propagates_through_outer_loop() {
    // Pass 0 succeeds; pass 1's first per-channel decode_symbol
    // closure errors mid-varblock. The outer driver propagates
    // the error and returns it cleanly.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
    let r = decode_multi_pass_three_channels_with_resolver(
        &grid,
        &mut nz,
        &resolver,
        |_p, _vb| Ok([0, 0, 0]),
        |p, _c, _pred| {
            if p == 1 {
                Ok(1)
            } else {
                Ok(0)
            }
        },
        |p, c, _coef| {
            if p == 1 && c == 0 {
                Err(oxideav_core::Error::InvalidData(
                    "pass-1 X-channel decode_symbol failure".into(),
                ))
            } else {
                Ok(0)
            }
        },
    );
    assert!(r.is_err());
    // Pass 0 ran to completion; its per-channel NonZeros stay at
    // 0 (raw_non_zeros = 0).
    for c in 0..3u32 {
        assert_eq!(nz.get(0, c, 0, 0).unwrap(), 0);
    }
}
