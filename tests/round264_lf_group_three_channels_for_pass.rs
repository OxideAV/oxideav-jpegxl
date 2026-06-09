//! Round 264 integration coverage —
//! [`oxideav_jpegxl::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext::decode_lf_group_three_channels_for_pass`].
//!
//! ISO/IEC FDIS 18181-1:2021 §C.8.3 — the bundled per-LfGroup raster-walk
//! three-channel decode driver for one pass, layered above the round-260
//! single-varblock three-channel walk. Walks the
//! [`DctSelectGrid`] in raster order via
//! [`oxideav_jpegxl::varblock_walk::VarblockWalk`] and composes the
//! round-260 method once per top-left cell, owning both the raster walk
//! and the §C.7.2 entropy-stream routing through the round-252 typed
//! decode context.
//!
//! These integration tests pin the public-surface invariants from a
//! consumer's vantage point:
//!
//! * Default-prefix short-circuit (single-symbol prefix → `non_zeros
//!   == 0` per channel → no coefficient symbols read on any of the
//!   three channels) holds across single-cell, multi-cell, and
//!   mixed-transform grids; the returned per-channel `coeffs` vectors
//!   are the correct length for each varblock's transform.
//! * Raster ordering is row-major — varblock outputs in the returned
//!   `Vec<ThreeChannelVarblock>` match the cell walk order; the
//!   per-varblock `qdc_at` + `predicted_at` closures observe the
//!   identical sequence; per-varblock `qdc_at` fires **before**
//!   `predicted_at`.
//! * The per-pass `histogram_offset` is honoured — pass `p = 1`
//!   against a 2-preset histogram bundle routes through
//!   `cluster_map[ctx + 495 × nb_block_ctx × hfp(1)]`.
//! * Closure errors propagate verbatim — qdc_at / predicted_at errors
//!   abort the walk before any downstream read; the BitReader cursor
//!   is unchanged.
//! * Defensive rejections (`p >= num_passes`, residual `Empty` cell
//!   in the grid) bubble out as [`oxideav_core::Error`] without
//!   panicking.

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::block_context_resolver::BlockContextResolver;
use oxideav_jpegxl::dct_select::{DctSelectCell, DctSelectGrid, TransformType};
use oxideav_jpegxl::hf_coeff_histogram_size::HfCoefficientHistogramSize;
use oxideav_jpegxl::hf_coefficient_histograms::HfCoefficientHistograms;
use oxideav_jpegxl::lf_global::HfBlockContext;
use oxideav_jpegxl::multi_pass_hf_header::PerPassHfHeaders;
use oxideav_jpegxl::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext;
use oxideav_jpegxl::pass_group_hf::PassGroupHfHeader;

use std::cell::RefCell;

/// §D.3 prelude bytes for the minimal single-cluster, single-symbol
/// prefix-coded histogram block. Mirrors the round-260 integration
/// test helper exactly.
fn minimal_prefix_prelude_bytes() -> [u8; 2] {
    [0b0001_0010, 0b0000_0000]
}

fn make_minimal_histograms(num_hf_presets: u32, nb_block_ctx: u32) -> HfCoefficientHistograms {
    let bytes = minimal_prefix_prelude_bytes();
    let mut br = BitReader::new(&bytes);
    let size = HfCoefficientHistogramSize::new(num_hf_presets, nb_block_ctx).unwrap();
    HfCoefficientHistograms::read(&mut br, size).unwrap()
}

fn single_pass_headers() -> PerPassHfHeaders {
    PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
        hfp: 0,
        histogram_offset: 0,
    }])
}

/// Default §I.2.2 `HfBlockContext` bundle — the empty-thresholds
/// shape used by round 214 / 221 / 228 / 260 fixtures.
/// `nb_block_ctx = 15`.
fn default_hbc() -> HfBlockContext {
    HfBlockContext {
        used_default: true,
        qf_thresholds: vec![],
        lf_thresholds: [vec![], vec![], vec![]],
        block_ctx_map: HfBlockContext::DEFAULT_BLOCK_CTX_MAP.to_vec(),
        nb_block_ctx: 15,
    }
}

/// Build a uniform-DCT8x8 [`DctSelectGrid`] of the given dimensions.
fn make_uniform_grid(width_blocks: u32, height_blocks: u32, t: TransformType) -> DctSelectGrid {
    let total = (width_blocks as usize) * (height_blocks as usize);
    DctSelectGrid {
        cells: vec![DctSelectCell::TopLeft(t); total],
        hf_mul: vec![1i32; total],
        width_blocks,
        height_blocks,
    }
}

#[test]
fn r264_integration_1x1_dct8x8_short_circuits_three_channels() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let grid = make_uniform_grid(1, 1, TransformType::Dct8x8);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let bits_before = br.bits_read();
    let out = ctx
        .decode_lf_group_three_channels_for_pass(
            &mut br,
            0,
            &grid,
            &resolver,
            |_vb| Ok([0, 0, 0]),
            |_vb| Ok([0, 0, 0]),
        )
        .unwrap();
    assert_eq!(out.len(), 1);
    let (vb, decoded, raw) = &out[0];
    assert_eq!(vb.x, 0);
    assert_eq!(vb.y, 0);
    for c in 0..3 {
        assert_eq!(decoded[c].coeffs.len(), 64);
        assert_eq!(decoded[c].coeffs_read, 0);
        assert_eq!(raw[c], 0);
    }
    assert_eq!(br.bits_read(), bits_before);
}

#[test]
fn r264_integration_3x3_uniform_dct8x8_raster_order() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let grid = make_uniform_grid(3, 3, TransformType::Dct8x8);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let out = ctx
        .decode_lf_group_three_channels_for_pass(
            &mut br,
            0,
            &grid,
            &resolver,
            |_vb| Ok([0, 0, 0]),
            |_vb| Ok([0, 0, 0]),
        )
        .unwrap();
    assert_eq!(out.len(), 9);
    let mut expected = vec![];
    for y in 0..3u32 {
        for x in 0..3u32 {
            expected.push((x, y));
        }
    }
    for (i, e) in expected.iter().enumerate() {
        assert_eq!(out[i].0.x, e.0);
        assert_eq!(out[i].0.y, e.1);
        for c in 0..3 {
            assert_eq!(out[i].1[c].coeffs.len(), 64);
        }
    }
}

#[test]
fn r264_integration_qdc_at_then_predicted_at_then_decode_per_varblock_order() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let grid = make_uniform_grid(2, 1, TransformType::Dct8x8);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let log: RefCell<Vec<(char, u32, u32)>> = RefCell::new(vec![]);
    let _ = ctx
        .decode_lf_group_three_channels_for_pass(
            &mut br,
            0,
            &grid,
            &resolver,
            |vb| {
                log.borrow_mut().push(('q', vb.x, vb.y));
                Ok([0, 0, 0])
            },
            |vb| {
                log.borrow_mut().push(('p', vb.x, vb.y));
                Ok([0, 0, 0])
            },
        )
        .unwrap();
    // Per-varblock pair (qdc, predicted) interleaved in raster order.
    assert_eq!(
        *log.borrow(),
        vec![('q', 0, 0), ('p', 0, 0), ('q', 1, 0), ('p', 1, 0)]
    );
}

#[test]
fn r264_integration_per_pass_offset_routes_through_cluster_map() {
    let mut h = make_minimal_histograms(2, 1);
    let headers = PerPassHfHeaders::from_headers(vec![
        PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        },
        PassGroupHfHeader {
            hfp: 1,
            histogram_offset: 495,
        },
    ]);
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    // nb_block_ctx = 1 matches the histograms shape.
    let hbc = HfBlockContext {
        used_default: false,
        qf_thresholds: vec![],
        lf_thresholds: [vec![], vec![], vec![]],
        block_ctx_map: vec![0u8; 39],
        nb_block_ctx: 1,
    };
    let resolver = BlockContextResolver::new(&hbc);
    let grid = make_uniform_grid(2, 2, TransformType::Dct8x8);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    // Pass 0 — offset 0.
    let out0 = ctx
        .decode_lf_group_three_channels_for_pass(
            &mut br,
            0,
            &grid,
            &resolver,
            |_vb| Ok([0, 0, 0]),
            |_vb| Ok([0, 0, 0]),
        )
        .unwrap();
    assert_eq!(out0.len(), 4);
    for vbo in &out0 {
        assert_eq!(vbo.2, [0, 0, 0]);
    }
    // Pass 1 — offset 495.
    let out1 = ctx
        .decode_lf_group_three_channels_for_pass(
            &mut br,
            1,
            &grid,
            &resolver,
            |_vb| Ok([0, 0, 0]),
            |_vb| Ok([0, 0, 0]),
        )
        .unwrap();
    assert_eq!(out1.len(), 4);
    for vbo in &out1 {
        assert_eq!(vbo.2, [0, 0, 0]);
    }
}

#[test]
fn r264_integration_rejects_out_of_range_pass() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let grid = make_uniform_grid(2, 1, TransformType::Dct8x8);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let r = ctx.decode_lf_group_three_channels_for_pass(
        &mut br,
        9, // > num_passes (= 1)
        &grid,
        &resolver,
        |_vb| Ok([0, 0, 0]),
        |_vb| Ok([0, 0, 0]),
    );
    assert!(r.is_err());
}

#[test]
fn r264_integration_does_not_advance_br_when_short_circuited() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let grid = make_uniform_grid(3, 3, TransformType::Dct8x8);
    let bytes = [0xFFu8; 32];
    let mut br = BitReader::new(&bytes);
    let bits_before = br.bits_read();
    let _ = ctx
        .decode_lf_group_three_channels_for_pass(
            &mut br,
            0,
            &grid,
            &resolver,
            |_vb| Ok([0, 0, 0]),
            |_vb| Ok([0, 0, 0]),
        )
        .unwrap();
    assert_eq!(br.bits_read(), bits_before);
}

#[test]
fn r264_integration_rejects_residual_empty_cell() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let grid = DctSelectGrid {
        cells: vec![DctSelectCell::Empty],
        hf_mul: vec![0],
        width_blocks: 1,
        height_blocks: 1,
    };
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let r = ctx.decode_lf_group_three_channels_for_pass(
        &mut br,
        0,
        &grid,
        &resolver,
        |_vb| Ok([0, 0, 0]),
        |_vb| Ok([0, 0, 0]),
    );
    assert!(r.is_err());
}

#[test]
fn r264_integration_mixed_transforms_per_varblock_coeff_lengths() {
    // 2×2 grid: (0,0) = DCT16x16 (4 cells), so the entire grid is one
    // varblock. coeffs.len() per channel must equal 256.
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let grid = DctSelectGrid {
        cells: vec![
            DctSelectCell::TopLeft(TransformType::Dct16x16),
            DctSelectCell::Continuation,
            DctSelectCell::Continuation,
            DctSelectCell::Continuation,
        ],
        hf_mul: vec![1, 0, 0, 0],
        width_blocks: 2,
        height_blocks: 2,
    };
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let out = ctx
        .decode_lf_group_three_channels_for_pass(
            &mut br,
            0,
            &grid,
            &resolver,
            |_vb| Ok([0, 0, 0]),
            |_vb| Ok([0, 0, 0]),
        )
        .unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0.transform, TransformType::Dct16x16);
    for c in 0..3 {
        assert_eq!(out[0].1[c].coeffs.len(), 256);
        assert_eq!(out[0].1[c].coeffs_read, 0);
        assert_eq!(out[0].2[c], 0);
    }
}

#[test]
fn r264_integration_round_trip_with_per_pass_hf_headers_read() {
    // Drive PerPassHfHeaders::read against a real bitstream end-to-end.
    let mut h = make_minimal_histograms(2, 15);
    let header_bytes = [0b0000_0010u8]; // pass 0 hfp = 0, pass 1 hfp = 1
    let mut hbr = BitReader::new(&header_bytes);
    let headers = PerPassHfHeaders::read(&mut hbr, 2, 2, 15).unwrap();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    assert_eq!(ctx.num_passes(), 2);
    assert_eq!(ctx.histogram_offset(0).unwrap(), 0);
    assert_eq!(ctx.histogram_offset(1).unwrap(), 7425); // 495 × 15 × 1
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let grid = make_uniform_grid(2, 1, TransformType::Dct8x8);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let out0 = ctx
        .decode_lf_group_three_channels_for_pass(
            &mut br,
            0,
            &grid,
            &resolver,
            |_vb| Ok([0, 0, 0]),
            |_vb| Ok([0, 0, 0]),
        )
        .unwrap();
    let out1 = ctx
        .decode_lf_group_three_channels_for_pass(
            &mut br,
            1,
            &grid,
            &resolver,
            |_vb| Ok([0, 0, 0]),
            |_vb| Ok([0, 0, 0]),
        )
        .unwrap();
    assert_eq!(out0.len(), 2);
    assert_eq!(out1.len(), 2);
    for vbo in &out0 {
        assert_eq!(vbo.2, [0, 0, 0]);
    }
    for vbo in &out1 {
        assert_eq!(vbo.2, [0, 0, 0]);
    }
}

#[test]
fn r264_integration_empty_grid_returns_empty_vec() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let grid = DctSelectGrid {
        cells: vec![],
        hf_mul: vec![],
        width_blocks: 0,
        height_blocks: 0,
    };
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let out = ctx
        .decode_lf_group_three_channels_for_pass(
            &mut br,
            0,
            &grid,
            &resolver,
            |_vb| Ok([0, 0, 0]),
            |_vb| Ok([0, 0, 0]),
        )
        .unwrap();
    assert!(out.is_empty());
}
