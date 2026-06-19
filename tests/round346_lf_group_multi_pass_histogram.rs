//! Round 346 — histogram-backed multi-pass per-LfGroup VarDCT decode +
//! reconstruction.
//!
//! Two new drivers under test, both closing the README-noted "§C.7.2
//! entropy-histogram materialisation that backs those closures" wiring
//! step:
//!
//! * [`oxideav_jpegxl::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext::decode_lf_group_multi_pass_three_channels`]
//!   — the histogram-backed sibling of the closure-based
//!   [`oxideav_jpegxl::multi_pass_decode::decode_multi_pass_three_channels_with_resolver`].
//!   It runs the full §C.8.3 outer-pass loop over the round-260
//!   three-channel varblock walk, threading the per-pass per-channel
//!   `PredictedNonZeros` read + `NonZeros(x, y)` writeback through the
//!   [`oxideav_jpegxl::per_pass_non_zeros::PerPassNonZerosGrids`]
//!   container, owning the §C.7.2 entropy-stream routing itself.
//! * [`oxideav_jpegxl::vardct_reconstruct::reconstruct_lf_group_from_histogram`]
//!   — fuses that decode with the round-340 cross-pass reconstruction in
//!   one call (the histogram-source sibling of
//!   [`oxideav_jpegxl::vardct_reconstruct::reconstruct_lf_group_from_entropy`]).
//!
//! The central correctness claim is **equivalence**: the histogram-backed
//! driver must produce bit-for-bit identical output to the closure-based
//! path when the latter's `read_non_zeros` / `decode_symbol` closures are
//! wired to an identical histogram context over the same §C.7.2 stream.
//! Both paths share the same `decode_block_at` predicted-read /
//! writeback sequence, the same Y → X → B channel order, and the same
//! cross-pass reconstruction, so they cannot diverge.
//!
//! Clean-room: all behaviour is derived from the FDIS spec PDF under
//! `docs/image/jpegxl/`; no external implementation source is consulted.

use std::cell::RefCell;

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::block_context_resolver::BlockContextResolver;
use oxideav_jpegxl::dct_quant_weights::materialise_default_dequant_set;
use oxideav_jpegxl::dct_select::{DctSelectCell, DctSelectGrid, TransformType};
use oxideav_jpegxl::frame_header::Passes;
use oxideav_jpegxl::hf_coeff_histogram_size::HfCoefficientHistogramSize;
use oxideav_jpegxl::hf_coefficient_histograms::HfCoefficientHistograms;
use oxideav_jpegxl::hf_dequant::QmScaleFactors;
use oxideav_jpegxl::lf_dequant::LfDequantOutput;
use oxideav_jpegxl::lf_global::{HfBlockContext, LfChannelCorrelation};
use oxideav_jpegxl::metadata_fdis::OpsinInverseMatrix;
use oxideav_jpegxl::multi_pass_decode::decode_multi_pass_three_channels_with_resolver;
use oxideav_jpegxl::multi_pass_hf_header::PerPassHfHeaders;
use oxideav_jpegxl::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext;
use oxideav_jpegxl::pass_group_hf::PassGroupHfHeader;
use oxideav_jpegxl::per_pass_non_zeros::PerPassNonZerosGrids;
use oxideav_jpegxl::vardct_reconstruct::{
    reconstruct_lf_group_cross_pass, reconstruct_lf_group_from_histogram, DequantContext,
};

/// §D.3 prelude bytes for the minimal single-cluster, single-symbol
/// prefix-coded histogram block (every decoded symbol is 0). Mirrors the
/// round-260 / round-264 integration helper exactly.
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

fn two_pass_headers() -> PerPassHfHeaders {
    PerPassHfHeaders::from_headers(vec![
        PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        },
        PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        },
    ])
}

/// The round-214/221/228 default HfBlockContext (`nb_block_ctx = 15`).
fn default_hbc() -> HfBlockContext {
    HfBlockContext {
        used_default: true,
        qf_thresholds: vec![],
        lf_thresholds: [vec![], vec![], vec![]],
        block_ctx_map: HfBlockContext::DEFAULT_BLOCK_CTX_MAP.to_vec(),
        nb_block_ctx: 15,
    }
}

fn grid(cells: Vec<DctSelectCell>, hf_mul: Vec<i32>, w: u32, h: u32) -> DctSelectGrid {
    DctSelectGrid {
        cells,
        hf_mul,
        width_blocks: w,
        height_blocks: h,
    }
}

fn uniform_grid(w: u32, h: u32, t: TransformType) -> DctSelectGrid {
    let total = (w * h) as usize;
    grid(
        vec![DctSelectCell::TopLeft(t); total],
        vec![1i32; total],
        w,
        h,
    )
}

fn qm() -> QmScaleFactors {
    QmScaleFactors {
        x_factor: 0.8,
        b_factor: 1.0,
    }
}

fn lf_uniform(value: f32, w: u32, h: u32) -> LfDequantOutput {
    let n = (w * h) as usize;
    LfDequantOutput {
        samples: [vec![value; n], vec![value; n], vec![value; n]],
        widths: [w, w, w],
        heights: [h, h, h],
    }
}

fn passes(num_passes: u32, shift: Vec<u32>) -> Passes {
    Passes {
        num_passes,
        num_ds: 0,
        shift,
        downsample: Vec::new(),
        last_pass: Vec::new(),
    }
}

// ---------------------------------------------------------------------
// decode_lf_group_multi_pass_three_channels — structural invariants.
// ---------------------------------------------------------------------

#[test]
fn r346_multi_pass_single_pass_3x3_default_prefix_short_circuits() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let g = uniform_grid(3, 3, TransformType::Dct8x8);
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 3, 3).unwrap();
    let bytes = [0u8; 8];
    let mut br = BitReader::new(&bytes);
    let bits_before = br.bits_read();

    let out = ctx
        .decode_lf_group_multi_pass_three_channels(&mut br, &g, &mut nz, &resolver, |_p, _vb| {
            Ok([0i32; 3])
        })
        .unwrap();

    assert_eq!(out.len(), 1, "one pass");
    assert_eq!(out[0].len(), 9, "9 varblocks in raster order");
    // Default single-symbol prefix → every NonZeros is 0 → no coefficient
    // symbols read → BitReader cursor unmoved.
    assert_eq!(br.bits_read(), bits_before);
    for (i, (vb, decoded, raw)) in out[0].iter().enumerate() {
        let ex = (i as u32 % 3, i as u32 / 3);
        assert_eq!((vb.x, vb.y), ex, "raster order at {i}");
        for c in 0..3 {
            assert_eq!(decoded[c].coeffs.len(), 64);
            assert_eq!(decoded[c].coeffs_read, 0);
            assert_eq!(raw[c], 0);
        }
    }
}

#[test]
fn r346_multi_pass_two_pass_uniform_lengths() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = two_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let g = uniform_grid(2, 1, TransformType::Dct8x8);
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 2, 1).unwrap();
    let bytes = [0u8; 8];
    let mut br = BitReader::new(&bytes);

    let out = ctx
        .decode_lf_group_multi_pass_three_channels(&mut br, &g, &mut nz, &resolver, |_p, _vb| {
            Ok([0i32; 3])
        })
        .unwrap();
    assert_eq!(out.len(), 2, "two passes");
    assert_eq!(out[0].len(), 2);
    assert_eq!(out[1].len(), 2);
}

#[test]
fn r346_multi_pass_rejects_pass_count_mismatch() {
    // ctx has one pass (single_pass_headers) but the NonZeros grids carry
    // two — the driver must reject before consuming any stream bytes.
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let g = uniform_grid(1, 1, TransformType::Dct8x8);
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);

    let res = ctx.decode_lf_group_multi_pass_three_channels(
        &mut br,
        &g,
        &mut nz,
        &resolver,
        |_p, _vb| Ok([0i32; 3]),
    );
    assert!(res.is_err(), "pass-count mismatch must be rejected");
}

#[test]
fn r346_multi_pass_qdc_at_observes_pass_and_raster_order() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = two_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let g = uniform_grid(2, 1, TransformType::Dct8x8);
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 2, 1).unwrap();
    let bytes = [0u8; 8];
    let mut br = BitReader::new(&bytes);

    let log: RefCell<Vec<(u32, u32, u32)>> = RefCell::new(vec![]);
    ctx.decode_lf_group_multi_pass_three_channels(&mut br, &g, &mut nz, &resolver, |p, vb| {
        log.borrow_mut().push((p, vb.x, vb.y));
        Ok([0i32; 3])
    })
    .unwrap();
    // Outer pass loop, inner raster walk.
    assert_eq!(
        *log.borrow(),
        vec![(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    );
}

// ---------------------------------------------------------------------
// Equivalence: the histogram-backed driver matches the closure-based
// path wired to an identical histogram context over the same stream.
// ---------------------------------------------------------------------

/// Drive the closure-based [`decode_multi_pass_three_channels_with_resolver`]
/// with `read_non_zeros` / `decode_symbol` closures routed to an
/// independent-but-identical [`HfHistogramDecodeContext`] over the same
/// §C.7.2 stream, then compare against the histogram-backed driver.
///
/// The closures need the histogram context + BitReader inside both the
/// `read_non_zeros` and `decode_symbol` closures; we wrap both in
/// `RefCell` so they can be borrowed independently per call. The
/// `nb_block_ctx` invariant is read off the resolver.
#[test]
fn r346_multi_pass_matches_closure_path_bit_for_bit() {
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let g = uniform_grid(2, 2, TransformType::Dct8x8);

    // A few non-zero stream bytes so at least some coefficient symbols
    // are read (the default prefix decodes every symbol to 0 regardless,
    // but the cursor still advances per `decode_block_for_pass_transform`
    // NonZeros reads if the predicted count is non-trivial; here the
    // point is that BOTH paths consume the identical stream).
    let stream = [0xA5u8, 0x3C, 0x7E, 0x11, 0x00, 0xFF, 0x42, 0x90];

    // ---- Histogram-backed driver ----
    let mut h_a = make_minimal_histograms(1, 15);
    let headers_a = single_pass_headers();
    let mut ctx_a = HfHistogramDecodeContext::new(&mut h_a, &headers_a).unwrap();
    let mut nz_a = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
    let mut br_a = BitReader::new(&stream);
    let hist = ctx_a
        .decode_lf_group_multi_pass_three_channels(
            &mut br_a,
            &g,
            &mut nz_a,
            &resolver,
            |_p, _vb| Ok([0i32; 3]),
        )
        .unwrap();

    // ---- Closure-based driver wired to an identical histogram ctx ----
    let mut h_b = make_minimal_histograms(1, 15);
    let headers_b = single_pass_headers();
    let ctx_b = RefCell::new(HfHistogramDecodeContext::new(&mut h_b, &headers_b).unwrap());
    let br_b = RefCell::new(BitReader::new(&stream));
    let mut nz_b = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();

    let closure = decode_multi_pass_three_channels_with_resolver(
        &g,
        &mut nz_b,
        &resolver,
        |_p, _vb| Ok([0i32; 3]),
        |p, _c, _predicted| {
            // The closure-based per-channel path reads NonZeros through a
            // ctx-routed call. `coeff_ctx` is the resolved context value;
            // the closure path passes the already-context-resolved value,
            // so we route through `decode_symbol_for_pass` directly with
            // the supplied predicted-derived context.
            let mut ctx = ctx_b.borrow_mut();
            let mut br = br_b.borrow_mut();
            ctx.decode_symbol_for_pass(&mut br, p, _predicted)
        },
        |p, _c, coeff_ctx| {
            let mut ctx = ctx_b.borrow_mut();
            let mut br = br_b.borrow_mut();
            ctx.decode_symbol_for_pass(&mut br, p, coeff_ctx)
        },
    )
    .unwrap();

    // Both produce the MultiPassThreeChannelOutput shape; compare coeffs.
    assert_eq!(hist.len(), closure.len(), "pass count");
    for (ph, pc) in hist.iter().zip(closure.iter()) {
        assert_eq!(ph.len(), pc.len(), "varblock count per pass");
        for ((vbh, dh, rh), (vbc, dc, rc)) in ph.iter().zip(pc.iter()) {
            assert_eq!((vbh.x, vbh.y), (vbc.x, vbc.y), "varblock position");
            assert_eq!(rh, rc, "raw non-zeros triple");
            for c in 0..3 {
                assert_eq!(dh[c].coeffs, dc[c].coeffs, "coeffs ch {c}");
                assert_eq!(dh[c].coeffs_read, dc[c].coeffs_read, "coeffs_read ch {c}");
            }
        }
    }
    // Both consumed the identical number of stream bits.
    assert_eq!(br_a.bits_read(), br_b.borrow().bits_read());
}

// ---------------------------------------------------------------------
// reconstruct_lf_group_from_histogram — end-to-end reconstruction.
// ---------------------------------------------------------------------

#[test]
fn r346_from_histogram_2x2_dct8x8_reconstructs_flat() {
    let set = materialise_default_dequant_set().unwrap();
    let p = passes(1, vec![]);
    let g = uniform_grid(2, 2, TransformType::Dct8x8);
    let lf = lf_uniform(3.0, 2, 2);
    let dq = DequantContext {
        set: &set,
        oim: &OpsinInverseMatrix::default(),
        qm: &qm(),
    };
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
    let bytes = [0u8; 8];
    let mut br = BitReader::new(&bytes);

    let planes = reconstruct_lf_group_from_histogram(
        &p,
        &g,
        &mut nz,
        &resolver,
        &mut ctx,
        &mut br,
        &lf,
        &dq,
        &[0i32; 1],
        &[0i32; 1],
        &LfChannelCorrelation::default(),
        |_p, _vb| Ok([0i32; 3]),
    )
    .unwrap();

    assert_eq!(planes.dims(), (16, 16));
    let v0 = planes.y().get(0, 0).unwrap();
    for y in 0..16 {
        for x in 0..16 {
            let v = planes.y().get(x, y).unwrap();
            assert!(v.is_finite(), "Y ({x},{y}) not finite: {v}");
            assert!((v - v0).abs() < 1e-3, "Y ({x},{y}) = {v} != {v0}");
        }
    }
}

/// The fused histogram-source driver matches the explicit two-call
/// (`decode_lf_group_multi_pass_three_channels` →
/// `reconstruct_lf_group_cross_pass`) path bit-for-bit, on a two-pass
/// non-square DCT8×16 frame.
#[test]
fn r346_from_histogram_two_pass_non_square_matches_explicit() {
    let set = materialise_default_dequant_set().unwrap();
    let p = passes(2, vec![1]);
    let g = grid(
        vec![
            DctSelectCell::TopLeft(TransformType::Dct8x16),
            DctSelectCell::Continuation,
        ],
        vec![1, 0],
        2,
        1,
    );
    let lf = LfDequantOutput {
        samples: [vec![3.0, 5.0], vec![3.0, 5.0], vec![3.0, 5.0]],
        widths: [2, 2, 2],
        heights: [1, 1, 1],
    };
    let dq = DequantContext {
        set: &set,
        oim: &OpsinInverseMatrix::default(),
        qm: &qm(),
    };
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);

    // Explicit two-call path.
    let mut h_a = make_minimal_histograms(1, 15);
    let headers_a = two_pass_headers();
    let mut ctx_a = HfHistogramDecodeContext::new(&mut h_a, &headers_a).unwrap();
    let mut nz_a = PerPassNonZerosGrids::new_uniform(2, 3, 2, 1).unwrap();
    let bytes = [0u8; 8];
    let mut br_a = BitReader::new(&bytes);
    let mp = ctx_a
        .decode_lf_group_multi_pass_three_channels(
            &mut br_a,
            &g,
            &mut nz_a,
            &resolver,
            |_p, _vb| Ok([0i32; 3]),
        )
        .unwrap();
    assert_eq!(mp.len(), 2);
    let explicit = reconstruct_lf_group_cross_pass(
        &p,
        &g,
        &lf,
        &dq,
        &[0i32; 1],
        &[0i32; 1],
        &LfChannelCorrelation::default(),
        &mp,
    )
    .unwrap();

    // Fused one-call path.
    let mut h_b = make_minimal_histograms(1, 15);
    let headers_b = two_pass_headers();
    let mut ctx_b = HfHistogramDecodeContext::new(&mut h_b, &headers_b).unwrap();
    let mut nz_b = PerPassNonZerosGrids::new_uniform(2, 3, 2, 1).unwrap();
    let mut br_b = BitReader::new(&bytes);
    let fused = reconstruct_lf_group_from_histogram(
        &p,
        &g,
        &mut nz_b,
        &resolver,
        &mut ctx_b,
        &mut br_b,
        &lf,
        &dq,
        &[0i32; 1],
        &[0i32; 1],
        &LfChannelCorrelation::default(),
        |_p, _vb| Ok([0i32; 3]),
    )
    .unwrap();

    assert_eq!(fused.dims(), (16, 8));
    assert_eq!(fused.dims(), explicit.dims());
    for ch in 0..3 {
        let (pe, pf) = match ch {
            0 => (explicit.x(), fused.x()),
            1 => (explicit.y(), fused.y()),
            _ => (explicit.b(), fused.b()),
        };
        for y in 0..8 {
            for x in 0..16 {
                let a = pe.get(x, y).unwrap();
                let b = pf.get(x, y).unwrap();
                assert_eq!(a.to_bits(), b.to_bits(), "ch {ch} ({x},{y}): {a} != {b}");
            }
        }
    }
}

#[test]
fn r346_from_histogram_rejects_pass_count_mismatch() {
    let set = materialise_default_dequant_set().unwrap();
    // passes.num_passes = 2 but nz carries 1.
    let p = passes(2, vec![1]);
    let g = uniform_grid(1, 1, TransformType::Dct8x8);
    let lf = lf_uniform(0.0, 1, 1);
    let dq = DequantContext {
        set: &set,
        oim: &OpsinInverseMatrix::default(),
        qm: &qm(),
    };
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);

    let res = reconstruct_lf_group_from_histogram(
        &p,
        &g,
        &mut nz,
        &resolver,
        &mut ctx,
        &mut br,
        &lf,
        &dq,
        &[0i32; 1],
        &[0i32; 1],
        &LfChannelCorrelation::default(),
        |_p, _vb| Ok([0i32; 3]),
    );
    assert!(res.is_err(), "passes/nz pass-count mismatch must reject");
}
