//! Round 343 — fused live-entropy VarDCT reconstruction.
//!
//! Drives the round-343 one-call fused driver
//! ([`oxideav_jpegxl::vardct_reconstruct::reconstruct_lf_group_from_entropy`])
//! which runs the §C.8.3 live multi-pass entropy decode AND the
//! cross-pass reconstruction (accumulation → F.3 dequant → §I.2.4 LLF
//! merge → §I.2.3.2 IDCT → §C.5.4 placement → Annex G CfL) in a single
//! call per LfGroup. The README previously flagged "feeding it the live
//! per-pass DecodedHfBlock stack from the §C.7.2 entropy stream rather
//! than a caller-supplied one" as the remaining wiring step; this test
//! file exercises the closed gap.
//!
//! The entropy side is fed deterministic stub closures (an all-zero
//! entropy stream) so the output is fully predictable: every varblock
//! reconstructs to a flat plane seeded only by its dequantised LF
//! sample. The point under test is the *wiring* — that the live decode
//! walk feeds the reconstruction in one call and matches the explicit
//! two-call path bit-for-bit.
//!
//! Clean-room: all behaviour is derived from the FDIS spec PDF under
//! `docs/image/jpegxl/`; no external implementation source is consulted.

use oxideav_jpegxl::block_context_resolver::BlockContextResolver;
use oxideav_jpegxl::dct_quant_weights::materialise_default_dequant_set;
use oxideav_jpegxl::dct_select::{DctSelectCell, DctSelectGrid, TransformType};
use oxideav_jpegxl::frame_header::Passes;
use oxideav_jpegxl::hf_dequant::QmScaleFactors;
use oxideav_jpegxl::lf_dequant::LfDequantOutput;
use oxideav_jpegxl::lf_global::{HfBlockContext, LfChannelCorrelation};
use oxideav_jpegxl::metadata_fdis::OpsinInverseMatrix;
use oxideav_jpegxl::multi_pass_decode::decode_multi_pass_three_channels_with_resolver;
use oxideav_jpegxl::per_pass_non_zeros::PerPassNonZerosGrids;
use oxideav_jpegxl::vardct_reconstruct::{
    reconstruct_lf_group_cross_pass, reconstruct_lf_group_from_entropy, DequantContext,
};

fn passes(num_passes: u32, shift: Vec<u32>) -> Passes {
    Passes {
        num_passes,
        num_ds: 0,
        shift,
        downsample: Vec::new(),
        last_pass: Vec::new(),
    }
}

fn qm() -> QmScaleFactors {
    QmScaleFactors {
        x_factor: 0.8,
        b_factor: 1.0,
    }
}

/// The round-214/221/228 default HfBlockContext — empty thresholds
/// collapse the qf/qdc knobs, the default 39-entry block_ctx_map.
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

fn grid(cells: Vec<DctSelectCell>, hf_mul: Vec<i32>, w: u32, h: u32) -> DctSelectGrid {
    DctSelectGrid {
        cells,
        hf_mul,
        width_blocks: w,
        height_blocks: h,
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

/// A 2×2 DCT8×8 grid (four varblocks, 16×16 px) reconstructs end to end
/// through the live entropy walk + cross-pass reconstruction in a single
/// call, with each block flat to its LF DC.
#[test]
fn from_entropy_2x2_dct8x8_grid_reconstructs() {
    let set = materialise_default_dequant_set().unwrap();
    let p = passes(1, vec![]);
    let g = grid(
        vec![
            DctSelectCell::TopLeft(TransformType::Dct8x8),
            DctSelectCell::TopLeft(TransformType::Dct8x8),
            DctSelectCell::TopLeft(TransformType::Dct8x8),
            DctSelectCell::TopLeft(TransformType::Dct8x8),
        ],
        vec![1, 1, 1, 1],
        2,
        2,
    );
    let lf = lf_uniform(3.0, 2, 2);
    let dq = DequantContext {
        set: &set,
        oim: &OpsinInverseMatrix::default(),
        qm: &qm(),
    };
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();

    let planes = reconstruct_lf_group_from_entropy(
        &p,
        &g,
        &mut nz,
        &resolver,
        &lf,
        &dq,
        &[0i32; 1],
        &[0i32; 1],
        &LfChannelCorrelation::default(),
        |_p, _vb| Ok([0i32; 3]),
        |_p, _c, _pred| Ok(0u32),
        |_p, _c, _coef| Ok(0u32),
    )
    .unwrap();
    // 2 cells × 8 px = 16 px on each axis.
    assert_eq!(planes.dims(), (16, 16));
    // Constant LF + zero HF → every Y sample is the shared flat DC.
    let v0 = planes.y().get(0, 0).unwrap();
    for y in 0..16 {
        for x in 0..16 {
            let v = planes.y().get(x, y).unwrap();
            assert!(v.is_finite(), "Y ({x},{y}) not finite: {v}");
            assert!((v - v0).abs() < 1e-3, "Y ({x},{y}) = {v} != {v0}");
        }
    }
}

/// A two-pass non-square DCT8×16 frame reconstructs end to end through
/// the live entropy walk. The fused one-call driver matches the explicit
/// two-call (`decode_multi_pass…` → `reconstruct_lf_group_cross_pass`)
/// path bit-for-bit.
#[test]
fn from_entropy_two_pass_non_square_matches_explicit() {
    let set = materialise_default_dequant_set().unwrap();
    // Two passes; pass 0 shift = 1 (×2), pass 1 (last) shift 0.
    let p = passes(2, vec![1]);
    // DCT8×16 footprint is (bcols=2, brows=1) → a 2×1 grid.
    let g = grid(
        vec![
            DctSelectCell::TopLeft(TransformType::Dct8x16),
            DctSelectCell::Continuation,
        ],
        vec![1, 0],
        2,
        1,
    );
    // LF image: cx=2 × cy=1 sub-block read by DCT8×16.
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
    let mut nz_a = PerPassNonZerosGrids::new_uniform(2, 3, 2, 1).unwrap();
    let mp = decode_multi_pass_three_channels_with_resolver(
        &g,
        &mut nz_a,
        &resolver,
        |_p, _vb| Ok([0i32; 3]),
        |_p, _c, _pred| Ok(0u32),
        |_p, _c, _coef| Ok(0u32),
    )
    .unwrap();
    assert_eq!(mp.len(), 2, "two passes decoded");
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
    let mut nz_b = PerPassNonZerosGrids::new_uniform(2, 3, 2, 1).unwrap();
    let fused = reconstruct_lf_group_from_entropy(
        &p,
        &g,
        &mut nz_b,
        &resolver,
        &lf,
        &dq,
        &[0i32; 1],
        &[0i32; 1],
        &LfChannelCorrelation::default(),
        |_p, _vb| Ok([0i32; 3]),
        |_p, _c, _pred| Ok(0u32),
        |_p, _c, _coef| Ok(0u32),
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

/// An entropy-side error (a `decode_symbol` closure that fails)
/// propagates verbatim out of the fused driver — the entropy decode and
/// the reconstruction share the same error channel.
#[test]
fn from_entropy_propagates_entropy_error() {
    let set = materialise_default_dequant_set().unwrap();
    let p = passes(1, vec![]);
    let g = grid(
        vec![DctSelectCell::TopLeft(TransformType::Dct8x8)],
        vec![1],
        1,
        1,
    );
    let lf = lf_uniform(0.0, 1, 1);
    let dq = DequantContext {
        set: &set,
        oim: &OpsinInverseMatrix::default(),
        qm: &qm(),
    };
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();

    // A non-zero NonZeros count forces at least one decode_symbol call,
    // which fails — the error must surface from the fused driver.
    let res = reconstruct_lf_group_from_entropy(
        &p,
        &g,
        &mut nz,
        &resolver,
        &lf,
        &dq,
        &[0i32; 1],
        &[0i32; 1],
        &LfChannelCorrelation::default(),
        |_p, _vb| Ok([0i32; 3]),
        |_p, _c, _pred| Ok(1u32),
        |_p, _c, _coef| {
            Err(oxideav_core::Error::InvalidData(
                "synthetic entropy failure".into(),
            ))
        },
    );
    assert!(res.is_err(), "entropy-side error must propagate");
}
