//! Round 322 — LF-aware per-LfGroup VarDCT three-channel residual-plane
//! reconstruction: wire the round-316 LLF-aware per-block decode
//! (`block_dequant::decode_block_to_residual_with_llf`) into the
//! round-306/309 spatial placement + Annex G chroma-from-luma stack.
//!
//! Earlier rounds landed every constituent in isolation:
//!
//! * round 129  — `vardct::compose_lf_to_llf_block`: extract a varblock's
//!   `cy × cx` LF sub-block from the dequantised LF image and run Listing
//!   I.16 to obtain its LLF coefficient block;
//! * round 316  — `block_dequant::decode_block_to_residual_with_llf`: F.3
//!   dequant → §I.2.4 LLF merge → §I.2.3.2 IDCT of the **complete**
//!   coefficient matrix;
//! * round 306/309 — `residual_plane::reconstruct_three_channel_planes`:
//!   walk the shared `DctSelectGrid`, place each varblock's residual block
//!   into the per-channel plane, then apply Annex G CfL — but driven by
//!   the **HF-only** per-block decode (`decode_block_to_residual`), so the
//!   DC subband never reached the assembled plane.
//!
//! Round 322 closes the seam: `reconstruct_three_channel_planes_with_lf`
//! (+ its non-CfL sibling `assemble_three_channel_planes_with_lf`) takes
//! the LfGroup's dequantised LF samples (`LfDequantOutput`) and, per
//! varblock per channel, composes that channel's LLF block at the
//! varblock's block-grid origin and threads it into the caller's
//! LLF-aware decode closure. The assembled planes now carry the full
//! LF + HF spatial residual.
//!
//! These tests compose the *real* LF→LLF step + the *real* dequant + LLF
//! merge + IDCT walk + the *real* Annex G CfL across all three channels,
//! proving the per-LfGroup LF-aware decode → place → CfL chain runs
//! end-to-end on actual dequantised residual samples.
//!
//! Source of truth: ISO/IEC FDIS 18181-1:2021 §C.5.4 (DctSelect
//! placement) + §C.8.3 (per-varblock decode order) + §I.2.4 / Listing
//! I.16 (LF → LLF, natural-order prefix) + §I.2.3.2 (coefficients →
//! samples) + §F.3 (HF dequant) + Annex G (chroma-from-luma).

use oxideav_jpegxl::block_dequant::{decode_block_to_residual_with_llf, merge_llf_into_block};
use oxideav_jpegxl::dct_quant_weights::materialise_default_dequant_set;
use oxideav_jpegxl::dct_select::{DctSelectCell, DctSelectGrid, TransformType};
use oxideav_jpegxl::hf_dequant::QmScaleFactors;
use oxideav_jpegxl::idct::idct_for_transform;
use oxideav_jpegxl::lf_dequant::LfDequantOutput;
use oxideav_jpegxl::lf_global::LfChannelCorrelation;
use oxideav_jpegxl::metadata_fdis::OpsinInverseMatrix;
use oxideav_jpegxl::pass_group_hf::DecodedHfBlock;
use oxideav_jpegxl::residual_plane::{
    assemble_three_channel_planes_with_lf, reconstruct_three_channel_planes_with_lf,
};
use oxideav_jpegxl::varblock_walk::Varblock;
use oxideav_jpegxl::vardct::compose_lf_to_llf_block;

fn qm() -> QmScaleFactors {
    QmScaleFactors {
        x_factor: 0.8,
        b_factor: 1.0,
    }
}

fn block(coeffs: Vec<i32>) -> DecodedHfBlock {
    DecodedHfBlock {
        coeffs,
        remaining_non_zeros: 0,
        coeffs_read: 0,
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

/// A CfL header that disables chroma-from-luma (zero base factors,
/// non-zero colour_factor so the divide is valid). With zero `x_from_y`
/// / `b_from_y` factors the X / B planes pass through unchanged.
fn cfl_identity() -> LfChannelCorrelation {
    LfChannelCorrelation {
        all_default: false,
        colour_factor: 84,
        base_correlation_x: 0.0,
        base_correlation_b: 0.0,
        x_factor_lf: 128,
        b_factor_lf: 128,
    }
}

/// Build a single-channel `LfDequantOutput` triple with identical dims,
/// each channel carrying its own `w*h` row-major LF samples.
fn lf_output(samples: [Vec<f32>; 3], w: u32, h: u32) -> LfDequantOutput {
    LfDequantOutput {
        samples,
        widths: [w, w, w],
        heights: [h, h, h],
    }
}

/// 1×1 grid: a single DCT8×8 varblock. Each channel's LF image is a
/// single sample (`cx = cy = 1`, so the LLF is the LF sample unchanged).
/// The assembled plane must equal the per-block LF-aware decode placed at
/// the origin — proving the driver threads each channel's LLF correctly.
#[test]
fn single_dct8x8_threads_per_channel_llf() {
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();

    let g = grid(
        vec![DctSelectCell::TopLeft(TransformType::Dct8x8)],
        vec![3],
        1,
        1,
    );
    // Distinct DC per channel so a channel mix-up is visible.
    let lf = lf_output([vec![10.0], vec![20.0], vec![30.0]], 1, 1);

    let planes = assemble_three_channel_planes_with_lf(&g, &lf, |c, vb, llf| {
        assert_eq!(llf.len(), 1, "DCT8×8 LLF is 1 cell");
        let b = block(vec![0i32; 64]);
        decode_block_to_residual_with_llf(&b, vb.transform, c, vb.hf_mul, &set, &oim, &qm(), llf)
    })
    .unwrap();

    // Reference: decode each channel's block directly with its own LLF.
    for c in 0..3 {
        let lf_val = lf.samples[c][0];
        let llf = compose_lf_to_llf_block(&[lf_val], 1, 1, 0, 0, TransformType::Dct8x8).unwrap();
        let b = block(vec![0i32; 64]);
        let expected = decode_block_to_residual_with_llf(
            &b,
            TransformType::Dct8x8,
            c,
            3,
            &set,
            &oim,
            &qm(),
            &llf,
        )
        .unwrap();
        let plane = &planes.planes[c];
        for r in 0..8 {
            for x in 0..8 {
                assert!(
                    (plane.get(x, r).unwrap() - expected[r * 8 + x]).abs() < 1e-4,
                    "channel {c} cell ({x},{r}) mismatch",
                );
            }
        }
        // Pure-DC block is flat and non-zero.
        let s0 = plane.get(0, 0).unwrap();
        assert!(s0 != 0.0, "channel {c} DC must propagate");
        assert!(
            (1..64).all(|i| (plane.get(i % 8, i / 8).unwrap() - s0).abs() < 1e-4),
            "channel {c} pure-DC block must be flat",
        );
    }
}

/// The LF-aware driver differs from the HF-only path by exactly the LLF
/// contribution (linearity of the IDCT): with the same HF block the
/// assembled plane equals `hf_only_residual + idct(llf_only_grid)` placed
/// at the origin. Verified on a DCT16×16 varblock with HF coefficients
/// set and a 2×2 LF image.
#[test]
fn lf_aware_equals_hf_only_plus_llf_idct() {
    use oxideav_jpegxl::block_dequant::decode_block_to_residual;

    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();

    let g = grid(
        vec![
            DctSelectCell::TopLeft(TransformType::Dct16x16),
            DctSelectCell::Continuation,
            DctSelectCell::Continuation,
            DctSelectCell::Continuation,
        ],
        vec![5, 0, 0, 0],
        2,
        2,
    );

    let mut coeffs = vec![0i32; 256];
    coeffs[40] = 6;
    coeffs[100] = -3;

    // 2×2 LF image per channel → 2×2 LLF block for DCT16×16.
    let lf = lf_output(
        [
            vec![3.0, -1.0, 0.5, 2.0],
            vec![1.0, 0.0, -2.0, 4.0],
            vec![-0.5, 1.5, 2.5, -3.0],
        ],
        2,
        2,
    );

    let cc = coeffs.clone();
    let planes = assemble_three_channel_planes_with_lf(&g, &lf, |c, vb, llf| {
        let b = block(cc.clone());
        decode_block_to_residual_with_llf(&b, vb.transform, c, vb.hf_mul, &set, &oim, &qm(), llf)
    })
    .unwrap();

    for c in 0..3 {
        let b = block(coeffs.clone());
        let hf_only =
            decode_block_to_residual(&b, TransformType::Dct16x16, c, 5, &set, &oim, &qm()).unwrap();
        let llf =
            compose_lf_to_llf_block(&lf.samples[c], 2, 2, 0, 0, TransformType::Dct16x16).unwrap();
        let mut llf_grid = vec![0.0f32; 256];
        merge_llf_into_block(&mut llf_grid, TransformType::Dct16x16, &llf).unwrap();
        let llf_contrib = idct_for_transform(TransformType::Dct16x16, &llf_grid).unwrap();

        let plane = &planes.planes[c];
        for r in 0..16 {
            for x in 0..16 {
                let got = plane.get(x, r).unwrap();
                let expected = hf_only[r * 16 + x] + llf_contrib[r * 16 + x];
                assert!(
                    (got - expected).abs() < 1e-3,
                    "channel {c} cell ({x},{r}): {got} != hf+llf {expected}",
                );
            }
        }
    }
}

/// Two side-by-side DCT8×8 varblocks: each samples its own LF cell, so
/// the assembled plane carries two distinct flat blocks. Confirms the
/// per-varblock LF addressing threads through the grid walk + placement.
#[test]
fn two_varblocks_address_distinct_lf_cells() {
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();

    let g = grid(
        vec![
            DctSelectCell::TopLeft(TransformType::Dct8x8),
            DctSelectCell::TopLeft(TransformType::Dct8x8),
        ],
        vec![3, 3],
        2,
        1,
    );
    // 2×1 LF image: left cell 5.0, right cell 50.0 (channel 1 / Y).
    let lf = lf_output([vec![5.0, 50.0], vec![5.0, 50.0], vec![5.0, 50.0]], 2, 1);

    let planes = assemble_three_channel_planes_with_lf(&g, &lf, |c, vb, llf| {
        let b = block(vec![0i32; 64]);
        decode_block_to_residual_with_llf(&b, vb.transform, c, vb.hf_mul, &set, &oim, &qm(), llf)
    })
    .unwrap();

    let plane = &planes.planes[1];
    // Left 8×8 block (origin 0,0) and right 8×8 block (origin 8,0) are each
    // flat, and the right block (LF 50) is larger magnitude than the left
    // (LF 5).
    let left = plane.get(0, 0).unwrap();
    let right = plane.get(8, 0).unwrap();
    assert!(left != 0.0 && right != 0.0);
    assert!(
        right.abs() > left.abs() * 5.0,
        "right block (LF 50) must dwarf left (LF 5): left={left} right={right}",
    );
    // Each block flat across its own 8×8 footprint.
    for r in 0..8 {
        for x in 0..8 {
            assert!((plane.get(x, r).unwrap() - left).abs() < 1e-4);
            assert!((plane.get(8 + x, r).unwrap() - right).abs() < 1e-4);
        }
    }
}

/// `reconstruct_three_channel_planes_with_lf` with a CfL-identity header
/// equals `assemble_three_channel_planes_with_lf` (no chroma restore).
/// Then a non-zero `x_from_y` factor shifts the X plane by `kX·Y` per
/// Listing G.1, proving the CfL step runs on the LF-aware planes.
#[test]
fn reconstruct_applies_cfl_over_lf_aware_planes() {
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();

    let g = grid(
        vec![DctSelectCell::TopLeft(TransformType::Dct8x8)],
        vec![3],
        1,
        1,
    );
    let lf = lf_output([vec![8.0], vec![40.0], vec![16.0]], 1, 1);

    let decode = |c: usize, vb: &Varblock, llf: &[f32]| {
        let b = block(vec![0i32; 64]);
        decode_block_to_residual_with_llf(&b, vb.transform, c, vb.hf_mul, &set, &oim, &qm(), llf)
    };

    // Identity CfL: planes equal the no-CfL assembly.
    let bare = assemble_three_channel_planes_with_lf(&g, &lf, decode).unwrap();
    let identity =
        reconstruct_three_channel_planes_with_lf(&g, &lf, &[0], &[0], &cfl_identity(), decode)
            .unwrap();
    assert_eq!(bare, identity, "CfL-identity must be a no-op");

    // Non-zero x_from_y over the single 64×64 CfL tile: X += kX·Y.
    let x_before = bare.planes[0].get(0, 0).unwrap();
    let y = bare.planes[1].get(0, 0).unwrap();
    let cfl = cfl_identity();
    let x_from_y = vec![32]; // one tile (8×8 plane ⊂ one 64×64 tile)
    let restored =
        reconstruct_three_channel_planes_with_lf(&g, &lf, &x_from_y, &[0], &cfl, decode).unwrap();
    let x_after = restored.planes[0].get(0, 0).unwrap();
    // kX = base_correlation_x + x_from_y / colour_factor = 0 + 32/84.
    let kx = 32.0f32 / 84.0;
    assert!(
        (x_after - (x_before + kx * y)).abs() < 1e-3,
        "CfL X restore: {x_after} != {x_before} + {kx}·{y}",
    );
    // Y plane unchanged by CfL.
    assert!((restored.planes[1].get(0, 0).unwrap() - y).abs() < 1e-6);
}

/// Mismatched LF channel dims are rejected before any decode work.
#[test]
fn mismatched_lf_dims_rejected() {
    let g = grid(
        vec![DctSelectCell::TopLeft(TransformType::Dct8x8)],
        vec![3],
        1,
        1,
    );
    let lf = LfDequantOutput {
        samples: [vec![1.0], vec![1.0, 2.0], vec![1.0]],
        widths: [1, 2, 1],
        heights: [1, 1, 1],
    };
    let err = assemble_three_channel_planes_with_lf(&g, &lf, |_, _, _| Ok(vec![0.0; 64]));
    assert!(err.is_err(), "mismatched LF dims must be rejected");
}
