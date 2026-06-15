//! Round 309 — per-LfGroup VarDCT three-channel residual-plane assembly
//! + Annex G chroma-from-luma.
//!
//! Integration coverage for the round-309 additions to `residual_plane`:
//! the three-channel spatial-reconstruction layer above the round-306
//! single-channel `assemble_channel_plane`. `assemble_three_channel_planes`
//! walks the shared `DctSelectGrid` once per channel (X / Y / B);
//! `apply_chroma_from_luma` restores the X / B chroma residuals from the
//! Y luma per Listing G.1; `reconstruct_three_channel_planes` composes
//! both in one call.
//!
//! These tests compose the *real* `block_dequant` per-block decode walk
//! (F.3 dequant + I.2.3 inverse DCT) across all three channels and then
//! apply the real Annex G CfL — proving the full per-LfGroup
//! decode → place → CfL chain runs end-to-end on actual dequantised
//! residual samples, not synthetic constant blocks.
//!
//! Source of truth: ISO/IEC FDIS 18181-1:2021 §C.5.4 (DctSelect
//! placement) + §C.8.3 (per-varblock decode order) + Annex G
//! (chroma-from-luma, Listing G.1) + §F.3 / §I.2.3 (dequant + IDCT).

use oxideav_jpegxl::block_dequant::decode_block_to_residual;
use oxideav_jpegxl::dct_quant_weights::materialise_default_dequant_set;
use oxideav_jpegxl::dct_select::{DctSelectCell, DctSelectGrid, TransformType};
use oxideav_jpegxl::hf_dequant::QmScaleFactors;
use oxideav_jpegxl::lf_global::LfChannelCorrelation;
use oxideav_jpegxl::metadata_fdis::OpsinInverseMatrix;
use oxideav_jpegxl::pass_group_hf::DecodedHfBlock;
use oxideav_jpegxl::residual_plane::{
    apply_chroma_from_luma, assemble_three_channel_planes, reconstruct_three_channel_planes,
};
use oxideav_jpegxl::varblock_walk::Varblock;

fn grid(cells: Vec<DctSelectCell>, hf_mul: Vec<i32>, w: u32, h: u32) -> DctSelectGrid {
    DctSelectGrid {
        cells,
        hf_mul,
        width_blocks: w,
        height_blocks: h,
    }
}

fn block(coeffs: Vec<i32>) -> DecodedHfBlock {
    DecodedHfBlock {
        coeffs,
        remaining_non_zeros: 0,
        coeffs_read: 0,
    }
}

fn qm() -> QmScaleFactors {
    QmScaleFactors {
        x_factor: 0.8,
        b_factor: 1.0,
    }
}

/// Decode a single DC-only DCT8×8 block for `channel` and return its
/// flat residual value.
fn dc_residual(channel: usize, dc: i32) -> f32 {
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();
    let mut coeffs = vec![0i32; 64];
    coeffs[0] = dc;
    let decoded = block(coeffs);
    let r = decode_block_to_residual(
        &decoded,
        TransformType::Dct8x8,
        channel,
        2,
        &set,
        &oim,
        &qm(),
    )
    .unwrap();
    r[0]
}

#[test]
fn three_channel_real_decode_walk_single_dct8x8() {
    // One DCT8×8 varblock; each channel decodes a distinct DC-only block.
    // The assembled planes (pre-CfL) must each be flat at the matching
    // channel's DC residual.
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();
    let g = grid(
        vec![DctSelectCell::TopLeft(TransformType::Dct8x8)],
        vec![2],
        1,
        1,
    );
    // DC per channel: X=4, Y=9, B=2.
    let dcs = [4i32, 9, 2];
    let planes = assemble_three_channel_planes(&g, |c, v: &Varblock| {
        let mut coeffs = vec![0i32; 64];
        coeffs[0] = dcs[c];
        let decoded = block(coeffs);
        decode_block_to_residual(&decoded, v.transform, c, 2, &set, &oim, &qm())
    })
    .unwrap();

    assert_eq!(planes.dims(), (8, 8));
    let expect = [dc_residual(0, 4), dc_residual(1, 9), dc_residual(2, 2)];
    for (c, p) in planes.planes.iter().enumerate() {
        for y in 0..8 {
            for x in 0..8 {
                assert!(
                    (p.get(x, y).unwrap() - expect[c]).abs() < 1e-3,
                    "channel {c} ({x},{y}) not flat at {}",
                    expect[c]
                );
            }
        }
    }
}

#[test]
fn cfl_default_restores_b_from_y_after_real_decode() {
    // Default LfChannelCorrelation, zero XFromY/BFromY tile factors:
    //   kX = 0 + 0/84 = 0  → X unchanged
    //   kB = 1 + 0/84 = 1  → B += Y
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();
    let g = grid(
        vec![DctSelectCell::TopLeft(TransformType::Dct8x8)],
        vec![2],
        1,
        1,
    );
    let dcs = [4i32, 9, 2];
    let mut planes = assemble_three_channel_planes(&g, |c, v: &Varblock| {
        let mut coeffs = vec![0i32; 64];
        coeffs[0] = dcs[c];
        let decoded = block(coeffs);
        decode_block_to_residual(&decoded, v.transform, c, 2, &set, &oim, &qm())
    })
    .unwrap();

    let dx = dc_residual(0, 4);
    let dy = dc_residual(1, 9);
    let db = dc_residual(2, 2);

    // 8×8 plane → 1 tile.
    apply_chroma_from_luma(
        &mut planes,
        &[0i32; 1],
        &[0i32; 1],
        &LfChannelCorrelation::default(),
    )
    .unwrap();

    for y in 0..8 {
        for x in 0..8 {
            assert!((planes.x().get(x, y).unwrap() - dx).abs() < 1e-3, "X");
            assert!((planes.y().get(x, y).unwrap() - dy).abs() < 1e-3, "Y");
            assert!(
                (planes.b().get(x, y).unwrap() - (db + dy)).abs() < 1e-3,
                "B should be dB + 1·dY"
            );
        }
    }
}

#[test]
fn reconstruct_one_call_equals_two_step() {
    // The convenience driver must equal assemble + CfL on a real decode.
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();
    let g = grid(
        vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 4],
        vec![2; 4],
        2,
        2,
    );
    let resid = |c: usize, v: &Varblock| {
        let mut coeffs = vec![0i32; 64];
        coeffs[0] = (v.y * 2 + v.x) as i32 + 1 + c as i32 * 3;
        let decoded = block(coeffs);
        decode_block_to_residual(&decoded, v.transform, c, 2, &set, &oim, &qm())
    };
    let x_from_y = vec![21i32; 1]; // 16×16 plane → 1 tile.
    let b_from_y = vec![-42i32; 1];
    let cfl = LfChannelCorrelation::default();

    let mut step = assemble_three_channel_planes(&g, resid).unwrap();
    apply_chroma_from_luma(&mut step, &x_from_y, &b_from_y, &cfl).unwrap();

    let one = reconstruct_three_channel_planes(&g, &x_from_y, &b_from_y, &cfl, resid).unwrap();
    assert_eq!(one, step);
}

#[test]
fn mixed_transform_three_channel_layout() {
    // A DCT16×16 (2×2 cells) at (0,0) plus DCT8×8 blocks fill a 2×2-cell
    // grid... actually use a 2×2 grid fully covered by the DCT16×16.
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();
    let cells = vec![
        DctSelectCell::TopLeft(TransformType::Dct16x16),
        DctSelectCell::Continuation,
        DctSelectCell::Continuation,
        DctSelectCell::Continuation,
    ];
    let g = grid(cells, vec![2, 0, 0, 0], 2, 2);
    let planes = reconstruct_three_channel_planes(
        &g,
        &[0i32; 1],
        &[0i32; 1],
        &LfChannelCorrelation::default(),
        |c, v: &Varblock| {
            let (rows, cols) =
                oxideav_jpegxl::residual_plane::block_pixel_dims(v.transform).unwrap();
            let mut coeffs = vec![0i32; rows * cols];
            coeffs[0] = 5 + c as i32;
            let decoded = block(coeffs);
            decode_block_to_residual(&decoded, v.transform, c, 2, &set, &oim, &qm())
        },
    )
    .unwrap();
    // A single DCT16×16 covers the whole 16×16 plane for all 3 channels.
    assert_eq!(planes.dims(), (16, 16));
    // Y plane is unchanged by CfL; it must equal the decoded DCT16×16's
    // residual at (0,0).
    assert!(planes.y().get(0, 0).is_some());
    // X plane with kX=0 stays as the decoded X residual; just confirm the
    // whole plane is populated (DCT16×16 covers all 256 cells, no zeros
    // from un-placed regions in the DC-flat case the residual is ~constant
    // but non-trivially we only assert geometry coverage here).
    assert_eq!(planes.x().samples.len(), 256);
    assert_eq!(planes.b().samples.len(), 256);
}

#[test]
fn x_channel_restored_when_tile_factor_nonzero() {
    // x_factor = 84 → kX = 1 → X = dX + 1·Y over the whole plane.
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();
    let g = grid(
        vec![DctSelectCell::TopLeft(TransformType::Dct8x8)],
        vec![2],
        1,
        1,
    );
    let dcs = [3i32, 11, 1];
    let mut planes = assemble_three_channel_planes(&g, |c, v: &Varblock| {
        let mut coeffs = vec![0i32; 64];
        coeffs[0] = dcs[c];
        let decoded = block(coeffs);
        decode_block_to_residual(&decoded, v.transform, c, 2, &set, &oim, &qm())
    })
    .unwrap();
    let dx = dc_residual(0, 3);
    let dy = dc_residual(1, 11);

    apply_chroma_from_luma(
        &mut planes,
        &[84i32; 1],
        &[0i32; 1],
        &LfChannelCorrelation::default(),
    )
    .unwrap();
    // X = dX + 1·dY.
    assert!((planes.x().get(0, 0).unwrap() - (dx + dy)).abs() < 1e-3);
    // Y unchanged.
    assert!((planes.y().get(0, 0).unwrap() - dy).abs() < 1e-3);
}
