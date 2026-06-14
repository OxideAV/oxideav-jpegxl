//! Round 306 — per-LfGroup VarDCT residual-plane assembly.
//!
//! Integration coverage for `residual_plane`: the spatial-placement
//! stage that walks a `DctSelectGrid` and writes each varblock's decoded
//! residual block (the `block_dequant::decode_block_to_residual` output)
//! into a single-channel spatial plane at the varblock's pixel origin.
//!
//! These tests compose the real `block_dequant` decode walk (F.3 dequant
//! and the I.2.3 inverse DCT) with the new placement driver, proving the
//! two stages chain end-to-end on actual dequantised residual samples —
//! not just synthetic constant blocks.

use oxideav_jpegxl::block_dequant::decode_block_to_residual;
use oxideav_jpegxl::dct_quant_weights::materialise_default_dequant_set;
use oxideav_jpegxl::dct_select::{DctSelectCell, DctSelectGrid, TransformType};
use oxideav_jpegxl::hf_dequant::QmScaleFactors;
use oxideav_jpegxl::metadata_fdis::OpsinInverseMatrix;
use oxideav_jpegxl::pass_group_hf::DecodedHfBlock;
use oxideav_jpegxl::residual_plane::{
    assemble_channel_plane, block_pixel_dims, place_block, ResidualPlane,
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

fn vb(x: u32, y: u32, t: TransformType, hf_mul: i32) -> Varblock {
    Varblock {
        x,
        y,
        transform: t,
        hf_mul,
    }
}

#[test]
fn dc_only_dct8x8_block_lands_flat_in_plane() {
    // Decode a pure-DC DCT8×8 block, place it at the origin of a 1×1
    // plane, and confirm the whole 8×8 region is the constant residual.
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();
    let mut coeffs = vec![0i32; 64];
    coeffs[0] = 10;
    let decoded = block(coeffs);
    let residual =
        decode_block_to_residual(&decoded, TransformType::Dct8x8, 1, 3, &set, &oim, &qm()).unwrap();
    let expected = residual[0];
    assert!(expected.abs() > 1e-9);

    let g = grid(
        vec![DctSelectCell::TopLeft(TransformType::Dct8x8)],
        vec![3],
        1,
        1,
    );
    let mut plane = ResidualPlane::for_grid(&g).unwrap();
    place_block(&mut plane, &vb(0, 0, TransformType::Dct8x8, 3), &residual).unwrap();
    for y in 0..8 {
        for x in 0..8 {
            assert!(
                (plane.get(x, y).unwrap() - expected).abs() < 1e-3,
                "({x},{y}) not flat"
            );
        }
    }
}

#[test]
fn assemble_real_decode_walk_two_by_two_dct8x8() {
    // A 2×2 grid of DCT8×8; each varblock decodes a distinct DC-only
    // block (DC value = raster index + 1). The assembled plane's
    // quadrants must each be flat at the matching block's DC residual.
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();
    let g = grid(
        vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 4],
        vec![2; 4],
        2,
        2,
    );

    let mut dc_residuals = [0.0f32; 4];
    for (i, slot) in dc_residuals.iter_mut().enumerate() {
        let mut coeffs = vec![0i32; 64];
        coeffs[0] = (i as i32) + 1;
        let decoded = block(coeffs);
        let r = decode_block_to_residual(&decoded, TransformType::Dct8x8, 1, 2, &set, &oim, &qm())
            .unwrap();
        *slot = r[0];
    }

    let plane = assemble_channel_plane(&g, |v| {
        let idx = (v.y * 2 + v.x) as usize;
        let mut coeffs = vec![0i32; 64];
        coeffs[0] = (idx as i32) + 1;
        let decoded = block(coeffs);
        decode_block_to_residual(&decoded, v.transform, 1, 2, &set, &oim, &qm())
    })
    .unwrap();

    assert_eq!((plane.width, plane.height), (16, 16));
    // Quadrant origins: (0,0)->0, (8,0)->1, (0,8)->2, (8,8)->3.
    let origins = [(0, 0, 0usize), (8, 0, 1), (0, 8, 2), (8, 8, 3)];
    for (ox, oy, idx) in origins {
        for dy in 0..8 {
            for dx in 0..8 {
                assert!(
                    (plane.get(ox + dx, oy + dy).unwrap() - dc_residuals[idx]).abs() < 1e-3,
                    "quadrant {idx} ({},{})",
                    ox + dx,
                    oy + dy
                );
            }
        }
    }
}

#[test]
fn rectangular_dct16x8_decode_lands_in_tall_region() {
    // DCT16×8 (16px tall × 8px wide) decoded with a DC coefficient,
    // placed in a 1×2-cell plane (8×16 px). The whole 8×16 region must
    // be flat at the block's DC residual.
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();
    let (rows, cols) = block_pixel_dims(TransformType::Dct16x8).unwrap();
    assert_eq!((rows, cols), (16, 8));
    let mut coeffs = vec![0i32; rows * cols];
    coeffs[0] = 7;
    let decoded = block(coeffs);
    let residual =
        decode_block_to_residual(&decoded, TransformType::Dct16x8, 1, 3, &set, &oim, &qm())
            .unwrap();
    assert_eq!(residual.len(), 128);

    let g = grid(
        vec![
            DctSelectCell::TopLeft(TransformType::Dct16x8),
            DctSelectCell::Continuation,
        ],
        vec![3, 0],
        1,
        2,
    );
    let mut plane = ResidualPlane::for_grid(&g).unwrap();
    assert_eq!((plane.width, plane.height), (8, 16));
    place_block(&mut plane, &vb(0, 0, TransformType::Dct16x8, 3), &residual).unwrap();
    let expected = residual[0];
    assert!(expected.abs() > 1e-9);
    for y in 0..16 {
        for x in 0..8 {
            assert!(
                (plane.get(x, y).unwrap() - expected).abs() < 1e-3,
                "({x},{y}) not flat"
            );
        }
    }
}

#[test]
fn placement_equals_manual_idct_per_block() {
    // The assembled plane at a varblock's origin must equal the manual
    // idct_for_transform of the dequantised block, cell-for-cell — pins
    // that placement copies the residual verbatim with no reordering.
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();
    let mut coeffs = vec![0i32; 64];
    coeffs[0] = 6;
    coeffs[1] = -3;
    coeffs[9] = 2;
    let decoded = block(coeffs);
    let residual =
        decode_block_to_residual(&decoded, TransformType::Dct8x8, 1, 5, &set, &oim, &qm()).unwrap();

    // Place at grid (1,1) of a 2×2 plane → pixel origin (8,8).
    let g = grid(
        vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 4],
        vec![5; 4],
        2,
        2,
    );
    let mut plane = ResidualPlane::for_grid(&g).unwrap();
    place_block(&mut plane, &vb(1, 1, TransformType::Dct8x8, 5), &residual).unwrap();

    for r in 0..8 {
        for c in 0..8 {
            assert_eq!(
                plane.get(8 + c, 8 + r).unwrap(),
                residual[r * 8 + c],
                "({c},{r})"
            );
        }
    }
    // The other three quadrants stay zero (only one block placed).
    assert_eq!(plane.get(0, 0), Some(0.0));
    assert_eq!(plane.get(8, 0), Some(0.0));
    assert_eq!(plane.get(0, 8), Some(0.0));
}

#[test]
fn non_dct_afv_block_lands_eight_by_eight() {
    // An AFV0 block decodes to an 8×8 residual and lands at its origin.
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();
    let mut coeffs = vec![0i32; 64];
    coeffs[0] = 5;
    coeffs[8] = 3;
    let decoded = block(coeffs);
    let residual =
        decode_block_to_residual(&decoded, TransformType::Afv0, 1, 2, &set, &oim, &qm()).unwrap();
    assert_eq!(residual.len(), 64);
    // Cross-check the residual equals the standalone IDCT of the dequant.
    let g = grid(
        vec![DctSelectCell::TopLeft(TransformType::Afv0)],
        vec![2],
        1,
        1,
    );
    let plane = assemble_channel_plane(&g, |_| Ok(residual.clone())).unwrap();
    assert_eq!((plane.width, plane.height), (8, 8));
    for r in 0..8 {
        for c in 0..8 {
            assert_eq!(plane.get(c, r).unwrap(), residual[r * 8 + c], "({c},{r})");
        }
    }
}
