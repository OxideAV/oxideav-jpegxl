//! Round 328 — §6.2 right/bottom crop of the padded VarDCT
//! reconstruction.
//!
//! Rounds 306..322 built the per-LfGroup VarDCT three-channel residual
//! reconstruction (`reconstruct_three_channel_planes_with_lf` and its
//! HF-only / non-CfL siblings). Every one of those drivers produces XYB
//! planes on the **padded block grid**: a plane is
//! `width_blocks·8 × height_blocks·8` with
//! `width_blocks = ceil(lf_w / 8)`, `height_blocks = ceil(lf_h / 8)`
//! (§C.5.4), so a varblock on the right/bottom edge covers pixels past
//! the channel's logical `lf_w × lf_h` extent. Every reconstruct
//! function's doc comment ended with "the caller crops to lf_w × lf_h" —
//! but no crop primitive existed.
//!
//! §6.2 (Group splitting) is the normative source: "The decoder ensures
//! the decoded image has the dimensions specified in SizeHeader by
//! cropping at the right and bottom as necessary." Round 328 lands
//! `ResidualPlane::crop_to` and `ChannelResidualPlanes::crop_to`, the
//! shrink-only top-left-rectangle crop that turns the padded
//! reconstruction into the logical channel extent.
//!
//! This test runs the *real* reconstruct path (round 306
//! `reconstruct_three_channel_planes`) on a 2×2-block grid (16×16 padded
//! pixels), then crops the assembled three-channel planes to a 13×11
//! logical extent and checks that every retained sample is exactly the
//! pre-crop sample at the same coordinate — no resampling, no edge
//! handling, just the §6.2 right/bottom truncation.
//!
//! Source of truth: ISO/IEC FDIS 18181-1:2021 §6.2 (group splitting /
//! right-bottom crop) + §C.5.4 (DctSelect placement padding).

use oxideav_jpegxl::dct_select::{DctSelectCell, DctSelectGrid, TransformType};
use oxideav_jpegxl::lf_global::LfChannelCorrelation;
use oxideav_jpegxl::residual_plane::reconstruct_three_channel_planes;
use oxideav_jpegxl::varblock_walk::Varblock;

/// A 2×2 grid of DCT8×8 varblocks → a 16×16-pixel padded plane.
fn grid_2x2() -> DctSelectGrid {
    DctSelectGrid {
        cells: vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 4],
        hf_mul: vec![1; 4],
        width_blocks: 2,
        height_blocks: 2,
    }
}

#[test]
fn reconstruct_then_crop_to_logical_extent() {
    let g = grid_2x2();

    // A per-(channel, varblock) residual block whose every sample encodes
    // (channel, vb origin) so a mis-addressed crop would be visible.
    let residual_at = |c: usize, vb: &Varblock| -> oxideav_core::Result<Vec<f32>> {
        let base = (c as f32) * 1000.0 + (vb.y as f32) * 100.0 + (vb.x as f32) * 10.0;
        Ok((0..64).map(|i| base + i as f32).collect())
    };

    // Identity CfL: x_from_y = b_from_y = 0 over the single 64×64 tile, so
    // the planes carry the placed residuals unchanged (the crop semantics
    // are what this test pins, not CfL).
    let x_from_y = vec![0i32; 1];
    let b_from_y = vec![0i32; 1];
    let cfl = LfChannelCorrelation::default();

    let full =
        reconstruct_three_channel_planes(&g, &x_from_y, &b_from_y, &cfl, residual_at).unwrap();
    assert_eq!(full.dims(), (16, 16));

    // §6.2 crop to a 13×11 logical extent (ceil/8 = 2×2 blocks, the same
    // padded grid — the right/bottom 3 columns and 5 rows are padding).
    let cropped = full.crop_to(13, 11).unwrap();
    assert_eq!(cropped.dims(), (13, 11));

    for ch in 0..3 {
        let src = &full.planes[ch];
        let dst = &cropped.planes[ch];
        assert_eq!((dst.width, dst.height), (13, 11), "channel {ch} dims");
        for y in 0..11 {
            for x in 0..13 {
                assert_eq!(
                    dst.get(x, y),
                    src.get(x, y),
                    "channel {ch} pixel ({x},{y}) changed by crop"
                );
            }
        }
        // The dropped right/bottom region is genuinely gone.
        assert_eq!(dst.get(13, 0), None, "channel {ch} kept a dropped column");
        assert_eq!(dst.get(0, 11), None, "channel {ch} kept a dropped row");
    }
}

#[test]
fn crop_to_exact_multiple_of_eight_is_identity() {
    // An LfGroup whose logical extent is already a multiple of 8 needs no
    // truncation: crop_to the padded dims returns the planes unchanged.
    let g = grid_2x2();
    let residual_at =
        |_c: usize, _vb: &Varblock| -> oxideav_core::Result<Vec<f32>> { Ok(vec![1.5f32; 64]) };
    let x_from_y = vec![0i32; 1];
    let b_from_y = vec![0i32; 1];
    let cfl = LfChannelCorrelation::default();

    let full =
        reconstruct_three_channel_planes(&g, &x_from_y, &b_from_y, &cfl, residual_at).unwrap();
    let cropped = full.crop_to(16, 16).unwrap();
    assert_eq!(cropped, full);
}
