//! Round-129 integration tests — `vardct::extract_lf_subblock` +
//! `vardct::compose_lf_to_llf_block` + `compose_lf_to_llf_block_3ch`
//! as the public composition surface for the §I.2.5 LF→LLF step.
//!
//! These tests pin the public-API contract added in round 129 and
//! verify it composes correctly with the round-12
//! [`oxideav_jpegxl::lf_dequant`] output and the round-121
//! [`oxideav_jpegxl::llf_from_lf`] pure-math step.
//!
//! Round-129 deliverable: the geometry helper that drives
//! `llf_from_lf` from a single channel's dequantised LF samples for
//! a single varblock placement. This is the glue a future round
//! will wire into `decode_codestream` once the HF coefficient ANS
//! decode lands (round 91+ blocker — see CHANGELOG).

use oxideav_jpegxl::dct_select::TransformType;
use oxideav_jpegxl::lf_dequant::LfDequantOutput;
use oxideav_jpegxl::llf_from_lf::scale_f;
use oxideav_jpegxl::vardct::{
    compose_lf_to_llf_block, compose_lf_to_llf_block_3ch, extract_lf_subblock,
};

/// `extract_lf_subblock` reads the correct row-major slice for a
/// DCT16×16 varblock at the centre of a 4×4 LF grid.
#[test]
fn extract_lf_subblock_dct16x16_centre_origin() {
    let lf: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let sub = extract_lf_subblock(&lf, 4, 4, 1, 1, TransformType::Dct16x16).unwrap();
    // Origin (1, 1); 2×2 sub-block reads LF positions
    // (1,1) (1,2) (2,1) (2,2) = LF indices 5, 6, 9, 10.
    assert_eq!(sub, vec![5.0, 6.0, 9.0, 10.0]);
}

/// `compose_lf_to_llf_block` produces a single-cell DC output equal
/// to the LF sample (up to f32 ε) for every DCT8×8 varblock at every
/// LF origin.
#[test]
fn compose_dct8x8_returns_input_sample_for_every_origin() {
    let lf: Vec<f32> = (0..16).map(|i| (i as f32) - 8.0).collect();
    for by in 0..4 {
        for bx in 0..4 {
            let llf = compose_lf_to_llf_block(&lf, 4, 4, bx, by, TransformType::Dct8x8).unwrap();
            assert_eq!(llf.len(), 1);
            let expected = lf[(by * 4 + bx) as usize];
            // DCT8×8 scales by `ScaleF(1, 8, 0)^2`; the spec value
            // is 1.0 but `dct_2d` for 1×1 input still loses ~ulp.
            assert!(
                (llf[0] - expected).abs() < 1e-5,
                "({bx}, {by}): got {} want {}",
                llf[0],
                expected,
            );
        }
    }
}

/// Constant DCT16×16 sub-block has only DC; the AC cells are zero
/// (within f32 ε). Pins the FDIS Listing I.16 scaling factor
/// `ScaleF(2, 16, 0)^2 ≈ 0.5` on the DC cell.
#[test]
fn compose_dct16x16_constant_sub_block_has_dc_only() {
    let lf: Vec<f32> = (0..16).map(|_| 9.0f32).collect();
    let llf = compose_lf_to_llf_block(&lf, 4, 4, 0, 0, TransformType::Dct16x16).unwrap();
    assert_eq!(llf.len(), 4);
    let sf = scale_f(2, 16, 0);
    let dc = 9.0 * sf * sf;
    assert!((llf[0] - dc).abs() < 1e-5, "DC = {} want {}", llf[0], dc);
    for v in &llf[1..] {
        assert!(v.abs() < 1e-4, "AC = {v}");
    }
}

/// `compose_lf_to_llf_block_3ch` rejects subsampled inputs where the
/// three channels have different LF dims.
#[test]
fn compose_3ch_rejects_subsampled_channels() {
    let lf = LfDequantOutput {
        samples: [vec![0.0; 16], vec![0.0; 4], vec![0.0; 16]],
        widths: [4, 2, 4],
        heights: [4, 2, 4],
    };
    let err = compose_lf_to_llf_block_3ch(&lf, 0, 0, TransformType::Dct8x8);
    assert!(err.is_err());
}

/// `compose_lf_to_llf_block_3ch` returns three independent LLF
/// blocks for a non-subsampled `LfDequantOutput`.
#[test]
fn compose_3ch_dct16x16_three_constant_channels() {
    let lf = LfDequantOutput {
        samples: [vec![1.0f32; 16], vec![2.0f32; 16], vec![-3.0f32; 16]],
        widths: [4, 4, 4],
        heights: [4, 4, 4],
    };
    let blocks = compose_lf_to_llf_block_3ch(&lf, 2, 2, TransformType::Dct16x16).unwrap();
    let sf = scale_f(2, 16, 0);
    for (c, &dc_in) in [1.0f32, 2.0, -3.0].iter().enumerate() {
        assert_eq!(blocks[c].len(), 4, "channel {c}");
        let want = dc_in * sf * sf;
        assert!(
            (blocks[c][0] - want).abs() < 1e-5,
            "channel {c}: DC {} want {}",
            blocks[c][0],
            want,
        );
        for v in &blocks[c][1..] {
            assert!(v.abs() < 1e-4, "channel {c}: AC {v}");
        }
    }
}

/// The non-DCT transforms (Hornuss / DCT2×2 / DCT4×4 / DCT4×8 /
/// DCT8×4 / AFV0..AFV3) all return a single-cell LLF equal to the
/// LF sample at the origin, per FDIS §I.2.5 closing sentence.
#[test]
fn compose_non_dct_pass_through_for_all_variants() {
    let lf: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 1.0).collect();
    for t in [
        TransformType::Hornuss,
        TransformType::Dct2x2,
        TransformType::Dct4x4,
        TransformType::Dct4x8,
        TransformType::Dct8x4,
        TransformType::Afv0,
        TransformType::Afv1,
        TransformType::Afv2,
        TransformType::Afv3,
    ] {
        let llf = compose_lf_to_llf_block(&lf, 4, 4, 1, 2, t).unwrap();
        assert_eq!(llf.len(), 1, "{t:?}");
        assert!(
            (llf[0] - lf[2 * 4 + 1]).abs() < 1e-6,
            "{t:?}: got {} want {}",
            llf[0],
            lf[2 * 4 + 1],
        );
    }
}

/// Rectangular DCT16×8 / DCT8×16 / DCT32×8 / DCT8×32 placements all
/// extract the correct sub-block + scale per Listing I.16.
#[test]
fn compose_rectangular_transforms_byte_exact() {
    // 8×8 LF grid, all samples = 4.0. For each rectangular
    // transform, the DC should be `4.0 * ScaleF(cy, bheight, 0) *
    // ScaleF(cx, bwidth, 0)` and every AC should be ε-zero.
    let lf = vec![4.0f32; 64];
    let cases = [
        (TransformType::Dct16x8, 1u32, 2u32, 8u32, 16u32),
        (TransformType::Dct8x16, 2, 1, 16, 8),
        (TransformType::Dct32x8, 1, 4, 8, 32),
        (TransformType::Dct8x32, 4, 1, 32, 8),
        (TransformType::Dct16x32, 4, 2, 32, 16),
        (TransformType::Dct32x16, 2, 4, 16, 32),
    ];
    for (t, cx, cy, bwidth, bheight) in cases {
        let llf = compose_lf_to_llf_block(&lf, 8, 8, 0, 0, t).unwrap();
        assert_eq!(llf.len(), (cx * cy) as usize, "{t:?}");
        let dc_want = 4.0 * scale_f(cy, bheight, 0) * scale_f(cx, bwidth, 0);
        assert!(
            (llf[0] - dc_want).abs() < 1e-5,
            "{t:?}: DC {} want {}",
            llf[0],
            dc_want,
        );
        for (i, v) in llf.iter().enumerate().skip(1) {
            assert!(v.abs() < 1e-4, "{t:?}: AC[{i}] = {v}");
        }
    }
}

/// Verify the round-129 helpers reject every kind of out-of-bounds
/// varblock placement.
#[test]
fn compose_rejects_oob_in_every_direction() {
    let lf = vec![0.0f32; 16]; // 4×4 grid

    // OOB in x dimension (DCT16x16 needs 2×2; origin (3, 0) reads
    // cols 3..5 > 4).
    assert!(compose_lf_to_llf_block(&lf, 4, 4, 3, 0, TransformType::Dct16x16).is_err(),);
    // OOB in y dimension (origin (0, 3)).
    assert!(compose_lf_to_llf_block(&lf, 4, 4, 0, 3, TransformType::Dct16x16).is_err(),);
    // OOB in both directions (origin (3, 3)).
    assert!(compose_lf_to_llf_block(&lf, 4, 4, 3, 3, TransformType::Dct16x16).is_err(),);
    // DCT32×32 (4×4 block) on a 4×4 grid fits at (0, 0) but not at
    // (1, 0).
    assert!(compose_lf_to_llf_block(&lf, 4, 4, 1, 0, TransformType::Dct32x32).is_err(),);
    assert!(compose_lf_to_llf_block(&lf, 4, 4, 0, 0, TransformType::Dct32x32).is_ok(),);
}

/// Dimension-mismatch in `lf_samples` (vs `lf_width * lf_height`) is
/// caught early; the function never index-panics on malformed input.
#[test]
fn extract_lf_subblock_handles_short_buffer_without_panic() {
    let lf = vec![0.0f32; 5]; // claim 4×4 = 16 needed
    let r = extract_lf_subblock(&lf, 4, 4, 0, 0, TransformType::Dct8x8);
    assert!(r.is_err());
    // Also a too-long buffer.
    let lf2 = vec![0.0f32; 17];
    let r2 = extract_lf_subblock(&lf2, 4, 4, 0, 0, TransformType::Dct8x8);
    assert!(r2.is_err());
}
