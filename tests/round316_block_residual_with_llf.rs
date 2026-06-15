//! Round 316 — per-block VarDCT residual decode **including** the LLF
//! (DC) coefficients folded in from the dequantised LF image.
//!
//! Earlier rounds landed every constituent primitive of the VarDCT
//! per-block path in isolation:
//!
//! * `vardct::compose_lf_to_llf_block` (round 129) — extract a varblock's
//!   `cy × cx` LF sub-block and run Listing I.16 to obtain the
//!   `cy × cx` LLF coefficient block;
//! * `block_dequant::dequant_block_for_transform` (round 286/300) — F.3
//!   dequantise the decoded HF coefficient block;
//! * `block_dequant::decode_block_to_residual` (round 286) — dequant +
//!   §I.2.3.2 inverse DCT of the **HF-only** coefficient block (every LLF
//!   cell zero).
//!
//! What was missing was the placement step that folds the separately
//! decoded LLF coefficients into the natural-order LLF prefix of the
//! coefficient grid before the inverse DCT — FDIS §I.2.4 (the natural
//! order is `LLF` followed by `HF`, with `LLF` the cells `(x < cx,
//! y < cy)`) feeding §I.2.3.2 (`samples = IDCT_2D(coefficients)` over the
//! *complete* matrix). Round 316 adds `block_dequant::merge_llf_into_block`
//! (the placement) and `block_dequant::decode_block_to_residual_with_llf`
//! (the LF-aware decode walk).
//!
//! These tests compose the *real* LF→LLF step (`compose_lf_to_llf_block`)
//! with the *real* dequant + IDCT walk, proving the full per-varblock
//! LF + HF → spatial-residual chain runs end-to-end.
//!
//! Source of truth: ISO/IEC FDIS 18181-1:2021 §I.2.4 (natural ordering,
//! LLF prefix) + §I.2.5 / Listing I.16 (LF → LLF) + §I.2.3.2
//! (coefficients → samples) + §F.3 (HF dequant).

use oxideav_jpegxl::block_dequant::{
    decode_block_to_residual, decode_block_to_residual_with_llf, merge_llf_into_block,
};
use oxideav_jpegxl::dct_quant_weights::materialise_default_dequant_set;
use oxideav_jpegxl::dct_select::TransformType;
use oxideav_jpegxl::hf_dequant::QmScaleFactors;
use oxideav_jpegxl::metadata_fdis::OpsinInverseMatrix;
use oxideav_jpegxl::pass_group_hf::DecodedHfBlock;
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

/// DCT8×8 pure-DC frame: a constant LF image feeds a single LLF (DC)
/// coefficient; with the HF stream empty the reconstructed varblock is a
/// flat 8×8 block. The whole chain (LF sub-block → Listing I.16 → merge
/// → IDCT) runs on the real primitives.
#[test]
fn dct8x8_pure_dc_from_lf_image_is_flat_block() {
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();

    // A 1×1 LF image (one 8×8 block) with a known DC value. For DCT8×8
    // cx = cy = 1, so the LLF block is the single LF sample unchanged
    // (Listing I.16 with the trivial 1×1 DCT_2D = identity, ScaleF = 1).
    let lf = vec![12.0f32];
    let llf = compose_lf_to_llf_block(&lf, 1, 1, 0, 0, TransformType::Dct8x8).unwrap();
    // ScaleF(1, 8, 0) is computed via trig and is 1.0 up to f32 rounding,
    // so the DCT8×8 LLF is the LF sample within float tolerance.
    assert_eq!(llf.len(), 1);
    assert!((llf[0] - 12.0).abs() < 1e-3, "llf={:?}", llf);

    // HF stream is empty.
    let b = block(vec![0i32; 64]);
    let residual =
        decode_block_to_residual_with_llf(&b, TransformType::Dct8x8, 1, 3, &set, &oim, &qm(), &llf)
            .unwrap();
    assert_eq!(residual.len(), 64);
    let s0 = residual[0];
    assert!(s0 != 0.0, "DC must propagate a non-zero spatial value");
    for (i, &v) in residual.iter().enumerate() {
        assert!(
            (v - s0).abs() < 1e-4,
            "pure-DC block must be flat: cell {i} = {v} != {s0}"
        );
    }
}

/// The LF-aware walk equals the HF-only walk **plus** a manual LLF merge
/// then IDCT — i.e. `decode_block_to_residual_with_llf` is exactly the
/// composition `dequant → merge_llf → idct`, with the HF tail decoded
/// identically to the HF-only path.
#[test]
fn with_llf_walk_equals_manual_merge_then_idct() {
    use oxideav_jpegxl::idct::idct_for_transform;

    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();

    // DCT16×16 HF block with a couple of non-DC coefficients set.
    let mut coeffs = vec![0i32; 256];
    coeffs[40] = 6; // an HF cell
    coeffs[100] = -3; // another HF cell
    let b = block(coeffs);

    // A 2×2 LF image → cy×cx = 2×2 LLF block for DCT16×16.
    let lf = vec![3.0f32, -1.0, 0.5, 2.0];
    let llf = compose_lf_to_llf_block(&lf, 2, 2, 0, 0, TransformType::Dct16x16).unwrap();
    assert_eq!(llf.len(), 4);

    // Reference: dequant via the HF-only path is NOT directly usable
    // (it IDCTs immediately). Instead reconstruct the dequant grid by
    // running the HF-only IDCT inverse is not available; rebuild the
    // dequant grid manually using merge on a fresh dequant block.
    // We use `decode_block_to_residual` to obtain the HF-only IDCT and
    // assert the LF-aware path differs by exactly the IDCT of the merged
    // grid (linearity of IDCT): residual_with_llf == residual_hf_only +
    // idct(grid_with_only_llf).
    let hf_only =
        decode_block_to_residual(&b, TransformType::Dct16x16, 0, 5, &set, &oim, &qm()).unwrap();

    let with_llf = decode_block_to_residual_with_llf(
        &b,
        TransformType::Dct16x16,
        0,
        5,
        &set,
        &oim,
        &qm(),
        &llf,
    )
    .unwrap();

    // IDCT of a grid carrying only the LLF coefficients (HF cells zero).
    let mut llf_only_grid = vec![0.0f32; 256];
    merge_llf_into_block(&mut llf_only_grid, TransformType::Dct16x16, &llf).unwrap();
    let llf_contribution = idct_for_transform(TransformType::Dct16x16, &llf_only_grid).unwrap();

    assert_eq!(with_llf.len(), 256);
    for i in 0..256 {
        let expected = hf_only[i] + llf_contribution[i];
        assert!(
            (with_llf[i] - expected).abs() < 1e-3,
            "cell {i}: with_llf={} != hf_only+llf={}",
            with_llf[i],
            expected
        );
    }
}

/// A varblock not at the origin: `compose_lf_to_llf_block` samples the LF
/// sub-block at the varblock's block-grid origin, and the result feeds
/// the merge + IDCT unchanged. Confirms the LF-grid addressing threads
/// through to the spatial residual.
#[test]
fn non_origin_varblock_samples_correct_lf_subblock() {
    let set = materialise_default_dequant_set().unwrap();
    let oim = OpsinInverseMatrix::default();

    // 4×4 LF image; DCT8×8 varblock at block-grid (2, 3) → LF sample
    // index 3 * 4 + 2 = 14.
    let mut lf = vec![0.0f32; 16];
    lf[14] = 7.5;
    let llf = compose_lf_to_llf_block(&lf, 4, 4, 2, 3, TransformType::Dct8x8).unwrap();
    assert_eq!(llf.len(), 1);
    assert!(
        (llf[0] - 7.5).abs() < 1e-3,
        "DCT8×8 LLF is the addressed LF sample: {:?}",
        llf
    );

    let b = block(vec![0i32; 64]);
    let residual =
        decode_block_to_residual_with_llf(&b, TransformType::Dct8x8, 1, 3, &set, &oim, &qm(), &llf)
            .unwrap();
    let s0 = residual[0];
    assert!(s0 != 0.0);
    assert!(residual.iter().all(|&v| (v - s0).abs() < 1e-4));
}
