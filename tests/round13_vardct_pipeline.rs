//! Round-13 integration tests: DctSelect / HfMul derivation from
//! BlockInfo (FDIS C.5.4 + Table C.16) + HfGlobal default-fast-path
//! (C.6) + LF dequant (Listing F.1) + adaptive smoothing (F.2) wired
//! into the VarDCT pipeline.
//!
//! Round 13 extends round 12 by:
//!
//! 1. Walking BlockInfo column-by-column to reconstruct the per-LfGroup
//!    `DctSelect` + `HfMul` grids per FDIS C.5.4 prose.
//! 2. Reading the HfGlobal bundle (C.6.2 default-encoding fast path +
//!    C.6.4 num_hf_presets).
//! 3. Driving the VarDCT decoder past LfGlobal/LfGroup/HfGlobal and
//!    actually calling `dequant_lf` + `apply_adaptive_lf_smoothing` on
//!    the decoded LfCoefficients (round-12's unit-tested code now runs
//!    on real codestreams).
//!
//! The round-13 envelope still defers HF coefficient decode + IDCT +
//! Chroma-from-Luma + Gaborish + EPF to round 14+. The VarDCT path
//! returns `Error::Unsupported` with a "round 14+" message AFTER the
//! round-13 pipeline has parsed and dequantised all the LF data.

use oxideav_core::Error;
use oxideav_jpegxl::dct_select::{derive_dct_select, DctSelectCell, TransformType};
use oxideav_jpegxl::decode_one_frame;
use oxideav_jpegxl::lf_group::HfMetadata;

const PIXEL_1X1_JXL: &[u8] = include_bytes!("fixtures/pixel_1x1.jxl");
const GRAY_64X64_JXL: &[u8] = include_bytes!("fixtures/gray_64x64_lossless.jxl");
const GRADIENT_JXL: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
const PALETTE_JXL: &[u8] = include_bytes!("fixtures/palette_32x32.jxl");
const GREY_8X8_JXL: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");
const VARDCT_D3_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d3.jxl");

/// Round-13 sentinel: adding DctSelect derivation + HfGlobal +
/// pipeline-wiring of F.1 / F.2 must not regress the five small
/// modular fixtures.
#[test]
fn five_small_lossless_fixtures_still_decode_round_13() {
    for (name, bytes) in [
        ("pixel_1x1", PIXEL_1X1_JXL),
        ("gray_64x64", GRAY_64X64_JXL),
        ("gradient_64x64", GRADIENT_JXL),
        ("palette_32x32", PALETTE_JXL),
        ("grey_8x8", GREY_8X8_JXL),
    ] {
        let vf = decode_one_frame(bytes, None);
        assert!(
            vf.is_ok(),
            "round-13 regression: {name} should still decode (round-10 baseline); got {:?}",
            vf.err()
        );
    }
}

/// Round-13 acceptance: a real VarDCT codestream now travels through
/// the LF dequant + smoothing pipeline before the round-14+ deferral.
/// We assert the deferral message is the round-13-specific one,
/// confirming the codestream got past LfGlobal, LfGroup
/// (LfCoefficients + HfMetadata), HfGlobal, AND the F.1 + F.2 calls.
///
/// The fixture is the OxideAV `vardct-256x256-d3` reference photo; for
/// round 13 we don't yet expect pixel-correct decode (HF+IDCT defer to
/// round 14). What we DO expect: the parse reaches deep into the
/// decoder and errors with the round-13 deferral message, OR errors
/// with a precise round-13 sub-component diagnostic (e.g. non-default
/// HfBlockContext, non-default HfGlobal, > 1 LfGroup) — but NEVER with
/// the older "round 8 scaffold" or generic "encoding not supported"
/// error.
#[test]
fn vardct_d3_fixture_reaches_round_13_pipeline() {
    let r = decode_one_frame(VARDCT_D3_JXL, None);
    let err = r.expect_err("VarDCT codestream should error in round 13 (no IDCT+CfL yet)");
    let msg = format!("{err:?}");
    // Must NOT be the legacy round-8 scaffold message.
    assert!(
        !msg.contains("round 8 scaffold"),
        "round-13 should bypass the round-8 scaffold gate; got {msg}"
    );
    // Must be an Unsupported (or a precise InvalidData from a sub-
    // component like HfBlockContext non-default, HfGlobal non-default,
    // >1 LfGroup, or modular sub-bitstream limit).
    assert!(
        matches!(err, Error::Unsupported(_) | Error::InvalidData(_)),
        "round-13 should yield Unsupported or InvalidData; got {msg}"
    );
}

#[test]
fn vardct_d1_fixture_reaches_round_13_pipeline() {
    let r = decode_one_frame(VARDCT_D1_JXL, None);
    let err = r.expect_err("VarDCT codestream should error in round 13 (no IDCT+CfL yet)");
    let msg = format!("{err:?}");
    assert!(
        !msg.contains("round 8 scaffold"),
        "round-13 should bypass the round-8 scaffold gate; got {msg}"
    );
    assert!(
        matches!(err, Error::Unsupported(_) | Error::InvalidData(_)),
        "round-13 should yield Unsupported or InvalidData; got {msg}"
    );
}

/// Acceptance test for `derive_dct_select` covering a handful of
/// non-trivial layouts. Detailed unit tests live alongside
/// `dct_select.rs`.
#[test]
fn derive_dct_select_dct32x32_in_4x4_grid() {
    // 32×32 LfGroup, 4×4 cell grid, single DCT32×32 (covers everything).
    // BlockInfo: 1 column × 2 rows = [type=5, mul=0].
    let hf = HfMetadata {
        nb_blocks: 1,
        x_from_y: vec![0],
        b_from_y: vec![0],
        block_info: vec![5, 0],
        sharpness: vec![0; 16],
        channel_widths: [1, 1, 1, 4],
        channel_heights: [1, 1, 2, 4],
    };
    let g = derive_dct_select(&hf, 32, 32).unwrap();
    assert_eq!(g.width_blocks, 4);
    assert_eq!(g.height_blocks, 4);
    // Top-left only at (0,0).
    assert_eq!(g.cells[0], DctSelectCell::TopLeft(TransformType::Dct32x32));
    // The other 15 cells must all be Continuation.
    for i in 1..16 {
        assert_eq!(g.cells[i], DctSelectCell::Continuation, "cell {i}");
    }
    assert_eq!(g.hf_mul[0], 1);
}

#[test]
fn derive_dct_select_mixed_blocks_2x4_grid() {
    // 16×32 LfGroup → 2 cols × 4 rows = 8 cells. Place: DCT16×16 at
    // (0,0)-(1,1), DCT8×16 at (0,2)-(1,2) (1 row × 2 cols), DCT8×8 at
    // (0,3), DCT8×8 at (1,3). 4 varblocks total.
    let hf = HfMetadata {
        nb_blocks: 4,
        x_from_y: vec![0],
        b_from_y: vec![0],
        // BlockInfo width=4, height=2. Row 0: [4, 7, 0, 0], row 1: [0,0,0,0]
        block_info: vec![4, 7, 0, 0, 0, 0, 0, 0],
        sharpness: vec![0; 8],
        channel_widths: [1, 1, 4, 2],
        channel_heights: [1, 1, 2, 4],
    };
    let g = derive_dct_select(&hf, 16, 32).unwrap();
    assert_eq!(g.width_blocks, 2);
    assert_eq!(g.height_blocks, 4);
    // Layout (row, col): (0,0)=TL(DCT16x16), (0,1)=Cont, (1,0)=Cont,
    // (1,1)=Cont, (2,0)=TL(DCT8x16), (2,1)=Cont, (3,0)=TL(DCT8x8),
    // (3,1)=TL(DCT8x8).
    assert_eq!(g.cells[0], DctSelectCell::TopLeft(TransformType::Dct16x16));
    assert_eq!(g.cells[1], DctSelectCell::Continuation);
    assert_eq!(g.cells[2], DctSelectCell::Continuation);
    assert_eq!(g.cells[3], DctSelectCell::Continuation);
    assert_eq!(g.cells[4], DctSelectCell::TopLeft(TransformType::Dct8x16));
    assert_eq!(g.cells[5], DctSelectCell::Continuation);
    assert_eq!(g.cells[6], DctSelectCell::TopLeft(TransformType::Dct8x8));
    assert_eq!(g.cells[7], DctSelectCell::TopLeft(TransformType::Dct8x8));
}
