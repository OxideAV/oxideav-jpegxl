//! Round-14 integration tests: HfBlockContext non-default-table branch
//! (ISO/IEC 18181-1:2024 §I.2.2 custom branch) + HfGlobal §I.2.4
//! non-default-encoding parse (Listing C.10 modes) +
//! `GetDCTQuantWeights` parameter ingestion.
//!
//! These two pieces are the **pre-flight** for round 15+'s HF
//! coefficient decode + IDCT dispatch. Round 14 lands the *parse*: the
//! per-slot encoding modes are recognised, parameters captured, and
//! Table I.5 valid-index constraints enforced.

use oxideav_core::Error;
use oxideav_jpegxl::decode_one_frame;

const PIXEL_1X1_JXL: &[u8] = include_bytes!("fixtures/pixel_1x1.jxl");
const GRAY_64X64_JXL: &[u8] = include_bytes!("fixtures/gray_64x64_lossless.jxl");
const GRADIENT_JXL: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
const PALETTE_JXL: &[u8] = include_bytes!("fixtures/palette_32x32.jxl");
const GREY_8X8_JXL: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");
const VARDCT_D3_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d3.jxl");

/// Round-14 sentinel: the new HfBlockContext custom branch +
/// HfGlobal C.6.2 dequant-matrix parse must not regress the five
/// small modular fixtures (which exercise the kModular path and
/// don't reach VarDCT-only code).
#[test]
fn five_small_lossless_fixtures_still_decode_round_14() {
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
            "round-14 regression: {name} should still decode (round-10 baseline); got {:?}",
            vf.err()
        );
    }
}

/// Round-14 acceptance: the d1 fixture is no longer blocked at
/// `HfBlockContext non-default-table branch (...) not yet supported
/// (round 12+)`. Either it progresses past LfGlobal entirely (next
/// downstream blocker) or it errors with a different precise message —
/// but NEVER the round-12-vintage HfBlockContext message.
#[test]
fn vardct_d1_fixture_is_past_hf_block_context_round_14() {
    let r = decode_one_frame(VARDCT_D1_JXL, None);
    let err = r.expect_err("VarDCT d1 still defers (no HF decode yet)");
    let msg = format!("{err:?}");
    assert!(
        !msg.contains("HfBlockContext non-default-table branch"),
        "round-14 should bypass the round-11/12 HfBlockContext gate; got {msg}"
    );
    assert!(
        matches!(err, Error::Unsupported(_) | Error::InvalidData(_)),
        "round-14 should yield Unsupported or InvalidData; got {msg}"
    );
}

/// Round-14 sentinel: d3 fixture still hits its FrameHeader UTF-8 issue
/// (independent of the HfBlockContext / HfGlobal work). Documenting it
/// here so a regression in either direction is visible.
#[test]
fn vardct_d3_fixture_still_errors_in_round_14() {
    let r = decode_one_frame(VARDCT_D3_JXL, None);
    let err = r.expect_err("VarDCT d3 still has its own (separate) blocker");
    let msg = format!("{err:?}");
    assert!(
        !msg.contains("HfBlockContext non-default-table branch"),
        "round-14: HfBlockContext gate should not be the d3 blocker; got {msg}"
    );
}
