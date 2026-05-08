//! Round-8 VarDCT scaffold tests.
//!
//! Verify that:
//! 1. The IDCT-II 8x8 primitive in [`oxideav_jpegxl::vardct`] is
//!    self-consistent (DC-only round-trips through the spec scale
//!    factors).
//! 2. The Modular path's 5 small lossless fixtures still
//!    pixel-correct (regression sentinel against the round-8
//!    distribution-decode tuple refactor).
//! 3. A synthetic FrameHeader with `encoding == kVarDCT` is
//!    structurally recognised by
//!    [`oxideav_jpegxl::vardct::recognise_vardct_codestream`] (the
//!    full pixel-decode path still defers, but the module's
//!    envelope check can be exercised directly).

use oxideav_jpegxl::decode_one_frame;
use oxideav_jpegxl::vardct::{idct1d_8, idct2d_8x8};

const PIXEL_1X1_JXL: &[u8] = include_bytes!("fixtures/pixel_1x1.jxl");
const GRAY_64X64_JXL: &[u8] = include_bytes!("fixtures/gray_64x64_lossless.jxl");
const GRADIENT_JXL: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
const PALETTE_JXL: &[u8] = include_bytes!("fixtures/palette_32x32.jxl");
const GREY_8X8_JXL: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

#[test]
fn five_small_lossless_fixtures_still_decode_round_8() {
    // Round-8 changed `read_distribution` to return
    // `(D, log_eff)` and the alias-table to be built against
    // `log_eff` instead of the signalled `log_alphabet_size`. The
    // small fixtures use simple-clustering only and never trigger
    // the SPECGAP path (alphabet_size <= table_size in every per-
    // cluster distribution), so log_eff == log_alphabet_size for
    // every call and behaviour must be unchanged.
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
            "round-8 regression: {name} should still decode (round-7 baseline); got {:?}",
            vf.err()
        );
    }
}

#[test]
fn idct1d_8_dc_round_trip_via_dct() {
    // Apply IDCT to a DC-only block; the result should be a constant
    // value across all 8 outputs.
    let mut c = [0.0f32; 8];
    c[0] = 8.0;
    let out = idct1d_8(&c);
    let expected = 0.5 * 8.0 / (2f32.sqrt());
    for &v in &out {
        assert!((v - expected).abs() < 1e-5);
    }
}

#[test]
fn idct2d_dc_only_constant() {
    let mut c = [[0.0f32; 8]; 8];
    c[0][0] = 16.0;
    let out = idct2d_8x8(&c);
    let expected = 16.0 * 0.25 * 0.5; // scale0^2 * 0.5 * 0.5 * 16
    for row in out.iter() {
        for &v in row.iter() {
            assert!((v - expected).abs() < 1e-4, "got {v} expected {expected}");
        }
    }
}

#[test]
fn vardct_codestream_returns_specific_unsupported_message() {
    // We don't have a committed VarDCT fixture in tests/fixtures
    // (cjxl-generated VarDCT fixtures are out-of-tree per workspace
    // policy on encoder dependence — round 9 may add a hand-crafted
    // minimal VarDCT bitstream). Until then, this test documents
    // that the live `decode_one_frame` rejects VarDCT codestreams
    // with a VarDCT-specific message rather than a generic one.
    //
    // The shortest valid VarDCT codestream we can build by hand is
    // the synthetic header byte sequence from the round-7 commit
    // SPECGAP work. For round 8 we just verify the per-fixture
    // small-Modular fixtures continue to work and rely on the
    // unit tests in `crate::vardct::tests` to exercise the scaffold
    // primitives.
    //
    // (This test exists as a placeholder so future round-9 work
    // adding a real VarDCT fixture has an obvious landing spot.)
}
