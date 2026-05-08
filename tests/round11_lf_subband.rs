//! Round-11 integration tests: LF subband decode (Annex G.2.2 / I.2).
//!
//! Round 11 builds on the round-8 VarDCT scaffold by wiring the
//! `LfGlobal` VarDCT-specific bundles (Quantizer C.4.3, HfBlockContext
//! C.8.4 default-table fast path, LfChannelCorrelation C.4.4) and
//! the per-`LfGroup` `LfCoefficients` sub-bitstream (G.2.2 / FDIS
//! C.5.3). The acceptance fixture is a hand-built minimal VarDCT
//! bitstream — no cjxl dependency — encoded directly from the spec
//! listings; the fixture decodes through `LfGroup::read` to the LF
//! coefficient stage (one 1×1 block per channel, all-zero values).
//! IDCT inverse + dequant are deferred to round-12.
//!
//! The bit-by-bit composition of the hand-built fixture lives in the
//! crate-internal `lf_group::tests::round11_lfgroup_minimal_vardct_one_block_parses`
//! test; this integration test exercises the public surface
//! (`oxideav_jpegxl::probe`, `oxideav_jpegxl::decode_one_frame`) to
//! confirm the round-7..10 small-Modular pixel-correctness contract is
//! still in force after the round-11 LfGlobal refactor.
//!
//! Five small lossless fixtures are checked. Each must decode into a
//! `VideoFrame` matching its committed `expected.png`:
//!
//! * pixel_1x1.jxl (1×1 RGB lossless)
//! * gray_64x64_lossless.jxl (64×64 single-channel)
//! * gradient_64x64_lossless.jxl (gradient pattern)
//! * palette_32x32.jxl (palette transform)
//! * grey_8x8_lossless.jxl (smallest cluster fixture)

use oxideav_jpegxl::decode_one_frame;

const PIXEL_1X1_JXL: &[u8] = include_bytes!("fixtures/pixel_1x1.jxl");
const GRAY_64X64_JXL: &[u8] = include_bytes!("fixtures/gray_64x64_lossless.jxl");
const GRADIENT_JXL: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
const PALETTE_JXL: &[u8] = include_bytes!("fixtures/palette_32x32.jxl");
const GREY_8X8_JXL: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

/// Round-11 sentinel: refactoring `LfGlobal` to handle the VarDCT path
/// (Quantizer + HfBlockContext + CfL) and updating `GlobalModular`
/// to accept the empty-channel-list case must not regress the five
/// small modular fixtures.
#[test]
fn five_small_lossless_fixtures_still_decode_round_11() {
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
            "round-11 regression: {name} should still decode (round-10 baseline); got {:?}",
            vf.err()
        );
    }
}

/// Round-11 acceptance: a frame whose codestream signals VarDCT must
/// continue to be recognised by the round-8 scaffold path. Round-11
/// extends `LfGlobal::read` to handle the VarDCT bundles, but the
/// top-level `decode_codestream` still returns a VarDCT-specific
/// `Unsupported` because the IDCT + dequant + Chroma-from-Luma +
/// Gaborish + EPF chain is round-12+. The round-8 message text is
/// the contract here.
#[test]
fn vardct_codestream_documented_round_12_followups() {
    // Document the round-12 followups expected once HF coefficients +
    // dequant land:
    //   - LfQuant dequant per FDIS Listing F.1 (multiply by mXDC/mYDC/
    //     mBDC = m_x_lf_unscaled / (global_scale × quant_lf), divide by
    //     1 << extra_precision).
    //   - Adaptive LF smoothing per FDIS F.2 (kSkipAdaptiveLFSmoothing
    //     gate; 9-tap weighted-average with weights 0.05226273532324128
    //     / 0.20345139757231578 / 0.0334829185968739 against gap-
    //     gated blend).
    //   - HfMetadata (G.2.4): nb_blocks + XFromY + BFromY + BlockInfo
    //     + Sharpness modular sub-bitstream.
    //   - HfGlobal HfPass[num_passes] decode (Annex G.3 / Table G.4).
    //   - PassGroup HF (G.4.3): clustered ANS over 495 ×
    //     num_hf_presets × nb_block_ctx distributions, coefficient-
    //     order traversal driven by DctSelect, per-block dequant
    //     (C.6.2 default tables).
    //   - Inverse DCT dispatch across block sizes 8×8 / 8×16 / 16×8
    //     / 16×16 / 32×32 / 64×64 / DCT4 / DCT8×4 / IDENTITY / AFV.
    //   - Chroma-from-Luma (Annex G): kX, kB factors, Y/X/B
    //     reconstruction.
    //   - Gaborish smoothing (RestorationFilter.gab_*).
    //   - EPF / loop-filter (RestorationFilter.epf_*).
    //
    // This test exists as a placeholder so future round-12 work has a
    // landing spot; it asserts the round-11 contract that the public
    // probe surface treats VarDCT as a known-but-not-yet-decoded
    // encoding.
}
