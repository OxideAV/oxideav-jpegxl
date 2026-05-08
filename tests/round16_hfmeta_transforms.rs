//! Round-16 integration tests: nested transforms (Squeeze / Palette /
//! RCT) inside the HfMetadata sub-bitstream (FDIS §C.5.4 + §C.9.4).
//!
//! Round 15 closed two stacked bugs (GlobalModular ModularHeader §C.9.1
//! N=0 gate + F.3.1 single-TOC-entry section chaining). The d1 fixture
//! then surfaced the round-12 deferral inside `HfMetadata::read`:
//! `nb_transforms > 0` errored out as
//! `"transforms inside HF metadata sub-bitstream not yet supported
//! (round 13+)"`.
//!
//! Round 16 wires the parse:
//! * `apply_transforms_to_channel_layout` is invoked on the four-channel
//!   HfMetadata baseline so the inner pixel-decode loop sees the
//!   transform-adjusted channel list (Squeeze residuals appended,
//!   Palette meta-channel inserted at the front, RCT pass-through).
//! * After per-channel decode, `apply_inverse_transforms` walks the
//!   transform sequence in reverse to recover the four-channel base
//!   layout [XFromY, BFromY, BlockInfo, Sharpness].
//!
//! The d1 fixture's HfMetadata sub-bitstream emits an explicit Squeeze
//! transform whose `SqueezeParam.begin_c` values reference channels
//! beyond the four-channel HfMetadata baseline (we observe `begin_c=39`
//! on the very first step). That violates the channel-count invariant
//! checked by `apply_transforms_to_channel_layout` — so the d1 fixture
//! still errors, but **at a strictly later point** than round 15. The
//! new error is the next round-17 candidate.

use oxideav_core::Error;
use oxideav_jpegxl::decode_one_frame;

const PIXEL_1X1_JXL: &[u8] = include_bytes!("fixtures/pixel_1x1.jxl");
const GRAY_64X64_JXL: &[u8] = include_bytes!("fixtures/gray_64x64_lossless.jxl");
const GRADIENT_JXL: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
const PALETTE_JXL: &[u8] = include_bytes!("fixtures/palette_32x32.jxl");
const GREY_8X8_JXL: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");

/// Round-16 sentinel: wiring HfMetadata transforms must not regress the
/// five small modular fixtures.
#[test]
fn five_small_lossless_fixtures_still_decode_round_16() {
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
            "round-16 regression: {name} should still decode (round-10 baseline); got {:?}",
            vf.err()
        );
    }
}

/// Round-16 acceptance: the d1 fixture is past the round-15 HfMetadata
/// `nb_transforms > 0` deferral. The parse now succeeds for
/// `nb_transforms` + the per-transform descriptors; the surfacing error
/// MUST come from a strictly later point in the pipeline.
#[test]
fn vardct_d1_fixture_is_past_hf_metadata_transform_parse_round_16() {
    let r = decode_one_frame(VARDCT_D1_JXL, None);
    let err = r.expect_err("VarDCT d1 still defers (HfMetadata transform values out of range)");
    let msg = format!("{err:?}");
    // Round-12/13 deferral message must NOT be the surface error any more.
    assert!(
        !msg.contains("transforms inside HF metadata sub-bitstream not yet"),
        "round-16 should bypass the round-12 HfMetadata transforms deferral; got {msg}"
    );
    // Round-15 sentinels must continue to hold.
    assert!(
        !msg.contains("TransformId: invalid value 3"),
        "round-16 should still bypass the round-14 TransformId=3 blocker; got {msg}"
    );
    assert!(
        !msg.contains("TOC slot 1 out of range"),
        "round-16 should still bypass the round-15 single-TOC-entry blocker; got {msg}"
    );
    assert!(
        matches!(err, Error::Unsupported(_) | Error::InvalidData(_)),
        "round-16 should yield Unsupported or InvalidData; got {msg}"
    );
}
