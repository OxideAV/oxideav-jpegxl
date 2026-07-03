//! Round-15 integration tests: GlobalModular zero-channel ModularHeader
//! gating (FDIS §C.9.1 last sentence: "In the trivial case where N is
//! zero, the decoder takes no action.") + single-TOC-entry section
//! chaining for the VarDCT pipeline.
//!
//! Round-14 left the d1 fixture stuck on `JXL TransformId: invalid value 3`
//! — diagnosis (round-15): our `GlobalModular::read` was unconditionally
//! reading the inner ModularHeader (`use_global_tree`, `WPHeader`,
//! `nb_transforms`, `TransformInfo[]`) even when the channel count was
//! zero. Bit-position trace of d1 confirmed the libjxl reference decoder
//! ends LfGlobal at the bit where our code starts reading
//! `inner_use_global_tree`. Fix: skip the entire inner ModularHeader
//! when `derive_channel_descs` returns an empty list.
//!
//! Second blocker uncovered + fixed in round 15: when the TOC has a
//! single entry (`num_groups == 1 && num_passes == 1`), all sections
//! (LfGlobal, LfGroup, HfGlobal, PassGroup) are concatenated bit-aligned
//! without byte alignment. Our `decode_vardct_round13` was slicing each
//! TOC slot into its own byte range — fine for multi-entry TOCs but
//! wrong for the single-entry case. Fix: chain the section reads on a
//! shared `BitReader` when `toc.entries.len() == 1`.

use oxideav_jpegxl::decode_one_frame;

const PIXEL_1X1_JXL: &[u8] = include_bytes!("fixtures/pixel_1x1.jxl");
const GRAY_64X64_JXL: &[u8] = include_bytes!("fixtures/gray_64x64_lossless.jxl");
const GRADIENT_JXL: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
const PALETTE_JXL: &[u8] = include_bytes!("fixtures/palette_32x32.jxl");
const GREY_8X8_JXL: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");

/// Round-15 sentinel: the GlobalModular ModularHeader gating + single-
/// TOC-entry section chaining must not regress the five small modular
/// fixtures.
#[test]
fn five_small_lossless_fixtures_still_decode_round_15() {
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
            "round-15 regression: {name} should still decode (round-10 baseline); got {:?}",
            vf.err()
        );
    }
}

/// Round-15 acceptance: the d1 fixture is past the
/// `JXL TransformId: invalid value 3` blocker (which was caused by the
/// inner ModularHeader being read inside an N=0 GlobalModular) AND past
/// the LfGlobal section boundary (single-TOC-entry path now chains).
/// The d1 fixture should now reach LfGroup territory and surface a
/// downstream HfMetadata round-13+ deferral message.
#[test]
fn vardct_d1_fixture_is_past_global_modular_round_15() {
    // Rounds 15–385 pinned a precise deferral here; round 389
    // reference-validated the integrated decode and lifted the public
    // pixel withhold, so being "past GlobalModular" now means pixels.
    let frame = decode_one_frame(VARDCT_D1_JXL, None)
        .expect("d1 decodes on the public path since round 389");
    assert_eq!(frame.planes.len(), 3);
}
