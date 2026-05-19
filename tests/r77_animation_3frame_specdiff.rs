//! Round-77 audit harness — `docs/image/jpegxl/fixtures/animation-3frame`.
//!
//! This test characterises the 1-bit SPECDIFF between ISO/IEC 18181-1:2021
//! FDIS Table C.9 (which our `RestorationFilter::read` currently follows)
//! and ISO/IEC 18181-1:2024 final Table J.1 (which the libjxl 0.12.0
//! `cjxl` binary uses when emitting the animation-3frame fixture). The
//! fixture is `docs/image/jpegxl/fixtures/animation-3frame/input.jxl`
//! (78 B, 3 RGB Regular Modular frames of 32×32 each with
//! `have_animation = 1`).
//!
//! ## Probe-level success
//!
//! [`oxideav_jpegxl::probe_fdis`] correctly recognises the fixture: it
//! reads the SizeHeader (32×32) and ImageMetadata (have_animation=true,
//! 8-bit RGB) without error. The pre-decoder header surface is
//! complete.
//!
//! ## Decode-level failure
//!
//! [`oxideav_jpegxl::decode_one_frame`] currently returns
//! `Error::InvalidData("JXL clustering: ...")` mid-decode, because the
//! TOC entry is parsed as 0 bytes (the size in the codestream is 16
//! bytes) — the parse cursor is off by one byte after FrameHeader,
//! cascading into a wrong section split.
//!
//! ## Root cause (audit finding)
//!
//! ISO/IEC 18181-1:**2024** Table J.1 prepends a leading `all_default
//! Bool()` field to the RestorationFilter bundle that ISO/IEC
//! 18181-1:**2021** FDIS Table C.9 does NOT carry. The 2024 spec
//! reads:
//!
//! ```text
//! condition           type      default    name
//!                     Bool()    true       all_default
//! !all_default        Bool()    true       gab
//! gab                 Bool()    false      gab_custom
//! gab_custom          F16()     0.115...   gab_x_weight1
//! ...
//! !all_default and epf_iters  u(32)  0  (ignored)   <-- new in 2024
//! ...
//! ```
//!
//! Our `RestorationFilter::read` follows 2021 (starting directly with
//! `gab`). When the animation fixture is decoded the missing 1 bit at
//! the top of RestorationFilter shifts the bit cursor: instead of FH
//! ending at bit 80 (= byte 10 boundary), it ends at bit 79. The
//! subsequent `permuted_toc` + byte-align + TOC-entry U32 then read
//! the wrong bits — the entry parses as value 0 instead of 16, the
//! single-section frame body collapses to zero length, and LfGlobal
//! parsing errors mid-clustering.
//!
//! The exact-bit reconstruction (LSB-first):
//!
//! * bytes 5..=11 of the codestream (after the FF 0A signature) are
//!   `08 00 92 09 00 00 40`.
//! * After ImageMetadata (35 bits) + pu0-to-byte (5 bits) we begin
//!   FrameHeader at absolute bit 40.
//! * Our 2021-FDIS path consumes 39 bits and lands at bit 79.
//!   Adding the missing 2024 RF.all_default bit lifts the count to
//!   40 and lands at bit 80.
//! * After permuted_toc (1 bit) + pu0 (7 bits = jump from bit 81 to
//!   the next byte boundary = bit 88), the U32 entry value is read
//!   from bits 88..=99 = `0,0 | 0,0,0,0,1,0,0,0,0,0` LSB-first =
//!   value 16. ✓ Matches the trace's `TOC ... sizes=16`.
//!
//! ## Why a one-line fix here is not enough
//!
//! Naively prepending `all_default = br.read_bool()?; if all_default
//! { return Ok(default) }` to `RestorationFilter::read` breaks the
//! `alpha-64x64` fixture (a cjxl 0.11.1 / 2021-FDIS encoded fixture
//! that emits RF without the leading all_default bit; with the
//! prepend, the first RF bit is misinterpreted as all_default=1 and
//! the rest of the FrameHeader misaligns). The seven small lossless
//! fixtures currently green were all encoded by cjxl 0.11.1 against
//! the 2021 FDIS layout; the animation fixture is encoded by cjxl
//! 0.12.0 against the 2024 layout.
//!
//! The clean fix needs:
//!
//! 1. A codestream-level **layout discriminator** to know which
//!    edition of the RestorationFilter table to apply.
//! 2. OR for the 2021 fixtures to be re-encoded with cjxl 0.12.0+
//!    so all on-disk fixtures share a single layout.
//! 3. AND the 2024-spec `u(32) (ignored)` field after
//!    `epf_channel_scale` when `epf_weight_custom == 1`.
//!
//! Path (2) is the cleanest answer if the parent project's docs
//! collaborator can supply re-encoded fixtures. Path (1) is harder
//! because the codestream does not carry an explicit edition tag —
//! it would require a probe-time heuristic (e.g. attempt 2024 parse,
//! fall back to 2021 on early failure) which conflicts with the
//! single-pass design.
//!
//! ## Forward leverage
//!
//! Future rounds should:
//!
//! 1. Ask the docs collaborator to re-encode the seven small
//!    lossless fixtures with cjxl 0.12.0+ to align them on a single
//!    spec edition (preferably 2024), then apply the RF.all_default
//!    + epf u(32) fix uniformly.
//! 2. Update the encoder-facing trace doc with the explicit
//!    cjxl-version-to-spec-edition mapping so we don't lose this
//!    finding.
//!
//! Spec citations:
//! * ISO/IEC 18181-1:2024 Table J.1 — RestorationFilter bundle
//!   (visible in `docs/image/jpegxl/ISO_IEC_18181-1-JPEG-XL-Core-2024.pdf`
//!   on the page enumerated as 70).
//! * ISO/IEC FDIS 18181-1:2021 Table C.9 — pdftotext extract at
//!   `/tmp/jxl_fdis2021.txt` lines 4088-4101 (no `all_default` row).
//! * Trace events at
//!   `docs/image/jpegxl/fixtures/animation-3frame/trace.txt` (3017 B).

use oxideav_jpegxl::probe_fdis;

// Fixture is also copied under `docs/image/jpegxl/fixtures/animation-3frame/input.jxl`
// in the workspace's `docs/` repository (provenance: cjxl v0.12.0 commit `950c327`,
// 78 B, SHA-256
// `68d00bf562eb4c3810777e6ec987b0ed7eeb1dedc9e1b7d9606edbce8610e76f`). The
// in-crate copy is required because library crates publish to crates.io
// without the workspace's `docs/` tree alongside.
const ANIM_FIXTURE: &[u8] = include_bytes!("fixtures/animation_3frame.jxl");

#[test]
fn animation_3frame_probe_succeeds_with_have_animation_flag() {
    let headers = probe_fdis(ANIM_FIXTURE).expect("probe must succeed");
    assert_eq!(headers.size.width, 32);
    assert_eq!(headers.size.height, 32);
    assert!(
        headers.metadata.have_animation,
        "fixture is a 3-frame animation per its trace.txt; probe must surface have_animation=true"
    );
    assert!(headers.metadata.extra_fields);
    assert_eq!(headers.metadata.bit_depth.bits_per_sample, 8);
    // Decoded animation's tps numerator/denominator are part of the
    // AnimationHeader sub-bundle. Verify it parsed (not the values —
    // those depend on cjxl's encoding defaults).
    assert!(
        headers.metadata.animation.is_some(),
        "have_animation=true must populate animation header bundle"
    );
}

#[test]
fn animation_3frame_decode_blocks_with_specdiff_signature() {
    // Document the current decode-time error so a future change
    // landing the SPECDIFF fix is immediately attributable.
    use oxideav_core::Error;
    let res = oxideav_jpegxl::decode_one_frame(ANIM_FIXTURE, None);
    let err = res.expect_err(
        "round 77 — animation decode must still fail until the 2024-spec RF.all_default bit \
         + u(32) (ignored) field are reconciled with the 2021-FDIS fixtures",
    );
    // Two acceptable failure shapes for the present audit:
    //   - InvalidData("JXL clustering: ...") if the bit cursor lands
    //     mid-LfGlobal (current observed behaviour).
    //   - Unsupported("...") if some intermediate guard rejects the
    //     fixture before the clustering path is reached.
    match &err {
        Error::InvalidData(_) | Error::Unsupported(_) => { /* documented */ }
        other => panic!(
            "round 77 audit — animation-3frame decode produced an unexpected error category \
             (expected InvalidData or Unsupported): {other:?}"
        ),
    }
}
