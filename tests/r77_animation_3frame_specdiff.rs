//! `docs/image/jpegxl/fixtures/animation-3frame` — RestorationFilter
//! edition resolution + first-frame decode.
//!
//! The fixture is `input.jxl` (78 B, 3 RGB Regular Modular frames of
//! 32×32 each with `have_animation = 1`). Its `expected.png` is a solid
//! red (255, 0, 0) 32×32 image.
//!
//! ## Background (resolved)
//!
//! This fixture is encoded against the **2024** edition of the
//! `RestorationFilter` sub-bundle (§C.2 → Annex J, Table J.1), which
//! opens with an `all_default` `Bool()` row that the earlier 2021 FDIS
//! Table C.9 did not carry, and replaces the two `F16`
//! `epf_pass{1,2}_zeroflush` fields with a single ignored `u(32)`. The
//! seven small lossless fixtures shipped alongside it are encoded
//! against the 2021 layout.
//!
//! Because the codestream has no explicit edition tag, the decoder
//! resolves the edition by a trial parse (2024 first, then 2021)
//! validated against the resulting TOC's self-consistency with the
//! remaining codestream bytes. For this fixture the 2024 parse yields a
//! `TOC ... sizes=16` entry that fits the codestream, so it is
//! accepted; the seven 2021 fixtures fall back to the 2021 layout.
//!
//! Spec citations:
//! * ISO/IEC 18181-1:2024 Annex J, Table J.1 — RestorationFilter bundle
//!   (`docs/image/jpegxl/ISO_IEC_18181-1-JPEG-XL-Core-2024.pdf`,
//!   printed page 70).
//! * ISO/IEC FDIS 18181-1:2021 Table C.9 — the earlier layout without
//!   the `all_default` row.
//! * Trace events at
//!   `docs/image/jpegxl/fixtures/animation-3frame/trace.txt`.

use oxideav_jpegxl::{decode_one_frame, probe_fdis};

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
fn animation_3frame_first_frame_decodes_to_solid_red() {
    // With the 2024-edition RestorationFilter layout resolved by the
    // trial parse, the first frame decodes cleanly. `expected.png` is a
    // solid red (255, 0, 0) 32×32 image, so the R plane is all 255 and
    // the G / B planes are all 0.
    let vf = decode_one_frame(ANIM_FIXTURE, None)
        .expect("2024-edition RestorationFilter must let the first frame decode");
    assert_eq!(vf.planes.len(), 3, "RGB fixture must yield three planes");
    for (i, plane) in vf.planes.iter().enumerate() {
        assert_eq!(plane.stride, 32, "plane {i} stride");
        assert_eq!(plane.data.len(), 32 * 32, "plane {i} sample count");
    }
    assert!(
        vf.planes[0].data.iter().all(|&v| v == 255),
        "R plane must be solid 255 (expected.png is solid red)"
    );
    assert!(
        vf.planes[1].data.iter().all(|&v| v == 0),
        "G plane must be solid 0"
    );
    assert!(
        vf.planes[2].data.iter().all(|&v| v == 0),
        "B plane must be solid 0"
    );
}
