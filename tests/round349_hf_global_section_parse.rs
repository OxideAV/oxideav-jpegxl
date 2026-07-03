//! Round 349 — the integrated VarDCT decode path now parses the full
//! §C.7 HfGlobal section (HfGlobal + HfPass sequence + §C.7.2
//! HF-coefficient histograms + ANS-state init) on a real codestream.
//!
//! Before round 349, `decode_vardct_round13` stopped after
//! `HfGlobal::read` (the §I.2.4 dequant matrices + §I.2.6
//! `num_hf_presets`) and never advanced through the §C.7.1 HfPass
//! sequence or the §C.7.2 histogram block. Round 349's
//! `hf_global_section::HfGlobalSection::read` chains all three pieces on
//! the same bit cursor; this test pins that the chain succeeds on the
//! staged `vardct_256x256_d1.jxl` fixture (a single-group single-pass
//! D1 VarDCT codestream) — i.e. the decode reaches the post-§C.7
//! `Error::Unsupported` boundary rather than failing inside the new
//! HfPass / histogram reads.
//!
//! This is the on-real-bytes confirmation of the
//! `hf_global_section` unit suite, which exercises the chain against
//! hand-packed prefix-coded fixtures.
//!
//! Round 355 update: the integrated decode now runs *past* the §C.7
//! boundary — the §C.7 HfGlobal section is consumed as a prerequisite
//! of the full §C.8.3 → §L.2.2 reconstruction, which then runs to
//! completion. So this test no longer asserts a stop *at* §C.7; it
//! asserts the §C.7 section is consumed cleanly (no error inside the
//! HfPass / histogram reads) by confirming the public path reaches the
//! reconstruction succeeding on the public path (round 389 lifted the
//! rounds-355–385 pixel withhold once the output was validated against
//! the staged reference decodes).

/// `vardct_256x256_d1.jxl` is consumed cleanly through the §C.7 HfGlobal
/// section: the integrated decode reads HfGlobal + the §C.7.1 HfPass
/// sequence + the §C.7.2 HF-coefficient histograms without error, then
/// proceeds through the full §C.8.3 → §L.2.2 chain. Since round 389 the
/// public path returns the reconstructed pixels; a regression inside
/// the HfPass / histogram reads (or anywhere after) surfaces as an
/// error here.
#[test]
fn vardct_d1_parses_through_hf_global_section() {
    const FIXTURE: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");
    let frame = oxideav_jpegxl::decode_one_frame(FIXTURE, None)
        .expect("public VarDCT decode succeeds (round 389 lifted the pixel withhold)");
    assert_eq!(frame.planes.len(), 3);
    assert_eq!(frame.planes[0].data.len(), 256 * 256);
}

/// Driving the integrated reconstruction directly confirms the §C.7
/// section feeds a successful per-LfGroup HF decode + reconstruction
/// (the on-real-bytes counterpart of the `hf_global_section` unit
/// suite): a correctly-shaped RGB frame comes out.
#[test]
fn vardct_d1_hf_global_section_feeds_reconstruction() {
    const FIXTURE: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");
    let frame = oxideav_jpegxl::decode_vardct_frame_from_codestream(FIXTURE, None)
        .expect("§C.7 section should feed a successful integrated reconstruction");
    assert_eq!(frame.planes.len(), 3);
    assert_eq!(frame.planes[0].data.len(), 256 * 256);
}
