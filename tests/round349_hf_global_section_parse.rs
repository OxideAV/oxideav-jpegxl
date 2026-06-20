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

use oxideav_core::Error;

/// `vardct_256x256_d1.jxl` parses cleanly through the §C.7 HfGlobal
/// section. The decode still ends in `Error::Unsupported` (the
/// per-LfGroup HF reconstruction is not yet fed by the frame loop), but
/// the message must be the post-§C.7 boundary, proving the new
/// HfGlobalSection read consumed the HfPass sequence + §C.7.2
/// histograms + ANS-state init without error.
#[test]
fn vardct_d1_parses_through_hf_global_section() {
    const FIXTURE: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");
    let r = oxideav_jpegxl::decode_one_frame(FIXTURE, None);
    match r {
        Err(Error::Unsupported(msg)) => {
            // The boundary message names the full §C.7 section parse,
            // so a regression that drops back to the pre-§C.7 stop point
            // (or fails inside the HfPass / histogram reads) is caught.
            assert!(
                msg.contains("full §C.7 HfGlobal section"),
                "expected the post-§C.7 Unsupported boundary, got: {msg}"
            );
            assert!(
                msg.contains("HF-coefficient histograms"),
                "boundary message must confirm the §C.7.2 histograms were read: {msg}"
            );
        }
        other => panic!("expected Err(Unsupported) at the post-§C.7 boundary, got {other:?}"),
    }
}
