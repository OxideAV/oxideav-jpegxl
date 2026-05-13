//! Round 32 — `noise-64x64-lossless` pixel-divergence bisect.
//!
//! ## Round-31 hand-off
//!
//! Round 31 fixed the §F.3 zero-pad on the single-TOC-entry LfGlobal
//! fast path so that the noise fixture decode-completes without an
//! EOF error. The pixel-divergence (~98 % of plane[0] bytes wrong
//! from sample 194 onward) was tagged as a separate latent
//! state-evolution bug and parked for round 32.
//!
//! ## What this test asserts
//!
//! Round-32 reproduced the divergence and bisected it down to a
//! single well-defined locus:
//!
//! * The first divergent plane-0 byte is at linear index **194**
//!   (i.e. y=3, x=2 in the 64-wide channel).
//! * That sample's MA-tree leaf has `predictor == 6` (Self-correcting
//!   weighted predictor, FDIS Annex H.5 / Annex E).
//! * It is the FIRST `predictor == 6` sample whose WP path uses
//!   `WW` and `NN` both as their in-image values; all earlier
//!   `predictor == 6` samples sit in row y=0 or y=1 (border:
//!   `NN = N`), or at x=0/x=1 in row y=2 (border: `WW = W`).
//! * The MA-tree leaf, the decoded raw token, the unpacked diff
//!   (`-55`), and `wp_max_error` are all consistent with the
//!   prior-sample state — only the WP weighted-sum output
//!   (`wp_pred8 = 717`) is off-by-1-in-quotient-of-`>>3` from
//!   what `expected.png` demands. Any value in `[709..716]` would
//!   yield the correct rounded predictor `89`, then
//!   `v = diff + p = -55 + 89 = 34` matching `expected.png[194]`.
//!
//! Knobs swept (none recovers all eight fixtures simultaneously):
//!
//! * `WP_ROUND_BIAS ∈ {0..=7}` — bias=3 (FDIS-literal) gives the
//!   latest first-divergence (sample 194); other values regress
//!   earlier.
//! * `s_init ∈ {(sw >> 1) - 1, (sw >> 1), sw, 0}` — `(sw >> 1) - 1`
//!   matches the round-3 code and is the best of the four.
//! * `subpred[3]` sign — FDIS Listing E.1 says `N + (… >> 5)`,
//!   round-3 code uses `N - (… >> 5)`; both regress earlier
//!   fixtures.
//! * Same-sign clamp predicate — `<= 0` (current) vs. `>= 0`
//!   regresses earlier.
//!
//! ## Why we do NOT fix here
//!
//! Fixing the divergence needs either (a) a behavioural trace of
//! the libjxl WP path at sample 194 captured by the docs
//! collaborator, or (b) the docs collaborator's promised libjxl
//! trace-reverse-engineering doc on §H.5.2 Sub-predictions
//! (referenced in the `project_jpegxl_pixel_blocked` memory note,
//! but the file does not yet exist in `docs/image/jpegxl/`).
//! The seven pre-round-32 fixtures stay pixel-correct.
//!
//! ## What this test does
//!
//! Asserts that the divergence boundary holds exactly where round
//! 32 left it (`plane[0]` first mismatch at linear index 194), so
//! when the docs collaborator delivers the libjxl trace and the
//! WP fix lands, this regression locks in that we did not
//! accidentally regress sample 194 in the interim.

use oxideav_jpegxl::decode_one_frame;
use std::io::Cursor;

const NOISE_JXL: &[u8] = include_bytes!("fixtures/noise_64x64_lossless.jxl");
const EXPECTED_PNG: &[u8] = include_bytes!("fixtures/noise_64x64_lossless_expected.png");

#[test]
fn r32_noise_first_divergence_locked_at_194() {
    let vf = decode_one_frame(NOISE_JXL, None).expect("noise fixture must decode");
    let decoder = png::Decoder::new(Cursor::new(EXPECTED_PNG));
    let mut reader = decoder.read_info().expect("png info");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).expect("png frame");
    let raw = buf[..info.buffer_size()].to_vec();

    let plane0 = &vf.planes[0].data;
    assert_eq!(
        plane0.len(),
        4096,
        "plane[0] must hold 64*64=4096 byte samples",
    );

    // The first 194 samples are byte-equal to expected.png.
    for i in 0..194 {
        assert_eq!(
            plane0[i],
            raw[i * 3],
            "round-32 regression: plane[0][{i}] dec={} exp={} \
             (the matching-prefix must extend through index 193 \
             exactly; any new earlier divergence is a regression)",
            plane0[i],
            raw[i * 3]
        );
    }

    // Sample 194 is the FIRST divergent byte (dec=35, expected=34).
    assert_ne!(
        plane0[194],
        raw[194 * 3],
        "round-32 regression: plane[0][194] is supposed to be the \
         first divergent byte. If it suddenly matches, the WP bug \
         may be fixed — promote this test to a full pixel-equality \
         assertion and update the fixture count to 8."
    );
    assert_eq!(
        plane0[194], 35,
        "round-32 regression: expected dec[194]=35 (off-by-1 from \
         the spec-correct 34, due to WP `(wp_pred8 + 3) >> 3` \
         rounding 717→90 instead of 712→89)"
    );
    assert_eq!(
        raw[194 * 3],
        34,
        "expected.png plane[0][194] must be 34 (the WP-correct \
         target); if this fails the expected.png fixture changed",
    );
}
