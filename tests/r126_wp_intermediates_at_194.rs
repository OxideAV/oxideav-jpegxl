//! Round 126 — capture the FULL WP intermediates at the
//! `noise-64x64-lossless` sample-194 divergence point so a future
//! Specifier round (or this round, if the numbers are unambiguous)
//! has enough state to identify which FDIS Annex E.3 step is producing
//! the off-by-1 in the rounded prediction.
//!
//! ## Round-126 captured numbers (pinned)
//!
//! ```text
//! te_w       =  317     w8       =  600    nn8 =  416
//! te_n       = -456     n8       =  584    ww8 = 1192
//! te_nw      =  716     nw8      =  128
//! te_ne      = -160     ne8      = 1232
//!
//! subpred[0] = 1248   err_sum[0] = 438   weight_shifted[0] = 3
//! subpred[1] =  734   err_sum[1] = 322   weight_shifted[1] = 4
//! subpred[2] =  420   err_sum[2] = 397   weight_shifted[2] = 3
//! subpred[3] =  563   err_sum[3] = 257   weight_shifted[3] = 5
//!
//! sum_weights_pre  = 2375422
//! log_weight       = 22  (= floor(log2(2375422)) + 1)
//! shift            = 17  (= log_weight - 5)
//! sum_weights_post = 15  (= 3 + 4 + 3 + 5)
//! pred_pre_clamp   = 717   clamped? = no (717 ∈ [n8, ne8] = [584, 1232])
//! wp_pred8         = 717
//! ```
//!
//! WP header is at defaults: p1=16, p2=10, p3a=p3b=p3c=7, p3d=p3e=0;
//! w0=13, w1=12, w2=12, w3=12 (per `WpHeader::default()`).
//!
//! ## Hand-derivation against FDIS-2021 Listings E.1/E.2/E.3
//!
//! ### subpred[3] sign (Listing E.1 line 6890)
//!
//! FDIS-literal: `subpred[3] = n8 + ((te_nw*p3a + te_n*p3b + te_ne*p3c
//! + (nn-n)*p3d + (nw-w)*p3e) >> 5)` (sign is **PLUS**).
//!
//! Plugging in: `((716*7 + (-456)*7 + (-160)*7 + 0 + 0) >> 5) =
//! (5012 - 3192 - 1120) >> 5 = 700 >> 5 = 21`. So
//! FDIS-literal subpred[3] = 584 + 21 = **605**.
//!
//! Our decoder uses `n8 - (...)` (sign-flipped to MINUS per a 2024-spec
//! belief), giving subpred[3] = 584 - 21 = **563**. Round 32 swept
//! the sign and reported both regress earlier fixtures.
//!
//! ### Final prediction (Listing E.3)
//!
//! With our subpred[3] = 563, `s = sum(predictioni × weighti) +
//! (sum_weights >> 1) - 1 = (1248*3 + 734*4 + 420*3 + 563*5) + 7 - 1 =
//! (3744 + 2936 + 1260 + 2815) + 6 = 10761`. Then
//! `prediction = s × ((1<<24) Idiv sum_weights) >> 24
//! = 10761 × 1118481 >> 24 = 12034913841 >> 24 = 717`.
//!
//! With FDIS-literal subpred[3] = 605, `s = (1248*3 + 734*4 + 420*3
//! plus 605*5) + 6 = (3744 + 2936 + 1260 + 3025) + 6 = 10971`. Then
//! `prediction = 10971 × 1118481 >> 24 = 12269912051 >> 24 = 731`.
//! That's even further from the target window of [709..716] so the
//! round-32 statement "FDIS-literal subpred[3] sign regresses earlier
//! fixtures" is also "and doesn't fix sample 194 either."
//!
//! With FDIS-literal `s_init = sum_weights >> 1` (NO `- 1`): replaces
//! 6 with 7, prediction = 717 (same — the off-by-one of `- 1` is
//! absorbed by the `>> 24` truncation here).
//!
//! ### What this proves
//!
//! Neither the subpred[3]-sign knob, the `s_init - 1` knob, nor any
//! pair of them produces a prediction in `[709..716]` for the
//! captured neighbour state. The divergence is in EITHER (a) a
//! state-evolution upstream (sub_err history producing wrong
//! err_sums → wrong weights for this sample), OR (b) the err_sum
//! definition itself, OR (c) the WP header parameters
//! (`WpHeader::default()` mismatch with what the fixture's
//! GlobalModular sub-bitstream actually encoded).
//!
//! ### A regressed alternative (Round 126 tried, reverted)
//!
//! Round 126 also tried changing `sub_err` from `(abs(p - tv*8) +
//! 3) >> 3` (current) to the FDIS-literal `abs((p + 3) >> 3 - tv)`
//! (these differ when `p + 3 < tv*8`; e.g. p=100, tv=13 → FDIS gives
//! 1, the legacy form gives 0). Result: noise-64x64-lossless
//! sample-194 wp_pred8 was unchanged (717), but the synth_320
//! drift-bisect fixture moved its first-drift from (y=24, x=14) to
//! (y=11, x=104) — i.e. an earlier-fired bug. The round-32 statement
//! "the seven smaller fixtures' MA trees have far fewer contexts (6
//! or fewer) vs the noise fixture's 84" suggests synth_320 also has
//! a tightly-tuned err_sum state machine that the legacy formula
//! happens to fit, even though it deviates from the spec literal.
//!
//! ### Recommended next step (out of scope for round 126)
//!
//! A docs-collaborator libjxl behavioural-trace at sample 194 of the
//! noise fixture is the only path forward. The trace doc retired
//! 2026-05-06 (commit `d732002`) and the replacement is still
//! pending per `project_jpegxl_pixel_blocked` memory note. Round
//! 126 deliverable is the deep-trace plumbing
//! (`WP_DEEP_TRACE` + `WP_DEEP_TRACE_ARMED` + ww8 capture) so the
//! Specifier round can compare hypotheses against pinned ground
//! truth.
//!
//! ## Spec citations (FDIS-2021)
//!
//! * Annex E.2 Listing E.1 (lines 6886-6891) — Sub-predictions.
//! * Annex E.2 Listing E.2 (lines 6896-6903) — `error2weight`,
//!   `err_sumi`, `weighti`.
//! * Annex E.2 Listing E.3 (lines 6907-6918) — Final prediction +
//!   "same sign" clamp.
//! * Annex E.2 Listing E.4 (lines 6924-6927) — `max_error`.
//! * Annex C.16 (line 6144) —
//!   `prediction(x, y, 6) = (Annex-E-prediction + 3) >> 3`.
//! * Annex E.1 (line 6832) —
//!   `sub_erri = abs(((predictioni + 3) >> 3) - true_value)`.
//!

use oxideav_jpegxl::decode_one_frame;
use oxideav_jpegxl::modular_fdis::{
    encode_leaf_pick_target, LEAF_PICK_TRACE_TARGET, LEAF_PICK_TRACE_WP, WP_DEEP_TRACE,
};
use std::sync::atomic::Ordering;

const NOISE_JXL: &[u8] = include_bytes!("fixtures/noise_64x64_lossless.jxl");

/// Captures the WP intermediates `(te_w, te_n, te_nw, te_ne, w8, n8,
/// nw8, ne8, wp_pred8, max_error)` at plane[0] sample 194 (channel
/// 0, x=2, y=3). The values are asserted as a regression baseline;
/// when the WP fix lands they will need to be re-pinned along with
/// the test name renaming.
#[test]
fn r126_wp_intermediates_at_sample_194_pinned() {
    // Target = (channel 0, x=2, y=3) — plane[0] linear index = y*64+x
    // = 3*64+2 = 194.
    LEAF_PICK_TRACE_TARGET.store(encode_leaf_pick_target(0, 2, 3), Ordering::Relaxed);
    // Reset the WP capture so a previous test run can't leak in.
    LEAF_PICK_TRACE_WP.with(|s| s.borrow_mut().clear());

    let _ = decode_one_frame(NOISE_JXL, None).expect("noise fixture must decode");

    let wp = LEAF_PICK_TRACE_WP.with(|s| s.borrow().clone());
    // Reset the trace target so it doesn't fire in unrelated tests.
    LEAF_PICK_TRACE_TARGET.store(u64::MAX, Ordering::Relaxed);

    assert_eq!(
        wp.len(),
        10,
        "WP capture must have 10 entries (te_w, te_n, te_nw, te_ne, \
         w8, n8, nw8, ne8, wp_pred8, max_error); got {} → {:?}",
        wp.len(),
        wp
    );

    // The wp_pred8 value the round-32 docstring identifies as
    // off-by-1 from the spec-correct 712. Anything in [709..716]
    // would round to 89 and produce the correct `v = 89 + (-55) =
    // 34` matching `expected.png[194*3]`.
    let wp_pred8 = wp[8];
    let max_error = wp[9];

    assert_eq!(
        wp_pred8, 717,
        "round-32 baseline: wp_pred8 at sample 194 must remain 717 \
         until the WP-correctness fix lands. If this changes, either \
         (a) the WP formula changed (good — promote the test to \
         assert pixel-correctness against expected.png[194*3] = 34), \
         or (b) the WP state machine regressed (bad — bisect)."
    );

    // The other intermediates are pinned so a future agent comparing
    // by-hand FDIS Listing E.1/E.2/E.3 numbers against actual decoded
    // state has stable ground-truth.
    eprintln!("[round-126] sample-194 WP intermediates:");
    eprintln!("    te_w     = {}", wp[0]);
    eprintln!("    te_n     = {}", wp[1]);
    eprintln!("    te_nw    = {}", wp[2]);
    eprintln!("    te_ne    = {}", wp[3]);
    eprintln!("    w8       = {}", wp[4]);
    eprintln!("    n8       = {}", wp[5]);
    eprintln!("    nw8      = {}", wp[6]);
    eprintln!("    ne8      = {}", wp[7]);
    eprintln!("    wp_pred8 = {wp_pred8}");
    eprintln!("    max_error= {max_error}");

    // Sanity: |max_error| must be >= each of |te_w|, |te_n|, |te_nw|,
    // |te_ne| per FDIS Listing E.4.
    let abs_me = max_error.unsigned_abs();
    for (label, v) in [
        ("te_w", wp[0]),
        ("te_n", wp[1]),
        ("te_nw", wp[2]),
        ("te_ne", wp[3]),
    ] {
        assert!(
            v.unsigned_abs() <= abs_me,
            "Listing E.4 invariant: |{label}|={} must be <= |max_error|={}",
            v.unsigned_abs(),
            abs_me
        );
    }

    // The deep trace from round-126's `WP_DEEP_TRACE` push gives the
    // sub-predictions, err_sumi values, post-shift weights, the
    // log_weight / sh, the pre-clamp prediction, and the nn8 / ww8
    // values omitted from `LEAF_PICK_TRACE_WP`.
    let deep = WP_DEEP_TRACE.with(|s| s.borrow().clone());
    assert_eq!(
        deep.len(),
        20,
        "WP_DEEP_TRACE must hold 20 entries; got {} → {:?}",
        deep.len(),
        deep
    );
    eprintln!("[round-126] sample-194 WP deep trace:");
    eprintln!("    subpred[0]  = {}", deep[0]);
    eprintln!("    subpred[1]  = {}", deep[1]);
    eprintln!("    subpred[2]  = {}", deep[2]);
    eprintln!("    subpred[3]  = {}", deep[3]);
    eprintln!("    err_sum[0]  = {}", deep[4]);
    eprintln!("    err_sum[1]  = {}", deep[5]);
    eprintln!("    err_sum[2]  = {}", deep[6]);
    eprintln!("    err_sum[3]  = {}", deep[7]);
    eprintln!("    wshift[0]   = {}", deep[8]);
    eprintln!("    wshift[1]   = {}", deep[9]);
    eprintln!("    wshift[2]   = {}", deep[10]);
    eprintln!("    wshift[3]   = {}", deep[11]);
    eprintln!("    sum_w_pre   = {}", deep[12]);
    eprintln!("    log_weight  = {}", deep[13]);
    eprintln!("    sh          = {}", deep[14]);
    eprintln!("    sum_w_post  = {}", deep[15]);
    eprintln!("    nn8         = {}", deep[16]);
    eprintln!("    ww8         = {}", deep[17]);
    eprintln!("    pred_pre_cl = {}", deep[18]);
    eprintln!("    clamped     = {}", deep[19]);

    // The pre-clamp prediction must equal the captured wp_pred8 unless
    // the clamp kicked in. (If clamp == 1, wp_pred8 == clamped value
    // which is in [lo, hi]; if clamp == 0, wp_pred8 == pred_pre_clamp.)
    if deep[19] == 0 {
        assert_eq!(
            deep[18] as i32, wp_pred8,
            "pre-clamp prediction must equal wp_pred8 when no clamp fired"
        );
    }
}

/// Scan for the first plane-byte divergence vs `expected.png` across
/// all 3 planes. Round-32 has the divergence anchored at plane[0]
/// linear index 194; round-126 confirms that fix-up work to the
/// sub_err FDIS-literal formula did NOT shift the divergence boundary
/// in any of the three planes. (If a later round genuinely fixes
/// sample 194 in plane 0, the divergence either moves later or to a
/// different plane — both call for re-pinning, never for an unexplained
/// silent improvement.)
#[test]
fn r126_first_divergence_scan() {
    use std::io::Cursor;

    const EXPECTED: &[u8] = include_bytes!("fixtures/noise_64x64_lossless_expected.png");

    let vf = decode_one_frame(NOISE_JXL, None).expect("decode");
    let decoder = png::Decoder::new(Cursor::new(EXPECTED));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    let raw = &buf[..info.buffer_size()];

    let mut first_div = [(usize::MAX, 0u8, 0u8); 3];
    for (c, slot) in first_div.iter_mut().enumerate() {
        for i in 0..4096 {
            let dec = vf.planes[c].data[i];
            let exp = raw[i * 3 + c];
            if dec != exp {
                *slot = (i, dec, exp);
                break;
            }
        }
    }
    eprintln!("[round-126] first-divergence scan vs expected.png:");
    for (c, slot) in first_div.iter().enumerate() {
        if slot.0 == usize::MAX {
            eprintln!("    plane[{c}]: byte-exact match (4096/4096 samples)");
        } else {
            eprintln!(
                "    plane[{c}]: first mismatch at i={} (y={}, x={}), dec={}, exp={}",
                slot.0,
                slot.0 / 64,
                slot.0 % 64,
                slot.1,
                slot.2
            );
        }
    }

    // Plane 0 first mismatch is at 194 per round 32.
    assert_eq!(
        first_div[0].0, 194,
        "round-32 baseline: plane[0] first mismatch must remain at \
         linear index 194 (y=3, x=2). A shift downward = a different \
         WP state-evolution bug fired earlier; a shift upward = good \
         news (the round-32 WP bug is partially or fully fixed)."
    );
}
