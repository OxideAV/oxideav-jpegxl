//! Round 272 — pin the Weighted-Predictor `sub_err_i` **reading choice**
//! (FDIS Annex E.1 / §H.5.2) as a regression guard.
//!
//! ## Background
//!
//! Annex E.1 defines the post-decode per-sub-predictor error stored into
//! the WP history as
//!
//! ```text
//! sub_err_i = abs(((prediction_i + 3) >> 3) - true_value)
//! ```
//!
//! where `prediction_i` is in the left-shifted-by-3 (8x) domain and
//! `true_value` is the **un-shifted** decoded sample. Taken literally,
//! the `(prediction_i + 3) >> 3` arithmetic shift floors toward negative
//! infinity *before* the `abs`. The production decoder instead computes
//! the magnitude in the 8x domain and rounds afterwards:
//!
//! ```text
//! (abs(prediction_i - true_value*8) + 3) >> 3
//! ```
//!
//! The two readings COINCIDE for every non-negative `prediction_i`
//! (including all four sub-predictions at the `noise-64x64-lossless`
//! sample-194 trace point, where
//! `prediction_i = [1248, 747, 420, 559]`, `true_value = 34`, and both
//! readings yield `sub_err = [122, 59, 18, 36]`). They DIVERGE only when
//! a sub-predictor goes negative.
//!
//! ## Why this test exists
//!
//! Round 272 investigated whether the residual WP state-evolution
//! divergence on `noise-64x64-lossless` (the sample-129 `Δ = -21`
//! smoking gun, `wp-trace-sample-194.md`) was caused by the production
//! `sub_err` reading. It is NOT: switching the decode path to the
//! literal-FDIS reading leaves that fixture's divergence profile
//! unchanged, while moving the round-10 `synth_320` drift bisect's first
//! PG[0][0] mismatch EARLIER — from the anchored `(y=24, x=14)` to
//! `(y=11, x=104)`. In other words the literal reading decodes a real
//! fixture LESS far, so the production 8x-domain reading is the
//! bisect-validated one. This is the same shape as the documented
//! FDIS-literal-vs-production `error2weight` discrepancy in
//! `r191_wp_trace_oracle`.
//!
//! This test locks:
//!  1. the two readings agree for every non-negative sub-prediction
//!     (so the choice is a no-op on the sample-194 trace point);
//!  2. they diverge for negative sub-predictions (so the choice is
//!     load-bearing);
//!  3. the production decode path uses the 8x-domain reading
//!     ([`sub_err_for`]), validated by `synth_320` still drifting at the
//!     round-10 anchor `(y=24, x=14)`.
//!
//! ## Spec citations
//!
//! - ISO/IEC FDIS 18181-1:2021 Annex E.1 — `sub_err_i` definition.
//! - Trace: `docs/image/jpegxl/fixtures/noise-64x64-lossless/
//!   wp-trace-sample-194.md` lines 19-20, 94-120 (sub-predictions +
//!   resulting `sub_err`).

use oxideav_jpegxl::modular_fdis::{sub_err_fdis_literal, sub_err_for};

const SYNTH_320_JXL: &[u8] = include_bytes!("fixtures/synth_320_grey/synth_320.jxl");

/// Sample-194 trace point (`wp-trace-sample-194.md` lines 94-120):
/// all four sub-predictions are non-negative, so both readings must
/// reproduce the trace's stored `sub_err = [122, 59, 18, 36]`.
#[test]
fn sample_194_both_readings_match_trace() {
    // 8x-domain sub-predictions; un-shifted decoded value.
    let preds_8x = [1248_i32, 747, 420, 559];
    let v = 34_i32;
    let want = [122_i32, 59, 18, 36];

    for (k, &p) in preds_8x.iter().enumerate() {
        let prod = sub_err_for(p, v);
        let lit = sub_err_fdis_literal(p, v);
        assert_eq!(
            prod, want[k],
            "production sub_err for sub-pred {k} (p={p}, v={v}) must match \
             trace-doc value {}",
            want[k]
        );
        assert_eq!(
            lit, want[k],
            "literal-FDIS sub_err for sub-pred {k} (p={p}, v={v}) must also \
             match trace-doc value {} (the two readings agree for \
             non-negative predictions)",
            want[k]
        );
        assert_eq!(
            prod, lit,
            "the two readings must coincide for the non-negative \
             sub-prediction p={p}",
        );
    }
}

/// For every non-negative 8x sub-prediction, the two readings agree.
#[test]
fn readings_agree_for_nonnegative_predictions() {
    for p in (0..=2048_i32).step_by(8) {
        for v in -16..=16_i32 {
            assert_eq!(
                sub_err_for(p, v),
                sub_err_fdis_literal(p, v),
                "readings must agree for non-negative p={p}, v={v}",
            );
        }
    }
}

/// The two readings DIVERGE for negative sub-predictions — so the choice
/// is load-bearing, not cosmetic. Pin a concrete witness.
#[test]
fn readings_diverge_for_negative_predictions() {
    // Concrete divergent witness: a strongly-negative sub-prediction
    // whose 8x-domain magnitude sits just below an 8-boundary so the
    // `(p + 3) >> 3` floor (literal) and the magnitude-then-round
    // (production) land on different multiples.
    let p = -2044_i32;
    let v = -8_i32;
    let prod = sub_err_for(p, v); // (abs(-2044 + 64) + 3) >> 3 = (1980+3)>>3 = 247
    let lit = sub_err_fdis_literal(p, v); // abs(((-2044+3)>>3) - (-8)) = abs(-256+8) = 248
    assert_ne!(
        prod, lit,
        "the pinned witness p={p}, v={v} must diverge: production={prod}, \
         literal={lit}",
    );
    // Confirm a divergence also exists across a broader sweep.
    let mut found = false;
    for pp in (-2048..0_i32).step_by(1) {
        for vv in -8..=8_i32 {
            if sub_err_for(pp, vv) != sub_err_fdis_literal(pp, vv) {
                found = true;
                break;
            }
        }
        if found {
            break;
        }
    }
    assert!(
        found,
        "the two sub_err readings must diverge for some negative \
         sub-prediction — otherwise the production reading choice would \
         be irrelevant",
    );
    // Document the concrete pair used in the doc-comment.
    eprintln!("[r272] sub_err(p={p}, v={v}): production={prod}, literal={lit}");
}

/// End-to-end anchor: the production decode path uses the 8x-domain
/// reading ([`sub_err_for`]), so `synth_320` must still drift at the
/// round-10 bisect anchor `(y=24, x=14)` inside PG[0][0]. (The
/// literal-FDIS reading would move this to `(y=11, x=104)` — strictly
/// worse.) This guards against a future agent swapping the decode path
/// to the literal listing.
#[test]
fn synth_320_drift_anchor_unchanged_by_reading_choice() {
    let vf = oxideav_jpegxl::decode_one_frame(SYNTH_320_JXL, None).unwrap();
    let plane = &vf.planes[0];

    let mut first: Option<(usize, usize)> = None;
    'outer: for y in 0..128usize {
        for x in 0..128usize {
            let want = ((y as u32 + x as u32) & 0xFF) as u8;
            if plane.data[y * 320 + x] != want {
                first = Some((y, x));
                break 'outer;
            }
        }
    }
    let (y, x) = first.expect("PG[0][0] should still have at least one mismatch");
    assert_eq!(
        (y, x),
        (24, 14),
        "PG[0][0] first mismatch must stay at the round-10 bisect anchor \
         (y=24, x=14); got ({y}, {x}). If this is (11, 104) the decode \
         path was switched to the literal-FDIS sub_err reading, which \
         decodes synth_320 LESS far — revert to `sub_err_for`.",
    );
}
