//! Round 195 — WP state-evolution bisect at sample 193
//!
//! ## Round-278 resolution (read this first)
//!
//! The upstream state-evolution defect chased below was fixed in
//! round 278 (`modular_fdis::wp_predict`: Listing E.2 `error2weight`
//! inner-Idiv-first operand order + `true_errNW → true_errN` fallback
//! at x == 0 — see `r32_noise_bisect.rs`). The assertions in this
//! file are re-pinned to the spec-conformant values: the shared
//! sample-129 stored true_err is now 737 (Δ = 0) and sample 193's
//! prediction is the trace's 896. The historical bisect narrative is
//! kept below because it documents HOW the defect was localised.
//!
//! ## Background (r191 finding)
//!
//! Round 191 established that the WP predictor itself is spec-conformant:
//! when fed the trace's spec-correct state inputs at sample 194, it
//! produces the correct output (prediction=709, max_error=737). The issue
//! is in **upstream state evolution** — the `true_err` values stored at
//! earlier samples are wrong.
//!
//! The r191 state gap at sample 194:
//! ```text
//! te_w  = 317 (should be 296, Δ = +21)
//! te_n  = -456 (correct!)
//! te_nw = 716 (should be 737, Δ = -21)
//! te_ne = -160 (should be -165, Δ = +5)
//! ```
//!
//! The +21/-21 symmetric pair in te_w/te_nw is the smoking gun. Both
//! values are stored at x=1 positions:
//! - te_w reads from (1, 3) = sample 193
//! - te_nw reads from (1, 2) = sample 129
//!
//! ## This test's hypothesis
//!
//! The error PROPAGATES through the state machine:
//! 1. Sample 129's true_err is -21 from spec (stored as 716, should be 737)
//! 2. Sample 193 reads te_n from sample 129, getting 716 instead of 737
//! 3. This wrong te_n causes sample 193's prediction to be wrong
//! 4. Sample 193's wrong prediction causes its stored true_err to be +21
//!    from spec (stored as 317, should be 296)
//!
//! If this hypothesis is correct:
//! - te_n at sample 193 should equal te_nw at sample 194 (both read from
//!   sample 129)
//! - With the corrected te_n=737, sample 193's prediction should be 896
//!   (matching the trace)
//!
//! ## Spec citations
//!
//! Trace doc: `docs/image/jpegxl/fixtures/noise-64x64-lossless/
//! wp-trace-sample-194.md`. The surrounding-sample context (s=188..200)
//! gives the spec-correct values at sample 193.

use oxideav_jpegxl::decode_one_frame;
use oxideav_jpegxl::modular_fdis::{
    encode_leaf_pick_target, WpHeader, LEAF_PICK_TRACE_TARGET, LEAF_PICK_TRACE_WP, WP_DEEP_TRACE,
    WP_DEEP_TRACE_ARMED,
};
use serial_test::serial;
use std::sync::atomic::Ordering;

const NOISE_JXL: &[u8] = include_bytes!("fixtures/noise_64x64_lossless.jxl");

/// Trace-doc spec-correct values at sample 193 (x=1, y=3).
/// From the surrounding-sample context at lines 129-149 of wp-trace-sample-194.md:
/// ```text
/// s=193 x=1  y=3  N=128  W=1192 NE=584  NW=1008 NN=968
///       err_sum = 222 343 338 200   weight = 973681 585254 585254 986899
///       pred_i  = 1648 365 1362 182   prediction=896    true_value=600   true_err=296
/// ```
mod trace_sample_193 {
    pub const X: i32 = 1;
    pub const Y: i32 = 3;

    // Neighbour samples in 8x scale (used by sub-prediction hand-derivation)
    pub const N8: i32 = 128;
    pub const W8: i32 = 1192;
    pub const NE8: i32 = 584;
    // NW8, NN8 not used by current tests but kept for reference:
    // pub const NW8: i32 = 1008;
    // pub const NN8: i32 = 968;

    // Spec-correct sub-predictions and final prediction
    pub const SUBPRED_EXPECTED: [i32; 4] = [1648, 365, 1362, 182];
    pub const PREDICTION_EXPECTED: i32 = 896;
    pub const TRUE_VALUE_8X: i32 = 600;
    pub const TRUE_ERR_EXPECTED: i32 = 296; // prediction - true_value = 896 - 600
}

/// Capture WP state at sample 193 and verify the error propagation
/// hypothesis: te_n at sample 193 should equal te_nw at sample 194
/// (both read from sample 129, both -21 from spec).
#[test]
#[serial]
fn r195_sample_193_te_n_equals_sample_194_te_nw() {
    // First capture sample 194's WP state
    LEAF_PICK_TRACE_TARGET.store(encode_leaf_pick_target(0, 2, 3), Ordering::Relaxed);
    LEAF_PICK_TRACE_WP.with(|s| s.borrow_mut().clear());
    let _ = decode_one_frame(NOISE_JXL, None).expect("decode");
    let wp_194 = LEAF_PICK_TRACE_WP.with(|s| s.borrow().clone());

    // Now capture sample 193's WP state
    LEAF_PICK_TRACE_TARGET.store(encode_leaf_pick_target(0, 1, 3), Ordering::Relaxed);
    LEAF_PICK_TRACE_WP.with(|s| s.borrow_mut().clear());
    let _ = decode_one_frame(NOISE_JXL, None).expect("decode");
    let wp_193 = LEAF_PICK_TRACE_WP.with(|s| s.borrow().clone());

    LEAF_PICK_TRACE_TARGET.store(u64::MAX, Ordering::Relaxed);

    assert_eq!(
        wp_194.len(),
        10,
        "sample 194 WP capture must have 10 entries"
    );
    assert_eq!(
        wp_193.len(),
        10,
        "sample 193 WP capture must have 10 entries"
    );

    // wp[0..4] = te_w, te_n, te_nw, te_ne
    let te_nw_at_194 = wp_194[2];
    let te_n_at_193 = wp_193[1];

    eprintln!("[r195] Verifying error propagation hypothesis:");
    eprintln!(
        "    sample 194: te_w={}, te_n={}, te_nw={}, te_ne={}",
        wp_194[0], wp_194[1], wp_194[2], wp_194[3]
    );
    eprintln!(
        "    sample 193: te_w={}, te_n={}, te_nw={}, te_ne={}",
        wp_193[0], wp_193[1], wp_193[2], wp_193[3]
    );

    // Both read from sample 129's stored true_err
    assert_eq!(
        te_n_at_193, te_nw_at_194,
        "te_n at sample 193 and te_nw at sample 194 both read from sample 129 \
         (position (1, 2)), so they MUST be identical. sample 193 te_n = {}, \
         sample 194 te_nw = {}",
        te_n_at_193, te_nw_at_194
    );

    // Round-278: the stored true_err at sample 129 is spec-exact
    // (737). Pre-fix production stored 716 (the Δ = -21 smoking gun).
    let spec_te_at_129 = 737;
    assert_eq!(
        te_n_at_193, spec_te_at_129,
        "te at sample 129 must match spec (737) from round 278 onward \
         (pre-fix production stored 716, Δ = -21). Got {}",
        te_n_at_193
    );

    eprintln!(
        "    ✓ te_n@193 == te_nw@194 == {} (spec: 737, Δ = 0)",
        te_n_at_193
    );
}

/// Verify that sample 193's prediction delta matches the propagation
/// formula: wrong te_n (-21) causes wrong sub-predictions, which cause
/// wrong final prediction (+21 in true_err).
#[test]
#[serial]
fn r195_sample_193_prediction_propagation() {
    use trace_sample_193::*;

    // Capture sample 193's deep trace
    LEAF_PICK_TRACE_TARGET.store(
        encode_leaf_pick_target(0, X as u32, Y as u32),
        Ordering::Relaxed,
    );
    LEAF_PICK_TRACE_WP.with(|s| s.borrow_mut().clear());
    WP_DEEP_TRACE.with(|s| s.borrow_mut().clear());
    WP_DEEP_TRACE_ARMED.with(|c| c.set(true));
    let _ = decode_one_frame(NOISE_JXL, None).expect("decode");
    WP_DEEP_TRACE_ARMED.with(|c| c.set(false));
    LEAF_PICK_TRACE_TARGET.store(u64::MAX, Ordering::Relaxed);

    let wp = LEAF_PICK_TRACE_WP.with(|s| s.borrow().clone());
    let deep = WP_DEEP_TRACE.with(|s| s.borrow().clone());

    assert!(wp.len() >= 9, "WP capture must include wp_pred8");
    let wp_pred8 = wp[8];

    eprintln!("[r195] Sample 193 production vs trace:");
    eprintln!(
        "    wp_pred8 = {} (spec: {})",
        wp_pred8, PREDICTION_EXPECTED
    );
    eprintln!(
        "    delta    = {} (0 from round 278 onward)",
        wp_pred8 - PREDICTION_EXPECTED
    );

    if deep.len() >= 4 {
        eprintln!(
            "    subpred  = [{}, {}, {}, {}]",
            deep[0], deep[1], deep[2], deep[3]
        );
        eprintln!("    expected = {:?}", SUBPRED_EXPECTED);
    }

    // The stored true_err at sample 193 = wp_pred8 - true_value_8x.
    let production_te = wp_pred8 - TRUE_VALUE_8X;
    let spec_te = TRUE_ERR_EXPECTED; // 296

    eprintln!(
        "    production true_err = {} (spec: {})",
        production_te, spec_te
    );

    // Round-278: sample 193's prediction and stored true_err are
    // spec-exact (pre-fix they carried the +21 delta propagated from
    // the -21 te_n error at sample 129).
    assert_eq!(
        production_te, spec_te,
        "Sample 193's stored true_err must match spec ({}) from round \
         278 onward. Got {}",
        spec_te, production_te
    );
    assert_eq!(
        wp_pred8, PREDICTION_EXPECTED,
        "Sample 193's prediction must match the trace's {} from round \
         278 onward. Got {}",
        PREDICTION_EXPECTED, wp_pred8
    );
}

/// Hand-derive what sample 193's prediction WOULD be with the spec-correct
/// te_n = 737 (instead of production's te_n = 716), using the trace's
/// neighbour samples.
///
/// This verifies that the -21 te_n error is SUFFICIENT to explain the +21
/// prediction error (no other state corruption needed).
#[test]
#[serial]
fn r195_sample_193_corrected_te_n_gives_spec_prediction() {
    use trace_sample_193::*;

    let wp = WpHeader::default();

    // Capture production's te_* values at sample 193
    LEAF_PICK_TRACE_TARGET.store(
        encode_leaf_pick_target(0, X as u32, Y as u32),
        Ordering::Relaxed,
    );
    LEAF_PICK_TRACE_WP.with(|s| s.borrow_mut().clear());
    let _ = decode_one_frame(NOISE_JXL, None).expect("decode");
    LEAF_PICK_TRACE_TARGET.store(u64::MAX, Ordering::Relaxed);
    let wp_cap = LEAF_PICK_TRACE_WP.with(|s| s.borrow().clone());

    let te_w = wp_cap[0];
    let te_n_prod = wp_cap[1]; // Production's te_n (should be 716)
    let te_nw = wp_cap[2];
    let te_ne = wp_cap[3];

    eprintln!("[r195] Hand-deriving sample 193 with corrected te_n:");
    eprintln!(
        "    production te_*: w={}, n={}, nw={}, ne={}",
        te_w, te_n_prod, te_nw, te_ne
    );

    // The spec-correct te_n at sample 193 = true_err at sample 129 = 737
    let te_n_spec = 737i32;
    eprintln!(
        "    spec te_n = {} (Δ = {})",
        te_n_spec,
        te_n_spec - te_n_prod
    );

    // Compute sub-predictions with PRODUCTION te values
    let p0_prod = W8 + NE8 - N8;
    let p1_prod =
        N8 - ((((te_w as i64 + te_n_prod as i64 + te_ne as i64) * wp.p1 as i64) >> 5) as i32);
    let p2_prod =
        W8 - ((((te_w as i64 + te_n_prod as i64 + te_nw as i64) * wp.p2 as i64) >> 5) as i32);
    let p3_prod = N8
        - (((te_nw as i64 * wp.p3a as i64
            + te_n_prod as i64 * wp.p3b as i64
            + te_ne as i64 * wp.p3c as i64)
            >> 5) as i32);

    // Compute sub-predictions with SPEC-CORRECT te_n
    let p0_spec = W8 + NE8 - N8; // p0 doesn't use te_*
    let p1_spec =
        N8 - ((((te_w as i64 + te_n_spec as i64 + te_ne as i64) * wp.p1 as i64) >> 5) as i32);
    let p2_spec =
        W8 - ((((te_w as i64 + te_n_spec as i64 + te_nw as i64) * wp.p2 as i64) >> 5) as i32);
    let p3_spec = N8
        - (((te_nw as i64 * wp.p3a as i64
            + te_n_spec as i64 * wp.p3b as i64
            + te_ne as i64 * wp.p3c as i64)
            >> 5) as i32);

    eprintln!(
        "    production subpred = [{}, {}, {}, {}]",
        p0_prod, p1_prod, p2_prod, p3_prod
    );
    eprintln!(
        "    corrected subpred  = [{}, {}, {}, {}]",
        p0_spec, p1_spec, p2_spec, p3_spec
    );
    eprintln!("    trace expected     = {:?}", SUBPRED_EXPECTED);

    // p0 should match (it doesn't use te_*)
    assert_eq!(
        p0_spec, SUBPRED_EXPECTED[0],
        "p0 (gradient) should match trace"
    );

    // Check the deltas in p1, p2, p3 from the te_n correction
    let delta_te_n = te_n_spec - te_n_prod; // Should be +21
    eprintln!(
        "    te_n delta = {} (spec {} - prod {})",
        delta_te_n, te_n_spec, te_n_prod
    );

    // p1 adjustment delta = -(delta_te_n * p1 >> 5) = -(21 * 16 >> 5) = -10
    let p1_adj_delta = -(((delta_te_n as i64) * wp.p1 as i64) >> 5) as i32;
    eprintln!("    p1 adj delta = {}", p1_adj_delta);

    // p2 adjustment delta = -(delta_te_n * p2 >> 5) = -(21 * 10 >> 5) = -6
    let p2_adj_delta = -(((delta_te_n as i64) * wp.p2 as i64) >> 5) as i32;
    eprintln!("    p2 adj delta = {}", p2_adj_delta);

    // p3 adjustment delta = -(delta_te_n * p3b >> 5) = -(21 * 7 >> 5) = -4
    let p3_adj_delta = -(((delta_te_n as i64) * wp.p3b as i64) >> 5) as i32;
    eprintln!("    p3 adj delta = {}", p3_adj_delta);

    // These are the expected differences between corrected and production subpreds
    assert_eq!(
        p1_spec - p1_prod,
        p1_adj_delta,
        "p1 delta should match te_n correction"
    );
    assert_eq!(
        p2_spec - p2_prod,
        p2_adj_delta,
        "p2 delta should match te_n correction"
    );
    assert_eq!(
        p3_spec - p3_prod,
        p3_adj_delta,
        "p3 delta should match te_n correction"
    );
}

/// Scan earlier samples (row y=2) to find where the true_err divergence
/// FIRST appears. This narrows down the root cause for the next round.
#[test]
#[serial]
fn r195_true_err_divergence_scan_row_2() {
    eprintln!("[r195] Scanning row y=2 for true_err divergence...");

    // We know sample 129 (x=1, y=2) has true_err = 716 (should be 737).
    // Scan x=0..63 at y=2 to find the first divergent sample.
    //
    // Since we don't have spec-correct true_err values for all samples,
    // we'll look for the symmetric +21/-21 pattern propagating backwards.

    // Capture sample 129's true_err (already known to be 716)
    LEAF_PICK_TRACE_TARGET.store(encode_leaf_pick_target(0, 1, 2), Ordering::Relaxed);
    LEAF_PICK_TRACE_WP.with(|s| s.borrow_mut().clear());
    let _ = decode_one_frame(NOISE_JXL, None).expect("decode");
    let wp_129 = LEAF_PICK_TRACE_WP.with(|s| s.borrow().clone());
    LEAF_PICK_TRACE_TARGET.store(u64::MAX, Ordering::Relaxed);

    // Sample 129's te_* values tell us about earlier samples
    if wp_129.len() >= 4 {
        let te_w_129 = wp_129[0]; // true_err at (0, 2) = sample 128
        let te_n_129 = wp_129[1]; // true_err at (1, 1) = sample 65
        let te_nw_129 = wp_129[2]; // true_err at (0, 1) = sample 64
        let te_ne_129 = wp_129[3]; // true_err at (2, 1) = sample 66

        eprintln!("    sample 129 (x=1, y=2) reads:");
        eprintln!("        te_w  = {} (from sample 128)", te_w_129);
        eprintln!("        te_n  = {} (from sample 65)", te_n_129);
        eprintln!("        te_nw = {} (from sample 64)", te_nw_129);
        eprintln!("        te_ne = {} (from sample 66)", te_ne_129);
    }

    // Capture sample 130's te_* to compare
    LEAF_PICK_TRACE_TARGET.store(encode_leaf_pick_target(0, 2, 2), Ordering::Relaxed);
    LEAF_PICK_TRACE_WP.with(|s| s.borrow_mut().clear());
    let _ = decode_one_frame(NOISE_JXL, None).expect("decode");
    let wp_130 = LEAF_PICK_TRACE_WP.with(|s| s.borrow().clone());
    LEAF_PICK_TRACE_TARGET.store(u64::MAX, Ordering::Relaxed);

    // Sample 130's te_n is from sample 66, same as sample 129's te_ne
    if wp_130.len() >= 4 {
        let te_n_130 = wp_130[1]; // true_err at (2, 1) = sample 66
        let te_nw_130 = wp_130[2]; // true_err at (1, 1) = sample 65

        eprintln!("    sample 130 (x=2, y=2) reads:");
        eprintln!("        te_n  = {} (from sample 66)", te_n_130);
        eprintln!("        te_nw = {} (from sample 65)", te_nw_130);

        // te_n at sample 130 == te_ne at sample 129 (both read sample 66)
        if wp_129.len() >= 4 {
            assert_eq!(
                te_n_130, wp_129[3],
                "te_n@130 should equal te_ne@129 (both read sample 66)"
            );
            eprintln!("    ✓ te_n@130 == te_ne@129 == {} (sample 66)", te_n_130);
        }

        // te_nw at sample 130 == te_n at sample 129 (both read sample 65)
        if wp_129.len() >= 4 {
            assert_eq!(
                te_nw_130, wp_129[1],
                "te_nw@130 should equal te_n@129 (both read sample 65)"
            );
            eprintln!("    ✓ te_nw@130 == te_n@129 == {} (sample 65)", te_nw_130);
        }
    }

    // The root cause must be before sample 129. Let's check sample 65 (x=1, y=1).
    LEAF_PICK_TRACE_TARGET.store(encode_leaf_pick_target(0, 1, 1), Ordering::Relaxed);
    LEAF_PICK_TRACE_WP.with(|s| s.borrow_mut().clear());
    let _ = decode_one_frame(NOISE_JXL, None).expect("decode");
    let wp_65 = LEAF_PICK_TRACE_WP.with(|s| s.borrow().clone());
    LEAF_PICK_TRACE_TARGET.store(u64::MAX, Ordering::Relaxed);

    if wp_65.len() >= 4 {
        eprintln!("    sample 65 (x=1, y=1) reads:");
        eprintln!("        te_w  = {} (from sample 64)", wp_65[0]);
        eprintln!("        te_n  = {} (from sample 1)", wp_65[1]);
        eprintln!("        te_nw = {} (from sample 0)", wp_65[2]);
        eprintln!("        te_ne = {} (from sample 2)", wp_65[3]);
    }

    eprintln!("\n[r195] Summary: the -21 error at sample 129 may propagate");
    eprintln!("    from sample 65 (te_n for sample 129). Next round should");
    eprintln!("    check if sample 65's true_err is also divergent, and");
    eprintln!("    continue the chain back to find the ROOT sample.");
}
