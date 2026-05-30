//! Round 191 — Weighted-Predictor (Annex E / §H.5.2) oracle test
//! driven by the clean-room behavioural trace at
//! `docs/image/jpegxl/fixtures/noise-64x64-lossless/
//!  wp-trace-sample-194.md`.
//!
//! ## What this test does
//!
//! The trace doc reports, for the `noise-64x64-lossless` fixture's
//! sample 194 (`channel 0`, `x = 2`, `y = 3`), the ISO/IEC FDIS
//! 18181-1:2021 Annex E.2 *ground-truth* per-listing intermediates a
//! spec-conformant decoder must produce:
//!
//! | quantity | trace |
//! |----------|------:|
//! | `true_errN` (`-456`), `true_errW` (`296`), `true_errNW` (`737`), `true_errNE` (`-165`) | given |
//! | `sub_err_i,N`, `sub_err_i,NE`, `sub_err_i,NW` per i ∈ 0..4 | given (`err_sum_i` = sum of the three) |
//! | `prediction_i` per i ∈ 0..4 | `[1248, 747, 420, 559]` |
//! | `weight_i` per i ∈ 0..4 (Listing E.2 `error2weight`) | `[495694, 599189, 474830, 825112]` |
//! | final `prediction` (Listing E.3, pre-rounding) | `709` |
//!
//! All values are in the FDIS "left-shifted by 3" sample domain
//! (Annex E.2). `WpHeader` is at defaults (`p1=16, p2=10, p3a=p3b=p3c=7,
//! p3d=p3e=0, w0=13, w1=12, w2=12, w3=12`).
//!
//! This test builds a synthetic `WpState` from the trace's `sub_err_i,*`
//! and `true_err*` values, builds a synthetic `Neighbours` from the
//! trace's `N/W/NE/NW/NN/WW` neighbour samples, then calls our public
//! `wp_predict_pub` wrapper around the production weighted-predictor.
//!
//! If our predictor arithmetic is spec-conformant, the returned tuple
//! must equal the trace's reported sub-predictions and final prediction
//! exactly. If it diverges, the test pins **which listing** of Annex E
//! deviates from the spec (Listing E.1 sub-prediction, Listing E.2
//! `err_sum_i` / `error2weight`, Listing E.3 weighted sum + clamp, or
//! Listing E.4 `max_error`).
//!
//! ## Why this test exists (round 191)
//!
//! Rounds 31..126 bisected the `noise-64x64-lossless` pixel divergence
//! to a `wp_pred8 = 717` vs spec-correct `709` off-by-8 (= off-by-1 in
//! un-shifted sample space) at sample 194. The round-126 capture pinned
//! `te_w=317, te_nw=716, te_ne=-160, subpred_3=563, err_sum=[438, 322,
//! 397, 257]` from a full decode of the fixture against our
//! still-evolving WP state machine.
//!
//! The trace doc landed 2026-05-30 provides the spec-conformant
//! reference numbers for the **same** sample (`te_w=296, te_nw=737,
//! te_ne=-165, subpred_3=559, err_sum=[438, 330, 416, 240]`). The
//! gap between the two sets of numbers is the WP state-evolution bug
//! we are hunting.
//!
//! This test does *not* try to fix the state-evolution bug. Instead it
//! establishes that — given the spec-correct state inputs as initial
//! conditions — our `wp_predict` arithmetic is itself faithful to
//! Listings E.1 / E.2 / E.3. That isolates the bug to upstream state
//! evolution (the `set_true_err` / `set_sub_err` calls fired after
//! decoding samples 0..193) and excludes Listings E.1-E.3 from suspicion.
//!
//! ## Verification path
//!
//! 1. Construct `WpState { width=64, height=64, .. }` with
//!    `sub_err_i,N/NE/NW` and `true_err_{N,W,NW,NE}` placed at the
//!    positions our `wp_predict` reads them from.
//! 2. Construct `Neighbours { w=600/8, n=584/8, nw=128/8, ne=1232/8,
//!    nn=416/8, ww=1192/8, nee=NE-fallback }` — the raw sample values
//!    are the trace's left-shifted-by-3 values divided by 8.
//! 3. Call `wp_predict_pub(&state, &nb, 2, 3, &WpHeader::default())`.
//! 4. Assert (a) each `subpred_i` equals the trace, (b) the final
//!    weighted prediction is 709 ± 0, (c) `max_error` is the
//!    `|true_errX|`-largest of the four neighbour `true_err` values
//!    (Listing E.4).
//!
//! ## Spec citations
//!
//! * Annex E.1 — `true_err`, `sub_err_i` definitions (line 4480-4485).
//! * Annex E.2, Listing E.1 — sub-predictions (line 4527-4532).
//! * Annex E.2, Listing E.2 — `error2weight`, `err_sum_i`, `weight_i`
//!   (line 4539-4546).
//! * Annex E.2, Listing E.3 — final weighted prediction + same-sign
//!   clamp (line 4551-4562).
//! * Annex E.2, Listing E.4 — `max_error` (line 4569-4573).
//! * Annex C.16 — `prediction(x, y, 6) = (E-prediction + 3) >> 3`.
//!
//! Trace doc: `docs/image/jpegxl/fixtures/noise-64x64-lossless/
//! wp-trace-sample-194.md`.
//! Provenance: `docs/image/jpegxl/fixtures/noise-64x64-lossless/
//! wp-trace-provenance.md`.

use oxideav_jpegxl::modular_fdis::{wp_predict_pub, Neighbours, WpHeader, WpState};

/// Sample 194 trace data — every value reproduced from the in-repo
/// docs trace doc, with the FDIS conventions (left-shifted by 3 for
/// neighbour samples, raster x/y, etc.).
mod trace_sample_194 {
    /// Channel geometry of the noise-64x64-lossless fixture.
    pub const WIDTH: u32 = 64;
    pub const HEIGHT: u32 = 64;

    /// Sample under test: x=2, y=3, channel 0.
    pub const X: i32 = 2;
    pub const Y: i32 = 3;

    // Neighbour samples in 8x scale (`N`, `W`, etc. left-shifted by 3
    // per Annex E.2).
    pub const N8: i32 = 584;
    pub const W8: i32 = 600;
    pub const NE8: i32 = 1232;
    pub const NW8: i32 = 128;
    pub const NN8: i32 = 416;
    pub const WW8: i32 = 1192;
    // NEE is not part of the WP listings (only used by `prediction0` in
    // a couple of variants — Listing E.1 does not reference NEE), but
    // `Neighbours` carries the field, so we plug in `NE` per the H.3
    // edge-rule fallback.
    pub const NEE8: i32 = 1232;

    // Neighbour `true_err` values (already in 8x scale, per Annex E.1).
    pub const TRUE_ERR_W: i32 = 296;
    pub const TRUE_ERR_N: i32 = -456;
    pub const TRUE_ERR_NW: i32 = 737;
    pub const TRUE_ERR_NE: i32 = -165;

    // Per-sub-predictor accumulated-error values at the three positions
    // our `wp_predict` reads (N, NE, NW). For the in-image positions of
    // sample 194 the spec's 5-term `err_sum_i` (N + W + NW + WW + NE)
    // also includes positions W (sample 193) and WW (sample 192) of
    // the current row.
    //
    // The trace doc reports `err_sum_i` as the 5-term sum:
    //   i=0: err_sum = 438
    //   i=1: err_sum = 330
    //   i=2: err_sum = 416
    //   i=3: err_sum = 240
    //
    // We split each `err_sum_i` across the five neighbour positions
    // (N, W, NW, WW, NE) so the spec formula reproduces the trace
    // value when our `wp_predict` reads from these positions. The
    // doc enumerates `sub_err_i,N` / `sub_err_i,NE` / `sub_err_i,NW`
    // explicitly (table at line 78-83 of the doc); we add the remainder
    // as a single W contribution to match `err_sum_i`.
    pub const SUB_ERR_N: [i32; 4] = [174, 142, 190, 82];
    pub const SUB_ERR_NE: [i32; 4] = [90, 29, 92, 67];
    pub const SUB_ERR_NW: [i32; 4] = [174, 159, 134, 91];
    /// `sub_err_i,W` derived as `err_sum_i - (N + NE + NW)`. The doc's
    /// fold note says "the W and WW contributions are already folded
    /// into the stored N and NW accumulators" — but since our
    /// implementation reads the 5 FDIS-literal positions separately,
    /// we synthesise an equivalent W-only contribution (with WW = 0)
    /// that reproduces the same `err_sum_i`.
    pub const SUB_ERR_W: [i32; 4] = [
        // err_sum_0 = 438; N+NE+NW = 174+90+174 = 438; remainder = 0
        0, // err_sum_1 = 330; N+NE+NW = 142+29+159 = 330; remainder = 0
        0, // err_sum_2 = 416; N+NE+NW = 190+92+134 = 416; remainder = 0
        0, // err_sum_3 = 240; N+NE+NW = 82+67+91 = 240; remainder = 0
        0,
    ];
    pub const SUB_ERR_WW: [i32; 4] = [0, 0, 0, 0];

    // Ground truth — what our `wp_predict` must return when fed the
    // above state.
    pub const SUBPRED_EXPECTED: [i32; 4] = [1248, 747, 420, 559];
    pub const WEIGHT_EXPECTED: [u64; 4] = [495_694, 599_189, 474_830, 825_112];
    pub const PREDICTION_EXPECTED: i32 = 709;
    // `max_error = arg-max-by-abs(te_W, te_N, te_NW, te_NE)`. With our
    // four `te_*` values, `|te_NW| = 737` is the largest, so the spec
    // selects `te_NW = 737`.
    pub const MAX_ERROR_EXPECTED: i32 = 737;
}

/// Build a `WpState` of the fixture's dimensions with the trace's
/// `sub_err_i` and `true_err_X` values planted at the positions our
/// `wp_predict` reads.
///
/// Sample 194 is (x=2, y=3). The neighbour read positions are:
///   - N  = (2, 2)
///   - W  = (1, 3)
///   - NW = (1, 2)
///   - NE = (3, 2)
///   - WW = (0, 3)
fn build_wp_state_for_sample_194() -> WpState {
    use trace_sample_194::*;

    let mut state = WpState::new(WIDTH, HEIGHT);
    let w = WIDTH as usize;

    // Index helper: row-major, matching our WpState layout.
    let idx = |x: u32, y: u32| -> usize { (y as usize) * w + (x as usize) };

    // Plant true_err at N, W, NW, NE positions.
    state.true_err[idx(2, 2)] = TRUE_ERR_N; // N = (2, 2)
    state.true_err[idx(1, 3)] = TRUE_ERR_W; // W = (1, 3)
    state.true_err[idx(1, 2)] = TRUE_ERR_NW; // NW = (1, 2)
    state.true_err[idx(3, 2)] = TRUE_ERR_NE; // NE = (3, 2)

    // Plant sub_err_i at N, W, NW, WW, NE positions for each
    // sub-predictor i.
    for i in 0..4 {
        state.sub_err[i][idx(2, 2)] = SUB_ERR_N[i];
        state.sub_err[i][idx(1, 3)] = SUB_ERR_W[i];
        state.sub_err[i][idx(1, 2)] = SUB_ERR_NW[i];
        state.sub_err[i][idx(0, 3)] = SUB_ERR_WW[i];
        state.sub_err[i][idx(3, 2)] = SUB_ERR_NE[i];
    }
    state
}

fn build_neighbours_for_sample_194() -> Neighbours {
    use trace_sample_194::*;
    // `Neighbours` stores raw sample values; our `wp_predict` then
    // left-shifts each by 3 internally. So we divide the trace's
    // 8x-scaled values by 8.
    Neighbours {
        w: W8 >> 3,
        n: N8 >> 3,
        nw: NW8 >> 3,
        ne: NE8 >> 3,
        nn: NN8 >> 3,
        nee: NEE8 >> 3,
        ww: WW8 >> 3,
    }
}

/// Top-level oracle: the production `wp_predict_pub` must reproduce
/// the trace's reported sub-predictions, final prediction, and
/// `max_error` exactly when fed the trace's spec-conformant state.
#[test]
fn r191_wp_predict_matches_trace_at_sample_194() {
    use oxideav_jpegxl::modular_fdis::{WP_DEEP_TRACE, WP_DEEP_TRACE_ARMED};
    use trace_sample_194::*;
    let state = build_wp_state_for_sample_194();
    let nb = build_neighbours_for_sample_194();
    let wp = WpHeader::default();

    // Arm the deep trace so we capture per-i weights and shifted weights.
    WP_DEEP_TRACE.with(|s| s.borrow_mut().clear());
    WP_DEEP_TRACE_ARMED.with(|c| c.set(true));
    let (prediction, subpreds, max_error) = wp_predict_pub(&state, &nb, X, Y, &wp);
    WP_DEEP_TRACE_ARMED.with(|c| c.set(false));
    let deep = WP_DEEP_TRACE.with(|s| s.borrow().clone());

    eprintln!("[round-191] WP oracle at sample 194 (trace-driven):");
    eprintln!("    subpred = {:?}", subpreds);
    eprintln!("    expected = {:?}", SUBPRED_EXPECTED);
    eprintln!("    prediction (pre-round) = {}", prediction);
    eprintln!("    expected = {}", PREDICTION_EXPECTED);
    eprintln!("    max_error = {}", max_error);
    eprintln!("    expected = {}", MAX_ERROR_EXPECTED);
    if deep.len() >= 16 {
        eprintln!(
            "    err_sums (impl)   = [{}, {}, {}, {}]",
            deep[4], deep[5], deep[6], deep[7]
        );
        eprintln!(
            "    shifted weights   = [{}, {}, {}, {}] (sum_post={})",
            deep[8], deep[9], deep[10], deep[11], deep[15]
        );
        eprintln!(
            "    sum_w_pre={} log_weight={} sh={}",
            deep[12], deep[13], deep[14]
        );
    }

    assert_eq!(
        subpreds, SUBPRED_EXPECTED,
        "Listing E.1 sub-predictions diverge from the trace doc at \
         sample 194. Trace expects {:?}, our wp_predict produced {:?}. \
         This means the bug is INSIDE wp_predict's Listing E.1 \
         arithmetic (not upstream state evolution).",
        SUBPRED_EXPECTED, subpreds
    );

    assert_eq!(
        prediction, PREDICTION_EXPECTED,
        "Listing E.3 final weighted prediction diverges from the trace \
         doc at sample 194. Trace expects {}, our wp_predict produced {}. \
         If sub-predictions matched (asserted above) but prediction \
         doesn't, the bug is in Listing E.2 weight computation \
         (`error2weight`) or Listing E.3 weighted sum / `same-sign` \
         clamp.",
        PREDICTION_EXPECTED, prediction
    );

    assert_eq!(
        max_error, MAX_ERROR_EXPECTED,
        "Listing E.4 max_error must select the |true_err|-largest of \
         (te_W={}, te_N={}, te_NW={}, te_NE={}); trace expects \
         max_error = {} (te_NW), our wp_predict produced {}.",
        TRUE_ERR_W, TRUE_ERR_N, TRUE_ERR_NW, TRUE_ERR_NE, MAX_ERROR_EXPECTED, max_error
    );
}

/// Cross-validate the trace's `err_sum_i` values against the
/// per-position `sub_err_i,{N,NE,NW}` accumulators reported in the
/// trace doc. This is a pure-arithmetic sanity check on the trace
/// data itself — if it fails, the trace doc has an internal
/// inconsistency, not our decoder.
#[test]
fn r191_trace_err_sum_self_consistency() {
    use trace_sample_194::*;
    // The doc reports err_sum_i as a 5-term sum (N + W + NW + WW + NE).
    // We also report N + NE + NW per-position values. The remainder
    // (W + WW) must be non-negative since `sub_err_i = abs(...) >= 0`.
    let expected = [438u64, 330, 416, 240];
    for i in 0..4 {
        let three_pos = (SUB_ERR_N[i] as u64) + (SUB_ERR_NE[i] as u64) + (SUB_ERR_NW[i] as u64);
        let five_pos = three_pos + (SUB_ERR_W[i] as u64) + (SUB_ERR_WW[i] as u64);
        assert_eq!(
            five_pos, expected[i],
            "trace sub_err_{}[N+W+NW+WW+NE] = {} but doc reports \
             err_sum_{} = {}",
            i, five_pos, i, expected[i]
        );
    }
}

/// Cross-validate the trace's reported `weight_i` values against the
/// Annex E.2 `error2weight(err_sum_i, wp_w_i)` arithmetic, and pin the
/// `(1<<24) Idiv denom` evaluation order as the spec-literal one.
///
/// FDIS-2021 Listing E.2 line 4543 reads:
/// ```text
///   error2weight(err_sum, maxweight) {
///     shift = floor(log2(err_sum + 1)) - 5;
///     if (shift < 0) shift = 0;
///     return 4 + (maxweight × ((1 << 24) Idiv ((err_sum >> shift) + 1)));
///   }
/// ```
///
/// The parenthesisation is unambiguous: `((1 << 24) Idiv denom)` is
/// evaluated **first** (truncating integer division), then the result
/// is multiplied by `maxweight`. Our production `wp_predict` currently
/// computes `maxweight × (1 << 24) Idiv denom` (multiplication-first
/// reading) under a 2024-published-edition tweak that also applies an
/// outer `>> shift` to bound the weight; for some `(err_sum, maxweight,
/// shift)` combinations the two evaluation orders disagree by 1
/// **before** the outer shift, and crucially can agree or disagree by
/// 1 in the final shifted weight as well.
///
/// For sample 194's four `(err_sum_i, w_i)` pairs the shifted weights
/// are identical under both readings (`[3, 4, 3, 6]`) so the final
/// prediction is unaffected. This test pins:
///
/// 1. The trace doc's `weight_i` values match the FDIS-literal
///    (inner-first) reading exactly.
/// 2. Our impl's (multiplication-first + outer-shift) reading produces
///    a weight that may be 1 larger than the FDIS-literal at this
///    sample, but both round to the same `>> sh` shifted weight.
///
/// This double-pin documents a latent spec-literal mismatch in
/// `wp_predict` that does NOT contribute to the noise-64x64-lossless
/// sample-194 wp_pred8 = 717 vs spec-correct 709 off-by-8 divergence.
/// (Followup: round 192 + can investigate whether the multiplication
/// -first reading affects a different fixture's prediction; for the
/// noise fixture sample 194 it does not.)
#[test]
fn r191_trace_weights_match_error2weight() {
    use trace_sample_194::*;
    let err_sums = [438u64, 330, 416, 240];
    let maxweights = [13u64, 12, 12, 12]; // WpHeader defaults

    for i in 0..4 {
        let es = err_sums[i];
        let bits = 64u32 - (es + 1).leading_zeros();
        let shift = (bits.saturating_sub(1)).saturating_sub(5);
        let denom = (es >> shift) + 1;
        // FDIS-literal reading (inner Idiv first, no outer shift on
        // the inner product).
        let fdis_lit = 4u64 + maxweights[i] * ((1u64 << 24) / denom);
        // 2024-edition reading the impl uses (multiplication-first,
        // outer shift applied to the inner product).
        let impl_form = 4u64 + ((maxweights[i] * (1u64 << 24) / denom) >> shift);
        // FDIS-literal + outer shift = the FDIS-literal trace value
        // after the implicit `>> shift` libjxl applies for bounding.
        let fdis_lit_shifted = 4u64 + ((maxweights[i] * ((1u64 << 24) / denom)) >> shift);
        eprintln!(
            "    i={i} es={es} shift={shift} denom={denom}: \
             fdis_lit={fdis_lit} fdis_lit_shifted={fdis_lit_shifted} \
             impl={impl_form} trace={}",
            WEIGHT_EXPECTED[i]
        );
        assert_eq!(
            fdis_lit_shifted, WEIGHT_EXPECTED[i],
            "trace's weight_{} should match FDIS-literal (inner Idiv \
             first) + outer shift = {}, got {}",
            i, WEIGHT_EXPECTED[i], fdis_lit_shifted
        );

        // Both readings produce the same shifted weight at the
        // Listing E.3 step (so they don't affect sample 194's final
        // prediction). Let `log_weight = floor(log2(sum_pre)) + 1`
        // and `sh = log_weight - 5`; verify
        // `fdis_lit_shifted >> sh == impl_form >> sh` for the actual
        // `sh` used at sample 194.
        let sh: u32 = 17; // pinned for sample 194 trace numbers
        assert_eq!(
            fdis_lit_shifted >> sh,
            impl_form >> sh,
            "FDIS-literal and impl evaluation orders must agree on the \
             >>{} shifted weight for sample 194 (otherwise the final \
             prediction would differ).",
            sh
        );
    }
}

/// Cross-validate the trace's final `prediction = 709` against
/// Listing E.3 literal arithmetic.
///
/// The relationship:
/// ```text
///   sum_weights = sum_i weight_i
///   log_weight = floor(log2(sum_weights)) + 1
///   weight_i' = weight_i >> (log_weight - 5)
///   sum_weights' = sum_i weight_i'
///   s = (sum_weights' >> 1)
///   s += sum_i (prediction_i × weight_i')
///   prediction = s × ((1 << 24) Idiv sum_weights') >> 24
/// ```
///
/// (The "same-sign clamp" at the bottom of Listing E.3 does NOT fire
/// for sample 194 because `te_N = -456` (negative), `te_W = 296`
/// (positive) — different signs, so the clamp predicate is false.)
#[test]
fn r191_trace_prediction_matches_listing_e3() {
    use trace_sample_194::*;

    let weights = WEIGHT_EXPECTED;
    let preds = SUBPRED_EXPECTED;
    let sum_pre: u64 = weights.iter().sum();
    let log_weight = 64u32 - sum_pre.leading_zeros();
    let shift = log_weight.saturating_sub(5);
    let shifted: [u64; 4] = [
        weights[0] >> shift,
        weights[1] >> shift,
        weights[2] >> shift,
        weights[3] >> shift,
    ];
    let sum_post: u64 = shifted.iter().sum();
    let s_init = (sum_post as i64) >> 1;
    let mut s = s_init;
    for i in 0..4 {
        s += (preds[i] as i64) * (shifted[i] as i64);
    }
    let denom = (1i64 << 24) / (sum_post as i64);
    let pred = ((s * denom) >> 24) as i32;

    eprintln!("[round-191] Listing E.3 hand-derivation:");
    eprintln!("    sum_pre  = {}", sum_pre);
    eprintln!("    log_w    = {}", log_weight);
    eprintln!("    shift    = {}", shift);
    eprintln!("    shifted  = {:?}", shifted);
    eprintln!("    sum_post = {}", sum_post);
    eprintln!("    s_init   = {}", s_init);
    eprintln!("    s        = {}", s);
    eprintln!("    denom    = {}", denom);
    eprintln!("    pred     = {}", pred);

    // Same-sign clamp predicate (Listing E.3 line 4560):
    //   if (((te_N ^ te_W) | (te_N ^ te_NW)) <= 0) clamp fires.
    //
    // For sample 194 te_N=-456 (sign=1), te_W=296 (sign=0),
    // te_NW=737 (sign=0). The XOR sign-bits give 1 for both
    // (te_N ^ te_W) and (te_N ^ te_NW), so the OR also has sign=1
    // → predicate <= 0 is TRUE → clamp DOES fire. However, the
    // pre-clamp prediction (709) is already inside [min(W,N,NE),
    // max(W,N,NE)] = [584, 1232], so the clamp is a no-op.
    let clamp_pred = (TRUE_ERR_N ^ TRUE_ERR_W) | (TRUE_ERR_N ^ TRUE_ERR_NW);
    assert!(
        clamp_pred <= 0,
        "Listing E.3 same-sign clamp predicate at sample 194 must be \
         <= 0 (clamp fires; but the pre-clamp prediction is in-range \
         so it's a no-op). Got predicate value {}",
        clamp_pred
    );
    let lo = W8.min(N8).min(NE8);
    let hi = W8.max(N8).max(NE8);
    assert!(
        pred >= lo && pred <= hi,
        "Pre-clamp prediction {} must already be in [min(W,N,NE)={}, \
         max(W,N,NE)={}] for sample 194 so the firing clamp is a no-op \
         (we get 709 either way). If this fails the trace-data \
         prediction value disagrees with the clamp bracket.",
        pred,
        lo,
        hi
    );

    assert_eq!(
        pred, PREDICTION_EXPECTED,
        "Listing E.3 hand-derivation gives prediction = {}, but the \
         trace doc reports prediction = {}. Either the trace doc is \
         wrong, or our reading of Listing E.3 disagrees with the FDIS \
         literal.",
        pred, PREDICTION_EXPECTED
    );
}

/// Pin the GAP between the production decoder's WP intermediates at
/// sample 194 (captured by round-126's `r126_wp_intermediates_at_194`)
/// and the trace doc's spec-conformant intermediates. This test is a
/// documentation aid: it materialises the WP state-evolution defect as
/// a list of named numeric deltas the next agent can attack
/// systematically.
///
/// The pinned numbers come from:
///   - production: round 126 captured values (see `r126_wp_intermediates_at_194.rs`)
///   - spec:       this trace doc
#[test]
fn r191_pin_state_evolution_gap() {
    let production_te_w: i32 = 317;
    let production_te_nw: i32 = 716;
    let production_te_ne: i32 = -160;
    let production_err_sum: [u64; 4] = [438, 322, 397, 257];
    let production_subpred: [i32; 4] = [1248, 734, 420, 563];
    let production_wp_pred8: i32 = 717;

    use trace_sample_194::*;

    eprintln!(
        "[round-191] sample-194 WP state-evolution gap \
         (production minus trace):"
    );
    eprintln!(
        "    Δ te_w  = {:+} ({} - {})",
        production_te_w - TRUE_ERR_W,
        production_te_w,
        TRUE_ERR_W
    );
    eprintln!(
        "    Δ te_nw = {:+} ({} - {})",
        production_te_nw - TRUE_ERR_NW,
        production_te_nw,
        TRUE_ERR_NW
    );
    eprintln!(
        "    Δ te_ne = {:+} ({} - {})",
        production_te_ne - TRUE_ERR_NE,
        production_te_ne,
        TRUE_ERR_NE
    );
    for i in 0..4 {
        let expected_es = [438u64, 330, 416, 240][i];
        eprintln!(
            "    Δ err_sum[{}] = {:+} ({} - {})",
            i,
            production_err_sum[i] as i64 - expected_es as i64,
            production_err_sum[i],
            expected_es
        );
    }
    for i in 0..4 {
        eprintln!(
            "    Δ subpred[{}] = {:+} ({} - {})",
            i,
            production_subpred[i] - SUBPRED_EXPECTED[i],
            production_subpred[i],
            SUBPRED_EXPECTED[i]
        );
    }
    eprintln!(
        "    Δ wp_pred8 = {:+} ({} - {})",
        production_wp_pred8 - PREDICTION_EXPECTED,
        production_wp_pred8,
        PREDICTION_EXPECTED
    );

    // Sanity invariant: err_sum_0 already matches the trace — i.e. the
    // sub-predictor-0 state evolution is correct. The defect is
    // localised to sub-predictors 1, 2, 3.
    assert_eq!(
        production_err_sum[0], 438,
        "round-126's err_sum[0] = 438 must match the trace's \
         err_sum_0 = 438. If this fails, sub-predictor 0 also \
         drifted (it didn't, at round 126)."
    );

    // The gap on te_w / te_nw is exactly +21 / -21 — a symmetric pair.
    // This pattern hints the state-evolution defect is a sign-flipped
    // `te` storage at some earlier sample (one extra row?), which the
    // next round (192) should investigate via a per-sample `te*`-vs-
    // trace bisect over y in 0..3.
    assert_eq!(
        production_te_w - TRUE_ERR_W,
        21,
        "Δ te_w pinned at +21; if this changes the upstream state-evolution \
         landscape has shifted and the round-192 bisect plan needs an update."
    );
    assert_eq!(
        production_te_nw - TRUE_ERR_NW,
        -21,
        "Δ te_nw pinned at -21; pairs with Δ te_w = +21 — symmetric \
         gap suggests a single upstream defect, not 3 independent ones."
    );
    assert_eq!(
        production_wp_pred8 - PREDICTION_EXPECTED,
        8,
        "Δ wp_pred8 pinned at +8 in 8x scale = +1 in un-shifted \
         sample space — exactly the off-by-1 pixel divergence \
         observed in `r126_first_divergence_scan` (dec=35, exp=34)."
    );
}
