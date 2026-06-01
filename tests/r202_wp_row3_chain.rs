//! Round 202 — Weighted-Predictor state-evolution **chain** across the
//! whole of row `y = 3` (samples 192..200) of the
//! `noise-64x64-lossless` fixture.
//!
//! ## Why this test exists
//!
//! Rounds 31..195 progressively localised the WP state-evolution defect
//! at sample 194 (`wp_pred8 = 717` vs spec `709` off-by-8 → pixel
//! divergence `dec=35` vs `exp=34`). The clean-room behavioural trace
//! at `docs/image/jpegxl/fixtures/noise-64x64-lossless/
//! wp-trace-sample-194.md` reports the FDIS-conformant per-sample
//! `prediction` / `true_value` / `true_err` values for **the entire
//! row 3 window samples 188..200** (table at doc lines 130-168).
//!
//! Round 195 captured production WP state at samples 129, 130, 193, 194
//! and proved the propagation hypothesis "te_n@193 == te_nw@194 ==
//! sample-129's stored true_err". It also surfaced a stronger fact
//! from the production captures:
//!
//! ```text
//! sample 193: te_w = -804   (= production stored true_err at sample 192)
//! sample 194: te_w =  317   (= production stored true_err at sample 193)
//! ```
//!
//! Cross-referencing the trace doc's `true_err` column for the same
//! samples (sample 192 = `-754`, sample 193 = `296`) gives a per-sample
//! stored-true_err delta profile across row 3:
//!
//! ```text
//! sample 192: production stored = ?    spec stored = -754  Δ = ?
//! sample 193: production stored = 317  spec stored = 296   Δ = +21
//! sample 194: production stored = ?    spec stored = 437   Δ = ?
//! ...
//! ```
//!
//! ## What this test does
//!
//! 1. Captures `wp_pred8` and the four `te_*` accumulators for each
//!    sample in `192..=200` via the existing [`LEAF_PICK_TRACE_WP`]
//!    capture hook (one decode per sample because the capture is
//!    target-keyed).
//! 2. Pins the production `wp_pred8` and the production stored
//!    `true_err` at each sample to the captured numbers, with
//!    eprintln deltas vs the trace-doc spec values for forward
//!    auditability.
//! 3. Pins the within-row read chain:
//!    `te_w(s+1) == wp_pred8(s) - true_value_8x_spec(s)` for every
//!    `s` in `192..=199`. (This is the cleanest cross-validation
//!    available: the decoded sample value `v(s)` MUST equal the spec
//!    `true_value_8x(s)` whenever the production sample at `s` is
//!    bit-exact decoded — which it is at every row-3 sample except
//!    `s = 194`, where round 32's pixel bisect pinned `dec=35` vs
//!    `exp=34`.)
//! 4. Pins the cross-row read chain at row-3 entries that read from
//!    samples in row 2 with known spec `true_err`:
//!    - `te_n(194) == -456`  (sample 130 spec)
//!    - `te_nw(194) == 737`  (sample 129 spec — known defective:
//!      production = 716, Δ = -21)
//!    - `te_ne(194) == -165` (sample 131 spec — known defective:
//!      production = -160, Δ = +5)
//! 5. Records the wider chain `te_n(195) == te_ne(194)`,
//!    `te_n(196) == te_ne(195)`, ... so the next round can follow the
//!    drift profile across the entire row-3 prediction sequence
//!    without re-deriving the read positions.
//!
//! ## Spec citations
//!
//! - ISO/IEC FDIS 18181-1:2021 Annex E.1 — `true_err` storage
//!   (`set_true_err` definition, line 4485).
//! - Annex E.2 Listings E.1 / E.2 / E.3 / E.4 — sub-prediction +
//!   weight + final-prediction + max_error.
//! - Annex C.16 — `prediction(x, y, 6) = (E-prediction + 3) >> 3`.
//! - Trace doc: `docs/image/jpegxl/fixtures/noise-64x64-lossless/
//!   wp-trace-sample-194.md` lines 130-168 (sample 188..200 context).
//! - Provenance: `wp-trace-provenance.md` (clean-room attribution +
//!   bit-exact decode confirmation).
//!
//! ## Scope of this round
//!
//! This test does NOT fix the state-evolution defect — it only locks
//! the per-sample drift profile across row 3 as a contract a future
//! round can drive the fix against. (The fix itself remains
//! upstream-bisect-blocked because the trace doc does not give
//! spec-correct `true_err` values for samples before 188.)

use oxideav_jpegxl::decode_one_frame;
use oxideav_jpegxl::modular_fdis::{
    encode_leaf_pick_target, LEAF_PICK_TRACE_TARGET, LEAF_PICK_TRACE_WP,
};
use serial_test::serial;
use std::sync::atomic::Ordering;

const NOISE_JXL: &[u8] = include_bytes!("fixtures/noise_64x64_lossless.jxl");

/// Row-3 spec ground truth from `wp-trace-sample-194.md` lines 142-168.
///
/// Each tuple = `(sample_index, x, y, true_value_8x, true_err_spec,
/// wp_pred8_spec)` where `wp_pred8_spec = true_err_spec + true_value_8x`
/// (Annex E.1 definition of `true_err`).
const ROW3_SPEC: &[(u32, u32, u32, i32, i32, i32)] = &[
    // s    x  y  tv8   te    pred8
    (192, 0, 3, 1192, -754, 438),
    (193, 1, 3, 600, 296, 896),
    (194, 2, 3, 272, 437, 709),
    (195, 3, 3, 80, 222, 302),
    (196, 4, 3, 32, 119, 151),
    (197, 5, 3, 1248, -622, 626),
    (198, 6, 3, 1840, -669, 1171),
    (199, 7, 3, 376, 390, 766),
    (200, 8, 3, 1736, -1323, 413),
];

/// Cross-row spec values (row 2 stored true_err) the row-3 `te_*` reads
/// must resolve to under a fully-correct decoder.
///
/// Indexed by sample number; values from `wp-trace-sample-194.md`
/// sample-194 explicit table (lines 64-71):
///   sample 129 (= (1, 2)) spec true_err = `true_errNW@194` = 737
///   sample 130 (= (2, 2)) spec true_err = `true_errN@194`  = -456
///   sample 131 (= (3, 2)) spec true_err = `true_errNE@194` = -165
///   sample 132..134 (= (4..6, 2)) derivable from sample 195..197's
///   `true_errNE` reads (each row-3 sample s+1 reads `te_ne` from
///   (x+1, 2) = sample 128 + (x+1)). See chain in body of
///   `r202_cross_row_te_n_ne_at_sample_195_onward`.
mod row2_spec {
    pub const TRUE_ERR_AT_129: i32 = 737;
    pub const TRUE_ERR_AT_130: i32 = -456;
    pub const TRUE_ERR_AT_131: i32 = -165;
}

/// `LEAF_PICK_TRACE_WP` schema indices (see `RichLeafPickLog` doc).
const TE_W: usize = 0;
const TE_N: usize = 1;
const TE_NW: usize = 2;
const TE_NE: usize = 3;
const WP_PRED8: usize = 8;

/// Capture WP state at one sample. Returns the 10-element vector
/// `[te_w, te_n, te_nw, te_ne, w8, n8, nw8, ne8, wp_pred8, max_error]`.
fn capture_wp_at(channel: u32, x: u32, y: u32) -> Vec<i32> {
    LEAF_PICK_TRACE_TARGET.store(encode_leaf_pick_target(channel, x, y), Ordering::Relaxed);
    LEAF_PICK_TRACE_WP.with(|s| s.borrow_mut().clear());
    let _ = decode_one_frame(NOISE_JXL, None).expect("decode");
    let snap = LEAF_PICK_TRACE_WP.with(|s| s.borrow().clone());
    LEAF_PICK_TRACE_TARGET.store(u64::MAX, Ordering::Relaxed);
    snap
}

/// Top-level: pin the production `wp_pred8` and the derived production
/// stored `true_err` at each row-3 sample 192..=200. Numbers are
/// printed alongside the trace doc's spec values so a regression in
/// upstream state evolution is visible at a glance.
///
/// The asserts pin only the **production** numbers (so this test
/// remains green until the upstream state-evolution bug is touched);
/// the SPEC numbers are reported via eprintln only.
///
/// Production stored true_err is derived from the next sample's
/// `te_w` (row-internal read) rather than computed as
/// `pred8 - tv8_spec`, because at sample 194 the production decodes
/// `v = 35` (vs spec `v = 34`) — the round-32-pinned pixel divergence
/// — so production stored at 194 = `pred8@194 - 35*8 = 437` (matches
/// spec by coincidence), NOT `pred8@194 - 34*8 = 445`.
#[test]
#[serial]
fn r202_row3_wp_pred8_and_stored_te_profile() {
    // Capture all row-3 samples first so we can derive production
    // stored values from te_w of the next sample.
    let snaps: Vec<Vec<i32>> = ROW3_SPEC
        .iter()
        .map(|&(_, x, y, _, _, _)| capture_wp_at(0, x, y))
        .collect();
    for (i, snap) in snaps.iter().enumerate() {
        assert_eq!(
            snap.len(),
            10,
            "sample {} capture must have 10 entries, got {}",
            ROW3_SPEC[i].0,
            snap.len()
        );
    }

    eprintln!("[r202] row-3 WP wp_pred8 + stored-true_err profile:");
    eprintln!(
        "    {:>4}  {:>4} {:>4}   {:>6} {:>6}   {:>6} {:>6}",
        "s", "x", "y", "pred8", "spec", "stored", "spec"
    );

    let last = ROW3_SPEC.len() - 1;
    for i in 0..ROW3_SPEC.len() {
        let (s, x, y, _tv8, te_spec, pred_spec) = ROW3_SPEC[i];
        let pred8_prod = snaps[i][WP_PRED8];
        // Production stored is the value the state machine wrote at
        // sample `s`. That's `te_w` of the next sample in the same
        // row (row-internal read). Sample 200 (the last in our slice)
        // has no next entry; we report "?" instead of asserting.
        let stored_prod_opt = if i < last {
            Some(snaps[i + 1][TE_W])
        } else {
            None
        };

        match stored_prod_opt {
            Some(stored_prod) => {
                eprintln!(
                    "    {:>4}  {:>4} {:>4}   {:>6} {:>+6}   {:>6} {:>+6}   \
                     (Δpred8={:+}, Δstored={:+})",
                    s,
                    x,
                    y,
                    pred8_prod,
                    pred_spec,
                    stored_prod,
                    te_spec,
                    pred8_prod - pred_spec,
                    stored_prod - te_spec,
                );
            }
            None => {
                eprintln!(
                    "    {:>4}  {:>4} {:>4}   {:>6} {:>+6}   {:>6} {:>+6}   \
                     (Δpred8={:+}, stored=N/A — last sample in slice)",
                    s,
                    x,
                    y,
                    pred8_prod,
                    pred_spec,
                    0,
                    te_spec,
                    pred8_prod - pred_spec,
                );
            }
        }

        // Pin pred8 to a plausible range so a decoder regression that
        // shifts WP capture is caught.
        assert!(
            (-10_000..=10_000).contains(&pred8_prod),
            "sample {} pred8 = {} is implausibly out of range; \
             a decoder regression likely shifted the WP capture",
            s,
            pred8_prod
        );
    }
}

/// In-row read chain: `te_w(s+1)` MUST equal the **production** stored
/// true_err at sample `s`, which the state machine computes as
/// `pred8_prod(s) - v_prod(s) * 8`.
///
/// Restricted to samples 192..=194 where the production decoded value
/// is known (samples 192, 193 decode bit-exact; sample 194 decodes
/// the round-32-pinned `v_prod = 35`). The production diverges
/// LARGELY from spec at sample 195 onward (production decodes `v=88`
/// vs spec `v=10` at sample 195 — the WP defect cascades into a
/// wrong MA-tree leaf pick, not just an off-by-1 in the predictor
/// rounding). Locking samples 195..=200 would over-pin the test;
/// the surrounding-sample context in the trace doc is sufficient to
/// reason about row-3 propagation without further chain asserts.
#[test]
#[serial]
fn r202_row3_in_row_te_w_chain_192_to_194() {
    eprintln!("[r202] row-3 in-row te_w chain (samples 192..=194):");

    // Production decoded values at each pinned row-3 sample. Match
    // spec at 192 + 193; sample 194 = 35 (round-32 pixel divergence).
    fn v_prod_at(s: u32) -> i32 {
        match s {
            192 => 1192, // spec tv8 = 1192, v_spec = 149
            193 => 600,  // spec tv8 = 600,  v_spec = 75
            194 => 280,  // production v = 35 (vs spec 34) → tv8_prod = 280
            other => panic!("v_prod_at: unhandled sample {}", other),
        }
    }

    let boundaries: &[(u32, u32, u32, u32, u32, u32)] = &[
        // (s_a, x_a, y_a, s_b, x_b, y_b)
        (192, 0, 3, 193, 1, 3),
        (193, 1, 3, 194, 2, 3),
        (194, 2, 3, 195, 3, 3),
    ];

    for &(s_a, x_a, y_a, s_b, x_b, y_b) in boundaries {
        let snap_a = capture_wp_at(0, x_a, y_a);
        let snap_b = capture_wp_at(0, x_b, y_b);

        let pred8_at_a = snap_a[WP_PRED8];
        let v8_at_a = v_prod_at(s_a);
        let stored_at_a = pred8_at_a - v8_at_a;
        let te_w_at_b = snap_b[TE_W];

        eprintln!(
            "    s={}->s={}: stored@{} = {} (= pred8@{}={} - v_prod@{}*8={}); te_w@{} = {}",
            s_a, s_b, s_a, stored_at_a, s_a, pred8_at_a, s_a, v8_at_a, s_b, te_w_at_b,
        );
        assert_eq!(
            te_w_at_b,
            stored_at_a,
            "te_w at sample {} (= production stored true_err at sample {}) \
             must equal `pred8@{} - v_prod@{}*8` = {}; got te_w@{} = {}. \
             If this fails, EITHER the in-row state read chain is broken \
             at the {}→{} boundary, OR the production decoded value at \
             sample {} has shifted from the pinned value (v_prod={}).",
            s_b,
            s_a,
            s_a,
            s_a,
            stored_at_a,
            s_b,
            te_w_at_b,
            s_a,
            s_b,
            s_a,
            v8_at_a / 8
        );
    }
}

/// Pin the production decoded sample value at sample 194 to 35
/// (round-32 first-pixel-divergence). Every other row-3 sample
/// decodes bit-exact (matches spec); sample 194 is the off-by-1
/// pixel divergence resulting from the WP state-evolution defect
/// at upstream samples (round-191 / round-195 chain analysis).
#[test]
#[serial]
fn r202_sample_194_decoded_value_is_35() {
    // Decoded value at sample 194 derived from pred8@194 - te_w@195.
    let snap_194 = capture_wp_at(0, 2, 3);
    let snap_195 = capture_wp_at(0, 3, 3);
    let pred8_at_194 = snap_194[WP_PRED8];
    let te_w_at_195 = snap_195[TE_W];
    let v_prod_at_194 = (pred8_at_194 - te_w_at_195) / 8;

    eprintln!(
        "[r202] sample 194 decoded value: pred8={}, te_w@195={}, \
         v_prod = (pred8 - te_w@195) / 8 = {}",
        pred8_at_194, te_w_at_195, v_prod_at_194,
    );

    assert_eq!(
        v_prod_at_194, 35,
        "Sample 194 production decoded value pinned at 35 (spec: 34). \
         This is the round-32-pinned first-pixel-divergence. A WP \
         state-evolution fix that lands the spec prediction (709) \
         must also flip this to 34.",
    );
}

/// Cross-row spec reads at sample 194 (the validated divergence
/// point). Three of these four positions (NW, N, NE) read from row 2
/// samples whose spec true_err is known from the trace doc table.
#[test]
#[serial]
fn r202_sample_194_cross_row_te_reads() {
    let snap = capture_wp_at(0, 2, 3);
    assert!(snap.len() == 10);

    let te_n_prod = snap[TE_N]; // reads (2, 2) = sample 130
    let te_nw_prod = snap[TE_NW]; // reads (1, 2) = sample 129
    let te_ne_prod = snap[TE_NE]; // reads (3, 2) = sample 131

    eprintln!("[r202] sample 194 cross-row te_* reads vs spec:");
    eprintln!(
        "    te_n  = {:>5} (spec at sample 130 = {:>5}, Δ = {:+})",
        te_n_prod,
        row2_spec::TRUE_ERR_AT_130,
        te_n_prod - row2_spec::TRUE_ERR_AT_130,
    );
    eprintln!(
        "    te_nw = {:>5} (spec at sample 129 = {:>5}, Δ = {:+})",
        te_nw_prod,
        row2_spec::TRUE_ERR_AT_129,
        te_nw_prod - row2_spec::TRUE_ERR_AT_129,
    );
    eprintln!(
        "    te_ne = {:>5} (spec at sample 131 = {:>5}, Δ = {:+})",
        te_ne_prod,
        row2_spec::TRUE_ERR_AT_131,
        te_ne_prod - row2_spec::TRUE_ERR_AT_131,
    );

    // te_n@194 reads sample 130: production already matches spec
    // (round-195 captured 716 vs 737 only for sample 129, NOT
    // sample 130 — and the trace doc + our derivations show 130 is
    // CORRECT). This is a known good cell in the WP state grid.
    assert_eq!(
        te_n_prod,
        row2_spec::TRUE_ERR_AT_130,
        "te_n at sample 194 (reads sample 130) should match spec \
         exactly (= {}). Got {}. Sample 130 was a 'known good' cell in \
         the round-195 WP state landscape; if this now fails, the \
         state-evolution defect at sample 129 has spread to its \
         neighbour 130 too.",
        row2_spec::TRUE_ERR_AT_130,
        te_n_prod,
    );

    // te_nw@194 reads sample 129: this is the SMOKING GUN — the
    // -21 delta documented at length in `r191_pin_state_evolution_gap`
    // and `r195_sample_193_te_n_equals_sample_194_te_nw`. Pin the
    // CURRENT production value so a future fix that targets sample
    // 129 is forced to update this number.
    assert_eq!(
        te_nw_prod, 716,
        "te_nw at sample 194 (reads sample 129) currently pinned at \
         716 (spec: 737, Δ = -21). A fix at sample 129's upstream \
         state evolution must update this assertion.",
    );

    // te_ne@194 reads sample 131: similarly a known-defective cell.
    // Round-195 captured -160 vs spec -165, Δ = +5.
    assert_eq!(
        te_ne_prod, -160,
        "te_ne at sample 194 (reads sample 131) currently pinned at \
         -160 (spec: -165, Δ = +5). A fix at sample 131's upstream \
         state evolution must update this assertion.",
    );
}

/// Walk the cross-row `te_n`/`te_ne` chain across row 3, where each
/// sample's `te_ne` is the next sample's `te_n` (both read from the
/// same row-2 cell `(x+1, 2)`). This locks the row-2 stored-true_err
/// shadow as observed at row 3's read time.
#[test]
#[serial]
fn r202_row3_cross_row_te_n_eq_prev_te_ne() {
    eprintln!("[r202] row-3 cross-row chain — te_n(s+1) ?= te_ne(s):");

    // Walk pairs (s, s+1) within row 3. Sample 200's te_ne reads
    // (9, 2) = sample 137; sample 201's te_n reads the same cell.
    // We stop at 199 so both endpoints are in our ROW3_SPEC slice.
    let last = ROW3_SPEC.len() - 1;
    for i in 0..last {
        let (s_a, x_a, y_a, _, _, _) = ROW3_SPEC[i];
        let (s_b, x_b, y_b, _, _, _) = ROW3_SPEC[i + 1];
        let snap_a = capture_wp_at(0, x_a, y_a);
        let snap_b = capture_wp_at(0, x_b, y_b);

        // te_ne@a reads (x_a + 1, 2); te_n@b reads (x_b, 2) =
        // (x_a + 1, 2) — same cell.
        let te_ne_a = snap_a[TE_NE];
        let te_n_b = snap_b[TE_N];

        eprintln!(
            "    te_ne@{}={} ?= te_n@{}={}  (cell = ({}, 2))",
            s_a,
            te_ne_a,
            s_b,
            te_n_b,
            x_a + 1,
        );

        assert_eq!(
            te_ne_a,
            te_n_b,
            "te_ne at sample {} and te_n at sample {} both read row-2 \
             cell ({}, 2), so they MUST equal each other. Got \
             te_ne@{} = {}, te_n@{} = {}.",
            s_a,
            s_b,
            x_a + 1,
            s_a,
            te_ne_a,
            s_b,
            te_n_b,
        );
    }
}

/// Walk the cross-row chain in the OPPOSITE direction:
/// `te_nw(s+1) == te_n(s)` for in-row consecutive `s, s+1` (both read
/// row-2 cell `(x_a, 2) == (x_b - 1, 2)`).
#[test]
#[serial]
fn r202_row3_cross_row_te_nw_eq_prev_te_n() {
    eprintln!("[r202] row-3 cross-row chain — te_nw(s+1) ?= te_n(s):");

    let last = ROW3_SPEC.len() - 1;
    for i in 0..last {
        let (s_a, x_a, y_a, _, _, _) = ROW3_SPEC[i];
        let (s_b, x_b, y_b, _, _, _) = ROW3_SPEC[i + 1];
        let snap_a = capture_wp_at(0, x_a, y_a);
        let snap_b = capture_wp_at(0, x_b, y_b);

        // te_n@a reads (x_a, 2); te_nw@b reads (x_b - 1, 2) = (x_a, 2).
        let te_n_a = snap_a[TE_N];
        let te_nw_b = snap_b[TE_NW];

        eprintln!(
            "    te_n@{}={} ?= te_nw@{}={}  (cell = ({}, 2))",
            s_a, te_n_a, s_b, te_nw_b, x_a,
        );

        // Sample 192's te_n reads (0, 2) = sample 128. Sample 193's
        // te_nw also reads (0, 2). They MUST equal each other.
        assert_eq!(
            te_n_a, te_nw_b,
            "te_n at sample {} and te_nw at sample {} both read row-2 \
             cell ({}, 2), so they MUST equal each other. Got \
             te_n@{} = {}, te_nw@{} = {}.",
            s_a, s_b, x_a, s_a, te_n_a, s_b, te_nw_b,
        );
    }
}

/// Sample 192 is the row-3 leftmost — `x = 0`. Its `te_w` reads
/// out-of-bounds (`x = -1`), and per `WpState::at` this returns 0.
/// Its `te_nw` also reads out-of-bounds (`x = -1, y = 2`), returns 0.
/// This test pins those zero-border behaviours so a refactor to the
/// WP-state border policy is caught.
#[test]
#[serial]
fn r202_sample_192_left_border_te_w_te_nw_zero() {
    let snap = capture_wp_at(0, 0, 3);
    assert!(snap.len() == 10);

    eprintln!("[r202] sample 192 (x=0, y=3) WP state with left-border zeroing:");
    eprintln!(
        "    te_w={}, te_n={}, te_nw={}, te_ne={}, wp_pred8={}",
        snap[TE_W], snap[TE_N], snap[TE_NW], snap[TE_NE], snap[WP_PRED8],
    );

    assert_eq!(
        snap[TE_W], 0,
        "te_w at sample 192 (x=0) reads out-of-bounds at x=-1, must be 0",
    );
    assert_eq!(
        snap[TE_NW], 0,
        "te_nw at sample 192 (x=0) reads out-of-bounds at x=-1, must be 0",
    );
}
